import sys
sys.path.append("../..")


from collections import Counter
import numpy as np
from network_ad.config import VAL_RATIO
from network_ad.config import MULTIClASS_CLASS_LABELS_TO_ID
from typing import List
import h5pickle as h5py
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from network_ad.config import MLP_CATEGORICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    NUMERICAL_STATS_PATH, CATEGORICAL_STATS_PATH, PREPROCESSED_DATA_PATH
from network_ad.config import MLP_PREPROCESSED_DATA_PATH






class MLP_Dataset(Dataset):
    def __init__(self,
                 hdf5_path,
                 label_type, #binary or multiclass
                 means,
                 stds,
                 mode: str):
        self.hdf5_path = hdf5_path
        self.means = means
        self.stds = stds
        self.h5file = h5py.File(self.hdf5_path, 'r')
        self.mode = mode
        self.label_type = label_type

        self.binary_labels = []
        self.multiclass_labels = []


        if self.mode == 'train':
            df = pd.read_csv(TRAIN_DATA_PATH)
            nb_train_samples = len(df) * (1-VAL_RATIO)
            df = df[:int(nb_train_samples)]
            self.binary_labels = df['Label'].tolist()
            self.multiclass_labels = df['Attack'].tolist()
        elif self.mode == 'val':
            df = pd.read_csv(TRAIN_DATA_PATH)
            nb_train_samples = len(df) * (1-VAL_RATIO)
            df = df[int(nb_train_samples):]
            self.binary_labels = df['Label'].tolist()
            self.multiclass_labels = df['Attack'].tolist()
        elif self.mode == 'test':
            df = pd.read_csv(TEST_DATA_PATH)
            self.binary_labels = df['Label'].tolist()
            self.multiclass_labels = df['Attack'].tolist()
        self.compute_class_weights()

    def compute_class_weights(self):
        """
        Computes class weights inversely proportional to the class frequencies for both binary and multiclass labels.
        Returns:
            tuple: (binary_class_weights, multiclass_class_weights), both as torch tensors.
        """

        # Compute weights for binary labels
        if self.label_type == "binary":
            binary_counts = Counter(self.binary_labels)
            total_binary = sum(binary_counts.values())
            binary_weights = {label: total_binary / count for label, count in binary_counts.items()}


            # Normalize to ensure average weight is 1
            mean_binary_weight = np.mean(list(binary_weights.values()))
            binary_weights = {label: weight / mean_binary_weight for label, weight in binary_weights.items()}

            # Convert to tensor and reorder by class label (assuming binary classes are 0 and 1)
            self.binary_class_weights = torch.tensor([binary_weights[i] for i in sorted(binary_weights.keys())],
                                                dtype=torch.float32)

        else:
            # Compute weights for multiclass labels
            multiclass_counts = Counter(self.multiclass_labels)
            total_multiclass = sum(multiclass_counts.values())
            multiclass_weights = {label: total_multiclass / count for label, count in multiclass_counts.items()}

            # Normalize multiclass weights
            mean_multiclass_weight = np.mean(list(multiclass_weights.values()))
            self.multiclass_weights = {label: weight / mean_multiclass_weight for label, weight in multiclass_weights.items()}



    def __len__(self):

        return self.h5file[self.mode].shape[0]

    def __getitem__(self, idx):
        """Return the features for a single sample."""
        features = self.h5file[self.mode][idx]
        #Load the file to get the labels

        if self.label_type == "binary":
            binary_label = self.binary_labels[idx]
            label = torch.tensor([binary_label], dtype=torch.float32)
        else:
            multiclass_label = self.multiclass_labels[idx]
            label = torch.tensor(MULTIClASS_CLASS_LABELS_TO_ID[multiclass_label])
        return torch.tensor(features), label

    def close(self):
        self.h5file.close()  # Ensure to close the HDF5 file when done


class MLP_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, val_ratio=0.1, num_workers=0, label_type = "binary"):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        assert label_type in ["binary", "multiclass"], f"Invalid label type: {label_type}. Must be 'binary' or 'multiclass'."

        self.label_type = label_type
        self.train_path = TRAIN_DATA_PATH
        self.test_path = TEST_DATA_PATH
        self.numerical_stats_path = NUMERICAL_STATS_PATH
        self.categorical_stats_path = CATEGORICAL_STATS_PATH
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.means = None
        self.stds = None
        self.categorical_encoders = {}
        self.num_workers = num_workers
        self.binary_labels_by_mode = {}
        self.multiclass_labels_by_mode = {}

    def load_data(self, mode):
        if mode == 'train':
            df = pd.read_csv(self.train_path)
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[:int(nb_train_samples)]
        elif mode == 'val':
            df = pd.read_csv(self.train_path)
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[int(nb_train_samples):]
        elif mode == 'test':
            df = pd.read_csv(self.test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Ensure that categorical features are read as strings
        for feature in MLP_CATEGORICAL_FEATURES:
            df[feature] = df[feature].astype(str)
        return df

    def get_binary_labels(self, mode)->List:
        """
        Only for visualization purposes.
        The training of the autoencoder is unsupervised.

        :param mode:
        :return:
        """
        if mode in self.binary_labels_by_mode:
            return self.binary_labels_by_mode[mode]

        if mode == 'train':
            df = pd.read_csv(self.train_path, usecols=['Label'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[:int(nb_train_samples)]
            df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
            self.binary_labels_by_mode[mode] = df['Label'].tolist()
            return df['Label'].tolist()

        elif mode == 'val':
            df = pd.read_csv(self.train_path, usecols=['Label'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[int(nb_train_samples):]
            df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
            self.binary_labels_by_mode[mode] = df['Label'].tolist()
            return df['Label'].tolist()
        elif mode == 'test':
            df = pd.read_csv(self.test_path, usecols=['Label'])
            df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
            self.binary_labels_by_mode[mode] = df['Label'].tolist()
            return df['Label'].tolist()

    def get_multiclass_labels(self, mode)->List:
        """
        Only for visualization purposes.
        :param mode:
        :return:
        """
        if mode in self.multiclass_labels_by_mode:
            return self.multiclass_labels_by_mode[mode]

        if mode == 'train':
            df = pd.read_csv(self.train_path, usecols=['Attack'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[:int(nb_train_samples)]
            self.multiclass_labels_by_mode[mode] = df['Attack'].tolist()
            return df['Attack'].tolist()

        elif mode == 'val':
            df = pd.read_csv(self.train_path, usecols=['Attack'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[int(nb_train_samples):]
            self.multiclass_labels_by_mode[mode] = df['Attack'].tolist()
            return df['Attack'].tolist()

        elif mode == 'test':
            df = pd.read_csv(self.test_path, usecols=['Attack'])
            self.multiclass_labels_by_mode[mode] = df['Attack'].tolist()
            return df['Attack'].tolist()



    def setup(self, stage=None):
        preprocessed_h5_path = MLP_PREPROCESSED_DATA_PATH

        if not preprocessed_h5_path.exists():
            raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_h5_path}.  Please run "
                                    " the script netflow_v2_preprocessing_autoencoder_data.py to preprocess the data.")



        # Create dataset objects for training, validation, and test
        self.train_data = MLP_Dataset(str(preprocessed_h5_path),self.label_type, self.means, self.stds, mode='train')
        self.val_data = MLP_Dataset(str(preprocessed_h5_path),self.label_type, self.means, self.stds, mode='val')
        self.test_data = MLP_Dataset(str(preprocessed_h5_path),self.label_type, self.means, self.stds, mode='test')

    def train_dataloader(self):
        #Balance the classes
        if self.label_type == "binary":
            weights = [self.train_data.binary_class_weights[label] for label in self.train_data.binary_labels]
        else :
            weights = [self.train_data.multiclass_weights[label] for label in self.train_data.multiclass_labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        return DataLoader(self.train_data, batch_size=self.batch_size,  num_workers=self.num_workers,
                          sampler=sampler,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=self.num_workers > 0)

    def teardown(self, stage):
        if stage == 'test':
            self.test_data.close()
            self.train_data.close()
            self.val_data.close()



# Main function to test the DataModule
if __name__ == "__main__":
    # Define the batch size and validation ratio
    BATCH_SIZE = 64

    # Initialize the DataModule
    data_module = MLP_DataModule(batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, label_type="multiclass")

    # Setup the data (loads, preprocesses, and splits the data)
    data_module.setup()

    # Test the train_dataloader
    print("Train DataLoader:")
    for batch in data_module.train_dataloader():
        features, labels = batch
        print(f"First batch", batch)
        print (f"Train Batch X: {features.shape}", f"Train Batch Y: {labels.shape}")
        break  # Only show one batch

    # Test the val_dataloader
    print("Validation DataLoader:")
    for batch in data_module.val_dataloader():
        features, labels = batch
        print(f"First batch", batch)
        print(f"Validation Batch X: {features.shape}", f"Validation Batch Y: {labels.shape}")
        break
    # Test the test_dataloader
    print("Test DataLoader:")
    for batch in data_module.test_dataloader():
        features, labels = batch
        print(f"First batch", batch)
        print(f"Test Batch X: {features.shape}", f"Test Batch Y: {labels.shape}")
        break
