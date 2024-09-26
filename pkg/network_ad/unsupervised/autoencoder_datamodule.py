import sys
sys.path.append("../..")
from typing import List
import h5pickle as h5py
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from network_ad.config import AE_CATEGORICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    NUMERICAL_STATS_PATH, CATEGORICAL_STATS_PATH, PREPROCESSED_DATA_PATH
from network_ad.config import AE_PREPROCESSED_DATA_PATH

class AutoencoderDataset(Dataset):
    def __init__(self, hdf5_path, means, stds, mode: str):
        self.hdf5_path = hdf5_path
        self.means = means
        self.stds = stds
        self.h5file = h5py.File(self.hdf5_path, 'r')
        self.mode = mode

    def __len__(self):
        return self.h5file[self.mode].shape[0]

    def __getitem__(self, idx):
        """Return the features for a single sample."""
        features = self.h5file[self.mode][idx]
        return torch.tensor(features)

    def close(self):
        self.h5file.close()  # Ensure to close the HDF5 file when done


class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, val_ratio=0.1, num_workers=0):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
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
        elif mode == 'test':
            df = pd.read_csv(self.test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Ensure that categorical features are read as strings
        for feature in AE_CATEGORICAL_FEATURES:
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
        preprocessed_h5_path = AE_PREPROCESSED_DATA_PATH

        if not preprocessed_h5_path.exists():
            raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_h5_path}.  Please run "
                                    " the script netflow_v2_preprocessing_autoencoder_data.py to preprocess the data.")



        # Create dataset objects for training, validation, and test
        self.train_data = AutoencoderDataset(str(preprocessed_h5_path), self.means, self.stds, mode='train')
        self.val_data = AutoencoderDataset(str(preprocessed_h5_path), self.means, self.stds, mode='val')
        self.test_data = AutoencoderDataset(str(preprocessed_h5_path), self.means, self.stds, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
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
    VAL_RATIO = 0.1

    # Initialize the DataModule
    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE, val_ratio=VAL_RATIO)

    # Setup the data (loads, preprocesses, and splits the data)
    data_module.setup()

    # Test the train_dataloader
    print("Train DataLoader:")
    for batch in data_module.train_dataloader():
        print(f"Train Batch X: {batch.shape}")
        break  # Only show one batch

    # Test the val_dataloader
    print("Validation DataLoader:")
    for batch in data_module.val_dataloader():
        print(f"Validation Batch X: {batch.shape}")
        break  # Only show one batch

    # Test the test_dataloader
    print("Test DataLoader:")
    for batch in data_module.test_dataloader():
        print(f"Test Batch X: {batch.shape}")
        break  # Only show one batch
