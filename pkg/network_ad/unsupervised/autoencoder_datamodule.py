import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json
import pytorch_lightning as pl
# Define the constant lists of categorical and non-categorical features
from network_ad.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    NUMERICAL_STATS_PATH, CATEGORICAL_STATS_PATH, SEED
from network_ad.preprocessing.netflow_v2_preprocessing import train_test_split


class AutoencoderDataset(Dataset):
    def __init__(self, data_df, means, stds, categorical_encoders):
        self.df = data_df
        # Ensure categorical features are strings
        for col in CATEGORICAL_FEATURES:
            self.df[col] = self.df[col].astype(str)

        self.means = means
        self.stds = stds
        self.categorical_encoders = categorical_encoders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return the encoded features for a single sample."""
        sample = self.df.iloc[idx]

        # One-hot encode categorical features
        categorical_data_list = []
        for feature in CATEGORICAL_FEATURES:
            encoded_feature = self.categorical_encoders[feature].transform([[sample[feature]]])
            categorical_data_list.append(encoded_feature)
        categorical_encoded = np.hstack(categorical_data_list).squeeze()

        # Normalize numerical features
        numerical_data = sample[NUMERICAL_FEATURES].values
        numerical_scaled = (numerical_data - self.means) / self.stds

        # Concatenate numerical and one-hot encoded categorical features
        features = np.hstack((numerical_scaled, categorical_encoded)).astype(np.float32)

        return torch.tensor(features)


class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, val_ratio=0.1):
        super().__init__()
        self.train_path = TRAIN_DATA_PATH
        self.test_path = TEST_DATA_PATH
        self.numerical_stats_path = NUMERICAL_STATS_PATH
        self.categorical_stats_path = CATEGORICAL_STATS_PATH
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.means = None
        self.stds = None
        self.categorical_encoders = {}

    def load_data(self, mode):
        if mode == 'train':
            return pd.read_csv(self.train_path)
        elif mode == 'test':
            return pd.read_csv(self.test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def setup(self, stage=None):
        """Load data and set up the encoders and statistics."""
        # Load statistics
        print("Loading statistics for numerical and categorical features...")
        numerical_stats = pd.read_csv(self.numerical_stats_path, index_col=0)
        self.means = numerical_stats['mean'].values
        self.stds = numerical_stats['std'].values

        with open(self.categorical_stats_path, 'r') as f:
            categorical_stats = json.load(f)

        # Create one-hot encoders for each categorical feature
        for feature in CATEGORICAL_FEATURES:
            self.categorical_encoders[feature] = OneHotEncoder(sparse=False, categories=[categorical_stats[feature]],
                                                               handle_unknown="ignore")
            self.categorical_encoders[feature].fit(np.array(categorical_stats[feature]).reshape(-1, 1))

        # Train and validation datasets
        print("Loading training data...")
        df_train = self.load_data('train')
        train_df, val_df = train_test_split(df_train, test_ratio=self.val_ratio)

        # Create dataset objects for training and validation
        self.train_data = AutoencoderDataset(train_df, self.means, self.stds, self.categorical_encoders)
        self.val_data = AutoencoderDataset(val_df, self.means, self.stds, self.categorical_encoders)

        # Test dataset
        print("Loading test data...")
        df_test = self.load_data('test')
        self.test_data = AutoencoderDataset(df_test, self.means, self.stds, self.categorical_encoders)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


# Main function to test the DataModule
if __name__ == "__main__":
    # Define the batch size and validation ratio
    BATCH_SIZE = 32
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
