import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Define the constant lists of categorical and non-categorical features
from network_ad.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    NUMERICAL_STATS_PATH, CATEGORICAL_STATS_PATH


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
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        """
        Load data, statistics, and split into train/val datasets.
        """
        # 1. Load train data and preprocess it
        print("1. Loading and preprocessing training data...")
        df_train = pd.read_csv(self.train_path)

        # Ensure the dataframe has all required features
        all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        if not all(col in df_train.columns for col in all_features):
            missing_cols = [col for col in all_features if col not in df_train.columns]
            raise ValueError(f"Missing columns in the dataset: {missing_cols}")

        # Load statistics
        print("2. Loading statistics for numerical and categorical features...")
        numerical_stats = pd.read_csv(self.numerical_stats_path, index_col=0)
        self.means = numerical_stats['mean'].values
        self.stds = numerical_stats['std'].values

        with open(self.categorical_stats_path, 'r') as f:
            categorical_stats = json.load(f)

        # Create one-hot encoders for each categorical feature
        for feature in CATEGORICAL_FEATURES:
            self.categorical_encoders[feature] = OneHotEncoder(sparse=False, categories=[categorical_stats[feature]], handle_unknown="ignore")

        # Process train data (categorical + numerical features)
        train_data = self._encode_features(df_train)

        # Split the train data into training and validation sets (90% train, 10% validation)
        X_train, X_val = train_test_split(train_data, test_size=self.val_ratio, random_state=42)

        # Convert to tensors
        self.train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        self.val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

        # 2. Load and preprocess test data
        print("3. Loading and preprocessing test data...")
        df_test = pd.read_csv(self.test_path)

        test_data = self._encode_features(df_test)
        self.test_data = TensorDataset(torch.tensor(test_data, dtype=torch.float32))

    def _encode_features(self, df):
        """Preprocess data by encoding categorical features and scaling numerical features."""
        # One-hot encode categorical features
        categorical_data_list = []
        for feature in CATEGORICAL_FEATURES:
            encoded_feature = self.categorical_encoders[feature].fit_transform(df[[feature]])
            categorical_data_list.append(encoded_feature)
        categorical_encoded = np.hstack(categorical_data_list)

        # Normalize numerical features with precomputed stats
        numerical_data = df[NUMERICAL_FEATURES]
        numerical_scaled = (numerical_data - self.means) / self.stds

        # Concatenate numerical and one-hot encoded categorical features
        features = np.hstack((numerical_scaled, categorical_encoded))

        return features

    def train_dataloader(self):
        """Returns the training data loader."""
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Returns the validation data loader."""
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        """Returns the test data loader."""
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
        print(f"Train Batch X: {batch[0].shape}")
        break  # Only show one batch

    # Test the val_dataloader
    print("Validation DataLoader:")
    for batch in data_module.val_dataloader():
        print(f"Validation Batch X: {batch[0].shape}")
        break  # Only show one batch

    # Test the test_dataloader
    print("Test DataLoader:")
    for batch in data_module.test_dataloader():
        print(f"Test Batch X: {batch[0].shape}")
        break  # Only show one batch
