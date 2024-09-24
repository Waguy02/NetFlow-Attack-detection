import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold

from network_ad.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH

class LightGBMDataset:
    def __init__(self, train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH):
        self.train_path = train_path
        self.test_path = test_path
        self.categorical_features = CATEGORICAL_FEATURES
        self.numerical_features = NUMERICAL_FEATURES
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, mode):
        """Load training or test data."""
        if mode == 'train':
            df = pd.read_csv(self.train_path)
        elif mode == 'test':
            df = pd.read_csv(self.test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Ensure categorical features are strings
        for feature in CATEGORICAL_FEATURES:
            df[feature] = df[feature].astype(str)
        return df

    def setup(self):
        """Load and prepare the data for LightGBM."""
        # Load the train and test data
        train_df = self.load_data('train')
        test_df = self.load_data('test')

        # Separate the features and target (assuming the target is in a column called 'target')
        self.X_train = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        self.y_train = train_df['target']
        self.X_test = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        self.y_test = test_df['target']

    def get_lgb_dataset(self, X, y):
        """Create a LightGBM dataset from given features and labels."""
        return lgb.Dataset(X, label=y, categorical_feature=self.categorical_features)
