import sys
sys.path.append("../..")
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from network_ad.config import AE_CATEGORICAL_FEATURES, AE_NUMERICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    BINARY_LABEL_COLUMN, MULTICLASS_LABEL_COLUMN,VAL_RATIO

class LightGBMDataset:
    def __init__(self, train_path=TRAIN_DATA_PATH,
                 test_path=TEST_DATA_PATH,
                 multiclass=False):
        self.train_path = train_path
        self.test_path = test_path
        self.categorical_features = AE_CATEGORICAL_FEATURES
        self.numerical_features = AE_NUMERICAL_FEATURES
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.multiclass = multiclass
        self.target_variable = MULTICLASS_LABEL_COLUMN if multiclass else BINARY_LABEL_COLUMN
        self.label_encoders = {feature: LabelEncoder() for feature in self.categorical_features}  # Encoders for each feature

    def load_data(self, mode):
        """Load training or test data."""
        if mode == 'train':
            df = pd.read_csv(self.train_path)
            num_train_samples = len(df)*(1-VAL_RATIO)
            df = df[:int(num_train_samples)]
        elif mode == 'val':
            df = pd.read_csv(self.train_path)
            num_train_samples = len(df)*(VAL_RATIO)
            df = df[int(num_train_samples):]
        elif mode == 'test':
            df = pd.read_csv(self.test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Ensure categorical features are strings
        for feature in self.categorical_features:
            df[feature] = df[feature].astype(str)
        return df

    def encode_categorical(self, df):
        """Encode the categorical features as integers using LabelEncoder."""
        for feature in self.categorical_features:
            df[feature] = self.label_encoders[feature].transform(df[feature])
        return df

    def setup(self):
        """Load and prepare the data for LightGBM."""
        # Load the train and test data
        train_df = self.load_data('train')
        val_df = self.load_data('val')
        test_df = self.load_data('test')

        # Fit the label encoders
        for feature in self.categorical_features:
            all_values = pd.concat([train_df[feature], val_df[feature], test_df[feature]])
            self.label_encoders[feature].fit(all_values)

        # Encode categorical features
        train_df = self.encode_categorical(train_df)
        val_df = self.encode_categorical(val_df)
        test_df = self.encode_categorical(test_df)

        # Separate the features and target
        self.X_train = train_df[self.numerical_features + self.categorical_features]
        self.y_train = train_df[self.target_variable]

        self.X_val = val_df[self.numerical_features + self.categorical_features]
        self.y_val = val_df[self.target_variable]

        self.X_test = test_df[self.numerical_features + self.categorical_features]
        self.y_test = test_df[self.target_variable]

    def get_lgb_dataset(self, X, y):
        """Create a LightGBM dataset from given features and labels."""
        return lgb.Dataset(X, label=y, categorical_feature=self.categorical_features)
