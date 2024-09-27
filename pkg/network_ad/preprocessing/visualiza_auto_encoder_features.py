# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import json

# Step 2: Load data paths and config variables
# Replace with actual paths in your environment
from network_ad.config import AE_CATEGORICAL_FEATURES, AE_NUMERICAL_FEATURES, NUMERICAL_STATS_PATH, TRAIN_DATA_PATH, \
    CATEGORICAL_STATS_PATH

# Load preprocessed numerical stats (e.g., means, stds)
numerical_stats = pd.read_csv(NUMERICAL_STATS_PATH, index_col=0)
means = numerical_stats['mean'].values
stds = numerical_stats['std'].values

# Step 3: Load the training data
df_train = pd.read_csv(TRAIN_DATA_PATH)


# Step 4: Function to visualize numerical features
def visualize_numerical_features(df, numerical_features):
    """Visualize numerical features with histograms."""
    print("Visualizing numerical features...")
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f'Histogram for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


# Step 5: Function to visualize one-hot encoded categorical features
def visualize_categorical_features(df, categorical_features, categorical_encoders):
    """Visualize one-hot encoded categorical features with bar plots."""
    print("Visualizing categorical features...")
    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        encoder = categorical_encoders[feature]
        transformed_data = encoder.transform(df[feature].values.reshape(-1, 1))
        categories = encoder.categories_[0]

        category_counts = np.sum(transformed_data, axis=0)  # Sum the one-hot encoded columns to get counts per category

        plt.bar(categories, category_counts)
        plt.title(f'One-hot Encoded Counts for {feature}')
        plt.xlabel(f'Categories of {feature}')
        plt.ylabel('Counts')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()


# Step 6: Load categorical encoders

if __name__ == "__main__":

    # Prepare OneHotEncoders for categorical features
    categorical_encoders = {}
    with open(CATEGORICAL_STATS_PATH, 'r') as f:
        categorical_stats = json.load(f)

    for feature in AE_CATEGORICAL_FEATURES:
        categorical_encoders[feature] = OneHotEncoder(sparse=False,
                                                      categories=[categorical_stats[feature]],
                                                      handle_unknown="ignore")
        categorical_encoders[feature].fit(np.array(categorical_stats[feature]).reshape(-1, 1))

    # Step 7: Visualize numerical features
    visualize_numerical_features(df_train, AE_NUMERICAL_FEATURES)

    # Step 8: Visualize categorical features
    visualize_categorical_features(df_train, AE_CATEGORICAL_FEATURES, categorical_encoders)
