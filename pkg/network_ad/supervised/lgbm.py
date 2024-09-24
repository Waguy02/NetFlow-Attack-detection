from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np


class LightGBMClassifier:
    def __init__(self, params=None, n_splits=5):
        """Initialize with LightGBM parameters and K-Fold setup."""
        if params is None:
            params = {
                'objective': 'binary',  # Assuming binary classification
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
        self.params = params
        self.n_splits = n_splits
        self.models = []
        self.best_iteration = []

    def train_kfold(self, X, y, dataset_class):
        """Perform K-fold cross-validation training."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold = 1

        for train_index, val_index in kf.split(X):
            print(f"Training fold {fold}/{self.n_splits}...")

            # Split data into train and validation sets for this fold
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Create LightGBM datasets
            lgb_train = dataset_class.get_lgb_dataset(X_train, y_train)
            lgb_val = dataset_class.get_lgb_dataset(X_val, y_val)

            # Train LightGBM model
            model = lgb.train(self.params, lgb_train, valid_sets=[lgb_train, lgb_val],
                              early_stopping_rounds=10, verbose_eval=False)

            # Save the model and best iteration
            self.models.append(model)
            self.best_iteration.append(model.best_iteration)

            print(f"Finished fold {fold}/{self.n_splits}, best iteration: {model.best_iteration}")
            fold += 1

    def predict(self, X_test):
        """Make predictions using the average of all K-fold models."""
        preds = np.zeros(len(X_test))

        for model in self.models:
            preds += model.predict(X_test, num_iteration=model.best_iteration)

        preds /= len(self.models)
        return preds

    def save_model(self, path):
        """Save the trained model."""
        for i, model in enumerate(self.models):
            model.save_model(f"{path}_fold_{i}.txt")

    def load_model(self, path, n_splits):
        """Load the trained model."""
        self.models = []
        for i in range(n_splits):
            model = lgb.Booster(model_file=f"{path}_fold_{i}.txt")
            self.models.append(model)
