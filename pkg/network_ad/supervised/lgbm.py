import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from network_ad.config import LOGS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH
from network_ad.supervised.lgbm_dataset import LightGBMDataset


class LightGBMClassifier:
    def __init__(self,
                 multiclass=True,
                 num_leaves=16,
                 max_depth=-1,
                 num_estimators=100,
                 learning_rate=0.05,
                 n_kfold_splits=5,
                 log_dir= LOGS_DIR / "lgbm"):
        """Initialize with LightGBM parameters and K-Fold setup."""
        self.params = {
            'objective': 'multiclass' if multiclass else 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'num_estimators': num_estimators,
            'learning_rate': learning_rate,
        }
        self.n_splits = n_kfold_splits
        self.models = []
        self.best_iteration = []

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

    def train_kfold(self, X, y, dataset_class, save_validation_performance_path="validation_metrics.json"):
        """Perform K-fold cross-validation training and save validation performance."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        validation_results = []

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
            model = lgb.train(self.params, lgb_train,
                              valid_sets=[lgb_train, lgb_val],
                              early_stopping_rounds=10,
                              verbose_eval=False)

            # Save the model and best iteration
            self.models.append(model)
            self.best_iteration.append(model.best_iteration)

            # Evaluate on validation set
            val_preds = np.argmax(model.predict(X_val, num_iteration=model.best_iteration), axis=1)
            accuracy = accuracy_score(y_val, val_preds)
            validation_results.append({"fold": fold, "accuracy": accuracy})

            print(f"Finished fold {fold}/{self.n_splits}, accuracy: {accuracy}, best iteration: {model.best_iteration}")

            # Log confusion matrix to TensorBoard for each fold
            cm = confusion_matrix(y_val, val_preds)
            self._log_confusion_matrix(cm, f'Confusion Matrix - Fold {fold}')

            fold += 1

        # Save validation results in JSON format
        with open(save_validation_performance_path, 'w') as f:
            json.dump(validation_results, f, indent=4)

    def predict(self, X_test):
        """Make predictions using the average of all K-fold models."""
        preds = np.zeros((len(X_test), len(self.models[0].predict(X_test, num_iteration=self.models[0].best_iteration)[0])))

        for model in self.models:
            preds += model.predict(X_test, num_iteration=model.best_iteration)

        preds /= len(self.models)
        return np.argmax(preds, axis=1)  # Return the class with the highest probability

    def evaluate_test(self, X_test, y_test, save_test_performance_path="test_metrics.json"):
        """Evaluate the model on the test set and save the results."""
        test_preds = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, test_preds)

        # Save performance metrics to JSON file
        test_results = {"accuracy": accuracy}
        with open(save_test_performance_path, 'w') as f:
            json.dump(test_results, f, indent=4)

        # Log confusion matrix to TensorBoard
        cm = confusion_matrix(y_test, test_preds)
        self._log_confusion_matrix(cm, 'Test Confusion Matrix')

        print(f"Test accuracy: {accuracy}")

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

    def _log_confusion_matrix(self, cm, tag):
        """Log confusion matrix to TensorBoard."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(tag)

        # Convert matplotlib plot to TensorBoard image format
        self.writer.add_figure(tag, fig)
        plt.close(fig)



if __name__ == "__main__":
    MULTICLASS = True

    # 1. Load and prepare dataset
    dataset = LightGBMDataset(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        multiclass=MULTICLASS)
    dataset.setup()

    # 2. Initialize LightGBM classifier
    classifier = LightGBMClassifier(
        multiclass=True,
        num_leaves=16,
        max_depth=-1,
        num_estimators=100,
        log_dir=LOGS_DIR / "lgbm" # Custom TensorBoard log directory
    )

    # 3. Train with K-Fold cross-validation
    classifier.train_kfold(dataset.X_train, dataset.y_train, dataset)

    # 4. Evaluate the model on the test set
    classifier.evaluate_test(dataset.X_test, dataset.y_test)

    # 5. Save the model
    classifier.save_model("lightgbm_model")
