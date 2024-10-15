import sys
sys.path.append("../..")
import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from network_ad.supervised.utils import compute_confusion_matrix
from network_ad.config import LOGS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, MULTIClASS_CLASS_NAMES, BINARY_CLASS_NAMES
from network_ad.supervised.lgbm_dataset import LightGBMDataset


class LightGBMClassifier:
    def __init__(self,
                 multiclass=True,
                 num_leaves=16,
                 max_depth=-1,
                 num_estimators=100,
                 learning_rate=0.05,
                 log_dir=LOGS_DIR / "lgbm"):
        """Initialize with LightGBM parameters."""
        self.params = {
            'objective': 'multiclass' if multiclass else 'binary',
            'metric': 'multi_logloss' if multiclass else 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'n_estimators': num_estimators,
            'learning_rate': learning_rate,
        }
        self.model = None
        self.multiclass = multiclass
        self.class_labels = MULTIClASS_CLASS_NAMES if multiclass else BINARY_CLASS_NAMES

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

    def train(self, X, y, X_val=None, y_val=None, save_training_performance_path="training_metrics.json"):
        """Train the model on the full dataset with an optional validation set and save training performance."""
        self.model = lgb.LGBMClassifier(**self.params, class_weight='balanced', n_jobs=-1)

        # Train the model with optional validation data
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="multi_logloss" if self.multiclass else "binary_logloss")
        else:
            self.model.fit(X, y)

        # Evaluate on training set
        train_preds = self.model.predict(X)
        accuracy = accuracy_score(y, train_preds)

        # Calculate F1 Score, Precision, and Recall
        labels = np.unique(y)  # Get unique labels in the training set
        classification_metrics = classification_report(y, train_preds,
                                                       labels=labels,
                                                       output_dict=True)
        f1_weighted = classification_metrics['weighted avg']['f1-score']
        f1_macro = classification_metrics['macro avg']['f1-score']
        precision = classification_metrics['weighted avg']['precision']
        recall = classification_metrics['weighted avg']['recall']

        training_results = {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall
        }

        # Save training results in JSON format
        with open(save_training_performance_path, 'w') as f:
            json.dump(training_results, f, indent=4)

        # Log confusion matrix to TensorBoard for training set
        cm = compute_confusion_matrix(y, train_preds, self.class_labels)
        self._log_confusion_matrix(cm, 'Training Confusion Matrix')

        # Cm normalized
        cm_norm = compute_confusion_matrix(y, train_preds, self.class_labels, normalize=True)
        self._log_confusion_matrix(cm_norm, 'Normalized Training Confusion Matrix', normalize=True)

        # Validation set performance logging if provided
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_preds)
            val_metrics = classification_report(y_val, val_preds, labels=np.unique(y_val), output_dict=True)
            val_f1_weighted = val_metrics['weighted avg']['f1-score']
            val_f1_macro = val_metrics['macro avg']['f1-score']
            val_precision = val_metrics['weighted avg']['precision']
            val_recall = val_metrics['weighted avg']['recall']

            print(f"Validation accuracy: {val_accuracy}, f1_weighted: {val_f1_weighted}, f1_macro: {val_f1_macro}, precision: {val_precision}, recall: {val_recall}")

            # Log validation confusion matrix to TensorBoard
            val_cm = compute_confusion_matrix(y_val, val_preds, self.class_labels)
            self._log_confusion_matrix(val_cm, 'Validation Confusion Matrix')

            # Cm normalized
            val_cm_norm = compute_confusion_matrix(y_val, val_preds,  labels=self.class_labels, normalize=True)
            self._log_confusion_matrix(val_cm_norm, 'Normalized Validation Confusion Matrix', normalize=True)

            with open(self.log_dir / "val_metrics.json", 'w') as f:
                json.dump({
                    "accuracy": val_accuracy,
                    "f1_weighted": val_f1_weighted,
                    "f1_macro": val_f1_macro,
                    "precision": val_precision,
                    "recall": val_recall
                }, f, indent=4)



    def evaluate_test(self, X_test, y_test):
        """Evaluate the model on the test set and save the results."""
        test_preds = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, test_preds)

        # Calculate F1 Score, Precision, and Recall
        labels = np.unique(y_test)  # Get unique labels in the test set
        classification_metrics = classification_report(y_test, test_preds,
                                                       labels=labels,
                                                       output_dict=True)
        f1_weighted = classification_metrics['weighted avg']['f1-score']
        f1_macro = classification_metrics['macro avg']['f1-score']
        precision = classification_metrics['weighted avg']['precision']
        recall = classification_metrics['weighted avg']['recall']

        # Save performance metrics to JSON file
        test_results = {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall
        }

        with open(self.log_dir / "test_metrics.json", 'w') as f:
            json.dump(test_results, f, indent=4)
        # Log confusion matrix to TensorBoard
        cm = compute_confusion_matrix(y_test, test_preds, self.class_labels)
        self._log_confusion_matrix(cm, 'Test Confusion Matrix')

        # Cm normalized
        cm_norm = compute_confusion_matrix(y_test, test_preds, self.class_labels, normalize=True)
        self._log_confusion_matrix(cm_norm, 'Normalized Test Confusion Matrix', normalize=True)

        print(f"Test accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}, precision: {precision}, recall: {recall}")

    def save_model(self, path):
        """Save the trained model."""
        self.model.booster_.save_model(path)

    def load_model(self, path):
        """Load a trained model."""
        self.model = lgb.Booster(model_file=path)

    def _log_confusion_matrix(self, fig, tag, normalize=False):
        """Log confusion matrix to TensorBoard."""
        # Convert matplotlib plot to TensorBoard image format
        self.writer.add_figure(tag, fig)
        plt.close(fig)


if __name__ == "__main__":
    MULTICLASS = False
    NUM_LEAVES = 20
    NUM_ESTIMATORS = 100
    MAX_DEPTH = -1

    # 1. Load and prepare dataset
    dataset = LightGBMDataset(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        multiclass=MULTICLASS)
    dataset.setup()

    # 2. Initialize LightGBM classifiera
    classifier = LightGBMClassifier(
        multiclass=MULTICLASS,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        num_estimators=NUM_ESTIMATORS,
        log_dir=LOGS_DIR / f"lgbm_nl.{NUM_LEAVES}_e.{NUM_ESTIMATORS}_{'multiclass' if MULTICLASS else 'binary'}"
    )

    # 3. Train on the full training dataset with validation set
    classifier.train(dataset.X_train, dataset.y_train, X_val=dataset.X_val, y_val=dataset.y_val)

    # 4. Evaluate the model on the test set
    classifier.evaluate_test(dataset.X_test, dataset.y_test)

    # 5. Save the model
    classifier.save_model(classifier.log_dir / "model.txt")
