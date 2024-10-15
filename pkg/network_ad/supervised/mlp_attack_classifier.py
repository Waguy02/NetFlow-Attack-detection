import sys
sys.path.append("../..")

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
from network_ad.config import MULTIClASS_CLASS_NAMES, BINARY_CLASS_NAMES, LOGS_DIR
from network_ad.supervised.utils import compute_confusion_matrix
from network_ad.config import VAL_RATIO
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import torch
from network_ad.supervised.mlp_datamodule import MLP_DataModule


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[128, 64], learning_rate=1e-3, dropout=0.2,
                 multiclass=True, max_training_steps = None
                 ):
        super(MLPClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.layers = nn.ModuleList()
        self.multiclass = multiclass

        if self.multiclass:
            self.output_dim = len(MULTIClASS_CLASS_NAMES)
        else:
            self.output_dim = 2

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], self.output_dim)
        )

        # Accuracy and other metrics
        self.train_acc = Accuracy(task="multiclass",num_classes=self.output_dim)
        self.val_acc = Accuracy(task="multiclass",num_classes=self.output_dim)
        self.validation_steps_outputs = []
        self.test_steps_outputs = []
        self.max_training_steps = max_training_steps
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        # Calculate F1 Scores
        f1_weighted = f1_score(y.cpu(), preds.cpu(), average='weighted')
        f1_macro = f1_score(y.cpu(), preds.cpu(), average='macro')

        # Log metrics
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1_weighted', f1_weighted, on_step=True)
        self.log('val_f1_macro', f1_macro, on_step=True)

        self.validation_steps_outputs.append({'val_loss': val_loss, 'val_acc': acc, 'preds': preds, 'targets': y})

        return {'val_loss': val_loss, 'val_acc': acc, 'preds': preds, 'targets': y}

    def on_validation_epoch_end(self):
        outputs=self.validation_steps_outputs
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])

        class_labels = MULTIClASS_CLASS_NAMES if self.multiclass else BINARY_CLASS_NAMES

        # Confusion Matrix
        cm_fig= compute_confusion_matrix(targets.cpu(), preds.cpu(), class_labels)
        self.logger.experiment.add_figure('Confusion Matrix', cm_fig, self.current_epoch)


        # Normalized Confusion Matrix
        cm_fig_norm = compute_confusion_matrix(targets.cpu(), preds.cpu(), class_labels, normalize=True)
        self.logger.experiment.add_figure('Normalized Confusion Matrix', cm_fig_norm, self.current_epoch)

        self.validation_steps_outputs = []

        # Compute F1 Scores
        f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted')
        self.log('val_f1_weighted_epoch', f1_weighted, on_epoch=True, prog_bar=True)

        f1_macro = f1_score(targets.cpu(), preds.cpu(), average='macro')
        self.log('val_f1_macro_epoch', f1_macro, on_epoch=True, prog_bar=True)




    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        test_loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        # Calculate F1 Scores
        f1_weighted = f1_score(y.cpu(), preds.cpu(), average='weighted')
        f1_macro = f1_score(y.cpu(), preds.cpu(), average='macro')

        # Log metrics
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_f1_weighted', f1_weighted, on_step=True)
        self.log('test_f1_macro', f1_macro, on_step=True)

        self.test_steps_outputs.append({'test_loss': test_loss, 'test_acc': acc, 'preds': preds, 'targets': y})

        return {'test_loss': test_loss, 'test_acc': acc, 'preds': preds, 'targets': y}

    def on_test_epoch_end(self):
        outputs=self.test_steps_outputs
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])

        class_labels = MULTIClASS_CLASS_NAMES if self.multiclass else BINARY_CLASS_NAMES

        # Confusion Matrix


        self.test_steps_outputs = []
        cm_fig = compute_confusion_matrix(targets.cpu(), preds.cpu(), class_labels)
        self.logger.experiment.add_figure('Test Confusion Matrix', cm_fig, self.current_epoch)

        cm_fig_norm = compute_confusion_matrix(targets.cpu(), preds.cpu(), class_labels, normalize=True)
        self.logger.experiment.add_figure('Test Normalized Confusion Matrix', cm_fig_norm, self.current_epoch)
        cm_fig_norm = compute_confusion_matrix(targets.cpu(), preds.cpu(), class_labels, normalize=True)
        self.logger.experiment.add_figure('Test Normalized Confusion Matrix', cm_fig_norm, self.current_epoch)

        # Compute F1 Scores
        f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted')
        self.log('test_f1_weighted_epoch', f1_weighted, on_epoch=True, prog_bar=True)

        f1_macro = f1_score(targets.cpu(), preds.cpu(), average='macro')
        self.log('test_f1_macro_epoch', f1_macro, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        print("Max training steps: ", self.max_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                        T_max=self.max_training_steps,
                                                                        eta_min=1e-6),
                "monitor": "train_loss",
                "interval": "step",  # step means "batch" here, default: epoch   # New!
                "frequency": 1,  # default
            },
        }

if __name__ == '__main__':
    # Define constants and paths
    BATCH_SIZE = 64
    HIDDEN_DIMS = [256, 128]
    LEARNING_RATE = 1e-3
    N_EPOCHS = 2
    DROPOUT_RATE = 0.1
    LABEL_TYPE = "binary"
    VERSION = f"mlp_classifier_{LABEL_TYPE}"


    # Initialize the DataModule
    data_module = MLP_DataModule(batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, label_type=LABEL_TYPE)
    data_module.setup()

    # Get input and output dimensions based on dataset
    sample_batch = next(iter(data_module.train_dataloader()))
    input_dim = sample_batch[0].shape[1]

    # Initialize the model
    model = MLPClassifier(input_dim=input_dim, hidden_dims=HIDDEN_DIMS,
                          max_training_steps=len(data_module.train_dataloader()) * N_EPOCHS,
                          learning_rate=LEARNING_RATE, dropout=DROPOUT_RATE,
                          multiclass=LABEL_TYPE=="multiclass"
                          )

    # Initialize TensorBoard logger
    logger = TensorBoardLogger(LOGS_DIR/"mlp_classifier/"/VERSION, name="mlp_classifier", version=VERSION)

    # Define checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=LOGS_DIR/"mlp_classifier/"/VERSION,
        filename='mlp_classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    # Define learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up the trainer
    trainer = Trainer(
        accelerator="auto",
        devices='auto',
        max_epochs=N_EPOCHS,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Optionally test the model with the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    trainer.test(model, datamodule=data_module, ckpt_path=best_model_path)
