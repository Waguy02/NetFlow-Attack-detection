import torch
import torch.nn as nn
import pytorch_lightning as pl
import umap
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from network_ad.config import LOGS_DIR, MAX_PLOT_POINTS, BINARY_CLASS_NAMES
from network_ad.unsupervised.autoencoder_datamodule import AutoencoderDataModule


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=32, learning_rate=1e-3,
                 dropout_rate=0.1,
                 max_training_steps = None
                 ):
        """
        :param input_dim: The input dimension (number of features) of the autoencoder
        :param hidden_dim1: The number of neurons in the first hidden layer
        :param hidden_dim2: The number of neurons in the second hidden layer
        :param latent_dim: The latent dimension (number of neurons in the bottleneck layer)
        :param learning_rate: The learning rate for the optimizer
        :param max_training_steps: The maximum number of training steps(used for the scheduler)
        """
        super(Autoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.max_training_steps = max_training_steps
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        # Encoder: 3 layers, input -> 256 -> 128 -> latent_dim (32)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, latent_dim)
        )

        # Decoder: 3 layers, latent_dim (32) -> 128 -> 256 -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )
        self.validation_step_outputs = np.zeros((0, latent_dim))
        self.test_step_outputs = np.zeros((0, latent_dim))

    def forward(self, x):
        # Forward pass: encoder -> decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        # Get inputs
        x = batch

        # Forward pass
        reconstructed = self.forward(x)

        # Loss: MSE (Mean Squared Error)
        loss = nn.functional.mse_loss(reconstructed, x)

        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        val_loss = nn.functional.mse_loss(reconstructed, x)

        # Save latent_dim representations and inputs for later  plotting
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Return encoded latent_dim for plotting
        self.validation_step_outputs = np.concatenate((self.validation_step_outputs, encoded.cpu().detach().numpy()), axis=0)
        return val_loss

    def on_validation_epoch_end(self):
        # Plotthe latent_dim representation
        self.plot_latent_space(self.validation_step_outputs[:MAX_PLOT_POINTS], mode='val')

        # Reset the validation step outputs
        self.validation_step_outputs = np.zeros((0, self.latent_dim))

    def plot_latent_space(self, encoded, mode='val', MULTICLASS_CLASS_NAMES=None):

        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(encoded)

        for label_type in ['binary', 'multiclass']:

            # Get the labels to color the plot
            if labels == 'binary':
                labels = self.trainer.datamodule.get_binary_labels(mode)[:encoded.shape[0]]  # Assuming a method get_labels exists in your datamodule
                colors = ["blue" if label == 0 else "red" for label in labels]
                cmap = mcolors.ListedColormap(colors)
                labels_names = BINARY_CLASS_NAMES

            else:
                labels = self.trainer.datamodule.get_multiclass_labels(mode)[:encoded.shape[0]]
                colors = labels
                cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))
                labels_names = MULTICLASS_CLASS_NAMES

            #pca
            # Plot UMAP result
            plt.figure(figsize=(10, 10))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5)
            # Create custom colorbar with two colors: blue (Normal) and red (Anomaly)

            bounds = [-0.5, 0.5, 1.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            cbar = plt.colorbar(mcolors.ScalarMappable(norm=norm, cmap=cmap), ticks=[0, 1])
            cbar.ax.set_yticklabels(labels_names)

            explained_variance = reducer.explained_variance_ratio_.tolist()
            explained_variance = [round(e, 3) for e in explained_variance]
            title = f'PCA projection of latent_dim - {label_type} - {mode}'
            plt.title(title + '\n' + f'Explained variance: {explained_variance}')

            # Log the plot in TensorBoard
            self.logger.experiment.add_figure(title, plt.gcf(), global_step=self.current_epoch)
            plt.close()



    def test_step(self, batch, batch_idx):
        x = batch
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        test_loss = nn.functional.mse_loss(reconstructed, x)

        # Log the test loss
        self.log('test_loss', test_loss)
        self.test_step_outputs = np.concatenate((self.test_step_outputs, encoded.cpu().detach().numpy()), axis=0)
        return test_loss

    def on_test_epoch_end(self) -> None:
        # Plot UMAP on the latent_dim representation
        self.plot_latent_space(self.test_step_outputs[:MAX_PLOT_POINTS], mode='test')

        # Reset the test step outputs
        self.test_step_outputs = np.zeros((0, self.latent_dim))

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        print("Max training steps: ", self.max_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max=self.max_training_steps,
                                                eta_min=1e-6),
                "monitor": "train_loss",
                "interval": "step", # step means "batch" here, default: epoch   # New!
                "frequency": 1, # default
            },
        }

if __name__ == "__main__":
    # Initialize the DataModule
    BATCH_SIZE = 64
    VAL_RATIO = 0.1

    HIDDEN_DIM_1 = 256
    HIDDEM_DIM_2 = 64
    LATENT_DIM = 32
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    N_EPOCHS=5
    DROPOUT_RATE = 0.1

    VERSION  = "v1"

    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE,
                                        val_ratio=VAL_RATIO,
                                        num_workers = NUM_WORKERS,
                                        )
    data_module.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get the input dimension (number of features)
    sample_batch = next(iter(data_module.train_dataloader()))
    input_dim = sample_batch.shape[1]
    print("AutoEncoder configuration: "
          f"Input dimension: {input_dim}, "
          f"Batch size: {BATCH_SIZE}, "
          f"Validation ratio: {VAL_RATIO}"
          )

    # Initialize Autoencoder model with input_dim
    model = Autoencoder(input_dim=input_dim,
                        hidden_dim1=HIDDEN_DIM_1,
                        hidden_dim2=HIDDEM_DIM_2,
                        latent_dim=LATENT_DIM,
                        max_training_steps= len(data_module.train_dataloader()) * N_EPOCHS,
                        learning_rate=LEARNING_RATE,
                        dropout_rate=DROPOUT_RATE
                        )

    # Define the TensorBoard logger
    logger = pl.loggers.TensorBoardLogger(LOGS_DIR, name="autoencoder",version=VERSION)

    # Define the ModelCheckpoint callback (Save the best model based on validation loss)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=LOGS_DIR/'autoencoder'/VERSION,
        filename='autoencoder-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        save_last=True,
        mode='min'
    )

    # Define the LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Define the Trainer, enabling TensorBoard logging and specifying the maximum epochs
    trainer = pl.Trainer(
        accelerator="auto",
        devices='auto',
        strategy="auto",
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=N_EPOCHS,
        num_sanity_val_steps=0
    )

    # # Train the model
    trainer.fit(model, data_module, ckpt_path="last")

    # Test the model
    best_checkpoint = checkpoint_callback.best_model_path
    trainer.test(model, data_module, ckpt_path=best_checkpoint)
