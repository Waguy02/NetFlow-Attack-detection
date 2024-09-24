import torch
import torch.nn as nn
import pytorch_lightning as pl
import umap
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from network_ad.config import LOGS_DIR
from network_ad.unsupervised.autoencoder_datamodule import AutoencoderDataModule


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=32, learning_rate=1e-3):
        super(Autoencoder, self).__init__()
        self.learning_rate = learning_rate

        # Encoder: 3 layers, input -> 256 -> 128 -> latent_dim (32)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, latent_dim)
        )

        # Decoder: 3 layers, latent_dim (32) -> 128 -> 256 -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

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

        # Save latent_dim representations and inputs for later UMAP plotting
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Return encoded latent_dim for UMAP and original data
        out = {'val_loss': val_loss, 'encoded': encoded.detach(), 'x': x.detach()}
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        # Concatenate latent_dim and input tensors from all batches
        encoded_list = [output['encoded'] for output in outputs]
        x_list = [output['x'] for output in outputs]

        encoded = torch.cat(encoded_list, dim=0).cpu().numpy()
        x = torch.cat(x_list, dim=0).cpu().numpy()

        # Plot UMAP on the latent_dim representation
        self.plot_umap(encoded, x)

    def plot_umap(self, encoded, x):
        # Perform UMAP on the encoded data
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='euclidean')
        embedding = reducer.fit_transform(encoded)

        # Plot UMAP result
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
        plt.title('UMAP projection of latent_dim')

        # Log the plot in TensorBoard
        self.logger.experiment.add_figure('UMAP of latent_dim', plt.gcf(), global_step=self.current_epoch)
        plt.close()

    def test_step(self, batch, batch_idx):
        x = batch
        reconstructed = self.forward(x)
        test_loss = nn.functional.mse_loss(reconstructed, x)

        # Log the test loss
        self.log('test_loss', test_loss)
        out = {'test_loss': test_loss, 'encoded': self.encoder(x).detach(), 'x': x.detach()}
        self.test_step_outputs.append(out)
        return test_loss

    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
        # Concatenate latent_dim and input tensors from all batches
        encoded_list = [output['encoded'] for output in outputs]
        x_list = [output['x'] for output in outputs]

        encoded = torch.cat(encoded_list, dim=0).cpu().numpy()
        x = torch.cat(x_list, dim=0).cpu().numpy()

        # Plot UMAP on the latent_dim representation
        self.plot_umap(encoded, x)

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Scheduler: ReduceLROnPlateau
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True),
            'monitor': 'val_loss',  # Scheduler will check this metric
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Initialize the DataModule
    BATCH_SIZE = 256
    VAL_RATIO = 0.1

    HIDDEN_DIM_1 = 256
    HIDDEM_DIM_2 = 64
    LATENT_DIM = 16
    INITIAL_LR = 1e-3

    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE,
                                        val_ratio=VAL_RATIO)


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

                        )

    # Define the TensorBoard logger
    logger = pl.loggers.TensorBoardLogger(LOGS_DIR, name="autoencoder",version="tb")

    # Define the ModelCheckpoint callback (Save the best model based on validation loss)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=LOGS_DIR,
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min'
    )

    # Define the LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Early stopping callback (Stop training early if the validation loss does not improve)
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Define the Trainer, enabling TensorBoard logging and specifying the maximum epochs
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    )

    # Train the model
    trainer.fit(model, data_module, ckpt_path="last")

    # Test the model
    trainer.test(model, data_module, ckpt_path="best")
