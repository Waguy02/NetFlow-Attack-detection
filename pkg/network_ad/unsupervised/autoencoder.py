import sys

sys.path.append("../..")
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Import tqdm for progress bar
from network_ad.config import LOGS_DIR, MAX_PLOT_POINTS, BINARY_CLASS_NAMES, MULTIClASS_CLASS_NAMES, VAL_RATIO
from network_ad.unsupervised.autoencoder_datamodule import AutoencoderDataModule


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, dropout_rate=0.1):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # TODO: Implement the encoder. It should contain two layers:
        # 1. A linear layer that maps `input_dim` to `hidden_dim`
        # 2. A ReLU activation and dropout with `dropout_rate`
        # 3. Another linear layer that maps `hidden_dim` to `latent_dim`
        self.encoder = nn.Sequential(
            # Your code here
        )

        # TODO: Implement the decoder. It should reverse the encoder steps:
        # 1. A linear layer that maps `latent_dim` to `hidden_dim`
        # 2. A ReLU activation and dropout with `dropout_rate`
        # 3. Another linear layer that maps `hidden_dim` back to `input_dim`
        # 4. Sigmoid activation function
        self.decoder = nn.Sequential(
            # Your code here
        )

    def forward(self, x):
        # TODO: Complete the forward function by encoding and then decoding `x`.
        encoded = self.encoder(x)
        decoded =  # Your code here
        return decoded


def train_autoencoder(model, dataloader, optimizer, scheduler, criterion, device, epoch, writer):
    model.train()
    train_loss = 0
    loop = tqdm(dataloader, desc=f'Epoch {epoch + 1} - Training', leave=False)
    for batch_idx, batch in enumerate(loop):
        batch = batch.to(device)
        optimizer.zero_grad()

        # TODO: Forward pass: feed the batch to the model
        reconstructed =  # Your code here

        # TODO: Compute the loss between the reconstructed output and the original batch
        loss =  # Your code here

        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the learning rate scheduler after each batch

        current_lr = optimizer.param_groups[0]['lr']
        train_loss += loss.item()

        # TODO: Log the training loss and learning rate in TensorBoard using `writer`
        writer.add_scalar('Loss/Train', loss.item(), epoch * len(dataloader) + batch_idx)
        writer.add_scalar('Learning Rate', current_lr, epoch * len(dataloader) + batch_idx)

        loop.set_postfix(loss=loss.item(), lr=current_lr)  # Display the current loss and learning rate in tqdm bar
    return train_loss / len(dataloader)


def validate_autoencoder(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0
    latent_representations = np.zeros((0, model.latent_dim))
    loop = tqdm(dataloader, desc=f'Epoch {epoch + 1} - Validating', leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loop):
            batch = batch.to(device)

            # TODO: Encode the batch using the model's encoder
            encoded =  # Your code here

            # TODO: Decode the encoded representation back to input dimensions
            reconstructed =  # Your code here

            # TODO: Compute the validation loss
            loss =  # Your code here
            val_loss += loss.item()

            # TODO: Log the validation loss using `writer`
            writer.add_scalar('Loss/Val', loss.item(), epoch * len(dataloader) + batch_idx)

            latent_representations = np.concatenate((latent_representations, encoded.cpu().numpy()), axis=0)
            loop.set_postfix(val_loss=loss.item())  # Display the current validation loss in tqdm bar
    return val_loss / len(dataloader), latent_representations


def plot_latent_space(writer, encoded, labels, class_names, label_type, epoch, mode='val'):
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(encoded)

    plt.figure(figsize=(5, 5))
    if label_type == 'binary':
        cmap = mcolors.ListedColormap(['royalblue', 'red'])
    else:
        cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))

    found_labels = np.unique(labels)
    labels_names = list(filter(lambda x: x in found_labels, class_names))
    labels_to_idx = {label: i for i, label in enumerate(labels_names)}

    colors = [cmap(labels_to_idx[label]) for label in labels]

    bounds = np.arange(len(np.unique(labels)) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(len(np.unique(labels))))

    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5)
    cbar.ax.set_yticklabels(labels_names)

    explained_variance = reducer.explained_variance_ratio_.tolist()
    explained_variance = [round(e, 3) for e in explained_variance]
    title = f'PCA projection of latent_dim - {label_type} - {mode}'
    plt.title(title + '\n' + f'Explained variance: {explained_variance}')

    # TODO: Save the plot to TensorBoard using `writer`
    writer.add_figure(title, plt.gcf(), global_step=epoch)
    plt.close()


def test_autoencoder(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    latent_representations = np.zeros((0, model.latent_dim))
    loop = tqdm(dataloader, desc='Testing', leave=False)
    with torch.no_grad():
        for batch in loop:
            batch = batch.to(device)

            # TODO: Encode the batch using the model's encoder
            encoded =  # Your code here

            # TODO: Decode the encoded representation back to input dimensions
            reconstructed =  # Your code here

            # TODO: Compute the test loss
            loss =  # Your code here
            test_loss += loss.item()

            latent_representations = np.concatenate((latent_representations, encoded.cpu().numpy()), axis=0)
            loop.set_postfix(test_loss=loss.item())  # Display the current test loss in tqdm bar
    return test_loss / len(dataloader), latent_representations


if __name__ == "__main__":
    BATCH_SIZE = 512
    HIDDEN_DIM = 256
    LATENT_DIM = 8
    LEARNING_RATE = 1e-4
    N_EPOCHS = 2
    DROPOUT_RATE = 0.1

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(LOGS_DIR / "autoencoder" / "tensorboard_logs")

    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE, val_ratio=VAL_RATIO)
    data_module.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Get the input dimension from a sample batch
    sample_batch = next(iter(data_module.train_dataloader()))
    input_dim =  # Your code here
    print(
        f"AutoEncoder configuration: Input dimension: {input_dim}, "
        f"Latent dimension: {LATENT_DIM}, Hidden dimension: {HIDDEN_DIM}, Dropout rate: {DROPOUT_RATE}, "
        f"Batch size: {BATCH_SIZE}, Validation ratio: {VAL_RATIO}")

    # TODO: Initialize the Autoencoder model
    model = Autoencoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # TODO: Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_module.train_dataloader()) * N_EPOCHS,
                                                     eta_min=1e-6)

    for epoch in range(N_EPOCHS):
        train_loss = train_autoencoder(model, data_module.train_dataloader(), optimizer, scheduler, criterion, device,
                                       epoch, writer)
        val_loss, val_latent_representations = validate_autoencoder(model, data_module.val_dataloader(), criterion,
                                                                    device, epoch, writer)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        # TODO: Plot latent space at the end of validation using the `plot_latent_space` function
        labels = data_module.get_binary_labels('val')[:val_latent_representations.shape[0]]
        plot_latent_space(writer, val_latent_representations[:MAX_PLOT_POINTS], labels, BINARY_CLASS_NAMES, 'binary',
                          epoch)

    test_loss, test_latent_representations = test_autoencoder(model, data_module.test_dataloader(), criterion, device)
    print(f"Test Loss: {test_loss}")

    # TODO: Log the test loss using `writer`
    writer.add_scalar('Loss/Test', test_loss, N_EPOCHS)

    # TODO: Plot latent space at the end of the test using the `plot_latent_space` function
    labels = data_module.get_binary_labels('test')[:test_latent_representations.shape[0]]
    plot_latent_space(writer, test_latent_representations[:MAX_PLOT_POINTS], labels, BINARY_CLASS_NAMES, 'binary',
                      N_EPOCHS)

    # Close the TensorBoard writer
    writer.close()
