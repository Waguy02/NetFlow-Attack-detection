import sys

sys.path.append("../..")
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
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

        # Encoder: 2 layers, input -> hidden_dim + activation -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder: 2 layers, latent_dim -> hidden_dim + activation -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, dataloader, optimizer, scheduler, criterion, device, epoch, writer):
    model.train()
    train_loss = 0
    loop = tqdm(dataloader, desc=f'Epoch {epoch + 1} - Training', leave=False)
    for batch_idx, batch in enumerate(loop):
        batch = batch.to(device)
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the learning rate scheduler after each batch

        # Log the loss and learning rate
        current_lr = optimizer.param_groups[0]['lr']
        train_loss += loss.item()
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
            encoded = model.encoder(batch)
            reconstructed = model.decoder(encoded)
            loss = criterion(reconstructed, batch)
            val_loss += loss.item()

            # Log validation loss
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
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(len(np.unique(labels))))

    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5)
    cbar.ax.set_yticklabels(labels_names)

    explained_variance = reducer.explained_variance_ratio_.tolist()
    explained_variance = [round(e, 3) for e in explained_variance]
    title = f'PCA projection of latent_dim - {label_type} - {mode}'
    plt.title(title + '\n' + f'Explained variance: {explained_variance}')

    # Save plot to TensorBoard
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
            encoded = model.encoder(batch)
            reconstructed = model.decoder(encoded)
            loss = criterion(reconstructed, batch)
            test_loss += loss.item()
            latent_representations = np.concatenate((latent_representations, encoded.cpu().numpy()), axis=0)
            loop.set_postfix(test_loss=loss.item())  # Display the current test loss in tqdm bar
    return test_loss / len(dataloader), latent_representations


if __name__ == "__main__":
    BATCH_SIZE = 256
    HIDDEN_DIM = 256
    LATENT_DIM = 8
    LEARNING_RATE = 1e-4
    N_EPOCHS = 2
    DROPOUT_RATE = 0.1
    NUM_WORKERS=4

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(LOGS_DIR / "autoencoder" / "tensorboard_logs")

    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE, val_ratio=VAL_RATIO,
                                        num_workers= NUM_WORKERS
                                        )
    data_module.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Get the input dimension (number of features)
    sample_batch = next(iter(data_module.train_dataloader()))
    input_dim = sample_batch.shape[1]
    print(
        f"AutoEncoder configuration: Input dimension: {input_dim}, "
        f"Latent dimension: {LATENT_DIM}, Hidden dimension: {HIDDEN_DIM}, Dropout rate: {DROPOUT_RATE}, "
        f"Batch size: {BATCH_SIZE}, Validation ratio: {VAL_RATIO}")

    model = Autoencoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_module.train_dataloader()) * N_EPOCHS,
                                                     eta_min=1e-6)

    for epoch in range(N_EPOCHS):
        train_loss = train_autoencoder(model, data_module.train_dataloader(), optimizer, scheduler, criterion, device,
                                       epoch, writer)
        val_loss, val_latent_representations = validate_autoencoder(model, data_module.val_dataloader(), criterion,
                                                                    device, epoch, writer)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Plot latent space at the end of validation
        labels = data_module.get_binary_labels('val')[:val_latent_representations.shape[0]]
        plot_latent_space(writer, val_latent_representations, labels, BINARY_CLASS_NAMES, 'binary',
                          epoch)

    test_loss, test_latent_representations = test_autoencoder(model, data_module.test_dataloader(), criterion, device)
    print(f"Test Loss: {test_loss}")

    # Log test loss to TensorBoard
    writer.add_scalar('Loss/Test', test_loss, N_EPOCHS)

    # Plot latent space at the end of test
    labels = data_module.get_binary_labels('test')[:test_latent_representations.shape[0]]
    plot_latent_space(writer, test_latent_representations, labels, BINARY_CLASS_NAMES, 'binary',
                      N_EPOCHS)

    # Close the TensorBoard writer
    writer.close()

    # S
