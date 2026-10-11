"""PyTorch autoencoder for behavioral keystroke anomaly detection."""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class KeystrokeAutoencoder(nn.Module):
    """Autoencoder network for reconstructing keystroke feature vectors."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Mean Squared Error (MSE) reconstruction loss per sample."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse_per_sample = torch.mean((x - reconstructed) ** 2, dim=-1)
            return mse_per_sample


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
    seed: int = RANDOM_SEED,
) -> Tuple[KeystrokeAutoencoder, float, Dict[str, float]]:
    """Train the autoencoder on normal behavioral keystroke data and calibrate anomaly threshold.

    Args:
        X_train: Scaled normal training features.
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        threshold_percentile: Percentile of normal training reconstruction error to set as threshold.
        seed: Random seed.

    Returns:
        Tuple of (trained_autoencoder, anomaly_threshold, metrics_dict).
    """
    torch.manual_seed(seed)
    input_dim = X_train.shape[1]
    model = KeystrokeAutoencoder(input_dim=input_dim)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    logger.info(f"Training PyTorch Autoencoder (input_dim={input_dim}, epochs={epochs})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        epoch_loss = total_loss / len(X_train)
        if epoch % 10 == 0 or epoch == epochs:
            logger.info(f"Autoencoder Epoch {epoch:02d}/{epochs:02d} - Reconstruction Loss: {epoch_loss:.5f}")

    # Calibrate anomaly threshold on training reconstruction errors
    model.eval()
    with torch.no_grad():
        train_errors = model.compute_reconstruction_error(train_tensor).cpu().numpy()

    threshold = float(np.percentile(train_errors, threshold_percentile))
    mean_err = float(np.mean(train_errors))
    std_err = float(np.std(train_errors))

    logger.info(
        f"Calibrated Anomaly Threshold ({threshold_percentile}th percentile): {threshold:.5f} "
        f"(Train Error Mean: {mean_err:.5f}, Std: {std_err:.5f})"
    )

    metrics = {
        "final_loss": float(epoch_loss),
        "mean_reconstruction_error": mean_err,
        "std_reconstruction_error": std_err,
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
    }
    return model, threshold, metrics


def detect_anomalies(
    model: KeystrokeAutoencoder,
    X: Union[np.ndarray, torch.Tensor],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate reconstruction error anomaly scores and anomaly classification.

    Args:
        model: Trained KeystrokeAutoencoder.
        X: Scaled feature matrix or tensor.
        threshold: Calibrated anomaly threshold.

    Returns:
        Tuple of (reconstruction_errors_array, is_anomaly_boolean_array).
    """
    model.eval()
    if isinstance(X, np.ndarray):
        x_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        x_tensor = X

    with torch.no_grad():
        errors = model.compute_reconstruction_error(x_tensor).cpu().numpy()

    is_anomaly = errors > threshold
    return errors, is_anomaly
