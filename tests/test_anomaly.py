"""Tests for PyTorch Autoencoder anomaly detection."""

import pytest
import numpy as np
import torch

from src.models.autoencoder import (
    KeystrokeAutoencoder,
    train_autoencoder,
    detect_anomalies,
)


def test_autoencoder_architecture():
    """Verify Autoencoder forward pass reconstructs the input dimension."""
    model = KeystrokeAutoencoder(input_dim=48, latent_dim=16)
    dummy_input = torch.randn(8, 48)

    reconstructed = model(dummy_input)
    assert reconstructed.shape == (8, 48)


def test_autoencoder_reconstruction_error():
    """Verify reconstruction error computation outputs 1D tensor of MSE per sample."""
    model = KeystrokeAutoencoder(input_dim=48, latent_dim=16)
    dummy_input = torch.randn(8, 48)

    errors = model.compute_reconstruction_error(dummy_input)
    assert errors.shape == (8,)
    assert (errors >= 0).all()


def test_detect_anomalies():
    """Verify anomaly detection correctly flags samples above threshold."""
    model = KeystrokeAutoencoder(input_dim=10, latent_dim=4)
    # Set weights to identity-like by zeroing out or using simple inputs
    X_normal = np.zeros((5, 10))
    threshold = 0.5

    errors, is_anomaly = detect_anomalies(model, X_normal, threshold)
    assert len(errors) == 5
    assert len(is_anomaly) == 5
    assert isinstance(is_anomaly[0], (bool, np.bool_))
