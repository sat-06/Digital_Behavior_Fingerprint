"""Tests for classical and neural network models."""

import pytest
import numpy as np
import torch

from src.models.classical import get_classical_models, train_and_evaluate_model
from src.models.neural_net import KeystrokeMLP, evaluate_neural_network
from src.models.persistence import save_classical_artifacts, load_inference_artifacts


def test_classical_models_initialization():
    """Verify all 5 required classical models are instantiated."""
    models = get_classical_models()
    expected_names = [
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
    ]
    for name in expected_names:
        assert name in models


def test_classical_model_train_and_evaluate():
    """Verify single model training and evaluation returns required metrics."""
    models = get_classical_models()
    rf = models["Random Forest"]

    X_train = np.random.randn(60, 10)
    y_train = np.array([0, 1, 2] * 20)
    X_test = np.random.randn(15, 10)
    y_test = np.array([0, 1, 2] * 5)

    metrics = train_and_evaluate_model(rf, X_train, y_train, X_test, y_test)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_pytorch_mlp_forward_and_proba():
    """Verify PyTorch KeystrokeMLP model forward pass and output probability shapes."""
    model = KeystrokeMLP(input_dim=48, num_classes=10)
    dummy_input = torch.randn(4, 48)

    logits = model(dummy_input)
    assert logits.shape == (4, 10)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (4, 10)
    # Sum of probabilities across classes should be 1
    np.testing.assert_almost_equal(probs.sum(dim=-1).numpy(), np.ones(4), decimal=5)
