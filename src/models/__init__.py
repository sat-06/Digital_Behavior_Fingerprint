"""Models module for keystroke dynamics."""

from src.models.classical import (
    get_classical_models,
    train_and_evaluate_model,
    train_and_compare_all,
)
from src.models.neural_net import (
    KeystrokeMLP,
    train_neural_network,
    evaluate_neural_network,
)
from src.models.autoencoder import (
    KeystrokeAutoencoder,
    train_autoencoder,
    detect_anomalies,
)
from src.models.persistence import (
    save_classical_artifacts,
    save_pytorch_classifier,
    save_autoencoder_artifacts,
    load_inference_artifacts,
)

__all__ = [
    "get_classical_models",
    "train_and_evaluate_model",
    "train_and_compare_all",
    "KeystrokeMLP",
    "train_neural_network",
    "evaluate_neural_network",
    "KeystrokeAutoencoder",
    "train_autoencoder",
    "detect_anomalies",
    "save_classical_artifacts",
    "save_pytorch_classifier",
    "save_autoencoder_artifacts",
    "load_inference_artifacts",
]
