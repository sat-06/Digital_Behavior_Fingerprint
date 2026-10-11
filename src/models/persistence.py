"""Model and artifact persistence module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import torch

from src.config import (
    MODELS_DIR,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    CLASSICAL_MODEL_PATH,
    PYTORCH_CLASSIFIER_PATH,
    AUTOENCODER_PATH,
    METADATA_PATH,
    ANOMALY_CONFIG_PATH,
    EVALUATION_METRICS_PATH,
    ALL_FEATURE_COLUMNS,
)
from src.models.neural_net import KeystrokeMLP
from src.models.autoencoder import KeystrokeAutoencoder

logger = logging.getLogger(__name__)


def ensure_models_dir(directory: Path = MODELS_DIR) -> None:
    """Ensure that the model artifacts directory exists."""
    directory.mkdir(parents=True, exist_ok=True)


def save_classical_artifacts(
    model: Any,
    scaler: Any,
    label_encoder: Any,
    feature_names: List[str] = ALL_FEATURE_COLUMNS,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save classical model, scaler, label encoder, and feature metadata."""
    ensure_models_dir()

    joblib.dump(model, CLASSICAL_MODEL_PATH, compress=3)
    logger.info(f"Saved classical model to: {CLASSICAL_MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH, compress=3)
    logger.info(f"Saved feature scaler to: {SCALER_PATH}")

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info(f"Saved label encoder to: {LABEL_ENCODER_PATH}")

    metadata = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "classes": [str(c) for c in label_encoder.classes_],
        "num_classes": len(label_encoder.classes_),
        "model_type": type(model).__name__,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature metadata to: {METADATA_PATH}")

    if metrics is not None:
        with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to: {EVALUATION_METRICS_PATH}")


def save_pytorch_classifier(model: KeystrokeMLP, path: Path = PYTORCH_CLASSIFIER_PATH) -> None:
    """Save PyTorch classifier model state and architecture metadata."""
    ensure_models_dir(path.parent)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
    }
    torch.save(payload, path)
    logger.info(f"Saved PyTorch classifier to: {path}")


def save_autoencoder_artifacts(
    model: KeystrokeAutoencoder,
    anomaly_config: Dict[str, Any],
    path: Path = AUTOENCODER_PATH,
    config_path: Path = ANOMALY_CONFIG_PATH,
) -> None:
    """Save PyTorch autoencoder weights and anomaly configuration."""
    ensure_models_dir(path.parent)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
    }
    torch.save(payload, path)
    logger.info(f"Saved Autoencoder model to: {path}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(anomaly_config, f, indent=2)
    logger.info(f"Saved anomaly config to: {config_path}")


def load_inference_artifacts() -> Dict[str, Any]:
    """Load all artifacts required for production/API inference.

    Returns:
        Dictionary containing loaded scaler, label_encoder, classical_model,
        autoencoder, anomaly_config, and feature_metadata.

    Raises:
        FileNotFoundError: If any required artifact file is missing.
    """
    missing = []
    for required in [SCALER_PATH, LABEL_ENCODER_PATH, CLASSICAL_MODEL_PATH, METADATA_PATH]:
        if not required.exists():
            missing.append(str(required))

    if missing:
        raise FileNotFoundError(
            f"Missing required model artifacts: {missing}. Run training first."
        )

    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    classical_model = joblib.load(CLASSICAL_MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)

    # Load autoencoder if available
    autoencoder = None
    anomaly_config = None
    if AUTOENCODER_PATH.exists() and ANOMALY_CONFIG_PATH.exists():
        payload = torch.load(AUTOENCODER_PATH, weights_only=True)
        autoencoder = KeystrokeAutoencoder(
            input_dim=payload["input_dim"], latent_dim=payload["latent_dim"]
        )
        autoencoder.load_state_dict(payload["state_dict"])
        autoencoder.eval()

        with open(ANOMALY_CONFIG_PATH, "r", encoding="utf-8") as f:
            anomaly_config = json.load(f)

    # Load PyTorch classifier if available
    pytorch_classifier = None
    if PYTORCH_CLASSIFIER_PATH.exists():
        payload = torch.load(PYTORCH_CLASSIFIER_PATH, weights_only=True)
        pytorch_classifier = KeystrokeMLP(
            input_dim=payload["input_dim"], num_classes=payload["num_classes"]
        )
        pytorch_classifier.load_state_dict(payload["state_dict"])
        pytorch_classifier.eval()

    return {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "classical_model": classical_model,
        "autoencoder": autoencoder,
        "anomaly_config": anomaly_config,
        "pytorch_classifier": pytorch_classifier,
        "feature_metadata": feature_metadata,
    }
