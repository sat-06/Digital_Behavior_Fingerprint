"""FastAPI backend application for Digital Behavior Fingerprint."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    RAW_TIMING_COLUMNS,
    ALL_FEATURE_COLUMNS,
    CLASSICAL_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    AUTOENCODER_PATH,
    METADATA_PATH,
    EVALUATION_METRICS_PATH,
    ANOMALY_CONFIG_PATH,
)
from src.features.extractor import extract_sample_features
from src.models.persistence import load_inference_artifacts
from src.models.autoencoder import detect_anomalies
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    AnomalyRequest,
    AnomalyResponse,
    ModelInfoResponse,
    HealthResponse,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# In-memory artifact storage
artifacts: Dict[str, Any] = {}


def load_artifacts_safe() -> bool:
    """Attempt to load saved model artifacts into memory."""
    global artifacts
    try:
        loaded = load_inference_artifacts()
        artifacts.update(loaded)
        logger.info("Successfully loaded all model inference artifacts.")
        return True
    except Exception as e:
        logger.warning(f"Could not load all model artifacts on startup: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load artifacts on application startup."""
    load_artifacts_safe()
    yield


app = FastAPI(
    title="Digital Behavior Fingerprint API",
    description=(
        "Production backend API for Keystroke Dynamics Behavioral Fingerprinting, "
        "Subject Classification, and Autoencoder Anomaly Detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_feature_array(
    raw_input: Union[Dict[str, float], List[float]],
    target_columns: List[str] = ALL_FEATURE_COLUMNS,
) -> np.ndarray:
    """Parse, engineer, and align an input feature payload into a standardized 2D numpy array.

    Handles:
    - Dict with 31 raw timing features -> engineers 17 statistical features -> 48 features.
    - Dict with all 48 features -> aligns to expected column ordering.
    - List with 31 raw timing floats -> converts to dict -> engineers 17 features -> 48 features.
    - List with 48 floats -> directly converts to array matching order.

    Args:
        raw_input: Dictionary or list of float features.
        target_columns: Expected column ordering.

    Returns:
        2D numpy array of shape (1, 48).

    Raises:
        HTTPException: If input cannot be validated or aligned.
    """
    try:
        if isinstance(raw_input, list):
            if len(raw_input) == len(RAW_TIMING_COLUMNS):
                feature_dict = dict(zip(RAW_TIMING_COLUMNS, raw_input))
                full_features = extract_sample_features(feature_dict)
            elif len(raw_input) == len(ALL_FEATURE_COLUMNS):
                full_features = dict(zip(ALL_FEATURE_COLUMNS, raw_input))
            else:
                raise ValueError(
                    f"Expected {len(RAW_TIMING_COLUMNS)} or {len(ALL_FEATURE_COLUMNS)} features, "
                    f"got {len(raw_input)}"
                )
        elif isinstance(raw_input, dict):
            # If input only has the 31 raw timing features, engineer the remaining 17
            has_raw = all(c in raw_input for c in RAW_TIMING_COLUMNS)
            has_all = all(c in raw_input for c in ALL_FEATURE_COLUMNS)

            if has_all:
                full_features = raw_input
            elif has_raw:
                full_features = extract_sample_features(raw_input)
            else:
                missing = [c for c in RAW_TIMING_COLUMNS if c not in raw_input]
                raise ValueError(f"Missing required raw timing features: {missing[:5]} (and more)")
        else:
            raise TypeError("Invalid feature payload format.")

        ordered_values = [float(full_features[col]) for col in target_columns]
        arr = np.array([ordered_values], dtype=np.float64)

        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Feature vector contains NaN or Infinite numerical values.")

        return arr

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature preparation error: {str(exc)}",
        )


@app.get("/", summary="Project Information")
async def root():
    """Return overview information about the Digital Behavior Fingerprint service."""
    return {
        "project": "Digital Behavior Fingerprint",
        "description": "Keystroke Dynamics Biometrics & Behavioral Anomaly Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "detect_anomaly": "/detect-anomaly",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="API Health Status")
async def health_check():
    """Check health and model artifact readiness."""
    artifacts_status = {
        "scaler": SCALER_PATH.exists(),
        "label_encoder": LABEL_ENCODER_PATH.exists(),
        "classical_model": CLASSICAL_MODEL_PATH.exists(),
        "autoencoder": AUTOENCODER_PATH.exists(),
        "metadata": METADATA_PATH.exists(),
    }
    all_core_ready = (
        "classical_model" in artifacts
        and "scaler" in artifacts
        and "label_encoder" in artifacts
    )
    overall_status = "healthy" if all_core_ready else "degraded"

    return HealthResponse(
        status=overall_status,
        models_loaded=all_core_ready,
        classifier_available="classical_model" in artifacts,
        autoencoder_available="autoencoder" in artifacts,
        artifacts=artifacts_status,
    )


@app.get("/model-info", response_model=ModelInfoResponse, summary="Model and Feature Info")
async def get_model_info():
    """Return information about the trained models, feature definitions, and anomaly thresholds."""
    if "classical_model" not in artifacts or "feature_metadata" not in artifacts:
        # Try loading once more
        if not load_artifacts_safe():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not yet loaded or trained.",
            )

    meta = artifacts.get("feature_metadata", {})
    anomaly_config = artifacts.get("anomaly_config", {})

    threshold = anomaly_config.get("threshold") if anomaly_config else None
    metrics_summary = None
    if EVALUATION_METRICS_PATH.exists():
        import json
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)

    return ModelInfoResponse(
        project="Digital Behavior Fingerprint",
        version="1.0.0",
        best_classifier=meta.get("model_type", "Unknown"),
        num_classes=meta.get("num_classes", 0),
        classes=meta.get("classes", []),
        feature_count=meta.get("feature_count", len(ALL_FEATURE_COLUMNS)),
        expected_features=meta.get("feature_names", ALL_FEATURE_COLUMNS),
        raw_timing_features=RAW_TIMING_COLUMNS,
        anomaly_threshold=threshold,
        metrics_summary=metrics_summary,
    )


@app.post("/predict", response_model=PredictResponse, summary="Predict Behavioral Subject")
async def predict_subject(payload: PredictRequest):
    """Predict user/subject identity from keystroke timing patterns."""
    if "classical_model" not in artifacts or "scaler" not in artifacts:
        if not load_artifacts_safe():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classifier model artifacts not available. Please train models first.",
            )

    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    model = artifacts["classical_model"]
    feature_names = artifacts["feature_metadata"]["feature_names"]

    # 1. Parse and engineer features
    X_raw = prepare_feature_array(payload.features, target_columns=feature_names)

    # 2. Scale features using the fitted scaler
    X_df = pd.DataFrame(X_raw, columns=feature_names)
    X_scaled = scaler.transform(X_df)

    # 3. Predict class and probabilities
    try:
        pred_idx = model.predict(X_scaled)[0]
        predicted_subject = str(label_encoder.inverse_transform([pred_idx])[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            top_indices = np.argsort(probs)[::-1][:5]
            top_classes = label_encoder.inverse_transform(top_indices)
            top_preds = {str(cls): float(probs[idx]) for cls, idx in zip(top_classes, top_indices)}
            confidence = float(probs[pred_idx])
        else:
            confidence = 1.0
            top_preds = {predicted_subject: 1.0}

        return PredictResponse(
            predicted_subject=predicted_subject,
            confidence=round(confidence, 4),
            top_predictions={k: round(v, 4) for k, v in top_preds.items()},
            model_used=type(model).__name__,
            status="success",
        )
    except Exception as exc:
        logger.error(f"Prediction inference failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post("/detect-anomaly", response_model=AnomalyResponse, summary="Detect Keystroke Anomaly")
async def detect_anomaly(payload: AnomalyRequest):
    """Assess whether a keystroke timing pattern is anomalous using the autoencoder."""
    if "autoencoder" not in artifacts or "scaler" not in artifacts:
        if not load_artifacts_safe() or "autoencoder" not in artifacts:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Autoencoder anomaly detection model not available. Please train models first.",
            )

    scaler = artifacts["scaler"]
    autoencoder = artifacts["autoencoder"]
    anomaly_config = artifacts.get("anomaly_config", {})
    threshold = float(anomaly_config.get("threshold", 0.5))
    feature_names = artifacts["feature_metadata"]["feature_names"]

    # 1. Prepare features
    X_raw = prepare_feature_array(payload.features, target_columns=feature_names)

    # 2. Scale features
    X_df = pd.DataFrame(X_raw, columns=feature_names)
    X_scaled = scaler.transform(X_df)

    # 3. Compute autoencoder reconstruction error
    try:
        errors, is_anomaly_arr = detect_anomalies(autoencoder, X_scaled, threshold)
        recon_error = float(errors[0])
        is_anomaly = bool(is_anomaly_arr[0])
        anomaly_score = float(recon_error / (threshold + 1e-6))

        assessment = (
            "Unusual or unfamiliar behavioral pattern detected (possible unauthorized user)."
            if is_anomaly
            else "Normal familiar behavioral keystroke pattern."
        )

        return AnomalyResponse(
            reconstruction_error=round(recon_error, 5),
            threshold=round(threshold, 5),
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anomaly,
            assessment=assessment,
            status="success",
        )
    except Exception as exc:
        logger.error(f"Anomaly detection failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection error: {str(exc)}",
        )
