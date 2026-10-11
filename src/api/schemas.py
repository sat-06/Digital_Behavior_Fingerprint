"""Pydantic schemas for the Digital Behavior Fingerprint FastAPI backend."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import math

from src.config import RAW_TIMING_COLUMNS, ALL_FEATURE_COLUMNS


class PredictRequest(BaseModel):
    """Input payload for behavioral classification prediction."""

    features: Union[Dict[str, float], List[float]] = Field(
        ...,
        description=(
            "Keystroke timing features. Can be a dictionary mapping 31 raw timing feature names "
            "(or 48 all feature names) to floats, OR a list of exactly 31 or 48 ordered float values."
        ),
        examples=[{
            "H.period": 0.1491, "DD.period.t": 0.3979, "UD.period.t": 0.2488,
            "H.t": 0.1069, "DD.t.i": 0.1674, "UD.t.i": 0.0605,
            "H.i": 0.1169, "DD.i.e": 0.2212, "UD.i.e": 0.1043,
            "H.e": 0.1008, "DD.e.five": 0.2801, "UD.e.five": 0.1793,
            "H.five": 0.1055, "DD.five.Shift.r": 0.3402, "UD.five.Shift.r": 0.2347,
            "H.Shift.r": 0.1141, "DD.Shift.r.o": 0.2709, "UD.Shift.r.o": 0.1568,
            "H.o": 0.0988, "DD.o.a": 0.2104, "UD.o.a": 0.1116,
            "H.a": 0.1349, "DD.a.n": 0.1484, "UD.a.n": 0.0135,
            "H.n": 0.0932, "DD.n.l": 0.3515, "UD.n.l": 0.2583,
            "H.l": 0.1338, "DD.l.Return": 0.3509, "UD.l.Return": 0.2171,
            "H.Return": 0.0742
        }]
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if isinstance(v, list):
            if len(v) not in (len(RAW_TIMING_COLUMNS), len(ALL_FEATURE_COLUMNS)):
                raise ValueError(
                    f"Features list must contain either {len(RAW_TIMING_COLUMNS)} raw timing values "
                    f"or {len(ALL_FEATURE_COLUMNS)} full feature values. Received: {len(v)}"
                )
            for i, val in enumerate(v):
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Feature at index {i} is invalid (NaN, Inf, or non-numeric): {val}")
        elif isinstance(v, dict):
            if len(v) == 0:
                raise ValueError("Features dictionary cannot be empty.")
            for key, val in v.items():
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Feature '{key}' is invalid (NaN, Inf, or non-numeric): {val}")
        else:
            raise ValueError("Features must be a dictionary or a list of floats.")
        return v


class PredictResponse(BaseModel):
    """Response payload for behavioral subject prediction."""

    predicted_subject: str = Field(..., description="Predicted subject identifier (e.g., 's002').")
    confidence: float = Field(..., description="Estimated probability/confidence score (0.0 to 1.0).")
    top_predictions: Dict[str, float] = Field(..., description="Top 5 subject candidates and their probabilities.")
    model_used: str = Field(..., description="Classifier model architecture used for inference.")
    status: str = Field(default="success")


class AnomalyRequest(BaseModel):
    """Input payload for behavioral anomaly detection."""

    features: Union[Dict[str, float], List[float]] = Field(
        ...,
        description="Keystroke timing features (31 raw or 48 engineered).",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        return PredictRequest.validate_features(v)


class AnomalyResponse(BaseModel):
    """Response payload for autoencoder behavioral anomaly detection."""

    reconstruction_error: float = Field(..., description="Mean squared reconstruction error of feature vector.")
    threshold: float = Field(..., description="Calibrated anomaly decision threshold.")
    anomaly_score: float = Field(..., description="Normalized anomaly ratio (error / threshold).")
    is_anomaly: bool = Field(..., description="True if reconstruction error exceeds threshold.")
    assessment: str = Field(..., description="Descriptive assessment of behavioral familiar/unfamiliar pattern.")
    status: str = Field(default="success")


class ModelInfoResponse(BaseModel):
    """Project and model metadata response."""

    project: str
    version: str
    best_classifier: str
    num_classes: int
    classes: List[str]
    feature_count: int
    expected_features: List[str]
    raw_timing_features: List[str]
    anomaly_threshold: Optional[float]
    metrics_summary: Optional[Dict[str, Any]]


class HealthResponse(BaseModel):
    """API health status response."""

    status: str
    models_loaded: bool
    classifier_available: bool
    autoencoder_available: bool
    artifacts: Dict[str, bool]
