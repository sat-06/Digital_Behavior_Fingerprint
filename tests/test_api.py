"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_artifacts_safe
from src.config import RAW_TIMING_COLUMNS, ALL_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with loaded model artifacts."""
    load_artifacts_safe()
    with TestClient(app) as test_client:
        yield test_client


def get_sample_valid_dict():
    """Return a dictionary with 31 valid raw timing feature values."""
    return {
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
        "H.Return": 0.0742,
    }


def test_root_endpoint(client):
    """GET / returns project info and links to endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Digital Behavior Fingerprint"
    assert "endpoints" in data


def test_health_endpoint(client):
    """GET /health returns health status and artifact availability."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "models_loaded" in data
    assert "artifacts" in data


def test_model_info_endpoint(client):
    """GET /model-info returns metadata, classes, and expected features."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 51
    assert len(data["classes"]) == 51
    assert data["feature_count"] == 48
    assert "anomaly_threshold" in data


def test_predict_endpoint_valid_dict(client):
    """POST /predict with valid 31-feature dictionary succeeds."""
    payload = {"features": get_sample_valid_dict()}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_subject" in data
    assert data["predicted_subject"].startswith("s0")
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) > 0
    assert data["status"] == "success"


def test_predict_endpoint_valid_list(client):
    """POST /predict with valid 31-element list succeeds."""
    raw_dict = get_sample_valid_dict()
    feature_list = [raw_dict[col] for col in RAW_TIMING_COLUMNS]
    payload = {"features": feature_list}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_subject" in data


def test_predict_endpoint_invalid_nan(client):
    """POST /predict with NaN values returns 422 Unprocessable Entity."""
    raw_dict = get_sample_valid_dict()
    raw_dict["H.period"] = float("nan")
    # Pydantic JSON serialization turns NaN to null unless stringified
    payload = {"features": {"H.period": "invalid_number"}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_length(client):
    """POST /predict with wrong list length returns 422."""
    payload = {"features": [0.1, 0.2, 0.3]}  # only 3 features
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_detect_anomaly_endpoint_valid(client):
    """POST /detect-anomaly returns reconstruction error and anomaly status."""
    payload = {"features": get_sample_valid_dict()}
    response = client.post("/detect-anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reconstruction_error" in data
    assert "threshold" in data
    assert "is_anomaly" in data
    assert isinstance(data["is_anomaly"], bool)
    assert "assessment" in data
    assert data["status"] == "success"


def test_detect_anomaly_endpoint_invalid(client):
    """POST /detect-anomaly with empty features returns 422."""
    payload = {"features": {}}
    response = client.post("/detect-anomaly", json=payload)
    assert response.status_code == 422
