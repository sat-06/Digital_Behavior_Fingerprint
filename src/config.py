"""Configuration constants and paths for Digital Behavior Fingerprint."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATASET_PATH = DATA_RAW_DIR / "DSL-StrongPasswordData.csv"
PROCESSED_FINGERPRINTS_PATH = DATA_PROCESSED_DIR / "keystroke_fingerprints.csv"

# Saved Model Artifact Paths
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
CLASSICAL_MODEL_PATH = MODELS_DIR / "best_classifier.joblib"
ALL_CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"
PYTORCH_CLASSIFIER_PATH = MODELS_DIR / "pytorch_classifier.pt"
AUTOENCODER_PATH = MODELS_DIR / "autoencoder.pt"
METADATA_PATH = MODELS_DIR / "feature_metadata.json"
ANOMALY_CONFIG_PATH = MODELS_DIR / "anomaly_config.json"
EVALUATION_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Column Definitions for DSL-StrongPasswordData
ID_COLUMNS = ["subject", "sessionIndex", "rep"]

HOLD_COLUMNS = [
    "H.period",
    "H.t",
    "H.i",
    "H.e",
    "H.five",
    "H.Shift.r",
    "H.o",
    "H.a",
    "H.n",
    "H.l",
    "H.Return",
]

DD_COLUMNS = [
    "DD.period.t",
    "DD.t.i",
    "DD.i.e",
    "DD.e.five",
    "DD.five.Shift.r",
    "DD.Shift.r.o",
    "DD.o.a",
    "DD.a.n",
    "DD.n.l",
    "DD.l.Return",
]

UD_COLUMNS = [
    "UD.period.t",
    "UD.t.i",
    "UD.i.e",
    "UD.e.five",
    "UD.five.Shift.r",
    "UD.Shift.r.o",
    "UD.o.a",
    "UD.a.n",
    "UD.n.l",
    "UD.l.Return",
]

RAW_TIMING_COLUMNS = HOLD_COLUMNS + DD_COLUMNS + UD_COLUMNS
EXPECTED_RAW_COLUMNS = ID_COLUMNS + RAW_TIMING_COLUMNS

# Summary Statistical Features added during Feature Engineering
ENGINEERED_FEATURE_NAMES = [
    "avg_hold_time",
    "avg_dd_time",
    "avg_ud_time",
    "hold_time_std",
    "dd_time_std",
    "ud_time_std",
    "min_hold_time",
    "max_hold_time",
    "median_hold_time",
    "min_dd_time",
    "max_dd_time",
    "median_dd_time",
    "min_ud_time",
    "max_ud_time",
    "median_ud_time",
    "typing_speed",
    "hold_to_dd_ratio",
]

# Total Feature Columns = 31 raw timing + 17 engineered = 48 features
ALL_FEATURE_COLUMNS = RAW_TIMING_COLUMNS + ENGINEERED_FEATURE_NAMES

# Reproducibility
RANDOM_SEED = 42

# Session Splitting Strategy
# Sessions 1-6 for training (75%), Sessions 7-8 for evaluation (25%)
TRAIN_SESSIONS = [1, 2, 3, 4, 5, 6]
TEST_SESSIONS = [7, 8]
