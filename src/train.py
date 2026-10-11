"""Central training and evaluation pipeline for Digital Behavior Fingerprint."""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_SESSIONS,
    TEST_SESSIONS,
    PROCESSED_FINGERPRINTS_PATH,
    EVALUATION_METRICS_PATH,
)
from src.preprocessing import (
    load_raw_dataset,
    session_based_split,
    fit_scaler,
    transform_features,
    fit_label_encoder,
)
from src.features import extract_features, generate_session_fingerprints
from src.models import (
    train_and_compare_all,
    train_neural_network,
    evaluate_neural_network,
    train_autoencoder,
    detect_anomalies,
    save_classical_artifacts,
    save_pytorch_classifier,
    save_autoencoder_artifacts,
)
from src.evaluation import compute_classification_metrics, evaluate_anomaly_detection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_pipeline")


def run_pipeline() -> dict:
    """Execute the end-to-end training and evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("Starting Digital Behavior Fingerprint Training & Evaluation Pipeline")
    logger.info("=" * 70)

    # 1. Load raw dataset
    df_raw = load_raw_dataset()

    # 2. Generate and persist session-level behavioral fingerprints
    logger.info("Generating session-level behavioral fingerprints...")
    fingerprints_df = generate_session_fingerprints(
        df_raw, window_reps=None, save_path=PROCESSED_FINGERPRINTS_PATH
    )
    logger.info(
        f"Saved {len(fingerprints_df)} session fingerprints to {PROCESSED_FINGERPRINTS_PATH} "
        f"({fingerprints_df['subject'].nunique()} subjects, 8 sessions per subject)."
    )

    # 3. Extract features across all observations
    logger.info("Extracting engineered behavioral features (timing + statistics)...")
    df_features = extract_features(df_raw)

    # 4. Session-based Train/Test Split (Zero session leakage)
    logger.info(
        f"Splitting dataset by sessions: Train sessions={TRAIN_SESSIONS}, Test sessions={TEST_SESSIONS}"
    )
    train_df, test_df = session_based_split(
        df_features, train_sessions=TRAIN_SESSIONS, test_sessions=TEST_SESSIONS
    )

    # Separate X and y
    X_train_df = train_df[ALL_FEATURE_COLUMNS]
    y_train_raw = train_df["subject"].values
    X_test_df = test_df[ALL_FEATURE_COLUMNS]
    y_test_raw = test_df["subject"].values

    # 5. Fit Preprocessing Objects Strictly on Training Set
    logger.info("Fitting StandardScaler on training data only...")
    scaler = fit_scaler(X_train_df)
    X_train_scaled = transform_features(scaler, X_train_df, ALL_FEATURE_COLUMNS)
    X_test_scaled = transform_features(scaler, X_test_df, ALL_FEATURE_COLUMNS)

    logger.info("Fitting LabelEncoder on training subject labels...")
    label_encoder = fit_label_encoder(y_train_raw)
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)

    logger.info(f"Features: {X_train_scaled.shape[1]} dimensions | Classes: {num_classes} subjects")
    logger.info(f"Train samples: {len(X_train_scaled)} | Test samples: {len(X_test_scaled)}")

    # 6. Train and Evaluate Classical ML Models
    logger.info("-" * 70)
    logger.info("Stage 1: Training Classical Machine Learning Models")
    logger.info("-" * 70)
    classical_models, comparison_df, best_name = train_and_compare_all(
        X_train_scaled, y_train, X_test_scaled, y_test, seed=RANDOM_SEED
    )
    best_classical_model = classical_models[best_name]

    # Compute comprehensive metrics on best classical model
    y_pred_best = best_classical_model.predict(X_test_scaled)
    classical_eval = compute_classification_metrics(y_test, y_pred_best)

    # 7. Train and Evaluate Deep Learning PyTorch Model
    logger.info("-" * 70)
    logger.info("Stage 2: Training PyTorch Neural Network (MLP)")
    logger.info("-" * 70)
    pytorch_model, pytorch_metrics = train_neural_network(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_test_scaled,
        y_val=y_test,
        num_classes=num_classes,
        epochs=35,
        batch_size=128,
        seed=RANDOM_SEED,
    )

    # 8. Train PyTorch Autoencoder for Anomaly Detection
    logger.info("-" * 70)
    logger.info("Stage 3: Training PyTorch Autoencoder for Anomaly Detection")
    logger.info("-" * 70)
    autoencoder_model, threshold, autoencoder_metrics = train_autoencoder(
        X_train=X_train_scaled,
        epochs=35,
        batch_size=128,
        threshold_percentile=95.0,
        seed=RANDOM_SEED,
    )

    # Evaluate Anomaly Detection
    # Legitimate test errors
    test_recon_errors, _ = detect_anomalies(autoencoder_model, X_test_scaled, threshold)
    # Simulated imposter/anomalous samples (perturbed timing patterns, e.g. Gaussian noise & random shift)
    rng = np.random.RandomState(RANDOM_SEED)
    noise = rng.normal(loc=1.5, scale=1.0, size=X_test_scaled.shape)
    X_imposter_scaled = X_test_scaled + noise
    imposter_recon_errors, _ = detect_anomalies(autoencoder_model, X_imposter_scaled, threshold)

    anomaly_eval = evaluate_anomaly_detection(test_recon_errors, imposter_recon_errors, threshold)
    logger.info(
        f"Anomaly Detection -> ROC-AUC: {anomaly_eval['roc_auc']:.4f}, "
        f"TPR (Imposter Detection): {anomaly_eval['tpr_detection_rate'] * 100:.1f}%, "
        f"FPR (False Alarm): {anomaly_eval['fpr_false_alarm_rate'] * 100:.1f}%"
    )

    # 9. Model Persistence
    logger.info("-" * 70)
    logger.info("Stage 4: Persisting Model Artifacts")
    logger.info("-" * 70)

    # Append PyTorch model to comparison table
    nn_row = pd.DataFrame([{
        "Model": "PyTorch Neural Network",
        "Accuracy": pytorch_metrics["accuracy"],
        "Precision (Macro)": pytorch_metrics["precision_macro"],
        "Recall (Macro)": pytorch_metrics["recall_macro"],
        "F1-Score (Macro)": pytorch_metrics["f1_macro"],
    }])
    full_comparison = pd.concat([comparison_df, nn_row], ignore_index=True).sort_values(
        by="F1-Score (Macro)", ascending=False
    )

    overall_metrics = {
        "best_classical_model": best_name,
        "classical_models_comparison": full_comparison.to_dict(orient="records"),
        "best_classical_metrics": {
            "accuracy": classical_eval["accuracy"],
            "precision_macro": classical_eval["precision_macro"],
            "recall_macro": classical_eval["recall_macro"],
            "f1_macro": classical_eval["f1_macro"],
        },
        "pytorch_neural_network_metrics": pytorch_metrics,
        "autoencoder_anomaly_metrics": anomaly_eval,
        "split_strategy": {
            "type": "Session-Based Split (Zero Session Leakage)",
            "train_sessions": TRAIN_SESSIONS,
            "test_sessions": TEST_SESSIONS,
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
            "num_classes": num_classes,
        },
    }

    save_classical_artifacts(
        model=best_classical_model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_names=ALL_FEATURE_COLUMNS,
        metrics=overall_metrics,
    )
    save_pytorch_classifier(pytorch_model)
    save_autoencoder_artifacts(autoencoder_model, anomaly_config=anomaly_eval)

    logger.info("=" * 70)
    logger.info("Training & Evaluation Pipeline Finished Successfully!")
    logger.info("=" * 70)
    logger.info("\nFinal Model Comparison Table:\n" + full_comparison.to_string(index=False))

    return overall_metrics


if __name__ == "__main__":
    run_pipeline()
