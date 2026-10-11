"""Comprehensive evaluation metrics and reporting for classification and anomaly detection."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multi-class classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Model predicted labels.
        labels: Optional label list for class ordering.

    Returns:
        Dictionary of metrics including accuracy, macro/weighted precision,
        recall, f1-score, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate_anomaly_detection(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate autoencoder anomaly detection on normal vs anomalous samples.

    Args:
        normal_errors: Array of reconstruction errors for legitimate in-distribution samples.
        anomaly_errors: Array of reconstruction errors for simulated imposter/anomalous samples.
        threshold: The decision threshold.

    Returns:
        Dictionary containing TPR, FPR, ROC-AUC, error distribution stats.
    """
    y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    y_pred_anomaly = all_errors > threshold

    tpr = float(np.mean(anomaly_errors > threshold))  # True Positive Rate (Sensitivity)
    fpr = float(np.mean(normal_errors > threshold))   # False Positive Rate (Fall-out)

    try:
        auc = float(roc_auc_score(y_true, all_errors))
    except Exception:
        auc = 0.5

    return {
        "roc_auc": auc,
        "tpr_detection_rate": tpr,
        "fpr_false_alarm_rate": fpr,
        "threshold": float(threshold),
        "normal_error_mean": float(np.mean(normal_errors)),
        "normal_error_std": float(np.std(normal_errors)),
        "anomaly_error_mean": float(np.mean(anomaly_errors)),
        "anomaly_error_std": float(np.std(anomaly_errors)),
        "error_separation_ratio": float(np.mean(anomaly_errors) / (np.mean(normal_errors) + 1e-6)),
    }
