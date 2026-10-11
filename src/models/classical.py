"""Classical machine learning models for keystroke dynamics subject classification."""

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def get_classical_models(seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Instantiate the 5 required classical machine learning models.

    Args:
        seed: Random state seed for reproducibility.

    Returns:
        Dictionary mapping model names to initialized scikit-learn estimators.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=16,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
        ),
    }
    return models


def train_and_evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Train a single model and compute standard classification metrics on test data.

    Args:
        model: Scikit-learn classifier.
        X_train: Scaled training features.
        y_train: Encoded training labels.
        X_test: Scaled test features.
        y_test: Encoded test labels.

    Returns:
        Dictionary of evaluation metrics: accuracy, precision_macro, recall_macro, f1_macro.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }


def train_and_compare_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Train and evaluate all classical models, returning models, comparison table, and best model name.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        X_test: Scaled test features.
        y_test: Test labels.
        seed: Random seed.

    Returns:
        Tuple of (trained_models_dict, comparison_df, best_model_name).
    """
    models = get_classical_models(seed=seed)
    results = {}
    metrics_list = []

    logger.info("Starting training and evaluation of classical ML models...")
    for name, model in models.items():
        logger.info(f"Training {name}...")
        metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = model
        metrics_list.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision (Macro)": metrics["precision_macro"],
            "Recall (Macro)": metrics["recall_macro"],
            "F1-Score (Macro)": metrics["f1_macro"],
        })
        logger.info(
            f"{name} -> Accuracy: {metrics['accuracy']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}"
        )

    comparison_df = pd.DataFrame(metrics_list).sort_values(by="F1-Score (Macro)", ascending=False)
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Best classical model: {best_model_name} with F1-Macro={comparison_df.iloc[0]['F1-Score (Macro)']:.4f}")

    return results, comparison_df, best_model_name
