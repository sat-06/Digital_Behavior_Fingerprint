"""Leakage-free dataset splitting and feature scaling module."""

import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import TRAIN_SESSIONS, TEST_SESSIONS

logger = logging.getLogger(__name__)


def session_based_split(
    df: pd.DataFrame,
    train_sessions: Optional[List[int]] = None,
    test_sessions: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by session indices to prevent temporal session leakage.

    This ensures observations from test sessions (e.g., sessions 7 and 8)
    never appear in training sessions (e.g., sessions 1 to 6), testing true
    longitudinal behavioral generalization across different typing dates.

    Args:
        df: Input DataFrame containing 'sessionIndex' and 'subject'.
        train_sessions: List of session indices for training. Defaults to TRAIN_SESSIONS (1-6).
        test_sessions: List of session indices for testing. Defaults to TEST_SESSIONS (7-8).

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If session splits overlap or are invalid.
    """
    train_s = train_sessions if train_sessions is not None else TRAIN_SESSIONS
    test_s = test_sessions if test_sessions is not None else TEST_SESSIONS

    overlap = set(train_s).intersection(set(test_s))
    if overlap:
        raise ValueError(f"Train and test sessions must not overlap. Found overlapping sessions: {overlap}")

    train_mask = df["sessionIndex"].isin(train_s)
    test_mask = df["sessionIndex"].isin(test_s)

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Resulting split has empty data: train rows={len(train_df)}, test rows={len(test_df)}"
        )

    # Check subjects in both splits
    train_subjects = set(train_df["subject"].unique())
    test_subjects = set(test_df["subject"].unique())
    diff = test_subjects - train_subjects
    if diff:
        logger.warning(f"Test set contains subjects not in training set: {diff}")

    logger.info(
        f"Session-based split completed: Train={len(train_df)} rows ({len(train_subjects)} subjects, sessions {train_s}), "
        f"Test={len(test_df)} rows ({len(test_subjects)} subjects, sessions {test_s})"
    )
    return train_df, test_df


def fit_scaler(X_train: Union[pd.DataFrame, np.ndarray]) -> StandardScaler:
    """Fit a standard scaler strictly on training data features only.

    Args:
        X_train: Training features.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(
    scaler: StandardScaler,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Transform features using a pre-fitted scaler.

    Args:
        scaler: Fitted StandardScaler.
        X: Feature matrix or DataFrame.
        feature_names: Optional feature ordering for DataFrames.

    Returns:
        Scaled numpy array.
    """
    if isinstance(X, pd.DataFrame) and feature_names is not None:
        X_df = X[feature_names]
    elif isinstance(X, np.ndarray) and feature_names is not None:
        X_df = pd.DataFrame(X, columns=feature_names)
    elif isinstance(X, pd.DataFrame):
        X_df = X
    else:
        return scaler.transform(X)
    return scaler.transform(X_df)


def fit_label_encoder(y_train: Union[pd.Series, List[str], np.ndarray]) -> LabelEncoder:
    """Fit a label encoder on subject string labels.

    Args:
        y_train: Array or Series of subject labels (e.g. 's002').

    Returns:
        Fitted LabelEncoder.
    """
    le = LabelEncoder()
    le.fit(y_train)
    return le
