"""Feature extraction module for keystroke dynamics."""

import logging
from typing import Dict, Union
import numpy as np
import pandas as pd

from src.config import (
    HOLD_COLUMNS,
    DD_COLUMNS,
    UD_COLUMNS,
    RAW_TIMING_COLUMNS,
    ENGINEERED_FEATURE_NAMES,
    ALL_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def extract_sample_features(sample: Dict[str, float]) -> Dict[str, float]:
    """Extract engineered behavioral features from a single dictionary of 31 raw timing features.

    Args:
        sample: Dictionary containing all 31 raw timing measurements.

    Returns:
        Dictionary containing all 48 features (31 raw + 17 engineered).

    Raises:
        ValueError: If any required raw timing column is missing.
    """
    missing = [c for c in RAW_TIMING_COLUMNS if c not in sample]
    if missing:
        raise ValueError(f"Missing required raw timing features: {missing}")

    hold_vals = np.array([sample[c] for c in HOLD_COLUMNS], dtype=float)
    dd_vals = np.array([sample[c] for c in DD_COLUMNS], dtype=float)
    ud_vals = np.array([sample[c] for c in UD_COLUMNS], dtype=float)

    avg_hold = float(np.mean(hold_vals))
    avg_dd = float(np.mean(dd_vals))
    avg_ud = float(np.mean(ud_vals))

    hold_std = float(np.std(hold_vals, ddof=0))
    dd_std = float(np.std(dd_vals, ddof=0))
    ud_std = float(np.std(ud_vals, ddof=0))

    min_hold = float(np.min(hold_vals))
    max_hold = float(np.max(hold_vals))
    median_hold = float(np.median(hold_vals))

    min_dd = float(np.min(dd_vals))
    max_dd = float(np.max(dd_vals))
    median_dd = float(np.median(dd_vals))

    min_ud = float(np.min(ud_vals))
    max_ud = float(np.max(ud_vals))
    median_ud = float(np.median(ud_vals))

    eps = 1e-6
    typing_speed = 1.0 / (avg_hold + avg_dd + eps)
    hold_to_dd = avg_hold / (avg_dd + eps)

    features = dict(sample)
    features.update({
        "avg_hold_time": avg_hold,
        "avg_dd_time": avg_dd,
        "avg_ud_time": avg_ud,
        "hold_time_std": hold_std,
        "dd_time_std": dd_std,
        "ud_time_std": ud_std,
        "min_hold_time": min_hold,
        "max_hold_time": max_hold,
        "median_hold_time": median_hold,
        "min_dd_time": min_dd,
        "max_dd_time": max_dd,
        "median_dd_time": median_dd,
        "min_ud_time": min_ud,
        "max_ud_time": max_ud,
        "median_ud_time": median_ud,
        "typing_speed": typing_speed,
        "hold_to_dd_ratio": hold_to_dd,
    })
    return features


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract engineered features across a DataFrame of keystroke dynamics data.

    Vectorized implementation for efficiency on large tabular datasets.

    Args:
        df: DataFrame containing at least the 31 raw timing columns.

    Returns:
        DataFrame with original columns plus 17 engineered behavioral features.
    """
    missing = [c for c in RAW_TIMING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw timing features in DataFrame: {missing}")

    result = df.copy()

    hold_data = result[HOLD_COLUMNS].values
    dd_data = result[DD_COLUMNS].values
    ud_data = result[UD_COLUMNS].values

    avg_hold = np.mean(hold_data, axis=1)
    avg_dd = np.mean(dd_data, axis=1)
    avg_ud = np.mean(ud_data, axis=1)

    result["avg_hold_time"] = avg_hold
    result["avg_dd_time"] = avg_dd
    result["avg_ud_time"] = avg_ud

    result["hold_time_std"] = np.std(hold_data, axis=1)
    result["dd_time_std"] = np.std(dd_data, axis=1)
    result["ud_time_std"] = np.std(ud_data, axis=1)

    result["min_hold_time"] = np.min(hold_data, axis=1)
    result["max_hold_time"] = np.max(hold_data, axis=1)
    result["median_hold_time"] = np.median(hold_data, axis=1)

    result["min_dd_time"] = np.min(dd_data, axis=1)
    result["max_dd_time"] = np.max(dd_data, axis=1)
    result["median_dd_time"] = np.median(dd_data, axis=1)

    result["min_ud_time"] = np.min(ud_data, axis=1)
    result["max_ud_time"] = np.max(ud_data, axis=1)
    result["median_ud_time"] = np.median(ud_data, axis=1)

    eps = 1e-6
    result["typing_speed"] = 1.0 / (avg_hold + avg_dd + eps)
    result["hold_to_dd_ratio"] = avg_hold / (avg_dd + eps)

    logger.debug(f"Extracted 17 engineered features. New shape: {result.shape}")
    return result
