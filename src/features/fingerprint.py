"""Session-level and windowed behavioral fingerprint generation module."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_FINGERPRINTS_PATH,
    HOLD_COLUMNS,
    DD_COLUMNS,
    UD_COLUMNS,
    RAW_TIMING_COLUMNS,
    ENGINEERED_FEATURE_NAMES,
)
from src.features.extractor import extract_features

logger = logging.getLogger(__name__)


def generate_session_fingerprints(
    df: pd.DataFrame,
    window_reps: Optional[int] = None,
    save_path: Optional[Path] = PROCESSED_FINGERPRINTS_PATH,
) -> pd.DataFrame:
    """Generate session-level behavioral fingerprints with multiple samples per subject.

    Instead of compressing all records for a subject into a single row,
    this function preserves session boundaries (and optional sub-session windows)
    to retain multiple observations per subject across distinct sessions.

    Args:
        df: Input DataFrame with raw keystroke measurements and ID columns.
        window_reps: Optional number of repetitions to group within each session
                     (e.g., 10 reps yields 5 fingerprints per session, 40 per subject = 2,040 rows).
                     If None, aggregates by (subject, sessionIndex), yielding 8 fingerprints
                     per subject (408 total rows).
        save_path: Destination path for the processed CSV. Set to None to skip saving.

    Returns:
        pd.DataFrame of behavioral fingerprints with multiple samples per subject.
    """
    df_feat = extract_features(df)

    group_cols = ["subject", "sessionIndex"]
    if window_reps is not None and window_reps > 0:
        # Assign chunk ID based on repetition index within session
        df_feat["window_id"] = (df_feat["rep"] - 1) // window_reps
        group_cols.append("window_id")

    # Group and aggregate timing features
    agg_dict = {col: "mean" for col in RAW_TIMING_COLUMNS}
    for eng_col in ENGINEERED_FEATURE_NAMES:
        agg_dict[eng_col] = "mean"

    # Also compute intra-session standard deviation of hold, dd, ud
    fingerprints = df_feat.groupby(group_cols)[list(agg_dict.keys())].mean().reset_index()

    # Calculate overall typing speed for the session fingerprint
    eps = 1e-6
    fingerprints["typing_speed"] = 1.0 / (
        fingerprints["avg_hold_time"] + fingerprints["avg_dd_time"] + eps
    )

    logger.info(
        f"Generated behavioral fingerprints: shape={fingerprints.shape}, "
        f"subjects={fingerprints['subject'].nunique()}, "
        f"observations per subject={len(fingerprints) / fingerprints['subject'].nunique():.1f}"
    )

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprints.to_csv(out_path, index=False)
        logger.info(f"Saved session-level fingerprints to: {out_path}")

    return fingerprints
