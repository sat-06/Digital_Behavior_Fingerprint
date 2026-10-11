"""Tests for feature extraction and behavioral fingerprint generation."""

import pytest
import pandas as pd
import numpy as np

from src.features.extractor import extract_sample_features, extract_features
from src.features.fingerprint import generate_session_fingerprints
from src.config import (
    RAW_TIMING_COLUMNS,
    HOLD_COLUMNS,
    DD_COLUMNS,
    UD_COLUMNS,
    ENGINEERED_FEATURE_NAMES,
    ALL_FEATURE_COLUMNS,
)


def create_sample_raw_dict():
    """Create dummy 31 raw timing features."""
    sample = {}
    for col in HOLD_COLUMNS:
        sample[col] = 0.10
    for col in DD_COLUMNS:
        sample[col] = 0.20
    for col in UD_COLUMNS:
        sample[col] = 0.10
    return sample


def test_extract_sample_features():
    """Verify single-sample feature extraction creates all 17 engineered features."""
    sample = create_sample_raw_dict()
    extracted = extract_sample_features(sample)

    assert len(extracted) == len(ALL_FEATURE_COLUMNS)
    for col in ALL_FEATURE_COLUMNS:
        assert col in extracted

    # Average hold should be 0.10, average dd should be 0.20
    assert abs(extracted["avg_hold_time"] - 0.10) < 1e-4
    assert abs(extracted["avg_dd_time"] - 0.20) < 1e-4
    assert extracted["typing_speed"] > 0


def test_extract_features_dataframe():
    """Verify DataFrame feature extraction correctly vectorizes over rows."""
    row = create_sample_raw_dict()
    df = pd.DataFrame([row, row, row])
    df_feat = extract_features(df)

    assert len(df_feat) == 3
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in df_feat.columns
    assert np.all(df_feat["typing_speed"].values > 0)


def test_generate_session_fingerprints():
    """Verify session fingerprints are generated without collapsing all data into 1 row."""
    rows = []
    for sess in [1, 2, 3]:
        for rep in range(1, 11):
            row = create_sample_raw_dict()
            row["subject"] = "s002"
            row["sessionIndex"] = sess
            row["rep"] = rep
            rows.append(row)
    df = pd.DataFrame(rows)

    fingerprints = generate_session_fingerprints(df, save_path=None)
    # Should have 3 fingerprints (one per session), NOT 1 sample
    assert len(fingerprints) == 3
    assert fingerprints["subject"].nunique() == 1
    assert "typing_speed" in fingerprints.columns
