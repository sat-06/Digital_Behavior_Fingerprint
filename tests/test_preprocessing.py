"""Tests for preprocessing, schema validation, and session splitting."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing.loader import (
    load_raw_dataset,
    validate_schema,
    clean_dataset,
    DatasetValidationError,
)
from src.preprocessing.splitter import (
    session_based_split,
    fit_scaler,
    transform_features,
    fit_label_encoder,
)
from src.config import EXPECTED_RAW_COLUMNS, RAW_TIMING_COLUMNS


def test_load_raw_dataset():
    """Verify raw dataset can be loaded cleanly."""
    df = load_raw_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df["subject"].nunique() == 51
    for col in EXPECTED_RAW_COLUMNS:
        assert col in df.columns


def test_validate_schema_missing_column():
    """Schema validation should raise error when required columns are missing."""
    bad_df = pd.DataFrame({"subject": ["s002"], "sessionIndex": [1]})
    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        validate_schema(bad_df)


def test_validate_schema_empty_df():
    """Schema validation should raise error on empty DataFrame."""
    empty_df = pd.DataFrame(columns=EXPECTED_RAW_COLUMNS)
    with pytest.raises(DatasetValidationError, match="empty"):
        validate_schema(empty_df)


def test_clean_dataset_removes_duplicates_and_nans():
    """Dataset cleaning should drop duplicates and nan rows."""
    dummy_data = {col: [0.1, 0.1, np.nan] for col in RAW_TIMING_COLUMNS}
    dummy_data["subject"] = ["s002", "s002", "s003"]
    dummy_data["sessionIndex"] = [1, 1, 1]
    dummy_data["rep"] = [1, 1, 2]
    df = pd.DataFrame(dummy_data)

    cleaned = clean_dataset(df)
    assert len(cleaned) == 1  # 1 duplicate dropped, 1 NaN row dropped
    assert cleaned.iloc[0]["subject"] == "s002"


def test_session_based_split():
    """Session split should strictly separate train and test sessions without data leakage."""
    dummy_rows = []
    for subj in ["s002", "s003"]:
        for sess in range(1, 9):
            row = {col: 0.1 for col in RAW_TIMING_COLUMNS}
            row["subject"] = subj
            row["sessionIndex"] = sess
            row["rep"] = 1
            dummy_rows.append(row)
    df = pd.DataFrame(dummy_rows)

    train_df, test_df = session_based_split(df, train_sessions=[1, 2, 3], test_sessions=[4, 5])
    assert set(train_df["sessionIndex"].unique()) == {1, 2, 3}
    assert set(test_df["sessionIndex"].unique()) == {4, 5}
    assert len(set(train_df["sessionIndex"]).intersection(set(test_df["sessionIndex"]))) == 0
    assert train_df["subject"].nunique() == 2
    assert test_df["subject"].nunique() == 2


def test_scaler_fitting_and_transformation():
    """Scaler must fit strictly on train and transform features with consistent shapes."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[2.0, 3.0]])

    scaler = fit_scaler(X_train)
    X_train_scaled = transform_features(scaler, X_train)
    X_test_scaled = transform_features(scaler, X_test)

    assert X_train_scaled.shape == (3, 2)
    assert X_test_scaled.shape == (1, 2)
    # Train scaled mean should be approximately 0
    np.testing.assert_almost_equal(X_train_scaled.mean(axis=0), [0.0, 0.0])
