"""Data loading and schema validation module for keystroke dynamics."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATASET_PATH,
    EXPECTED_RAW_COLUMNS,
    ID_COLUMNS,
    RAW_TIMING_COLUMNS,
    HOLD_COLUMNS,
    DD_COLUMNS,
    UD_COLUMNS,
)

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Exception raised when dataset schema or contents are invalid."""
    pass


def load_raw_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Keystroke Dynamics dataset from disk.

    Args:
        path: Optional file path. Defaults to RAW_DATASET_PATH.

    Returns:
        pd.DataFrame containing the raw keystroke dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        DatasetValidationError: If the dataset does not meet schema requirements.
    """
    file_path = Path(path) if path else RAW_DATASET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {file_path}")

    logger.info(f"Loading raw keystroke dataset from: {file_path}")
    df = pd.read_csv(file_path)

    validate_schema(df)
    df_clean = clean_dataset(df)
    logger.info(
        f"Successfully loaded dataset with {len(df_clean)} rows and {len(df_clean.columns)} columns across "
        f"{df_clean['subject'].nunique()} subjects."
    )
    return df_clean


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame contains expected columns and appropriate types.

    Args:
        df: DataFrame to validate.

    Returns:
        True if valid.

    Raises:
        DatasetValidationError: If validation fails.
    """
    missing_cols = [col for col in EXPECTED_RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Missing required columns in dataset: {missing_cols}")

    if df.empty:
        raise DatasetValidationError("Dataset is empty.")

    # Check ID column types
    if not pd.api.types.is_object_dtype(df["subject"]) and not pd.api.types.is_string_dtype(df["subject"]):
        raise DatasetValidationError("Column 'subject' must be string/object type.")

    for num_col in ["sessionIndex", "rep"]:
        if not pd.api.types.is_numeric_dtype(df[num_col]):
            raise DatasetValidationError(f"Column '{num_col}' must be numeric.")

    # Check timing columns are numeric
    for timing_col in RAW_TIMING_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[timing_col]):
            raise DatasetValidationError(f"Timing column '{timing_col}' must be float/numeric.")

    return True


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the keystroke dataset: handle duplicates, missing values, and infinities.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned pd.DataFrame.
    """
    df_clean = df.copy()

    # Drop exact duplicate rows if any
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found and removing {duplicate_count} duplicate rows.")
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Check and handle NaN / Null values
    null_counts = df_clean.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values across dataset. Dropping null rows.")
        df_clean = df_clean.dropna().reset_index(drop=True)

    # Check for infinite values in numerical timing columns
    timing_vals = df_clean[RAW_TIMING_COLUMNS].values
    if np.isinf(timing_vals).any():
        logger.warning("Found infinite values in timing columns. Filtering them out.")
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Ensure subject is string type (e.g. 's002')
    df_clean["subject"] = df_clean["subject"].astype(str)
    df_clean["sessionIndex"] = df_clean["sessionIndex"].astype(int)
    df_clean["rep"] = df_clean["rep"].astype(int)

    return df_clean
