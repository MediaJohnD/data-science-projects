"""Data validation utilities for ingestion."""

from typing import Iterable

import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that required columns exist in the DataFrame.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

