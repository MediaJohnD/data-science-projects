"""Data splitting utilities for time-aware and geographic holdout."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def time_aware_split(
    df: pd.DataFrame,
    time_column: str,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/test respecting temporal order."""
    df_sorted = df.sort_values(time_column)
    split_point = int(len(df_sorted) * (1 - test_size))
    train_idx = df_sorted.index[:split_point].to_numpy()
    test_idx = df_sorted.index[split_point:].to_numpy()
    return train_idx, test_idx


def geo_holdout_split(
    df: pd.DataFrame,
    geo_column: str,
    holdout_geos: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices holding out specified geographic segments."""
    test_mask = df[geo_column].isin(holdout_geos)
    test_idx = df[test_mask].index.to_numpy()
    train_idx = df[~test_mask].index.to_numpy()
    return train_idx, test_idx
