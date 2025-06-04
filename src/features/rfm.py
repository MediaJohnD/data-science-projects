"""Utility functions to compute RFM (Recency, Frequency, Monetary) features."""

from __future__ import annotations

from datetime import datetime
import pandas as pd


def compute_rfm(
    df: pd.DataFrame,
    customer_id_col: str,
    timestamp_col: str,
    amount_col: str,
) -> pd.DataFrame:
    """Return a DataFrame of RFM features aggregated by customer ID."""
    if not {customer_id_col, timestamp_col, amount_col}.issubset(df.columns):
        missing = {customer_id_col, timestamp_col, amount_col} - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    df = df[[customer_id_col, timestamp_col, amount_col]].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    if df[timestamp_col].isna().any():
        raise ValueError("Invalid timestamps detected")

    now = df[timestamp_col].max()
    grouped = df.groupby(customer_id_col)
    recency = (now - grouped[timestamp_col].max()).dt.days.rename("recency")
    frequency = grouped[timestamp_col].count().rename("frequency")
    monetary = grouped[amount_col].mean().rename("monetary")

    result = pd.concat([recency, frequency, monetary], axis=1).reset_index()
    return result
