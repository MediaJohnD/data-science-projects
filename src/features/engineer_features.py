"""Feature engineering utilities for OptiReveal."""

from __future__ import annotations

import pandas as pd


def run(visits: pd.DataFrame) -> pd.DataFrame:
    """Derive basic visit features.

    Parameters
    ----------
    visits:
        Raw visit records with ``device_id`` and ``poi_id``.

    Returns
    -------
    pandas.DataFrame
        Feature table aggregated by ``device_id``.
    """

    features = (
        visits.groupby("device_id")
        .agg(visit_count=("poi_id", "size"), unique_pois=("poi_id", "nunique"))
        .reset_index()
    )
    return features
