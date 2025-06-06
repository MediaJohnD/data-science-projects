"""Contextual trigger utilities."""

from __future__ import annotations

import pandas as pd


def run(features: pd.DataFrame) -> pd.DataFrame:
    """Create simple triggers based on visit counts."""

    return features.assign(trigger=features["visit_count"] > 1)
