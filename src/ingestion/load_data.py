"""Data ingestion utilities for the OptiReveal pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def run(path: Optional[str] = None) -> pd.DataFrame:
    """Load raw visit data.

    Parameters
    ----------
    path:
        Optional path to a CSV file containing visit events. When omitted, a
        small in-memory data set is returned which is useful for unit tests
        and demonstrations.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``device_id``, ``timestamp`` and ``poi_id`` columns.
    """

    if path is None:
        # Create a small synthetic data set with multiple devices so that
        # model evaluation techniques have enough samples.
        frames = []
        devices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        for device_id in devices:
            frames.append(
                pd.DataFrame(
                    {
                        "device_id": [device_id, device_id],
                        "timestamp": pd.date_range(
                            "2023-01-01", periods=2, freq="h"
                        ),
                        "poi_id": (
                            [1, 2]
                            if device_id not in {"J", "K", "L"}
                            else [1, 1]
                        ),
                    }
                )
            )

        return pd.concat(frames, ignore_index=True)

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)
