"""Specialized CSV loaders for different dataset types."""

from pathlib import Path
import pandas as pd

from .csv_loader import load_csv_with_schema
from .schemas import (
    TRANSACTION_COLUMNS,
    LOCATION_COLUMNS,
    MEDIA_EXPOSURE_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    CAMPAIGN_RESULTS_COLUMNS,
)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transactional data and validate schema."""
    return load_csv_with_schema(path, TRANSACTION_COLUMNS)


def load_locations(path: str | Path) -> pd.DataFrame:
    """Load location data and validate schema."""
    return load_csv_with_schema(path, LOCATION_COLUMNS)


def load_media_exposures(path: str | Path) -> pd.DataFrame:
    """Load media exposure data and validate schema."""
    return load_csv_with_schema(path, MEDIA_EXPOSURE_COLUMNS)


def load_demographics(path: str | Path) -> pd.DataFrame:
    """Load demographic data and validate schema."""
    return load_csv_with_schema(path, DEMOGRAPHIC_COLUMNS)


def load_campaign_results(path: str | Path) -> pd.DataFrame:
    """Load campaign results and validate schema."""
    return load_csv_with_schema(path, CAMPAIGN_RESULTS_COLUMNS)
