import pandas as pd
from pathlib import Path
from typing import Iterable, Union

from .validators import validate_columns


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV: {exc}") from exc
    return df


def load_csv_with_schema(path: Union[str, Path], required_columns: Iterable[str]) -> pd.DataFrame:
    """Load a CSV and validate that required columns are present."""
    df = load_csv(path)
    validate_columns(df, required_columns)
    return df
