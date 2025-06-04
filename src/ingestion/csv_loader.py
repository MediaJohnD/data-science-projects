import pandas as pd
from pathlib import Path
from typing import Union


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return df
