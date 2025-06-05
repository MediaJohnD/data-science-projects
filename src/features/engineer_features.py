import pandas as pd
from typing import Iterable


def add_total_column(
    df: pd.DataFrame, columns: Iterable[str], output_column: str = "total"
) -> pd.DataFrame:
    """Add a column that is the row-wise sum of ``columns``."""
    df = df.copy()
    df[output_column] = df[list(columns)].sum(axis=1)
    return df
