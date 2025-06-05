import pandas as pd


def link(df: pd.DataFrame, key_columns: list) -> pd.DataFrame:
    """Drop duplicate rows based on key columns."""
    return df.drop_duplicates(subset=key_columns)
