import pandas as pd


def deduplicate(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Drop duplicate rows based on ``id_col``."""
    return df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
