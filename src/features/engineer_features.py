import pandas as pd


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple aggregated features."""
    numeric_cols = df.select_dtypes(include="number")
    df = df.copy()
    if not numeric_cols.empty:
        df["feature_sum"] = numeric_cols.sum(axis=1)
        df["feature_mean"] = numeric_cols.mean(axis=1)
    return df
