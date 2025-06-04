"""Feature generation utilities with validation and normalization."""

from typing import Tuple

import pandas as pd
from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing target values and fill numeric NaNs."""
    df = df.copy()
    df = df.dropna(subset=df.columns)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def generate_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Generate features with scaling and one-hot encoding.

    Returns the transformed feature matrix, target vector, and fitted pipeline.
    """
    df = _clean_dataframe(df)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = RobustScaler()
    if version.parse(sklearn_version) >= version.parse("1.2"):
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    try:
        X_processed = pipeline.fit_transform(X)
    except Exception as exc:
        raise ValueError(f"Feature generation failed: {exc}") from exc

    return X_processed, y, pipeline

