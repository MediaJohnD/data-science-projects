"""Feature generation utilities."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def generate_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Generate features with scaling and one-hot encoding.

    Returns the transformed feature matrix, target vector, and fitted pipeline.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    X_processed = pipeline.fit_transform(X)

    return X_processed, y, pipeline
