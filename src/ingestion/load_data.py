"""Data ingestion utilities."""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
import pandera as pa
from pandera import Column, DataFrameSchema
import s3fs

def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataframe schema."""
    schema = DataFrameSchema({"target": Column(int)}, strict=False)
    return schema.validate(df)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load data from the given path or built-in dataset."""
    # Path can be provided directly or via the DATA_PATH environment variable
    path = path or os.getenv("DATA_PATH")
    if path and path.startswith("s3://"):
        df = pd.read_csv(path, storage_options={"anon": False})
    elif path:
        df = pd.read_csv(path)
    else:
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        df["target"] = data.target

    df = _validate(df)

    # Persist raw data for downstream stages
    storage_path = os.getenv("LOCAL_STORAGE", "data/raw.csv")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    df.to_csv(storage_path, index=False)
    return df
