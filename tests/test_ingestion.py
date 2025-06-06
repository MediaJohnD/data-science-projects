import os
import pandas as pd
import pytest

from src.ingestion.load_data import load_csv


def test_load_csv_missing_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_csv(str(csv_path), expected_columns=["a", "b", "c"])


def test_load_csv_bad_s3_path():
    with pytest.raises(FileNotFoundError):
        load_csv("s3://bucket/file.csv")
