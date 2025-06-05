import pandas as pd
from src.ingestion.load_data import load_csv


def test_load_csv(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    df = load_csv(str(csv))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
