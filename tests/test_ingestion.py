from src.ingestion.csv_loader import load_csv
from pathlib import Path


def test_load_csv(tmp_path):
    csv_path = tmp_path / 'temp.csv'
    csv_path.write_text('a,b\n1,2')
    df = load_csv(csv_path)
    assert list(df.columns) == ['a', 'b']
