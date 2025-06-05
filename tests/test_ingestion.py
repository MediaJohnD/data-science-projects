import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.load_data import load_data  # noqa: E402


def test_load_data(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
    df = load_data(str(csv))
    assert 'target' in df.columns
