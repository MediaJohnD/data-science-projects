from pathlib import Path

import pandas as pd

from src.ingestion.csv_loader import load_csv, load_csv_with_schema


def test_load_csv(tmp_path):
    csv_path = tmp_path / 'temp.csv'
    csv_path.write_text('a,b\n1,2')
    df = load_csv(csv_path)
    assert list(df.columns) == ['a', 'b']


def test_load_csv_with_schema(tmp_path):
    csv_path = tmp_path / 'temp.csv'
    pd.DataFrame({'col1': [1], 'col2': [2]}).to_csv(csv_path, index=False)
    df = load_csv_with_schema(csv_path, ['col1', 'col2'])
    assert 'col1' in df.columns

    # trigger missing column error
    bad_path = tmp_path / 'bad.csv'
    pd.DataFrame({'col1': [1]}).to_csv(bad_path, index=False)
    try:
        load_csv_with_schema(bad_path, ['col1', 'col2'])
    except ValueError as e:
        assert 'Missing columns' in str(e)
    else:
        raise AssertionError('Expected ValueError for missing columns')
