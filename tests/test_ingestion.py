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

from src.ingestion.data_loaders import load_transactions


def test_load_transactions(tmp_path):
    csv_path = tmp_path / 'tx.csv'
    pd.DataFrame({
        'customer_id': ['c1'],
        'transaction_timestamp': ['2024-01-01'],
        'merchant_latitude': [0.0],
        'merchant_longitude': [0.0],
        'merchant_zip_plus4': ['12345'],
        'transaction_amount': [10.0],
    }).to_csv(csv_path, index=False)
    df = load_transactions(csv_path)
    assert 'transaction_amount' in df.columns

    # missing column
    bad = tmp_path / 'bad_tx.csv'
    pd.DataFrame({'customer_id': ['c1']}).to_csv(bad, index=False)
    try:
        load_transactions(bad)
    except ValueError as e:
        assert 'Missing columns' in str(e)
    else:
        raise AssertionError('Expected error for missing columns')
