import pandas as pd
from src.resolution.identity_linker import deduplicate


def test_deduplicate():
    df = pd.DataFrame({'id': [1, 1, 2], 'val': [10, 10, 20]})
    result = deduplicate(df, 'id')
    assert result['id'].tolist() == [1, 2]
