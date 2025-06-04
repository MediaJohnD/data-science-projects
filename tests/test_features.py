import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.features.feature_generator import generate_features


def test_generate_features():
    df = pd.DataFrame({
        'num': [1, 2, 3],
        'cat': ['a', 'b', 'a'],
        'target': [0, 1, 0]
    })
    X, y, pipeline = generate_features(df, 'target')
    assert X.shape[0] == 3
    assert len(y) == 3
    # ensure robust scaler is used
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
    assert isinstance(scaler, RobustScaler)
