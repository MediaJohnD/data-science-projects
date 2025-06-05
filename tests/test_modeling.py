import pandas as pd
from src.modeling.opti_shift import train_linear_model
from pytest import approx


def test_train_linear_model():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
    model = train_linear_model(df, 'y')
    pred = model.predict([[4]])
    assert pred[0] == approx(8.0, rel=1e-3)
