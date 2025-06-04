from src.models.xgb_model import train_model_cv, evaluate_model
import pandas as pd


def test_train_model_cv():
    df = pd.DataFrame({
        'f1': [1, 2, 3, 4],
        'f2': [10, 20, 30, 40],
        'target': [0, 0, 1, 1]
    })
    X = df[['f1', 'f2']]
    y = df['target']
    model, params, score = train_model_cv(X, y, {'max_depth': [3]}, cv=2)
    metrics = evaluate_model(model, X, y)
    assert metrics['accuracy'] >= 0.5
