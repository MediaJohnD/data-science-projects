from xgboost import XGBRegressor
import pandas as pd


def train_regressor(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Train a simple XGBoost regressor."""
    model = XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X, y)
    return model


def predict(model: XGBRegressor, X: pd.DataFrame):
    """Generate predictions using the trained model."""
    return model.predict(X)
