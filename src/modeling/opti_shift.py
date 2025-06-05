from sklearn.linear_model import LinearRegression
import pandas as pd


def train_linear_model(df: pd.DataFrame, target: str) -> LinearRegression:
    """Train a linear regression model using all other columns as features."""
    X = df.drop(columns=[target])
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model
