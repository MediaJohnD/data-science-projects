from sklearn.ensemble import IsolationForest
import pandas as pd


def detect_anomalies(data: pd.DataFrame, contamination: float = 0.1) -> IsolationForest:
    """Train an IsolationForest model and return it."""
    model = IsolationForest(contamination=contamination, random_state=0)
    model.fit(data)
    return model
