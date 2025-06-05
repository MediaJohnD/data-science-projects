"""Feature engineering utilities."""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def _add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create recency, frequency and rolling mean features from a datetime column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df["recency"] = (df[date_col].max() - df[date_col]).dt.days
    df["frequency"] = df.groupby(df[date_col].dt.to_period("M")).cumcount()
    df["rolling_mean_target"] = (
        df["target"].rolling(window=3, min_periods=1).mean()
    )
    return df


def split_and_scale(
    df: pd.DataFrame, target_col: str = "target", date_col: str | None = None
):
    """Split dataframe, encode categories and scale numeric features."""
    if date_col and date_col in df.columns:
        df = _add_time_features(df, date_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = [c for c in X.columns if is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not is_numeric_dtype(X[c])]

    preprocessor = ColumnTransformer(
        [
            ("num", RobustScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor
