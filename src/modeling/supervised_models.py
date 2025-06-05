"""Collection of supervised learning algorithms."""

import mlflow
from sklearn.metrics import accuracy_score
import joblib


def train_model(X_train, y_train, algorithm="xgboost"):
    """Train a supervised model and log metrics to MLflow."""
    if algorithm == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif algorithm == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier()
    elif algorithm == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(verbose=False)
    elif algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    return {"accuracy": acc}


def save_model(model, path: str):
    """Persist the trained model to disk."""
    joblib.dump(model, path)
