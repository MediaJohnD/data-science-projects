import joblib
import optuna
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import mlflow


def _objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    model = xgb.XGBClassifier(
        **params, use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


def tune_hyperparameters(X_train, y_train):
    """Search best hyperparameters using Optuna."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X_tr, y_tr, X_val, y_val), n_trials=20
    )
    return study.best_params


def train_model(X_train, y_train):
    """Train an XGBoost model with hyperparameter tuning and MLflow tracking."""
    params = tune_hyperparameters(X_train, y_train)
    with mlflow.start_run():
        mlflow.log_params(params)
        model = xgb.XGBClassifier(
            **params, use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id),
                              "dspipeline")
    return model


def evaluate_model(model, X_test, y_test):
    """Return evaluation metrics for the given model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    return metrics


def save_model(model, path: str):
    joblib.dump(model, path)
