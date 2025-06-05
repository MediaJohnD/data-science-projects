"""Prefect flows implementing an FTI (Feature-Training-Inference) pipeline."""

from prefect import flow, task
from prefect.blocks.notifications import AppriseNotificationBlock

from src.ingestion.load_data import load_data
from src.features.engineer_features import split_and_scale
from src.modeling.supervised_models import (
    train_model,
    evaluate_model,
    save_model,
)
from src.monitoring.log_metrics import log_accuracy
from src.monitoring.drift_detection import monitor_drift
from src.contextual_triggers.trigger_engine import should_trigger_retrain


@task(retries=2, retry_delay_seconds=30)
def ingestion_task(path: str | None = None):
    """Load and validate raw data."""
    return load_data(path)


@task(retries=2, retry_delay_seconds=30)
def feature_task(df):
    """Perform feature engineering."""
    return split_and_scale(df)


@task(retries=2, retry_delay_seconds=30)
def train_task(X_train, y_train, algorithm: str = "xgboost"):
    """Train a model using the specified algorithm."""
    return train_model(X_train, y_train, algorithm=algorithm)


@task(retries=2, retry_delay_seconds=30)
def eval_task(model, X_test, y_test):
    """Evaluate the trained model."""
    return evaluate_model(model, X_test, y_test)


@flow
def feature_flow(path: str | None = None):
    """Feature engineering stage."""
    df = ingestion_task(path)
    return feature_task(df)


@flow
def training_flow(X_train, y_train, algorithm: str = "xgboost"):
    """Training stage."""
    model = train_task(X_train, y_train, algorithm)
    save_model(model, "model.joblib")
    return model


@flow
def inference_flow(model, X_test, y_test):
    """Inference and evaluation stage."""
    metrics = eval_task(model, X_test, y_test)
    log_accuracy(metrics["accuracy"])
    drift_stat, drift = monitor_drift(X_test)
    print(f"Drift statistic: {drift_stat:.3f}")
    if should_trigger_retrain(metrics["accuracy"]) or drift:
        print("Retraining recommended.")
    else:
        print("Model performance acceptable.")
    return metrics


@flow(name="fti-pipeline", log_prints=True, timeout_seconds=3600)
def run_pipeline(path: str | None = None, algorithm: str = "xgboost"):
    """Execute the full Feature-Training-Inference pipeline."""
    X_train, X_test, y_train, y_test, _ = feature_flow(path)
    model = training_flow(X_train, y_train, algorithm)
    metrics = inference_flow(model, X_test, y_test)
    try:
        notifier = AppriseNotificationBlock.load("default")
        notifier.notify(
            f"Pipeline completed with accuracy: {metrics['accuracy']:.3f}"
        )
    except Exception:
        print("Notification block not configured")
    return metrics


if __name__ == "__main__":
    import os

    algo = os.getenv("MODEL_ALGORITHM", "xgboost")
    run_pipeline(algorithm=algo)
