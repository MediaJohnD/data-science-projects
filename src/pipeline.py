"""Prefect flows implementing an FTI (Feature-Training-Inference) pipeline."""

from prefect import flow, task

from src.ingestion.load_data import load_data
from src.features.engineer_features import split_and_scale
from src.modeling.opti_shift import train_model, evaluate_model, save_model
from src.monitoring.log_metrics import log_accuracy
from src.contextual_triggers.trigger_engine import should_trigger_retrain


@task
def ingestion_task(path: str | None = None):
    """Load and validate raw data."""
    return load_data(path)


@task
def feature_task(df):
    """Perform feature engineering."""
    return split_and_scale(df)


@task
def train_task(X_train, y_train):
    """Train a model with hyperparameter tuning."""
    return train_model(X_train, y_train)


@task
def eval_task(model, X_test, y_test):
    """Evaluate the trained model."""
    return evaluate_model(model, X_test, y_test)


@flow
def feature_flow(path: str | None = None):
    """Feature engineering stage."""
    df = ingestion_task(path)
    return feature_task(df)


@flow
def training_flow(X_train, y_train):
    """Training stage."""
    model = train_task(X_train, y_train)
    save_model(model, "model.joblib")
    return model


@flow
def inference_flow(model, X_test, y_test):
    """Inference and evaluation stage."""
    accuracy = eval_task(model, X_test, y_test)
    log_accuracy(accuracy)
    if should_trigger_retrain(accuracy):
        print("Retraining recommended.")
    else:
        print("Model performance acceptable.")
    return accuracy


@flow
def run_pipeline(path: str | None = None):
    """Execute the full Feature-Training-Inference pipeline."""
    X_train, X_test, y_train, y_test, _ = feature_flow(path)
    model = training_flow(X_train, y_train)
    accuracy = inference_flow(model, X_test, y_test)
    return accuracy


if __name__ == "__main__":
    run_pipeline()
