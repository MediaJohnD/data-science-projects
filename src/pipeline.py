from src.ingestion.load_data import load_data
from src.features.engineer_features import split_and_scale
from src.modeling.opti_shift import train_model, evaluate_model, save_model
from src.monitoring.log_metrics import log_accuracy
from src.contextual_triggers.trigger_engine import should_trigger_retrain
from prefect import flow, task


@task
def ingestion_task(path: str | None = None):
    return load_data(path)


@task
def feature_task(df):
    return split_and_scale(df)


@task
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task
def eval_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@flow
def run_pipeline(path: str | None = None):
    """Execute the entire ML pipeline.

    Parameters
    ----------
    path: optional path or URL to the dataset. If not provided, a built-in
        sample dataset is used.
    """
    df = ingestion_task(path)
    X_train, X_test, y_train, y_test, scaler = feature_task(df)
    model = train_task(X_train, y_train)
    accuracy = eval_task(model, X_test, y_test)
    log_accuracy(accuracy)
    save_model(model, "model.joblib")
    if should_trigger_retrain(accuracy):
        print("Retraining recommended.")
    else:
        print("Model performance acceptable.")


if __name__ == "__main__":
    run_pipeline()
