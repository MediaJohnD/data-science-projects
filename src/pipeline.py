from src.ingestion.load_data import load_data
from src.features.engineer_features import split_and_scale
from src.modeling.opti_shift import train_model, evaluate_model, save_model
from src.monitoring.log_metrics import log_accuracy
from src.contextual_triggers.trigger_engine import should_trigger_retrain


def run_pipeline(path: str | None = None):
    """Execute the entire ML pipeline.

    Parameters
    ----------
    path: optional path or URL to the dataset. If not provided, a built-in
        sample dataset is used.
    """
    df = load_data(path)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    log_accuracy(accuracy)
    save_model(model, "model.joblib")
    if should_trigger_retrain(accuracy):
        print("Retraining recommended.")
    else:
        print("Model performance acceptable.")


if __name__ == "__main__":
    run_pipeline()
