"""Simple Prefect orchestration for the data pipeline."""

from prefect import flow, task

from ingestion.load_data import load_csv
from features.engineer_features import create_basic_features
from modeling.opti_shift import train_regressor
from monitoring.log_metrics import log_metric


@task
def ingest(path: str):
    """Load the dataset from ``path``."""
    return load_csv(path)


@task
def engineer(df):
    """Generate features from the raw dataframe."""
    return create_basic_features(df)


@task
def train(df):
    """Train the regression model using the dataframe."""
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    model = train_regressor(X, y)
    return model


@task
def deploy(model):
    """Placeholder deployment step that logs model type."""
    log_metric("model_type", type(model).__name__)


@flow
def main_flow(path: str) -> None:
    """Orchestrate ingestion, training and deployment tasks."""
    df = ingest(path)
    df = engineer(df)
    model = train(df)
    deploy(model)


if __name__ == "__main__":
    main_flow("data.csv")
