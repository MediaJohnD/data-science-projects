from prefect import flow, task

from contextual_triggers.trigger_engine import run as trigger_run
from features.engineer_features import run as feature_run
from ingestion.load_data import run as ingest_run
from modeling.opti_shift import train as model_train
from monitoring.log_metrics import run as log_run
from scoring.api import create_app


@task
def ingest():
    return ingest_run()


@task
def feature_engineering(visits):
    return feature_run(visits)


@task
def train(features):
    target = (features["unique_pois"] > 1).astype(int)
    return model_train(features[["visit_count", "unique_pois"]], target)


@task
def log_metric(acc):
    log_run("training_accuracy", acc)


@task
def triggers(features):
    return trigger_run(features)


@task
def deploy(model):
    app = create_app(model)
    return app


@flow
def main_flow():
    """Orchestrate ingestion, training and deployment tasks."""
    visits = ingest()
    feats = feature_engineering(visits)
    model, acc = train(feats)
    log_metric(acc)
    triggers(feats)
    deploy(model)


if __name__ == "__main__":
    main_flow()
