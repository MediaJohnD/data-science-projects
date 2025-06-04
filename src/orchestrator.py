from prefect import flow, task

from ingestion.load_data import run as ingest_run
from modeling.opti_shift import train as model_train
from scoring.api import deploy as deploy_api


@task
def ingest():
    ingest_run()


@task
def train():
    model_train()


@task
def deploy():
    deploy_api()


@flow
def main_flow():
    """Orchestrate ingestion, training and deployment tasks."""
    ingest()
    train()
    deploy()


if __name__ == "__main__":
    main_flow()
