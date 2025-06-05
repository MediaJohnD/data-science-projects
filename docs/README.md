# Data Science Pipeline

This project demonstrates an end-to-end pipeline for a machine learning
workflow. The pipeline supports external data ingestion with schema validation,
hyperparameter tuning via Optuna, MLflow tracking, Prefect orchestration, and
Docker-based deployment automation.

Use `python -m src.pipeline` to run the pipeline locally.

A `Dockerfile` and `docker-compose.yml` are provided to deploy the API in a
containerized environment.
