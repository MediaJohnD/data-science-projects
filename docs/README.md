# Data Science Pipeline

This project demonstrates a production-oriented machine learning workflow. It
follows an **FTI** (Feature, Training, Inference) architecture with Prefect
flows orchestrating each stage. External data ingestion with schema validation,
hyperparameter tuning via Optuna, MLflow tracking, and Docker-based deployment
are all supported.

Use `python -m src.pipeline` to run the pipeline locally. Set
`MLFLOW_TRACKING_URI` to point at your tracking server if necessary.

A `Dockerfile` and `docker-compose.yml` are provided to deploy the API in a
containerized environment.

Refer to the project [README](../README.md) for instructions on configuring a
GitHub remote and running the CI pipeline.

Prefect deployments can schedule the pipeline to run automatically every day.
