# Architecture Overview

The pipeline orchestrates the following steps:

1. **Ingestion** – Load data from local files or URLs, validate the schema, and
   store raw data for reproducibility.
2. **Feature Engineering** – Split the data and scale numeric features.
3. **Modeling** – Tune an XGBoost classifier with Optuna while tracking runs in
   MLflow.
4. **Monitoring** – Log metrics both to MLflow and the console.
5. **Deployment** – Serve predictions through a FastAPI application with Docker
   automation.

The API can be containerized using Docker and run with Docker Compose.
