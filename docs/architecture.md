# Architecture Overview

The pipeline follows an **FTI** (Feature, Training, Inference) design. Prefect
flows orchestrate the following stages:

1. **Ingestion** – Load data from local files, URLs, or S3 buckets, validate
   the schema with Pandera, and store raw data for reproducibility.
2. **Feature Engineering** – Split the data and scale numeric features.
3. **Modeling** – Tune an XGBoost classifier with Optuna while tracking runs in
   MLflow.
4. **Monitoring** – Log metrics both to MLflow and the console.
5. **Model Registry** – Store versioned models in MLflow and deploy the
   best-performing model automatically.
6. **Orchestration** – Manage the end-to-end workflow with Prefect.
7. **Deployment** – Serve predictions through a FastAPI application with Docker
   automation and a GitHub Actions workflow for CI.
8. **Scheduling** – Prefect schedules run the pipeline daily with automatic
   retries and optional notifications.

MLflow's model registry records each trained model with metadata so that the
CI workflow can deploy the latest approved version.

The API can be containerized using Docker and run with Docker Compose.
