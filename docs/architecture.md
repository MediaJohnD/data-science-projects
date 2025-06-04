# Architecture Overview

This document outlines the system architecture for Fringe Audience AI.

## Data Flow

1. **Ingestion** – Raw CSV files are loaded via the ingestion module.
2. **Feature Engineering** – Numerical fields are scaled, categorical fields are one-hot encoded, and a training matrix is produced.
3. **Modeling** – The system supports multiple algorithms including XGBoost, KNN, Random Forest, DBSCAN clustering, Isolation Forest, autoencoders, and RNNs. Cross-validation and grid search are applied where appropriate and metrics such as accuracy and ROC AUC are reported.
4. **Deployment** – Trained models can be served via FastAPI for real-time scoring. MLflow tracks experiments and stores model artifacts.

## Deployment Steps

1. Build the Docker image with `docker build -t fringe-ai .`.
2. Run the pipeline using `python -m src.pipelines.main` or `make run`.
3. Use `mlflow` to track experiments and store model artifacts.
4. Deploy the FastAPI service to expose scoring endpoints for downstream systems.
