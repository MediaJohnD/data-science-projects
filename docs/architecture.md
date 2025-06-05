# Architecture Overview

The pipeline follows an **FTI** (Feature, Training, Inference) design. Prefect
flows orchestrate the following stages:

1. **Ingestion** – Load data from local files, URLs, or S3 buckets, validate
   the schema with Pandera, and store raw data for reproducibility.
2. **Feature Engineering** – Scale numeric features with `RobustScaler`, encode
   categoricals, and build time-series aggregates.
3. **Unsupervised Learning** – Cluster audiences using DBSCAN, K-Means,
   Agglomerative clustering, and perform dimensionality reduction with PCA,
   t-SNE, and UMAP. Detect anomalies with Isolation Forest and One-Class SVM.
4. **Modeling** – Train multiple supervised models (XGBoost, LightGBM,
   CatBoost, Random Forest, Logistic Regression) while tracking runs in MLflow.
5. **Monitoring** – Log metrics both to MLflow and the console.
6. **Drift Detection** – Compare new data against a saved baseline using a
   Kolmogorov-Smirnov test. Trigger retraining if drift exceeds a threshold.
7. **Model Registry** – Store versioned models in MLflow and deploy the
   best-performing model automatically.
8. **Orchestration** – Manage the end-to-end workflow with Prefect.
9. **Deployment** – Serve predictions through a FastAPI application with a
   GitHub Actions workflow for CI.
10. **Scheduling** – Prefect schedules run the pipeline daily with automatic
   retries and optional notifications.

Environment-specific settings are stored in `prefect/envs/` and loaded by
the deployment configuration in `prefect/deployments/run_pipeline.yaml`.

MLflow's model registry records each trained model with metadata so that the
CI workflow can deploy the latest approved version.

