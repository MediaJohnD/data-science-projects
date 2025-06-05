# Data Science Projects

![CI](https://github.com/MediaJohnD/data-science-projects/actions/workflows/ci.yml/badge.svg)

This repository contains an end-to-end machine learning pipeline. The project
follows an **FTI** (Feature, Training, Inference) architecture so that each
stage can be orchestrated and scaled independently. The code demonstrates how
to ingest data from external sources, validate schemas, engineer features,
train and tune a model while tracking experiments with MLflow, and expose
predictions via a FastAPI service. Continuous integration is handled through
the included GitHub Actions workflow.
The pipeline also detects data drift using a Kolmogorov-Smirnov test and logs
the statistic for monitoring.
See [docs/architecture.md](docs/architecture.md) for a detailed design overview.

## Features

- Robust feature scaling and categorical encoding via `RobustScaler` and
  `OneHotEncoder`.
- Time series feature engineering (recency, frequency, rolling averages).
- Optional utilities provide clustering algorithms (DBSCAN, K-Means,
  Agglomerative) and dimensionality reduction (PCA, t-SNE, UMAP).
- Supervised models including XGBoost, LightGBM, CatBoost, Random Forest,
  and Logistic Regression with MLflow experiment tracking.
- Optional anomaly detection modules (Isolation Forest, One-Class SVM,
  Autoencoders).
- Model explainability through SHAP.

## Running Locally

The pipeline is orchestrated with [Prefect](https://docs.prefect.io/) and uses
MLflow for experiment tracking. To run the complete FTI pipeline locally,
specify the tracking server if desired and execute:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000  # optional
python -m src.pipeline  # MODEL_ALGORITHM can override the default model
```

### Environment Configuration

Sample environment files for local, staging, and production deployments are
located in the `prefect/envs/` directory. Adjust the values as needed and load
them before running or scheduling the pipeline.

## Scheduling

The pipeline is packaged with a Prefect deployment that schedules a daily run.
Configure a Prefect API and register the deployment using the provided
configuration:

```bash
prefect deployment build src/pipeline.py:run_pipeline -n daily-fti \
  -o prefect/deployments/run_pipeline.yaml
prefect deployment apply prefect/deployments/run_pipeline.yaml
```
The schedule can be customized in `src/pipeline.py`.

## GitHub Remote

The project does not ship with a Git remote configured. To collaborate and
enable CI/CD you should connect it to your own GitHub repository. Use one of
the following commands to add a remote and verify it:

### HTTPS

```bash
git remote add origin https://github.com/MediaJohnD/data-science-projects.git
git push -u origin main
git remote -v  # verify
```

### SSH

```bash
git remote add origin git@github.com:MediaJohnD/data-science-projects.git
git push -u origin main
git remote -v  # verify
```

### GitHub CLI

```bash
gh repo clone MediaJohnD/data-science-projects
cd data-science-projects
git remote -v
```

Continuous integration is provided via a GitHub Actions workflow defined in
`.github/workflows/ci.yml`.
