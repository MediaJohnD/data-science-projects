# Data Science Pipeline

This project demonstrates a production-oriented machine learning workflow. It
follows an **FTI** (Feature, Training, Inference) architecture with Prefect
flows orchestrating each stage. External data ingestion with schema validation,
hyperparameter tuning via Optuna, and MLflow tracking are all supported.
Optional clustering and anomaly detection modules are available for exploratory
analysis, while multiple supervised models can be trained and compared. A
drift detection module monitors incoming data against a saved baseline to
identify significant distribution changes.

Use `python -m src.pipeline` to run the pipeline locally. Set
`MLFLOW_TRACKING_URI` to point at your tracking server if necessary.

Create and activate a Python virtual environment, then install the required
packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Refer to the project [README](../README.md) for instructions on configuring the
GitHub remote (`https://github.com/MediaJohnD/data-science-projects.git`) and
running the CI pipeline.

Prefect deployments can schedule the pipeline to run automatically every day.
An example deployment configuration is provided in
`prefect/deployments/run_pipeline.yaml`.
