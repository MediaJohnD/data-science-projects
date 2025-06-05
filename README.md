# Data Science Projects

[![Lint](https://github.com/MediaJohnD/data-science-projects/actions/workflows/lint.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/lint.yml)
[![Test](https://github.com/MediaJohnD/data-science-projects/actions/workflows/test.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/test.yml)
[![Docker Build](https://github.com/MediaJohnD/data-science-projects/actions/workflows/docker-build.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/docker-build.yml)

This repository is hosted at [https://github.com/MediaJohnD/data-science-projects](https://github.com/MediaJohnD/data-science-projects).

## Local development

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run lint and tests:

```bash
flake8 src tests
pytest -q
```

## Core Pipeline Steps

1. **Ingestion** (`src/ingestion/load_data.py`)
   - `load_csv(path)` loads raw CSV data into a DataFrame.
2. **Feature Engineering** (`src/features/engineer_features.py`)
   - `create_basic_features(df)` generates aggregate features.
3. **Modeling** (`src/modeling/opti_shift.py`)
   - `train_regressor(X, y)` trains an XGBoost model.
4. **Scoring API** (`src/scoring/api.py`)
   - A FastAPI application exposing a `/predict` endpoint.

## Optional Utilities

- **Monitoring** (`src/monitoring/log_metrics.py`)
  - `log_metric(name, value)` prints simple metrics.
- **Identity Resolution** (`src/resolution/identity_linker.py`)
  - `link(df, key_columns)` removes duplicates.
- **Contextual Triggers** (`src/contextual_triggers/trigger_engine.py`)
  - `check_threshold(value, threshold)` evaluates thresholds.
- **Clustering** (`src/modeling/clustering.py`)
  - `cluster_kmeans(data, n_clusters)` performs KMeans clustering.
- **Anomaly Detection** (`src/modeling/anomaly_detection.py`)
  - `detect_anomalies(data)` trains an IsolationForest model.

### Example: Clustering
```python
from src.modeling.clustering import cluster_kmeans
import pandas as pd

# ``df`` is a pandas DataFrame of numeric values
model = cluster_kmeans(df, n_clusters=4)
labels = model.labels_
```

### Example: Anomaly Detection
```python
from src.modeling.anomaly_detection import detect_anomalies
import pandas as pd

model = detect_anomalies(df, contamination=0.05)
# Scores available via model.decision_function(df)
```
