# Fringe Audience AI

**AiOpti Media | AI Solution for Discovering and Activating High-Propensity Fringe Audiences**

## Overview

Fringe Audience AI is a proprietary, production-grade machine learning system that identifies and activates non-obvious but highly valuable audience segments. These "fringe" audiences are often missed by conventional targeting due to data sparsity, anomalous behavior, or lack of deterministic matches.

---

## Why It Matters

Marketers often over-rely on high-confidence identity matches and predictive scores. However, high-intent, high-propensity users often live just outside these boundaries. This system uncovers those audiences through anomaly detection, behavioral modeling, and hybrid ML architecture.

---

## What Makes This Unique

- **Hybrid AI Modeling:** Combines lookalike expansion, anomaly detection, RNNs, and latent variable models.
- **Multi-Source Intelligence:** Unifies CRM, geo-behavior, transaction data, and engagement signals.
- **Real-Time Activation:** Deploys through API integrations to DSPs and marketing systems.
- **Feedback Loop:** Constantly improves through attribution data from visits and transactions.

---

## Core Components

- **Data Ingestion & Normalization**  
  Aggregates and harmonizes disparate first- and third-party data streams.

- **Identity Resolution Layer**  
  Links user identifiers across devices, cookies, MAIDs, and hashed IDs.

- **Feature Engineering & Selection**  
  Leverages time-based behaviors, content context, and psychographic indicators.

- **Modeling Stack**  
  - Gradient Boosted Trees (e.g., XGBoost)  
  - K-Nearest Neighbors  
  - Autoencoders for latent signals  
  - RNNs for behavioral sequences  
  - Isolation Forest for anomaly classification

- **Scoring & Activation**  
  Scoring exposed via containerized FastAPI endpoint and integrated with DSPs.

- **Attribution Measurement**  
  Incorporates footfall and transaction lift to evaluate model performance and trigger retraining.

---

## Deployment Architecture

- Python, FastAPI, Docker, MLflow
- Model versioning and data pipelines
- Optional orchestration via Prefect or Airflow

See `docs/architecture.md` for full technical blueprint.

---

## Getting Started

```bash
make install   # Install dependencies
make run       # Run full pipeline
```

The default pipeline loads a CSV dataset, performs feature scaling and one-hot
encoding, then trains an XGBoost model using cross-validation and hyperparameter
tuning. Results are printed to the console and the trained model artifact is
stored on disk.

---

This repository follows a modular structure:

- `docs/` and `configs/` store documentation and configuration files.
- `data/` contains `raw/`, `interim/`, and `processed/` data tiers.
- `src/` hosts ingestion, feature engineering, modeling, and pipeline modules.
- `tests/` includes the test suite.
- Specialized ingestion utilities validate schemas for multiple dataset types.
- RFM feature helper computes recency, frequency, and monetary metrics.

Feel free to expand each section with your own implementation details.

## Usage Example

1. Place your training data as a CSV file under `data/raw/`.
2. Update `configs/default_config.yaml` with the correct file path and parameters. This repo ships with a small sample dataset in `data/raw/example.csv` for testing.
3. Run the pipeline:

```bash
python -m src.pipelines.main
```

The trained model will be saved to `model.joblib` (or the path specified in the config).

### Configuration

`configs/default_config.yaml` contains example parameters for the pipeline and
the XGBoost model, including an optional hyperparameter grid for cross
validation. You can pass your own configuration dictionary to
`run_pipeline()` or edit this file when integrating with orchestration tools
like Prefect or Airflow.

## Testing

Run the test suite with:

```bash
pytest -q
```

This runs unit tests covering ingestion, feature generation, model training, and
the overall pipeline to ensure everything works end to end.

## License

This project and its code are proprietary to AiOpti. All rights are reserved. Redistribution or use in source or binary form, with or without modification, is prohibited without express written permission from AiOpti.
