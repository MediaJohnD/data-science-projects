# System Architecture

This document outlines the overall architecture of the project.

## Core Pipeline
1. **Ingestion** - Load raw data using `load_csv`.
2. **Feature Engineering** - Derive aggregates with `create_basic_features`.
3. **Modeling** - Train models via `train_regressor`.
4. **Scoring** - Serve predictions through the FastAPI app.

## Optional Utilities
- **Monitoring** - Basic metric logging with `log_metric`.
- **Identity Resolution** - Deduplicate records using `link`.
- **Contextual Triggers** - Evaluate thresholds via `check_threshold`.
- **Clustering** - Cluster data points with `cluster_kmeans`.
- **Anomaly Detection** - Train `detect_anomalies` for outlier detection.

A high-level diagram is available in `configs/system_diagram.png`.
