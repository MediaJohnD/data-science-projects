# Project Documentation

## Orchestration

This project uses **Prefect** to orchestrate the data ingestion, feature engineering and model training tasks.

### Running the orchestrator

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the flow using the pipeline entry point:
   ```bash
   python src/pipeline.py path/to/data.csv
   ```

The `main_flow` defined in `src/orchestrator.py` will load the dataset, engineer features, train a model and log a simple metric.

