# Project Documentation

## Orchestration

This project uses **Prefect** to orchestrate the data ingestion, model training
and deployment tasks.

### Running the orchestrator

1. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Execute the flow using the pipeline entry point:

   ```bash
   python src/pipeline.py
   ```

The `main_flow` defined in `src/orchestrator.py` will run each task in order.
