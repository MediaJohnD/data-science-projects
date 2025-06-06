# Architecture Overview

The OptiReveal prototype is organised as a series of Prefect tasks executed by
``main_flow``. Each module is responsible for a single stage of the workflow:

1. **Ingestion** (`ingestion.load_data`)
2. **Feature Engineering** (`features.engineer_features`)
3. **Modeling** (`modeling.opti_shift`)
4. **Monitoring** (`monitoring.log_metrics`)
5. **Contextual Triggers** (`contextual_triggers.trigger_engine`)
6. **Deployment** (`scoring.api`)

The system is intentionally lightweight and suitable for demonstration
purposes. It can be extended with richer data sources and additional modeling
techniques.
