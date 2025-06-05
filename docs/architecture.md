# Architecture

The project is organized into small modules under the `src` directory.  The
modules implement a tiny yet complete data science workflow:

1. **Ingestion** – loading raw data.
2. **Feature engineering** – constructing features from raw data.
3. **Modeling** – training simple models.
4. **Scoring** – serving predictions through an API.
5. **Monitoring** – logging runtime metrics.
6. **Resolution** – deduplicating and linking records.
7. **Contextual triggers** – reacting to events.

The typical flow is:

1. Load data with the ingestion utilities.
2. Generate additional features.
3. Train a model using the prepared features.
4. Serve predictions via the scoring API while monitoring metrics.
