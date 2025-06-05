# Architecture

The project is organized into small modules under the `src` directory. Each
module represents a piece of a typical data science workflow:

1. **Ingestion** – loading raw data.
2. **Feature engineering** – constructing features from raw data.
3. **Modeling** – training simple models.
4. **Scoring** – serving predictions through an API.
5. **Monitoring** – logging runtime metrics.
6. **Resolution** – deduplicating and linking records.
7. **Contextual triggers** – reacting to events.
