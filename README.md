# Data Science Projects

This repository contains a collection of simple modules that showcase a
lightweight data science workflow.  Each module is intentionally small so the
repository can be used as a starting point for experiments or tutorials.

## Project Structure

- `src/ingestion` – utilities to load datasets.
- `src/features` – basic feature engineering helpers.
- `src/modeling` – example model training code.
- `src/scoring` – a minimal FastAPI service used for scoring.
- `src/monitoring` – runtime metrics logging.
- `src/resolution` – entity resolution helpers.
- `src/contextual_triggers` – lightweight event trigger logic.
- `docs` – additional documentation.

### Running the tests

Install the dependencies and execute the test suite:

```bash
pytest -q
```

### Launch the scoring service

The `scoring` module exposes a very small FastAPI application.  You can run it
locally after installing the requirements:

```bash
uvicorn src.scoring.api:app --reload
```

Alternatively, build the provided Docker image:

```bash
docker build -t scoring-service ./docker
docker run -p 8000:8000 scoring-service
```
