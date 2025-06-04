# Data Science Projects

This repository contains an end-to-end machine learning pipeline with
deployment automation. The code demonstrates how to ingest data from external
sources, validate schemas, engineer features, train and tune a model while
tracking experiments with MLflow, and expose predictions via a FastAPI service.
Deployment can be automated using the provided `deploy.sh` script.

## Running Locally

```bash
python -m src.pipeline
```

## Running with Docker

```bash
cd docker
docker-compose up --build
```

## GitHub Remote

The repository is configured with a GitHub remote at
`https://github.com/example/data-science-projects.git` for collaboration and
CI/CD integration.
