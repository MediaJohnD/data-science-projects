# Data Science Projects

This repository contains an end-to-end machine learning pipeline with
deployment automation. The code demonstrates how to ingest data from external
sources, validate schemas, engineer features, train and tune a model while
tracking experiments with MLflow, and expose predictions via a FastAPI service.
Deployment can be automated using the provided `deploy.sh` script.

## Running Locally

The pipeline is orchestrated with [Prefect](https://docs.prefect.io/). Execute
the flow with:

```bash
python -m src.pipeline
```

## Running with Docker

```bash
cd docker
docker-compose up --build
```

## GitHub Remote

This repository can be connected to your GitHub account for collaboration and
continuous integration. Use one of the following methods to configure the
remote:

### HTTPS

```bash
git remote add origin https://github.com/MediaJohnD/data-science-projects.git
git push -u origin main
```

### SSH

```bash
git remote add origin git@github.com:MediaJohnD/data-science-projects.git
git push -u origin main
```

### GitHub CLI

```bash
gh repo clone MediaJohnD/data-science-projects
```

Continuous integration is provided via a GitHub Actions workflow defined in
`.github/workflows/ci.yml`.
