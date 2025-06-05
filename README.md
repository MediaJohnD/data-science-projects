# Data Science Projects

This repository contains an end-to-end machine learning pipeline with
deployment automation. The project follows an **FTI** (Feature, Training,
Inference) architecture so that each stage can be orchestrated and scaled
independently. The code demonstrates how to ingest data from external sources,
validate schemas, engineer features, train and tune a model while tracking
experiments with MLflow, and expose predictions via a FastAPI service.
Deployment can be automated using the provided `deploy.sh` script or via the
included GitHub Actions workflow.
See [docs/architecture.md](docs/architecture.md) for a detailed design overview.

## Running Locally

The pipeline is orchestrated with [Prefect](https://docs.prefect.io/) and uses
MLflow for experiment tracking. To run the complete FTI pipeline locally,
specify the tracking server if desired and execute:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000  # optional
python -m src.pipeline
```

## Running with Docker

```bash
cd docker
docker-compose up --build
```

## Scheduling

The pipeline is packaged with a Prefect deployment that schedules a daily run.
Configure a Prefect API and register the deployment:

```bash
prefect deployment build src/pipeline.py:run_pipeline -n daily-fti
prefect deployment apply run_pipeline-deployment.yaml
```
The schedule can be customized in `src/pipeline.py`.

## GitHub Remote

The project does not ship with a Git remote configured. To collaborate and
enable CI/CD you should connect it to your own GitHub repository. Use one of
the following commands to add a remote and verify it:

### HTTPS

```bash
git remote add origin https://github.com/MediaJohnD/data-science-projects.git
git push -u origin main
git remote -v  # verify
```

### SSH

```bash
git remote add origin git@github.com:MediaJohnD/data-science-projects.git
git push -u origin main
git remote -v  # verify
```

### GitHub CLI

```bash
gh repo clone MediaJohnD/data-science-projects
cd data-science-projects
git remote -v
```

Continuous integration is provided via a GitHub Actions workflow defined in
`.github/workflows/ci.yml`.
