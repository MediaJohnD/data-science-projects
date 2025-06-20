# Data Science Projects

[![Lint](https://github.com/MediaJohnD/data-science-projects/actions/workflows/lint.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/lint.yml)
[![Test](https://github.com/MediaJohnD/data-science-projects/actions/workflows/test.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/test.yml)

This repository is hosted at [https://github.com/MediaJohnD/data-science-projects](https://github.com/MediaJohnD/data-science-projects).

> **Note**
> This project is still under active development and is **not** finalized for submission to the UT Austin or Georgia Tech data science programs.

Clone the repository:

```bash
git clone https://github.com/MediaJohnD/data-science-projects.git
cd data-science-projects
```

## Local development

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run lint and tests:

```bash
flake8 src tests
pytest -q
```

## Prefect Orchestration

This project implements a small demonstration of the OptiReveal workflow using
**Prefect** for orchestration. A minimal sample data set is bundled with the
code so that the full pipeline can be executed without external dependencies.
The flow consists of the following stages:

1. **Ingest** – load raw visit events.
2. **Feature Engineering** – aggregate visits into device level features.
3. **Model Training** – train a logistic regression model predicting visitor
   propensity.
4. **Monitoring** – log basic metrics.
5. **Triggers** – create simple contextual triggers.
6. **Deployment** – expose a FastAPI scoring service.

Run the end-to-end flow locally:

```bash
python src/pipeline.py
```

## 📓 Notebook Walkthrough

An end-to-end example of the pipeline can be found in
[`full_pipeline_summary.ipynb`](full_pipeline_summary.ipynb).



