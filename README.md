# Data Science Projects

[![Lint](https://github.com/MediaJohnD/data-science-projects/actions/workflows/lint.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/lint.yml)
[![Test](https://github.com/MediaJohnD/data-science-projects/actions/workflows/test.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/test.yml)
[![Docker Build](https://github.com/MediaJohnD/data-science-projects/actions/workflows/docker-build.yml/badge.svg)](https://github.com/MediaJohnD/data-science-projects/actions/workflows/docker-build.yml)

This repository is hosted at [https://github.com/MediaJohnD/data-science-projects](https://github.com/MediaJohnD/data-science-projects).

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

