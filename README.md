# Machine Learning

A reproducible machine-learning project organized using the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) conventions.

## Project structure

```text
.
├── data/
│   ├── external/          # Data from third-party sources
│   ├── interim/           # Intermediate transformed data
│   ├── processed/         # Final modeling-ready data
│   └── raw/               # Immutable original data
├── docs/                  # Project documentation
├── models/                # Trained models and serialized artifacts
├── notebooks/             # Exploration and experiment notebooks
├── references/            # Data dictionaries and reference material
├── reports/
│   └── figures/           # Figures for reports
├── src/
│   └── machine_learning/  # Reusable project Python package
│       ├── config.py
│       ├── dataset.py
│       ├── features.py
│       ├── modeling/
│       │   ├── predict.py
│       │   └── train.py
│       └── plots.py
├── tests/                 # Automated tests
├── .env.example           # Example environment settings
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/t115378068/machine-learning.git
cd machine-learning
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Create `.env` from `.env.example` if you need local configuration. Never commit secrets or private datasets.

## Workflow

1. Store original inputs under `data/raw/` and keep them immutable.
2. Prepare clean modeling datasets in `data/processed/`.
3. Put exploratory analysis in `notebooks/`; move reusable code into `src/machine_learning/`.
4. Save trained models under `models/` and figures under `reports/figures/`.
5. Run `pytest` before committing code.

`make setup` installs the development dependencies; `make test` runs tests. On Windows without Make, use the Python commands above.

This repository provides the **project scaffold only**: model training and dataset logic are placeholders ready for your experiments.
