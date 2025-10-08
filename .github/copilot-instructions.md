# Copilot Instructions for AI Coding Agents

## Project Overview
This repository applies data science techniques to churn analysis, following the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) template. The codebase is organized for modularity and reproducibility, with clear separation of data, features, models, notebooks, and documentation.

## Architecture & Key Components
- **src/**: Main source code, organized by domain:
  - `data/`: Data ingestion and processing (see `make_dataset.py`).
  - `features/`: Feature engineering scripts.
  - `models/`: Model training (`train_model.py`) and prediction (`predict_model.py`).
  - `visualization/`: Plotting and reporting utilities.
- **data/**: Layered data storage:
  - `raw/`: Immutable source data.
  - `interim/`: Transformed intermediate data.
  - `processed/`: Final datasets for modeling.
- **notebooks/**: Jupyter notebooks for exploration and modeling. Naming convention: `<number>-<initials>-<description>`.
- **reports/**: Generated figures and analysis outputs.
- **docs/**: Sphinx documentation. Build with `make html` or `docs/make.bat html`.

## Developer Workflows
- **Environment Setup**:
  - Install dependencies: `make requirements` (uses `requirements.txt`).
  - Test environment: `make test_environment` (runs `test_environment.py`).
  - Supports both Conda and virtualenv (see `Makefile` targets).
- **Data Processing**:
  - Generate processed data: `make data` (runs `src/data/make_dataset.py`).
  - Sync data with S3: `make sync_data_to_s3` / `make sync_data_from_s3` (requires AWS CLI, see `Makefile` and `docs/commands.rst`).
- **Linting**:
  - Run `make lint` (uses flake8, config in `tox.ini`).
- **Documentation**:
  - Sphinx docs in `docs/`. Build with `make html` or `docs/make.bat html`.

## Conventions & Patterns
- **Imports**: Use relative imports within `src/` modules.
- **Data Flow**: Data moves from `data/raw` → `data/interim` → `data/processed` via scripts in `src/data/` and `src/features/`.
- **Modeling**: Models are trained and serialized in `models/`, with code in `src/models/`.
- **Notebooks**: For exploration and prototyping; production code should be moved to `src/` modules.
- **Naming**: Follow the cookiecutter template for directory and file naming.

## Integration Points
- **AWS S3**: Data sync via Makefile commands and AWS CLI.
- **Sphinx**: Documentation generation in `docs/`.
- **Joblib**: Used for data and model serialization.

## Examples
- To process data: `make data`
- To train a model: run scripts in `src/models/` or use notebooks in `notebooks/`
- To lint code: `make lint`
- To build docs: `make html` (from `docs/`)

## References
- See `README.md` for project structure and conventions.
- See `Makefile` and `docs/commands.rst` for workflow commands.
- See `requirements.txt` for dependencies.

---

**If any section is unclear or missing important project-specific details, please provide feedback so this guide can be improved.**
