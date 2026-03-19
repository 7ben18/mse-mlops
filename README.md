# mse-mlops

[![Build status](https://img.shields.io/github/actions/workflow/status/7ben18/mse-mlops/main.yml?branch=main)](https://github.com/7ben18/mse-mlops/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/7ben18/mse-mlops)](https://github.com/7ben18/mse-mlops/commits/main)
[![License](https://img.shields.io/github/license/7ben18/mse-mlops)](https://github.com/7ben18/mse-mlops/blob/main/LICENSE)

DINOv3 fine-tuning setup for melanoma skin cancer classification.

## Resources

- 🚀 [Repository](https://github.com/7ben18/mse-mlops)
- 📖 [Documentation](https://7ben18.github.io/mse-mlops/)

## Data Lifecycle

The project is organized around four dataset buckets:

- `train`: supervised data used for fitting model weights.
- `val`: held-out validation data used during experimentation and future hyperparameter tuning.
- `future`: newly collected production data. Keep it separate until it has been reviewed and selected samples are promoted into `train` for later fine-tuning.
- `test`: final hold-out set. Do not use it during normal development; touch it only for the final pre-production evaluation.

Long-term target layout:

```text
data/
  raw/
    melanoma_cancer_dataset/
      train/
        benign/
        malignant/
      future/
      test/
        benign/
        malignant/
```

Current default training does not use `test` for validation. It uses `val_mode: split` and carves validation data out of `train`, while keeping `test` untouched until the final evaluation.

If a dedicated validation directory is introduced later, switch to `val_mode: test` and point `val_subdir` at that held-out validation folder.

Dataset provenance and local acquisition notes live in:

`data/raw/melanoma_cancer_dataset/README.md`

## Training

Model:

`facebook/dinov3-vits16-pretrain-lvd1689m`

Before the first training run:

1. Request access to the model on Hugging Face.
2. Log in locally with the Hugging Face CLI:

`hf auth login`

3. Download the pretrained backbone locally:

`uv run python scripts/download_model.py`

The default training config expects the downloaded model at:

`outputs/pretrained/dinov3-vits16-pretrain-lvd1689m`

Run training:

`uv run train-dinov3`

Run training in Docker:

`docker compose --profile train run --build --rm train`

The train container is opt-in only. A plain `docker compose up` will not start training.

Docker training mounts these host folders into the container:

- `config/` -> `/app/config` (read-only)
- `data/` -> `/app/data`
- `outputs/` -> `/app/outputs`

That means edits to `config/train.yaml` apply to the next Docker training run without rebuilding the image. Rebuilds are still needed after code or dependency changes.

On Apple Silicon Macs, local host training can use `mps` when `device: auto` resolves it, but Docker training currently does not expose MPS and should be assumed to run on CPU.

Training settings:

`config/train.yaml`

Training artifacts are written under `outputs/`:

- `best_model.pt`: best checkpoint selected on validation ROC AUC.
- `checkpoints/epoch_*.pt`: resumable epoch checkpoints.
- `history.json`: per-epoch training and validation metrics.

## Serving

Serving now follows the main project layout instead of living as a nested standalone app. Importable API and UI code lives under `src/mse_mlops/serving`, while service Dockerfiles live under `docker/`.

Start the inference API and Streamlit UI from the repo root:

`docker compose up --build`

This starts:

- `api`: FastAPI on `http://localhost:8000`
- `ui`: Streamlit on `http://localhost:7777`

The `train` service is still opt-in only and is not started by a plain `docker compose up`.

For local development outside Docker:

- `uv run --group api serve-api`
- `uv run --group ui serve-ui`

The API expects a trained checkpoint at:

`outputs/dinov3_melanoma/best_model.pt`

## Project Conventions

- Importable and reusable Python code belongs under `src/mse_mlops`.
- Exploratory notebooks belong under `notebooks/`.
- Reusable notebook helper code belongs under `src/mse_mlops`, not under `notebooks/`.
- Raw data, derived tables, and dataset metadata belong under `data/` and should be managed outside git with DVC or another data-management layer.
- Explanatory project material belongs under `docs/`.
- Reports and exported analysis outputs belong under `reports/`.
- Service-specific Dockerfiles belong under `docker/`, not in nested app subtrees.

## Structure

    ├── .github
    │   ├── actions        <- Github Actions configuration.
    │   └── workflows      <- Github Actions workflows.
    │
    ├── config            <- Training and experiment configuration.
    ├── docker            <- Service Dockerfiles for API and UI.
    ├── scripts           <- Utility scripts for local and overnight runs.
    ├── src/mse_mlops     <- Source code for this project.
    │   └── serving       <- FastAPI API, Streamlit UI, and serving helpers.
    ├── data
    │   ├── raw            <- DVC-managed source data and curated splits.
    │   └── processed      <- Derived datasets and exported tables.
    │
    ├── docs               <- MkDocs documentation for the project.
    ├── models             <- Model checkpoints, predictions, metrics, and summaries.
    ├── notebooks          <- Exploratory notebooks only.
    ├── outputs            <- Local training outputs and resumable checkpoints.
    ├── logs               <- Local training and smoke-test logs.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    ├── tests              <- Unit tests for the project.
    ├── .gitignore         <- Files to be ignored by git.
    ├── docker/train.Dockerfile <- Dockerfile for the training image.
    ├── LICENSE            <- MIT License.
    ├── Makefile           <- Makefile with commands like `make install` or `make test`.
    ├── mkdocs.yml         <- MkDocs configuration.
    ├── pyproject.toml     <- Package build configuration.
    ├── README.md          <- The top-level README for this project.
    └── uv.lock            <- Lock file for uv.
