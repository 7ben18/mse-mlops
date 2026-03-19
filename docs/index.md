# mse-mlops

[![Build status](https://img.shields.io/github/actions/workflow/status/7ben18/mse-mlops/main.yml?branch=main)](https://github.com/7ben18/mse-mlops/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/7ben18/mse-mlops)](https://github.com/7ben18/mse-mlops/commits/main)
[![License](https://img.shields.io/github/license/7ben18/mse-mlops)](https://github.com/7ben18/mse-mlops/blob/main/LICENSE)

DINOv3 fine-tuning setup for melanoma skin cancer classification.

## Data Lifecycle

- `train`: supervised data used for fitting model weights.
- `val`: held-out validation data used during experimentation.
- `future`: newly collected production data kept separate until curated for later fine-tuning.
- `test`: final hold-out set used only for the final pre-production evaluation.

## Project Conventions

- Reusable Python code belongs under `src/mse_mlops`.
- Exploratory notebooks belong under `notebooks/`.
- Reusable helper code and dataset-processing logic belong under `src/mse_mlops`, not under `notebooks/`.
- Raw data and derived dataset artifacts belong under `data/` and should be managed outside git.

## Training Prerequisite

Before the first training run:

1. Request access to `facebook/dinov3-vits16-pretrain-lvd1689m` on Hugging Face.
2. Log in locally with:

`hf auth login`

3. Download the pretrained backbone locally:

`uv run python scripts/download_model.py`

The default config expects the downloaded model at:

`outputs/pretrained/dinov3-vits16-pretrain-lvd1689m`

## Docker Training

Run Docker training with:

`docker compose --profile train run --build --rm train`

The `train` service is opt-in and is not started by a plain `docker compose up`.

The container mounts:

- `config/` at `/app/config` read-only
- `data/` at `/app/data`
- `outputs/` at `/app/outputs`

So config changes in `config/train.yaml` are picked up on the next Docker training run without rebuilding the image.

## Serving

Serving code now lives under `src/mse_mlops/serving`, and the repo-root `compose.yaml` orchestrates the API and UI.

Start the serving stack with:

`docker compose up --build`

This starts:

- FastAPI at `http://localhost:8000`
- Streamlit at `http://localhost:7777`
