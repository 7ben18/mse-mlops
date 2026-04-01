# API Reference

Auto-generated from Python docstrings. All modules are part of the
`mse_mlops` package installed from `src/`.

---

## Core

### `mse_mlops.paths`

Project-wide path constants and configuration loader helpers.

::: mse_mlops.paths

---

### `mse_mlops.data_processing`

HAM10000 lesion-level dataset splitting and image/mask copying utilities.

::: mse_mlops.data_processing

---

### `mse_mlops.train`

DINOv3 fine-tuning training loop, config dataclass, and training helpers.

::: mse_mlops.train

---

## Analysis

### `mse_mlops.analysis.ham10000`

HAM10000 EDA helpers: image/mask triplet building, sampling, and visualisation.

::: mse_mlops.analysis.ham10000

---

### `mse_mlops.analysis.melanoma_dataset`

Melanoma dataset overview loader and per-class statistics helpers.

::: mse_mlops.analysis.melanoma_dataset

---

## Serving

### `mse_mlops.serving.api`

FastAPI application: prediction endpoint, feedback submission, and listing routes.

::: mse_mlops.serving.api

---

### `mse_mlops.serving.inference`

`DinoV3Classifier` model class, `load_model`, `predict`, and device resolution helpers.

::: mse_mlops.serving.inference

---

### `mse_mlops.serving.feedback_store`

JSONL-backed feedback persistence: append, load, and write helpers.

::: mse_mlops.serving.feedback_store

---

### `mse_mlops.serving.ui`

Streamlit frontend for skin lesion image upload and doctor feedback review.

::: mse_mlops.serving.ui

---

## Tracking

### `mse_mlops.tracking.mlflow_tracker`

MLflow run lifecycle helpers: `init_mlflow`, `log_run_params`, `log_epoch_metrics`,
`log_summary_metrics`, and `log_final_artifacts`.

::: mse_mlops.tracking.mlflow_tracker
