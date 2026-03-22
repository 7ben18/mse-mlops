# MLflow Tracking

Stage 3 of the MLOps pipeline. Captures training runs from `src/mse_mlops/train.py` in MLflow, including
configuration, epoch metrics, summary metrics, and model/history artifacts.

## What it is

Training is MLflow-enabled by default and always runs inside an MLflow run context.

Start MLflow server:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:./mlartifacts --host 127.0.0.1 --port 5000
```

Start training:

```bash
uv run train-dinov3 --config config/train.yaml
```

Open UI:

`http://127.0.0.1:5000`

Tracking code lives in:

- `mse_mlops.tracking.mlflow_tracker`
- `mse_mlops.train`

## Architecture

```text
train.py
  -> init_mlflow(args)
      -> set_tracking_uri
      -> set_experiment
      -> start_run
  -> log_run_params(...)
  -> epoch loop
      -> log_epoch_metrics(..., step=epoch)
  -> finalize
      -> log_summary_metrics(...)
      -> log_final_artifacts(...)
          - training/history.json
          - model/ (mlflow.pytorch.log_model)
```

The run closes automatically when the MLflow context exits.

---

## What is logged

### Parameters (once per run)

- Dataset/split config (`data_dir`, `val_mode`, `val_split`, fractions/samples)
- Training config (`epochs`, `batch_size`, `lr`, scheduler settings, etc.)
- Execution controls (`device`, `resume_from_checkpoint`, `save_total_limit`)
- Derived metadata (`train_count`, `val_count`, `class_names`)

### Metrics (per epoch)

- `train_loss`, `train_acc`
- `val_loss`, `val_acc`, `val_precision`, `val_recall`, `val_f1`, `val_roc_auc`
- `optimizer_steps`

### Summary metrics (run-level)

- `best_val_roc_auc`
- `best_epoch`

### Artifacts

- `training/history.json`
- `model/` MLflow model artifact from:

```python
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
)
```

## Configuration

`config/train.yaml`:

| Key | Required | Example |
|-----|----------|---------|
| `tracking.mlflow_tracking_uri` | yes | `http://127.0.0.1:5000` |
| `tracking.mlflow_experiment_name` | yes | `mse-mlops-training` |
| `tracking.mlflow_run_name` | no | `baseline-2026-03-22` |
| `tracking.mlflow_tags` | no | `{project: mse-mlops}` |

CLI overrides:

- `--mlflow-tracking-uri`
- `--mlflow-experiment-name`
- `--mlflow-run-name`
- `--mlflow-tags`

If tracking URI or experiment name is empty, training fails fast with `ValueError`.

## Checkpoint and resume

- Local checkpoints remain at `outputs/.../checkpoints/epoch_XXX.pt`.
- Checkpoints store `history` and optional `best_model_state`.
- Resume restores model/optimizer/scheduler and metric history.

## Model Registry note

This stage does not auto-register models.
You can register later from a run artifact (for example `runs:/<RUN_ID>/model`) after manual validation.
