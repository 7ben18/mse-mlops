# MLflow Tracking

MLflow is the central tracking backend for both standard training and Ray Tune sessions.

## What is tracked

- `scripts/train.py` starts a single MLflow run for one training job.
- `scripts/tune.py` starts one parent MLflow run for the tuning session.
- Each Ray Tune trial starts its own nested child run.

Docker services talk to MLflow at `http://mlflow:5001`. Local runs can point to `http://127.0.0.1:5001`.

## Core flow

```text
scripts/train.py
  -> mse_mlops.train.run_training(config)
      -> tracking.init_mlflow(...)
      -> tracking.log_run_params(...)
      -> epoch loop
          -> tracking.log_epoch_metrics(..., step=epoch)
      -> tracking.log_summary_metrics(...)
      -> tracking.log_final_artifacts(...)

scripts/tune.py
  -> mse_mlops.tune.run_tuning(config)
      -> tracking.start_run(...)               # parent tuning run
      -> Ray Tune trial orchestration
          -> mse_mlops.train.run_training(...)
              -> tracking.init_mlflow(..., tags={"mlflow.parentRunId": ...})
      -> tracking.log_dict_artifact(...)       # best config, leaderboard
      -> tracking.log_local_artifact(...)      # best trial model checkpoint
```

## Training runs

Each normal training run logs:

- Parameters: dataset paths, split settings, optimizer/scheduler settings, execution controls, train/val counts, class names.
- Per-epoch metrics: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_precision`, `val_recall`, `val_f1`, `val_roc_auc`, `optimizer_steps`.
- Summary metrics: `best_val_roc_auc`, `best_epoch`.
- Artifacts:
  - `training/history.json`
  - `model/` MLflow model artifact for the promoted best checkpoint

Local checkpoint files are still written under `models/finetuned/.../checkpoints/`, and the promoted serving checkpoint remains `best_model.pt`.

## Tuning runs

Each tuning session logs one parent run plus one nested child run per trial.

Parent run artifacts:

- `tuning/session.json`
- `tuning/best_config.yaml`
- `tuning/leaderboard.json`
- `tuning/best_model/*` copied from the best trial's promoted checkpoint

Nested child trial runs log:

- trial-specific parameters after config merge
- per-epoch metrics from the shared training codepath
- summary metrics for the trial
- `training/history.json`

Only the best trial's promoted checkpoint is logged back to MLflow as an artifact.

## Configuration

Training config keys:

- `tracking.mlflow_tracking_uri`
- `tracking.mlflow_experiment_name`
- `tracking.mlflow_run_name`
- `tracking.mlflow_tags`

Tuning config:

- `config/tune.yaml` is standalone and defines `base_run`, `search_space`, `tune`, and `output`.
- The recommended tuning experiment name is separate from standard training, for example `mse-mlops-tuning`.

## Docker usage

Start MLflow:

```bash
make mlflow-up
```

Run standard training:

```bash
make train-docker
```

Run tuning:

```bash
docker compose --profile train run --build --rm tune
```

MLflow state stays in the repo-local paths:

- `mlflow.db`
- `mlartifacts/`

`make docker-down` does not delete them.
