# Ray Tune

Stage 2 of the MLOps pipeline. Runs hyperparameter search against the existing DINOv3 training
workflow, tracks tuning sessions in MLflow, and exports the best trial configuration.

## What it is

The tuning stack is opt-in and runs only when you start it explicitly.

Local run:

```bash
uv run python scripts/tune.py --config config/tune.yaml
```

Docker run:

```bash
docker compose --profile train run --build --rm tune
```

Tuning depends on:

| Service | URL | Purpose |
|---------|-----|---------|
| MLflow | [http://localhost:5001](http://localhost:5001) | Parent/trial run tracking |
| Ray Tune | local process or Ray cluster | Trial orchestration |

Before a local run, start MLflow separately and use a local tracking URI in `config/tune.yaml`
or via config edits:

```bash
uv run mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Without an MLflow server, tuning fails before the first trial starts.

The tuning code is part of the main Python package:

- `mse_mlops.tune`
- `mse_mlops.train`
- `mse_mlops.tracking.mlflow_tracker`

## Architecture

```text
scripts/tune.py
  -> mse_mlops.tune.load_tune_config(...)
  -> mse_mlops.tune.run_tuning(config)
      -> start MLflow parent run
      -> initialize Ray / connect to Ray cluster
      -> build Ray search space from config/tune.yaml
      -> run N trials
          -> reuse mse_mlops.train.run_training(...)
          -> log one MLflow child run per trial
          -> report per-epoch metrics back to Ray Tune
      -> select best trial
      -> write best_config.yaml + leaderboard.json
      -> log best-trial artifacts to MLflow
```

The tuning path does not implement a second training pipeline. Each trial reuses the same training
codepath as `scripts/train.py`.

---

## User-facing behavior

### Standalone config

Tuning uses `config/tune.yaml` and does not inherit from `config/train.yaml`.

It defines:

- `base_run` — fixed model/data/training/tracking settings for each trial
- `search_space` — tunable parameters and distributions
- `tune` — metric, mode, scheduler, sampling, concurrency, resources, and Ray address
- `output` — Ray results directory and exported best-config/leaderboard paths

### Trial execution

Each Ray trial:

- merges sampled hyperparameters into `base_run`
- resolves repo-relative dataset/model paths before starting training
- writes trial-local checkpoints under the configured Ray results tree
- logs metrics and artifacts into MLflow as a child run

### Best-trial export

After completion, tuning writes:

- `reports/tuning/best_config.yaml`
- `reports/tuning/leaderboard.json`

The best trial's promoted checkpoint is also logged to MLflow as a tuning artifact.

---

## Configuration

Primary config file:

- `config/tune.yaml`

Key fields:

| Key | Example | Description |
|-----|---------|-------------|
| `base_run.model.model_name` | `models/pretrained/dinov3-vits16-pretrain-lvd1689m` | Backbone path or HF model ID |
| `base_run.data.metadata_csv` | `data/processed/ham10000/metadata.csv` | Processed metadata table |
| `base_run.data.images_dir` | `data/processed/ham10000/HAM10000_images` | Split image root |
| `base_run.tracking.mlflow_tracking_uri` | `http://mlflow:5001` | MLflow endpoint |
| `base_run.tracking.mlflow_experiment_name` | `mse-mlops-tuning` | Dedicated tuning experiment |
| `search_space.training.lr` | `loguniform` | Learning-rate distribution |
| `search_space.training.batch_size` | `choice` | Batch-size candidates |
| `tune.metric` | `val_roc_auc` | Objective optimized by Ray Tune |
| `tune.mode` | `max` | Optimization direction |
| `tune.num_samples` | `4` | Number of sampled trials |
| `tune.scheduler` | `fifo` | Trial scheduler |
| `tune.resources` | `{cpu: 1, gpu: 0}` | Per-trial Ray resources |
| `tune.ray_address` | `null` | Local Ray or remote cluster address |
| `output.ray_results_dir` | `reports/ray_results` | Ray output root |

Supported search-space types:

- `choice`
- `uniform`
- `loguniform`
- `randint`

## MLflow behavior

Tuning logs:

- one parent run for the overall tuning session
- one child run per Ray trial
- per-epoch trial metrics reported through the shared training loop
- summary artifacts:
  - `tuning/session.json`
  - `tuning/best_config.yaml`
  - `tuning/leaderboard.json`
  - `tuning/best_model/*`

Use a separate MLflow experiment for tuning, for example `mse-mlops-tuning`.

## Outputs

Local files written by tuning include:

- Ray result state under `reports/ray_results/`
- exported best config at `reports/tuning/best_config.yaml`
- exported leaderboard at `reports/tuning/leaderboard.json`
- per-trial checkpoints under the Ray results tree

These outputs are not used directly by serving. Serving still expects a promoted checkpoint in
`models/finetuned/.../best_model.pt`.

## Docker and Ray cluster usage

Docker tuning reuses the same image as training and adds a dedicated `tune` service in
`compose.yaml`.

Run it with:

```bash
docker compose --profile train run --build --rm tune
```

For multi-node Ray, set `tune.ray_address` in `config/tune.yaml` to your cluster address before
starting the run.

## Further reading

- Pipeline overview: [pipeline.md](pipeline.md)
- MLflow parent/child tracking model: [mlflow-tracking.md](mlflow-tracking.md)
- Source package: `src/mse_mlops/tune.py`
