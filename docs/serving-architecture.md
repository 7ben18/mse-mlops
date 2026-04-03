# Serving Architecture

The serving layer is part of the main `mse_mlops` package and is deployed from the repo root.

## Layout

```text
compose.yaml
docker/
  api.Dockerfile
  train.Dockerfile
  ui.Dockerfile
src/
  mse_mlops/
    serving/
      api.py
      feedback_store.py
      inference.py
      ui.py
scripts/
  serve_api.py
  serve_ui.py
```

## Runtime Topology

`make ui-up` starts the serving stack:

- `mlflow`: MLflow tracking UI on port `5001`
- `api`: FastAPI inference API on port `8000`
- `ui`: Streamlit web UI on port `7777`

Without `make ui-up`, the serving stack stays off. Use `make mlflow-up` if you only want MLflow.

The training service remains opt-in under the `train` profile:

`make train-docker`

## Mounted State

- `api` mounts `./models` read-only at `/app/models`
- `api` uses the checkpoint at `/app/models/finetuned/dinov3_ham10000/best_model.pt`
- `api` persists feedback data in the named Docker volume `feedback_data`

This keeps trained artifacts and serving state separate:

- model artifacts stay in `models/`
- feedback JSONL and uploaded images stay in the volume mounted at `/feedback`

## Package Boundaries

- `mse_mlops.serving.api`: FastAPI routes and request handling
- `mse_mlops.serving.inference`: checkpoint loading, preprocessing, inference, integration stubs
- `mse_mlops.serving.feedback_store`: JSONL-backed feedback persistence
- `mse_mlops.serving.ui`: Streamlit interface
- `scripts/serve_api.py`: CLI wrapper for local API runs
- `scripts/serve_ui.py`: CLI wrapper for local UI runs

## Local Development

Run the API:

```bash
uv run --group api python scripts/serve_api.py
```

Run the UI in a second terminal:

```bash
uv run --group ui python scripts/serve_ui.py
```

Set overrides via environment variables as needed:

- `MODEL_PATH`
- `FEEDBACK_DIR`
- `API_URL`
- `DOCTOR_PASSWORD`
