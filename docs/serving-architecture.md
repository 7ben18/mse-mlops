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
      launchers.py
      ui.py
```

## Runtime Topology

`docker compose up --build` starts two default services:

- `api`: FastAPI inference API on port `8000`
- `ui`: Streamlit web UI on port `7777`

The training service remains opt-in under the `train` profile:

`docker compose --profile train run --build --rm train`

## Mounted State

- `api` mounts `./outputs` read-only at `/app/outputs`
- `api` uses the checkpoint at `/app/outputs/dinov3_melanoma/best_model.pt`
- `api` persists feedback data in the named Docker volume `feedback_data`

This keeps trained artifacts and serving state separate:

- model artifacts stay in `outputs/`
- feedback JSONL and uploaded images stay in the volume mounted at `/feedback`

## Package Boundaries

- `mse_mlops.serving.api`: FastAPI routes and request handling
- `mse_mlops.serving.inference`: checkpoint loading, preprocessing, inference, integration stubs
- `mse_mlops.serving.feedback_store`: JSONL-backed feedback persistence
- `mse_mlops.serving.ui`: Streamlit interface
- `mse_mlops.serving.launchers`: script entrypoints for local runs

## Local Development

Run the API:

```bash
uv run --group api serve-api
```

Run the UI in a second terminal:

```bash
uv run --group ui serve-ui
```

Set overrides via environment variables as needed:

- `MODEL_PATH`
- `FEEDBACK_DIR`
- `API_URL`
- `DOCTOR_PASSWORD`
