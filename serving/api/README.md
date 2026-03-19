# serving/api/

FastAPI inference API for the melanoma skin cancer classifier. Handles predictions, feedback
collection, and labeled image ingestion. Exposes a REST interface consumed by the Streamlit UI
and any external tooling.

## Routes

| Method | Path | Body / Params | Response | Description |
|--------|------|---------------|----------|-------------|
| `GET` | `/health` | — | `{"status": "ok"}` | Liveness check |
| `POST` | `/predict` | `multipart/form-data` — `file` (image) | Prediction JSON | Run inference, save entry to feedback store (`label=null`) |
| `GET` | `/feedback` | — | Array of feedback entries | Return entire feedback store |
| `POST` | `/feedback` | JSON `{image_id, label, source}` | `{"status": "ok"}` | Assign a verified label to a prediction entry |
| `POST` | `/upload-labeled` | `multipart/form-data` — `file` (image) + `label` (form field) | `{"status": "ok", "image_id", "label"}` | Save image + label directly (no inference) |

### `POST /predict` — response shape

```json
{
  "class":         "benign",
  "confidence":    0.9241,
  "probabilities": {"benign": 0.9241, "malignant": 0.0759},
  "image_id":      "550e8400-e29b-41d4-a716-446655440000"
}
```

### Feedback entry shape

```json
{
  "image_id":      "550e8400-e29b-41d4-a716-446655440000",
  "filename":      "lesion_42.jpg",
  "timestamp":     "2026-03-19T10:00:00+00:00",
  "prediction":    "benign",
  "confidence":    0.9241,
  "probabilities": {"benign": 0.9241, "malignant": 0.0759},
  "label":         null,
  "source":        "predict"
}
```

`label` is `null` until a doctor assigns a ground-truth label via `POST /feedback`.
`source` is one of `predict`, `doctor_review`, or `upload_labeled`.

## Files

### `model.py`

Owns everything related to the model:

- `DinoV3Classifier` — PyTorch `nn.Module` (mirrors `src/mse_mlops/train.py`)
- `load_model(path)` — loads checkpoint, rebuilds model, builds eval transform
- `predict(image_bytes) → dict` — preprocesses image, runs forward pass, returns prediction
- `log_prediction_to_mlflow(image_id, prediction)` — **no-op stub**, ready for MLflow
- `notify_curation_pipeline(image_id, label)` — **no-op stub**, ready for retraining trigger

**Preprocessing pipeline** (must match training):
```
Resize(max(image_size, image_size * 256/224))
→ CenterCrop(image_size)
→ ToTensor()
→ Normalize(mean, std)   ← derived from AutoImageProcessor, not hard-coded
```

### `main.py`

FastAPI application. Key internals:

- `_save_feedback(entry)` — appends one JSON line to `/feedback/feedback.jsonl`
- `_load_feedback()` — reads and parses the full JSONL file
- CORS is enabled for all origins (Streamlit needs cross-origin requests)
- `/feedback` directory and `/feedback/images/` are created at startup if missing

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/best_model.pt` | Path to trained checkpoint |
| `FEEDBACK_DIR` | `/feedback` | Root of the feedback volume |

## Dependencies

Managed with `uv` via `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `torch` + `torchvision` | Inference |
| `transformers` | HuggingFace model + processor loading |
| `pillow` | Image decoding |
| `python-multipart` | Multipart file upload parsing |
| `pydantic` | Request body validation |

## Running locally

```bash
cd serving/api
MODEL_PATH=../../models/best_model.pt uv run uvicorn main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Adding a new route

1. Define the handler in `main.py` using `@app.get` / `@app.post`.
2. If the route reads or writes feedback data, use `_save_feedback()` / `_load_feedback()`.
3. If the route should eventually notify MLflow or the retraining pipeline, call
   `log_prediction_to_mlflow()` or `notify_curation_pipeline()` from `model.py`.
4. Add the route to the table in this README.

## Swapping the feedback store backend

All store I/O is in two functions in `main.py`:

```python
def _save_feedback(entry: dict) -> None: ...
def _load_feedback() -> list[dict]: ...
```

Replace the JSONL read/write logic with database calls or MLflow artifact logging.
No other code needs to change.
