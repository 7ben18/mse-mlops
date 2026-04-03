# Serving Layer

Stage 4 of the MLOps pipeline. Exposes the trained DINOv3 HAM10000 classifier to patients and
medical staff through a REST API and a web UI, and collects verified labels back into the
feedback store.

## What it is

The serving stack is opt-in and starts only when you enable its profile:

```bash
make ui-up
```

| Service | URL | Audience |
|---------|-----|----------|
| MLflow | http://localhost:5001 | Experiment browsing |
| Streamlit UI | http://localhost:7777 | Patients + doctors |
| FastAPI API | http://localhost:8000 | UI + external tooling |

Without `make ui-up`, the serving stack stays off. Use `make mlflow-up` if you only want MLflow.

Stop the whole Docker stack with:

```bash
make docker-down
```

`make docker-down` removes Compose containers, the Compose network, and named volumes such as
`feedback_data`. It does not delete repo-local bind-mounted files like `mlflow.db` or
`mlartifacts/`, and it does not remove local Docker images.

Stop only the API and UI while keeping MLflow running:

```bash
make ui-down
```

The serving code is now part of the main Python package:

- `mse_mlops.serving.api`
- `mse_mlops.serving.inference`
- `mse_mlops.serving.feedback_store`
- `mse_mlops.serving.ui`

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Docker Compose                          │
│                                                              │
│  ┌──────────────────┐   HTTP    ┌──────────────────────────┐ │
│  │  ui (Streamlit)  │ ────────► │     api (FastAPI)        │ │
│  │  port 7777       │           │     port 8000            │ │
│  └──────────────────┘           │                          │ │
│  models/ (read-only) ──────────►│  DinoV3Classifier        │ │
│                                 │  + eval preprocessing    │ │
│                                 │                          │ │
│  feedback_data (volume) ───────►│  /feedback/              │ │
│                                 │  ├── feedback.jsonl      │ │
│                                 │  └── images/             │ │
│                                 └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

The UI is stateless — it never touches the disk. All data flows through the API.

---

## User-facing features

### Tab 1 — Skin Analysis (public)

Any user can upload a skin lesion photo. The model returns:

- **Prediction** — benign or malignant
- **Confidence** — model certainty as a percentage
- **Probability breakdown** — bar for each class

A medical disclaimer reminds users this is not a clinical diagnosis.

### Tab 2 — Review & Label (doctors)

Password-protected. Doctors see a queue of AI predictions that have not yet been verified.
For each entry they assign a ground-truth label (benign / malignant). The label is written
back to the feedback store and the entry is removed from the queue.

### Tab 3 — Bulk Dataset Upload (doctors)

Password-protected. Two modes:

- **Single image** — upload one image with a label.
- **ZIP + label sheet** — upload a ZIP of images alongside a CSV or Excel file mapping
  `filename → label`. All matched images are uploaded with a progress bar.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Upload image → get prediction + save to feedback store |
| `GET` | `/feedback` | Return all feedback entries |
| `POST` | `/feedback` | Assign a verified label to a prediction entry |
| `POST` | `/upload-labeled` | Save image + label directly (no inference) |

## Feedback store

Every prediction and labeled upload is appended to `/feedback/feedback.jsonl` as a JSON line:

```json
{
  "image_id":      "uuid4",
  "filename":      "lesion.jpg",
  "timestamp":     "2026-03-19T10:00:00+00:00",
  "prediction":    "benign",
  "confidence":    0.92,
  "probabilities": {"benign": 0.92, "malignant": 0.08},
  "label":         null,
  "source":        "predict | doctor_review | upload_labeled"
}
```

`label` starts as `null` and is filled in when a doctor reviews the entry.

Images uploaded via Tab 3 are stored at `/feedback/images/{image_id}{ext}`.

The store is designed to be swapped out: replace `append_feedback_entry()`,
`load_feedback_entries()`, and `write_feedback_entries()` in
`mse_mlops.serving.feedback_store` to point at a database or MLflow artifact store.

## Configuration

| Variable | Compose Value | Service | Description |
|----------|---------------|---------|-------------|
| `MODEL_PATH` | `/app/models/finetuned/dinov3_ham10000/best_model.pt` | API | Path to trained `.pt` checkpoint inside the container |
| `FEEDBACK_DIR` | `/feedback` | API | Root of the feedback volume |
| `API_URL` | `http://api:8000` | UI | API base URL inside the Compose network |
| `DOCTOR_PASSWORD` | `doctor123` | UI | Password for doctor tabs — change in production |

Set overrides in the repo-root `.env`:

```
DOCTOR_PASSWORD=your_secure_password
```

## Model checkpoint

The API expects a `.pt` file written by the training pipeline (`scripts/train.py`, backed by `mse_mlops.train`) (`best_model.pt`):

```python
{
    "model_state_dict": dict,   # DinoV3Classifier weights
    "class_names":      list,   # ["benign", "malignant"]
    "model_name":       str,    # HuggingFace ID used at training time
    "image_size":       int,
    "freeze_backbone":  bool,
}
```

Preprocessing (resize → center crop → normalize) is derived from the checkpoint's `model_name`
at load time, ensuring it always matches what was used during training.

## Future integration points

Two no-op stubs in `mse_mlops.serving.inference` are already wired into the request lifecycle:

```python
def log_prediction_to_mlflow(image_id, prediction): pass  # TODO
def notify_curation_pipeline(image_id, label):      pass  # TODO
```

When MLflow and the retraining pipeline are ready, fill these in — no other code changes needed.

## Further reading

- Full architecture note: [serving-architecture.md](serving-architecture.md)
- Source package: `src/mse_mlops/serving`
