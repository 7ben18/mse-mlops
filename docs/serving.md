# Serving Layer

Stage 4 of the MLOps pipeline. Exposes the trained DINOv3 melanoma classifier to patients and
medical staff through a REST API and a web UI, and collects verified labels back into the
feedback store.

---

## What it is

Two Docker containers, one command:

```bash
cd serving && docker compose up --build
```

| Service | URL | Audience |
|---------|-----|----------|
| Streamlit UI | http://localhost:7777 | Patients + doctors |
| FastAPI API | http://localhost:8000 | UI + external tooling |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Docker Compose                          │
│                                                              │
│  ┌──────────────────┐   HTTP    ┌──────────────────────────┐ │
│  │  ui (Streamlit)  │ ────────► │     api (FastAPI)        │ │
│  │  port 7777       │           │     port 8000            │ │
│  └──────────────────┘           │                          │ │
│                                 │  DinoV3Classifier        │ │
│  ../models/ (read-only) ───────►│  + eval preprocessing    │ │
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

Full route documentation and request/response shapes: [`serving/api/README.md`](../serving/api/README.md).

---

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

The store is designed to be swapped out: replace `_save_feedback()` and `_load_feedback()` in
`serving/api/main.py` to point at a database or MLflow artifact store.

---

## Configuration

| Variable | Default | Service | Description |
|----------|---------|---------|-------------|
| `MODEL_PATH` | `/models/best_model.pt` | API | Path to trained `.pt` checkpoint |
| `FEEDBACK_DIR` | `/feedback` | API | Root of the feedback volume |
| `API_URL` | `http://api:8000` | UI | API base URL |
| `DOCTOR_PASSWORD` | `doctor123` | UI | Password for doctor tabs — change in production |

Set overrides in `serving/.env`:

```
DOCTOR_PASSWORD=your_secure_password
```

---

## Model checkpoint

The API expects a `.pt` file written by `src/mse_mlops/train.py` (`best_model.pt`):

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

---

## Future integration points

Two no-op stubs in `serving/api/model.py` are already wired into the request lifecycle:

```python
def log_prediction_to_mlflow(image_id, prediction): pass  # TODO
def notify_curation_pipeline(image_id, label):      pass  # TODO
```

When MLflow and the retraining pipeline are ready, fill these in — no other code changes needed.

---

## Further reading

| File | Contents |
|------|---------|
| [`serving/AGENT.md`](../serving/AGENT.md) | AI/developer guide: design decisions, extension patterns, pitfalls |
| [`serving/Architecture.md`](../serving/Architecture.md) | Detailed component diagram and all data flows |
| [`serving/api/README.md`](../serving/api/README.md) | API routes, model internals, how to add endpoints |
| [`serving/ui/README.md`](../serving/ui/README.md) | UI tabs, auth, how to extend |
