# Serving Layer — Architecture

## Overview

The serving layer consists of two containers (FastAPI API + Streamlit UI) sharing a single
named Docker volume for the feedback store. It is fully self-contained and can be started with
`docker compose up` from the `serving/` directory.

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                             │
│  ┌──────────────────┐          ┌──────────────────────────┐ │
│  │   ui (Streamlit) │  HTTP    │     api (FastAPI)        │ │
│  │   port 7777      │ ──────►  │     port 8000            │ │
│  └──────────────────┘          │                          │ │
│                                │  ┌────────────────────┐  │ │
│  ../models/ (:ro) ─────────────┼─►│   model.py         │  │ │
│                                │  │   DinoV3Classifier  │  │ │
│                                │  └────────────────────┘  │ │
│  feedback_data (volume) ───────┼─► /feedback/             │ │
│                                │   ├── feedback.jsonl     │ │
│                                │   └── images/            │ │
│                                └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         ▲                                   │
    localhost:7777                    localhost:8000
    (browser / patient)               (curl / monitoring)
```

---

## Components

### `api` — FastAPI + Uvicorn

| File | Responsibility |
|------|---------------|
| `model.py` | Model loading, preprocessing, inference, stub hooks |
| `main.py` | Route definitions, feedback store I/O, CORS |

The model is loaded once at startup (`@app.on_event("startup")`) and held in module-level
globals. All inference calls are single-image synchronous operations on the CPU (or GPU if
available).

### `ui` — Streamlit

| Tab | Audience | Auth required |
|-----|----------|--------------|
| Skin Analysis | Patients / public | No |
| Review & Label | Doctors | Yes (password) |
| Bulk Dataset Upload | Doctors | Yes (password) |

The UI is stateless — it holds no data itself. All reads and writes go through the API.
Doctor authentication is enforced client-side in session state (`st.session_state.doctor_auth`).

---

## Data flows

### Flow 1 — Patient prediction

```
Browser → POST /predict (multipart image)
              │
              ▼
         model.py::predict()
              │  preprocessing: Resize → CenterCrop → Normalize
              │  inference:     DinoV3Classifier forward pass
              │  output:        {class, confidence, probabilities}
              │
              ▼
         _save_feedback()  →  /feedback/feedback.jsonl
         (label=null, source="predict")
              │
              ▼
         log_prediction_to_mlflow()  [no-op stub]
              │
              ▼
         JSON response  →  Streamlit Tab 1
```

### Flow 2 — Doctor labels a prediction

```
Streamlit Tab 2 → GET /feedback
                      │
                      ▼
                 _load_feedback()  ←  /feedback/feedback.jsonl
                 (filters label=null, prediction≠null)
                      │
                      ▼ (doctor selects label)

Streamlit Tab 2 → POST /feedback  {image_id, label, source}
                      │
                      ▼
                 Update matching entry in JSONL
                 notify_curation_pipeline()  [no-op stub]
```

### Flow 3 — Doctor bulk uploads labeled dataset

```
Streamlit Tab 3 → ZIP + CSV/Excel
                      │
                      ▼
                 Parse CSV/Excel → {filename: label} map
                 Iterate images in ZIP
                      │  for each matched image:
                      ▼
                 POST /upload-labeled  (multipart image + label)
                      │
                      ▼
                 Save image → /feedback/images/{uuid}{ext}
                 _save_feedback()  →  /feedback/feedback.jsonl
                 (prediction=null, source="upload_labeled")
                 notify_curation_pipeline()  [no-op stub]
```

---

## API reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | — | Liveness check |
| `POST` | `/predict` | — | Infer from uploaded image, log to feedback store |
| `GET` | `/feedback` | — | Return all feedback entries as JSON array |
| `POST` | `/feedback` | — | Update or create a feedback entry (assign label) |
| `POST` | `/upload-labeled` | — | Save image + label directly to feedback store |

---

## Feedback store schema

Each line of `/feedback/feedback.jsonl` is a JSON object:

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

`label` is `null` until a doctor assigns a verified ground truth.
`source` is one of `predict`, `doctor_review`, or `upload_labeled`.

---

## Plug-in points for future integration

Two stub functions in `model.py` are called on every prediction / label event but currently
do nothing. Fill them in to connect to MLflow and the retraining pipeline:

```python
# model.py
def log_prediction_to_mlflow(image_id: str, prediction: dict) -> None:
    pass  # TODO: mlflow.log_metric / mlflow.log_artifact

def notify_curation_pipeline(image_id: str, label: str) -> None:
    pass  # TODO: trigger DVC repro / Airflow DAG / webhook
```

The feedback store backend (`_save_feedback` / `_load_feedback` in `main.py`) is similarly
isolated — swap the JSONL implementation for a database or MLflow artifact store by editing
only those two functions.

---

## Configuration

All runtime configuration is passed via environment variables:

| Variable | Default | Container | Description |
|----------|---------|-----------|-------------|
| `MODEL_PATH` | `/models/best_model.pt` | `api` | Path to the trained `.pt` checkpoint |
| `FEEDBACK_DIR` | `/feedback` | `api` | Root of the feedback store volume |
| `API_URL` | `http://api:8000` | `ui` | Base URL the UI uses to reach the API |
| `DOCTOR_PASSWORD` | `doctor123` | `ui` | Password for doctor-only tabs |

Set `DOCTOR_PASSWORD` in a `.env` file next to `compose.yml` for production:

```
DOCTOR_PASSWORD=your_secure_password
```
