from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_model, log_prediction_to_mlflow, notify_curation_pipeline, predict

FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "/feedback"))
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
IMAGES_DIR = FEEDBACK_DIR / "images"

app = FastAPI(title="Melanoma Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_feedback(entry: dict) -> None:
    """Append a feedback entry to the JSONL store. Swap this for DB/MLflow later."""
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    entries = []
    with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    result = predict(image_bytes)
    image_id = str(uuid.uuid4())

    entry = {
        "image_id": image_id,
        "filename": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": result["class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "label": None,
        "source": "predict",
    }
    _save_feedback(entry)
    log_prediction_to_mlflow(image_id, result)

    return {**result, "image_id": image_id}


class FeedbackRequest(BaseModel):
    image_id: str
    label: str
    source: str = "label_existing"


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    entries = _load_feedback()
    updated = False
    new_entries = []

    for entry in entries:
        if entry.get("image_id") == req.image_id:
            entry["label"] = req.label
            entry["source"] = req.source
            updated = True
        new_entries.append(entry)

    if not updated:
        # image_id not found — create a new stub entry
        new_entries.append({
            "image_id": req.image_id,
            "filename": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": None,
            "confidence": None,
            "probabilities": None,
            "label": req.label,
            "source": req.source,
        })

    FEEDBACK_FILE.write_text(
        "\n".join(json.dumps(e) for e in new_entries) + "\n",
        encoding="utf-8",
    )
    notify_curation_pipeline(req.image_id, req.label)
    return {"status": "ok", "image_id": req.image_id}


@app.get("/feedback")
def get_feedback() -> list[dict]:
    return _load_feedback()


@app.post("/upload-labeled")
async def upload_labeled(
    file: UploadFile = File(...),
    label: str = Form(...),
) -> dict:
    image_bytes = await file.read()
    image_id = str(uuid.uuid4())

    ext = Path(file.filename).suffix if file.filename else ".jpg"
    image_path = IMAGES_DIR / f"{image_id}{ext}"
    image_path.write_bytes(image_bytes)

    entry = {
        "image_id": image_id,
        "filename": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": None,
        "confidence": None,
        "probabilities": None,
        "label": label,
        "source": "upload_labeled",
    }
    _save_feedback(entry)
    notify_curation_pipeline(image_id, label)

    return {"status": "ok", "image_id": image_id, "label": label}
