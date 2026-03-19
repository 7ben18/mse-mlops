from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mse_mlops.serving.feedback_store import (
    append_feedback_entry,
    load_feedback_entries,
    write_feedback_entries,
)
from mse_mlops.serving.inference import (
    load_model,
    log_prediction_to_mlflow,
    notify_curation_pipeline,
    predict,
)

FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "/feedback"))
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
IMAGES_DIR = FEEDBACK_DIR / "images"
UPLOAD_FILE = File(...)
UPLOAD_LABEL = Form(...)

app = FastAPI(title="Melanoma Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    image_id: str
    label: str
    source: str = "label_existing"


@app.on_event("startup")
def startup() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = UPLOAD_FILE) -> dict[str, Any]:
    image_bytes = await file.read()
    result = predict(image_bytes)
    image_id = str(uuid.uuid4())

    entry = {
        "image_id": image_id,
        "filename": file.filename,
        "timestamp": datetime.now(UTC).isoformat(),
        "prediction": result["class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "label": None,
        "source": "predict",
    }
    append_feedback_entry(FEEDBACK_FILE, entry)
    log_prediction_to_mlflow(image_id, result)

    return {**result, "image_id": image_id}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict[str, str]:
    entries = load_feedback_entries(FEEDBACK_FILE)
    updated = False
    new_entries = []

    for entry in entries:
        if entry.get("image_id") == req.image_id:
            entry["label"] = req.label
            entry["source"] = req.source
            updated = True
        new_entries.append(entry)

    if not updated:
        new_entries.append({
            "image_id": req.image_id,
            "filename": None,
            "timestamp": datetime.now(UTC).isoformat(),
            "prediction": None,
            "confidence": None,
            "probabilities": None,
            "label": req.label,
            "source": req.source,
        })

    write_feedback_entries(FEEDBACK_FILE, new_entries)
    notify_curation_pipeline(req.image_id, req.label)
    return {"status": "ok", "image_id": req.image_id}


@app.get("/feedback")
def get_feedback() -> list[dict[str, Any]]:
    return load_feedback_entries(FEEDBACK_FILE)


@app.post("/upload-labeled")
async def upload_labeled(
    file: UploadFile = UPLOAD_FILE,
    label: str = UPLOAD_LABEL,
) -> dict[str, str]:
    image_bytes = await file.read()
    image_id = str(uuid.uuid4())

    ext = Path(file.filename).suffix if file.filename else ".jpg"
    image_path = IMAGES_DIR / f"{image_id}{ext}"
    image_path.write_bytes(image_bytes)

    entry = {
        "image_id": image_id,
        "filename": file.filename,
        "timestamp": datetime.now(UTC).isoformat(),
        "prediction": None,
        "confidence": None,
        "probabilities": None,
        "label": label,
        "source": "upload_labeled",
    }
    append_feedback_entry(FEEDBACK_FILE, entry)
    notify_curation_pipeline(image_id, label)

    return {"status": "ok", "image_id": image_id, "label": label}
