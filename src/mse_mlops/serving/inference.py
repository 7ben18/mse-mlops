from __future__ import annotations

import io
import os

import torch
from PIL import Image
from torchvision import transforms

from mse_mlops.modeling import (
    DinoV3Classifier,
    is_mps_available,
    load_processor_mean_std,
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "models/finetuned/dinov3_ham10000/best_model.pt"
)
MODEL_NOT_LOADED_ERROR = "Model not loaded. Call load_model() first."

_model: DinoV3Classifier | None = None
_class_names: list[str] = []
_transform: transforms.Compose | None = None
_device: torch.device = torch.device("cpu")


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str = MODEL_PATH) -> None:
    global _model, _class_names, _transform, _device

    _device = resolve_device()
    checkpoint = torch.load(path, map_location=_device)

    model_name: str = checkpoint["model_name"]
    image_size: int = int(checkpoint["image_size"])
    freeze_backbone: bool = bool(checkpoint["freeze_backbone"])
    _class_names = list(checkpoint["class_names"])

    _model = DinoV3Classifier(
        model_name=model_name,
        num_labels=len(_class_names),
        freeze_backbone=freeze_backbone,
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()

    mean, std = load_processor_mean_std(model_name)
    resize_size = max(image_size, int(image_size * 256 / 224))
    _transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def predict(image_bytes: bytes) -> dict[str, object]:
    if _model is None or _transform is None:
        raise RuntimeError(MODEL_NOT_LOADED_ERROR)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(pixel_values=tensor).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    return {
        "class": _class_names[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {name: round(prob, 4) for name, prob in zip(_class_names, probs, strict=False)},
    }


def log_prediction_to_mlflow(image_id: str, prediction: dict[str, object]) -> None:
    _ = image_id, prediction


def notify_curation_pipeline(image_id: str, label: str) -> None:
    _ = image_id, label
