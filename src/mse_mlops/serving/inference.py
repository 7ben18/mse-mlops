from __future__ import annotations

import io
import os
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/finetuned/dinov3_ham10000/best_model.pt")
HIDDEN_SIZE_ERROR = "Could not infer hidden size from model config."
OUTPUT_FORMAT_ERROR = "Backbone output format is not supported."
MODEL_NOT_LOADED_ERROR = "Model not loaded. Call load_model() first."

_model: DinoV3Classifier | None = None
_class_names: list[str] = []
_transform: transforms.Compose | None = None
_device: torch.device = torch.device("cpu")


class DinoV3Classifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_backbone: bool) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.backbone.config, "hidden_sizes"):
            hidden_size = self.backbone.config.hidden_sizes[-1]
        if hidden_size is None:
            raise ValueError(HIDDEN_SIZE_ERROR)

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True) -> DinoV3Classifier:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, tuple) and outputs:
            pooled = outputs[0][:, 0]
        else:
            raise ValueError(OUTPUT_FORMAT_ERROR)

        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


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

    processor = AutoImageProcessor.from_pretrained(model_name)
    mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]

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
