from __future__ import annotations

import io
from types import SimpleNamespace
from typing import ClassVar

import torch
from PIL import Image
from torch import nn

from mse_mlops.serving import inference


class DummyProcessor:
    image_mean: ClassVar[tuple[float, float, float]] = (0.5, 0.5, 0.5)
    image_std: ClassVar[tuple[float, float, float]] = (0.5, 0.5, 0.5)


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.proj = nn.Linear(3, 4)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        pooled = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(pooler_output=self.proj(pooled))


def test_load_model_accepts_promoted_best_model_checkpoint(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "mse_mlops.modeling.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: DummyBackbone(),
    )
    monkeypatch.setattr(
        "mse_mlops.modeling.AutoImageProcessor.from_pretrained",
        lambda *_args, **_kwargs: DummyProcessor(),
    )
    monkeypatch.setattr(
        inference,
        "resolve_device",
        lambda: torch.device("cpu"),
    )

    model = inference.DinoV3Classifier(
        model_name="models/pretrained/dummy",
        num_labels=2,
        freeze_backbone=True,
    )
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "epoch": 3,
            "metric_name": "val_roc_auc",
            "metric_value": 0.91,
            "model_state_dict": model.state_dict(),
            "class_names": ["benign", "malignant"],
            "model_name": "models/pretrained/dummy",
            "image_size": 16,
            "freeze_backbone": True,
        },
        checkpoint_path,
    )

    inference.load_model(str(checkpoint_path))

    image_bytes = io.BytesIO()
    Image.new("RGB", (16, 16), color="white").save(image_bytes, format="JPEG")
    prediction = inference.predict(image_bytes.getvalue())

    assert prediction["class"] in {"benign", "malignant"}
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert set(prediction["probabilities"]) == {"benign", "malignant"}
