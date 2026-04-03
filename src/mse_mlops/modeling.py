from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel

HIDDEN_SIZE_ERROR = "Could not infer hidden size from model config."
OUTPUT_FORMAT_ERROR = "Backbone output format is not supported."


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_mean_std(
    processor: AutoImageProcessor,
) -> tuple[Sequence[float], Sequence[float]]:
    mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
    return mean, std


def load_processor_mean_std(
    model_name: str,
) -> tuple[Sequence[float], Sequence[float]]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    return resolve_mean_std(processor)


class DinoV3Classifier(nn.Module):
    """Primary classifier wrapper for DINOv3 backbone checkpoints."""

    def __init__(
        self, model_name: str, num_labels: int, freeze_backbone: bool
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None and hasattr(
            self.backbone.config, "hidden_sizes"
        ):
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
        if (
            hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, tuple) and outputs:
            pooled = outputs[0][:, 0]
        else:
            raise ValueError(OUTPUT_FORMAT_ERROR)

        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)


__all__ = [
    "HIDDEN_SIZE_ERROR",
    "OUTPUT_FORMAT_ERROR",
    "DinoV3Classifier",
    "is_mps_available",
    "load_processor_mean_std",
    "resolve_mean_std",
]
