from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from torch import nn

import mlflow


class TrainTrackingConfig(Protocol):
    config: Path
    metadata_csv: Path
    images_dir: Path
    label_column: str
    train_set: str
    val_set: str
    train_fraction: float
    val_fraction: float
    train_samples: int | None
    val_samples: int | None
    model_name: str
    epochs: int
    batch_size: int
    image_size: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    gradient_accumulation_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    max_train_batches: int | None
    max_val_batches: int | None
    resume_from_checkpoint: str | None
    save_total_limit: int | None
    freeze_backbone: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_run_name: str | None
    mlflow_tags: str | dict[str, str]


def _coerce_tags(raw_tags: object) -> dict[str, str]:
    if raw_tags is None:
        return {}
    if isinstance(raw_tags, Mapping):
        return {str(key): str(value) for key, value in raw_tags.items()}
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.strip()
        if not raw_tags:
            return {}
        parsed = json.loads(raw_tags)
        if not isinstance(parsed, dict):
            raise ValueError("--mlflow-tags must be a JSON object.")
        return {str(key): str(value) for key, value in parsed.items()}
    raise ValueError("--mlflow-tags must be a JSON object string or mapping.")


def _sanitize_params(payload: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        elif isinstance(value, (dict, list, tuple)):
            sanitized[key] = json.dumps(value, sort_keys=True)
        else:
            sanitized[key] = value
    return sanitized


@contextmanager
def init_mlflow(config: TrainTrackingConfig):
    tracking_uri = str(config.mlflow_tracking_uri).strip()
    experiment_name = str(config.mlflow_experiment_name).strip()

    if not tracking_uri:
        raise ValueError("--mlflow-tracking-uri must be set.")
    if not experiment_name:
        raise ValueError("--mlflow-experiment-name must be set.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_name_value = config.mlflow_run_name
    run_name = run_name_value.strip() if isinstance(run_name_value, str) else None
    tags = _coerce_tags(config.mlflow_tags)

    with mlflow.start_run(run_name=run_name or None, tags=tags):
        yield


def log_run_params(
    config: TrainTrackingConfig,
    train_count: int,
    val_count: int,
    class_names: list[str],
) -> None:
    payload = {
        "config": config.config,
        "metadata_csv": config.metadata_csv,
        "images_dir": config.images_dir,
        "label_column": config.label_column,
        "train_set": config.train_set,
        "val_set": config.val_set,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "train_samples": config.train_samples,
        "val_samples": config.val_samples,
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "device": config.device,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "max_train_batches": config.max_train_batches,
        "max_val_batches": config.max_val_batches,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "save_total_limit": config.save_total_limit,
        "freeze_backbone": config.freeze_backbone,
        "train_count": train_count,
        "val_count": val_count,
        "class_names": class_names,
    }
    mlflow.log_params(_sanitize_params(payload))


def log_epoch_metrics(metrics: dict[str, float], epoch: int, optimizer_steps: int) -> None:
    metric_payload = dict(metrics)
    metric_payload["optimizer_steps"] = float(optimizer_steps)
    mlflow.log_metrics(metric_payload, step=epoch)


def log_summary_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_final_artifacts(
    best_model: nn.Module | None,
    history_payload: list[dict[str, object]],
) -> None:
    with tempfile.TemporaryDirectory(prefix="mse-mlops-mlflow-") as tmp_dir:
        tmp_root = Path(tmp_dir)

        history_path = tmp_root / "history.json"
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history_payload, file, indent=2)
        mlflow.log_artifact(str(history_path), artifact_path="training")

        if best_model is not None:
            mlflow.pytorch.log_model(
                pytorch_model=best_model,
                name="model",
            )
