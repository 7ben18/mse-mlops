from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
from torch import nn


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
def init_mlflow(config: object):
    tracking_uri = str(getattr(config, "mlflow_tracking_uri")).strip()
    experiment_name = str(getattr(config, "mlflow_experiment_name")).strip()

    if not tracking_uri:
        raise ValueError("--mlflow-tracking-uri must be set.")
    if not experiment_name:
        raise ValueError("--mlflow-experiment-name must be set.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_name_value = getattr(config, "mlflow_run_name")
    run_name = run_name_value.strip() if isinstance(run_name_value, str) else None
    tags = _coerce_tags(getattr(config, "mlflow_tags"))

    with mlflow.start_run(run_name=run_name or None, tags=tags):
        yield


def log_run_params(config: object, train_count: int, val_count: int, class_names: list[str]) -> None:
    payload = {
        "config": getattr(config, "config"),
        "metadata_csv": getattr(config, "metadata_csv"),
        "images_dir": getattr(config, "images_dir"),
        "label_column": getattr(config, "label_column"),
        "train_set": getattr(config, "train_set"),
        "val_set": getattr(config, "val_set"),
        "train_fraction": getattr(config, "train_fraction"),
        "val_fraction": getattr(config, "val_fraction"),
        "train_samples": getattr(config, "train_samples"),
        "val_samples": getattr(config, "val_samples"),
        "model_name": getattr(config, "model_name"),
        "epochs": getattr(config, "epochs"),
        "batch_size": getattr(config, "batch_size"),
        "image_size": getattr(config, "image_size"),
        "lr": getattr(config, "lr"),
        "weight_decay": getattr(config, "weight_decay"),
        "num_workers": getattr(config, "num_workers"),
        "seed": getattr(config, "seed"),
        "device": getattr(config, "device"),
        "gradient_accumulation_steps": getattr(config, "gradient_accumulation_steps"),
        "warmup_ratio": getattr(config, "warmup_ratio"),
        "lr_scheduler_type": getattr(config, "lr_scheduler_type"),
        "max_grad_norm": getattr(config, "max_grad_norm"),
        "max_train_batches": getattr(config, "max_train_batches"),
        "max_val_batches": getattr(config, "max_val_batches"),
        "resume_from_checkpoint": getattr(config, "resume_from_checkpoint"),
        "save_total_limit": getattr(config, "save_total_limit"),
        "freeze_backbone": getattr(config, "freeze_backbone"),
        "load_best_model_at_end": getattr(config, "load_best_model_at_end"),
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
                artifact_path="model",
            )
