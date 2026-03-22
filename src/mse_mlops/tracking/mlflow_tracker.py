from __future__ import annotations

import json
import tempfile
from argparse import Namespace
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
def init_mlflow(args: Namespace):
    tracking_uri = str(args.mlflow_tracking_uri).strip()
    experiment_name = str(args.mlflow_experiment_name).strip()

    if not tracking_uri:
        raise ValueError("--mlflow-tracking-uri must be set.")
    if not experiment_name:
        raise ValueError("--mlflow-experiment-name must be set.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_name = args.mlflow_run_name.strip() if isinstance(args.mlflow_run_name, str) else None
    tags = _coerce_tags(args.mlflow_tags)

    with mlflow.start_run(run_name=run_name or None, tags=tags):
        yield


def log_run_params(args: Namespace, train_count: int, val_count: int, class_names: list[str]) -> None:
    payload = {
        "config": args.config,
        "data_dir": args.data_dir,
        "train_subdir": args.train_subdir,
        "val_subdir": args.val_subdir,
        "val_mode": args.val_mode,
        "val_split": args.val_split,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "save_total_limit": args.save_total_limit,
        "freeze_backbone": args.freeze_backbone,
        "load_best_model_at_end": args.load_best_model_at_end,
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
