from __future__ import annotations

import json
from pathlib import Path

from mse_mlops.tracking import mlflow_tracker
from mse_mlops.train import TrainConfig


def build_config() -> TrainConfig:
    return TrainConfig(
        config=Path("config/train.yaml"),
        metadata_csv=Path("data/processed/ham10000/metadata.csv"),
        images_dir=Path("data/processed/ham10000/HAM10000_images"),
        label_column="mb",
        train_set="train",
        val_set="val",
        train_fraction=1.0,
        val_fraction=1.0,
        train_samples=None,
        val_samples=None,
        model_name="models/pretrained/dinov3-vits16-pretrain-lvd1689m",
        output_dir=Path("models/finetuned/dinov3_ham10000"),
        epochs=1,
        batch_size=4,
        image_size=224,
        lr=5e-5,
        weight_decay=0.01,
        num_workers=0,
        seed=42,
        device="cpu",
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        max_train_batches=None,
        max_val_batches=None,
        resume_from_checkpoint=None,
        save_total_limit=5,
        freeze_backbone=True,
        load_best_model_at_end=True,
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment_name="mse-mlops-training",
        mlflow_run_name=None,
        mlflow_tags="{}",
    )


def test_log_run_params_uses_ham10000_metadata_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_log_params(payload: dict[str, object]) -> None:
        captured.update(payload)

    monkeypatch.setattr(mlflow_tracker.mlflow, "log_params", fake_log_params)

    mlflow_tracker.log_run_params(
        config=build_config(),
        train_count=10,
        val_count=4,
        class_names=["benign", "malignant"],
    )

    assert captured["metadata_csv"] == "data/processed/ham10000/metadata.csv"
    assert captured["images_dir"] == "data/processed/ham10000/HAM10000_images"
    assert captured["train_set"] == "train"
    assert captured["val_set"] == "val"
    assert "data_dir" not in captured
    assert "val_mode" not in captured


def test_log_final_artifacts_is_independent_of_local_output_dir(monkeypatch):
    logged_artifacts: list[tuple[str, str]] = []

    def fake_log_artifact(path: str, artifact_path: str) -> None:
        logged_artifacts.append((path, artifact_path))
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload == [{"epoch": 1, "val_roc_auc": 0.9}]

    monkeypatch.setattr(mlflow_tracker.mlflow, "log_artifact", fake_log_artifact)

    mlflow_tracker.log_final_artifacts(
        best_model=None,
        history_payload=[{"epoch": 1, "val_roc_auc": 0.9}],
    )

    assert len(logged_artifacts) == 1
    assert logged_artifacts[0][1] == "training"
