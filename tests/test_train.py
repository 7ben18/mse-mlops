from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from mse_mlops import train as train_module
from mse_mlops.train import TrainConfig, build_dataloaders, choose_indices


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


def write_rgb_image(path: Path, size: tuple[int, int] = (12, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_train_config_yaml(path: Path, payload: dict[str, object] | None = None) -> None:
    config_payload = payload or {
        "model": {"model_name": "models/pretrained/dinov3-vits16-pretrain-lvd1689m"},
        "data": {
            "metadata_csv": "data/processed/ham10000/metadata.csv",
            "images_dir": "data/processed/ham10000/HAM10000_images",
            "label_column": "mb",
            "train_set": "train",
            "val_set": "val",
            "train_fraction": 1.0,
            "val_fraction": 1.0,
            "train_samples": None,
            "val_samples": None,
        },
        "training": {
            "output_dir": "models/finetuned/dinov3_ham10000",
            "epochs": 10,
            "batch_size": 16,
            "image_size": 224,
            "lr": 0.00005,
            "weight_decay": 0.01,
            "num_workers": 0,
            "seed": 42,
            "device": "auto",
            "freeze_backbone": True,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0,
            "max_train_batches": None,
            "max_val_batches": None,
            "resume_from_checkpoint": None,
            "save_total_limit": 5,
            "load_best_model_at_end": True,
        },
        "tracking": {
            "mlflow_tracking_uri": "http://127.0.0.1:5000",
            "mlflow_experiment_name": "mse-mlops-training",
            "mlflow_run_name": None,
            "mlflow_tags": {"project": "mse-mlops"},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")


def patch_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mse_mlops.train.AutoImageProcessor.from_pretrained", lambda *_args, **_kwargs: DummyProcessor()
    )


def patch_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mse_mlops.train.AutoModel.from_pretrained", lambda *_args, **_kwargs: DummyBackbone())


def test_choose_indices_fraction():
    indices = list(range(100))
    result = choose_indices(indices, fraction=0.5, max_samples=None, seed=42)
    assert len(result) == 50


def test_choose_indices_max_samples():
    indices = list(range(100))
    result = choose_indices(indices, fraction=1.0, max_samples=10, seed=42)
    assert len(result) == 10


def test_load_train_config_uses_repo_default_when_config_path_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config" / "train.yaml"
    write_train_config_yaml(config_path)
    monkeypatch.setattr(train_module.paths, "resolve_train_config_path", lambda: config_path)

    config = train_module.load_train_config()

    assert config.config == config_path
    assert config.metadata_csv == Path("data/processed/ham10000/metadata.csv")
    assert config.images_dir == Path("data/processed/ham10000/HAM10000_images")
    assert config.output_dir == Path("models/finetuned/dinov3_ham10000")
    assert config.epochs == 10


def test_load_train_config_applies_explicit_overrides(tmp_path: Path):
    config_path = tmp_path / "config" / "train.yaml"
    write_train_config_yaml(config_path)

    config = train_module.load_train_config(
        config_path=config_path,
        overrides={
            "epochs": 3,
            "device": "cpu",
            "output_dir": Path("models/finetuned/debug"),
        },
    )

    assert config.config == config_path
    assert config.epochs == 3
    assert config.device == "cpu"
    assert config.output_dir == Path("models/finetuned/debug")


def test_load_train_config_rejects_non_mapping_section(tmp_path: Path):
    config_path = tmp_path / "config" / "train.yaml"
    write_train_config_yaml(config_path, payload={"training": ["invalid"]})

    with pytest.raises(ValueError, match="Config section 'training' must be a YAML mapping"):
        train_module.load_train_config(config_path=config_path)


def test_load_train_config_rejects_missing_required_setting(tmp_path: Path):
    config_path = tmp_path / "config" / "train.yaml"
    payload = {
        "model": {"model_name": "models/pretrained/dinov3-vits16-pretrain-lvd1689m"},
        "data": {
            "images_dir": "data/processed/ham10000/HAM10000_images",
            "label_column": "mb",
            "train_set": "train",
            "val_set": "val",
            "train_fraction": 1.0,
            "val_fraction": 1.0,
            "train_samples": None,
            "val_samples": None,
        },
        "training": {
            "output_dir": "models/finetuned/dinov3_ham10000",
            "epochs": 10,
            "batch_size": 16,
            "image_size": 224,
            "lr": 0.00005,
            "weight_decay": 0.01,
            "num_workers": 0,
            "seed": 42,
            "device": "auto",
            "freeze_backbone": True,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0,
            "max_train_batches": None,
            "max_val_batches": None,
            "resume_from_checkpoint": None,
            "save_total_limit": 5,
            "load_best_model_at_end": True,
        },
        "tracking": {
            "mlflow_tracking_uri": "http://127.0.0.1:5000",
            "mlflow_experiment_name": "mse-mlops-training",
            "mlflow_run_name": None,
            "mlflow_tags": {"project": "mse-mlops"},
        },
    }
    write_train_config_yaml(config_path, payload=payload)

    with pytest.raises(ValueError, match="metadata_csv"):
        train_module.load_train_config(config_path=config_path)


def test_build_dataloaders_reads_ham10000_metadata_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata_csv = tmp_path / "data" / "processed" / "ham10000" / "metadata.csv"
    images_dir = tmp_path / "data" / "processed" / "ham10000" / "HAM10000_images"

    write_metadata_csv(
        metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "mb": "benign", "set": "train"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0002", "mb": "malignant", "set": "train"},
            {"lesion_id": "HAM_0003", "image_id": "ISIC_0003", "mb": "benign", "set": "val"},
            {"lesion_id": "HAM_0004", "image_id": "ISIC_0004", "mb": "malignant", "set": "val"},
            {"lesion_id": "HAM_0005", "image_id": "ISIC_0005", "mb": "benign", "set": "test"},
            {"lesion_id": "HAM_0006", "image_id": "ISIC_0006", "mb": "malignant", "set": "future"},
        ],
    )

    for split_name, image_id in [
        ("train", "ISIC_0001"),
        ("train", "ISIC_0002"),
        ("val", "ISIC_0003"),
        ("val", "ISIC_0004"),
        ("test", "ISIC_0005"),
        ("future", "ISIC_0006"),
    ]:
        write_rgb_image(images_dir / split_name / f"{image_id}.jpg")

    patch_processor(monkeypatch)

    train_loader, val_loader, class_names, train_count, val_count = build_dataloaders(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        batch_size=2,
        num_workers=0,
        image_size=16,
        model_name="models/pretrained/dummy",
        device=torch.device("cpu"),
        seed=13,
        label_column="mb",
        train_set="train",
        val_set="val",
        train_fraction=1.0,
        val_fraction=1.0,
        train_samples=None,
        val_samples=None,
    )

    assert class_names == ["benign", "malignant"]
    assert train_count == 2
    assert val_count == 2
    assert all(record.image_path.parent.name == "train" for record in train_loader.dataset.records)
    assert all(record.image_path.parent.name == "val" for record in val_loader.dataset.records)
    assert {record.label_index for record in train_loader.dataset.records} == {0, 1}


def test_build_dataloaders_rejects_missing_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata_csv = tmp_path / "metadata.csv"
    images_dir = tmp_path / "HAM10000_images"

    write_metadata_csv(
        metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "mb": "benign", "set": "train"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0002", "mb": "malignant", "set": "val"},
        ],
    )
    write_rgb_image(images_dir / "val" / "ISIC_0002.jpg")

    patch_processor(monkeypatch)

    with pytest.raises(FileNotFoundError, match="Image not found for split 'train'"):
        build_dataloaders(
            metadata_csv=metadata_csv,
            images_dir=images_dir,
            batch_size=2,
            num_workers=0,
            image_size=16,
            model_name="models/pretrained/dummy",
            device=torch.device("cpu"),
            seed=13,
            label_column="mb",
            train_set="train",
            val_set="val",
            train_fraction=1.0,
            val_fraction=1.0,
            train_samples=None,
            val_samples=None,
        )


def test_build_dataloaders_rejects_invalid_mb_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata_csv = tmp_path / "metadata.csv"
    images_dir = tmp_path / "HAM10000_images"

    write_metadata_csv(
        metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "mb": "benign", "set": "train"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0002", "mb": "unknown", "set": "val"},
        ],
    )
    write_rgb_image(images_dir / "train" / "ISIC_0001.jpg")
    write_rgb_image(images_dir / "val" / "ISIC_0002.jpg")

    patch_processor(monkeypatch)

    with pytest.raises(ValueError, match=r"unsupported 'mb' labels"):
        build_dataloaders(
            metadata_csv=metadata_csv,
            images_dir=images_dir,
            batch_size=2,
            num_workers=0,
            image_size=16,
            model_name="models/pretrained/dummy",
            device=torch.device("cpu"),
            seed=13,
            label_column="mb",
            train_set="train",
            val_set="val",
            train_fraction=1.0,
            val_fraction=1.0,
            train_samples=None,
            val_samples=None,
        )


def test_build_dataloaders_rejects_invalid_split_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata_csv = tmp_path / "metadata.csv"
    images_dir = tmp_path / "HAM10000_images"

    write_metadata_csv(
        metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "mb": "benign", "set": "train"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0002", "mb": "malignant", "set": "holdout"},
        ],
    )
    write_rgb_image(images_dir / "train" / "ISIC_0001.jpg")

    patch_processor(monkeypatch)

    with pytest.raises(ValueError, match="unsupported split values"):
        build_dataloaders(
            metadata_csv=metadata_csv,
            images_dir=images_dir,
            batch_size=2,
            num_workers=0,
            image_size=16,
            model_name="models/pretrained/dummy",
            device=torch.device("cpu"),
            seed=13,
            label_column="mb",
            train_set="train",
            val_set="val",
            train_fraction=1.0,
            val_fraction=1.0,
            train_samples=None,
            val_samples=None,
        )


def test_build_dataloaders_rejects_inconsistent_lesion_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata_csv = tmp_path / "metadata.csv"
    images_dir = tmp_path / "HAM10000_images"

    write_metadata_csv(
        metadata_csv,
        [
            {"lesion_id": "HAM_0000118", "image_id": "ISIC_0001", "mb": "benign", "set": "train"},
            {"lesion_id": "HAM_0000118", "image_id": "ISIC_0002", "mb": "benign", "set": "val"},
            {"lesion_id": "HAM_0007409", "image_id": "ISIC_0003", "mb": "malignant", "set": "train"},
        ],
    )
    write_rgb_image(images_dir / "train" / "ISIC_0001.jpg")
    write_rgb_image(images_dir / "val" / "ISIC_0002.jpg")
    write_rgb_image(images_dir / "train" / "ISIC_0003.jpg")

    patch_processor(monkeypatch)

    with pytest.raises(ValueError, match="Each lesion_id must belong to exactly one set"):
        build_dataloaders(
            metadata_csv=metadata_csv,
            images_dir=images_dir,
            batch_size=2,
            num_workers=0,
            image_size=16,
            model_name="models/pretrained/dummy",
            device=torch.device("cpu"),
            seed=13,
            label_column="mb",
            train_set="train",
            val_set="val",
            train_fraction=1.0,
            val_fraction=1.0,
            train_samples=None,
            val_samples=None,
        )


def test_run_training_and_mlflow_hooks_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata_csv = tmp_path / "data" / "processed" / "ham10000" / "metadata.csv"
    images_dir = tmp_path / "data" / "processed" / "ham10000" / "HAM10000_images"
    output_dir = tmp_path / "models" / "finetuned" / "dinov3_ham10000"

    write_metadata_csv(
        metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "mb": "benign", "set": "train"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0002", "mb": "malignant", "set": "train"},
            {"lesion_id": "HAM_0003", "image_id": "ISIC_0003", "mb": "benign", "set": "val"},
            {"lesion_id": "HAM_0004", "image_id": "ISIC_0004", "mb": "malignant", "set": "val"},
        ],
    )
    for split_name, image_id in [
        ("train", "ISIC_0001"),
        ("train", "ISIC_0002"),
        ("val", "ISIC_0003"),
        ("val", "ISIC_0004"),
    ]:
        write_rgb_image(images_dir / split_name / f"{image_id}.jpg", size=(16, 16))

    patch_processor(monkeypatch)
    patch_backbone(monkeypatch)

    config = TrainConfig(
        config=Path("config/train.yaml"),
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        label_column="mb",
        train_set="train",
        val_set="val",
        train_fraction=1.0,
        val_fraction=1.0,
        train_samples=None,
        val_samples=None,
        model_name="models/pretrained/dummy",
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        image_size=16,
        lr=1e-3,
        weight_decay=0.0,
        num_workers=0,
        seed=13,
        device="cpu",
        gradient_accumulation_steps=1,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        max_train_batches=None,
        max_val_batches=None,
        resume_from_checkpoint=None,
        save_total_limit=2,
        freeze_backbone=True,
        load_best_model_at_end=True,
        mlflow_tracking_uri="file:///tmp/mlruns",
        mlflow_experiment_name="mse-mlops-training",
        mlflow_run_name=None,
        mlflow_tags="{}",
    )

    calls: dict[str, object] = {
        "mlflow_entered": False,
        "run_params": None,
        "epoch_metrics": [],
        "summary_metrics": None,
        "final_artifacts": None,
    }

    @contextmanager
    def fake_init_mlflow(_config: object):
        calls["mlflow_entered"] = True
        yield

    def fake_log_run_params(*, config: object, train_count: int, val_count: int, class_names: list[str]) -> None:
        calls["run_params"] = {
            "config": config,
            "train_count": train_count,
            "val_count": val_count,
            "class_names": class_names,
        }

    def fake_log_epoch_metrics(*, metrics: dict[str, float], epoch: int, optimizer_steps: int) -> None:
        calls["epoch_metrics"].append({
            "metrics": metrics,
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
        })

    def fake_log_summary_metrics(metrics: dict[str, float]) -> None:
        calls["summary_metrics"] = metrics

    def fake_log_final_artifacts(*, best_model: nn.Module | None, history_payload: list[dict[str, object]]) -> None:
        calls["final_artifacts"] = {
            "best_model": best_model,
            "history_payload": history_payload,
        }

    monkeypatch.setattr(train_module.tracking, "init_mlflow", fake_init_mlflow)
    monkeypatch.setattr(train_module.tracking, "log_run_params", fake_log_run_params)
    monkeypatch.setattr(train_module.tracking, "log_epoch_metrics", fake_log_epoch_metrics)
    monkeypatch.setattr(train_module.tracking, "log_summary_metrics", fake_log_summary_metrics)
    monkeypatch.setattr(train_module.tracking, "log_final_artifacts", fake_log_final_artifacts)

    train_module.run_training(config)

    checkpoint_path = output_dir / "checkpoints" / "epoch_001.pt"
    assert calls["mlflow_entered"] is True
    assert calls["run_params"] == {
        "config": config,
        "train_count": 2,
        "val_count": 2,
        "class_names": ["benign", "malignant"],
    }
    assert len(calls["epoch_metrics"]) == 1
    assert calls["epoch_metrics"][0]["epoch"] == 1
    assert calls["summary_metrics"] is not None
    assert calls["summary_metrics"]["best_epoch"] == 1.0
    assert calls["final_artifacts"] is not None
    assert calls["final_artifacts"]["best_model"] is not None
    assert len(calls["final_artifacts"]["history_payload"]) == 1
    assert checkpoint_path.is_file()
