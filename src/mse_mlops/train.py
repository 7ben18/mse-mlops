from __future__ import annotations

import math
import random
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import get_scheduler

from mse_mlops import paths, tracking
from mse_mlops.modeling import (
    DinoV3Classifier,
    is_mps_available,
    load_processor_mean_std,
)

DEFAULT_CONFIG_PATH = paths.DEFAULT_CONFIG_PATH
CONFIG_SECTIONS = ("model", "data", "training", "tracking")
MB_LABEL_COLUMN = "mb"
IMAGE_EXTENSION = ".jpg"
BEST_MODEL_FILENAME = "best_model.pt"
REQUIRED_METADATA_COLUMNS = frozenset({"lesion_id", "image_id", "set"})
ALLOWED_METADATA_SETS = frozenset({"train", "val", "test", "future"})
ALLOWED_MB_LABELS = frozenset({"benign", "malignant"})

SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
)


@dataclass
class Metrics:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_roc_auc: float


@dataclass
class BestModelState:
    metric_name: str
    metric_value: float
    epoch: int
    model_state_dict: dict[str, torch.Tensor]
    class_names: list[str]
    model_name: str
    image_size: int
    freeze_backbone: bool


@dataclass
class TrainingRunResult:
    config: TrainConfig
    history: list[Metrics]
    best_model_state: BestModelState | None
    promoted_model_path: Path | None
    checkpoint_dir: Path
    train_count: int
    val_count: int

    @property
    def best_metric_name(self) -> str | None:
        if self.best_model_state is None:
            return None
        return self.best_model_state.metric_name

    @property
    def best_metric_value(self) -> float | None:
        if self.best_model_state is None:
            return None
        return self.best_model_state.metric_value


@dataclass
class TrainConfig:
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
    output_dir: Path
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
    config: Path


@dataclass(frozen=True)
class ImageRecord:
    label_index: int
    image_path: Path


class MetadataImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, records: list[ImageRecord], transform: transforms.Compose
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image), record.label_index


def _read_train_config_mapping(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: {config_path}"
        )

    return data


def _flatten_train_config_sections(
    config_data: dict[str, object],
) -> dict[str, object]:
    flattened: dict[str, object] = {}

    for section_name in CONFIG_SECTIONS:
        section_payload = config_data.get(section_name)
        if section_payload is None:
            continue
        if not isinstance(section_payload, dict):
            raise ValueError(
                f"Config section '{section_name}' must be a YAML mapping."
            )
        flattened.update(section_payload)

    for key, value in config_data.items():
        if key not in CONFIG_SECTIONS:
            flattened[key] = value

    return flattened


def _as_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _as_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config setting '{field_name}' must be a boolean.")


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def load_train_config(
    config_path: Path | str | None = None,
    overrides: dict[str, object] | None = None,
) -> TrainConfig:
    resolved_config_path = (
        paths.resolve_train_config_path()
        if config_path is None
        else Path(config_path)
    )
    return load_train_config_from_mapping(
        config_data=_read_train_config_mapping(resolved_config_path),
        config_path=resolved_config_path,
        overrides=overrides,
    )


def load_train_config_from_mapping(
    config_data: dict[str, object],
    *,
    config_path: Path | str,
    overrides: dict[str, object] | None = None,
) -> TrainConfig:
    resolved_config_path = Path(config_path)
    config_values = _flatten_train_config_sections(config_data)

    if overrides:
        if "config" in overrides:
            raise ValueError("Overrides must not contain 'config'.")
        for key, value in overrides.items():
            if value is not None:
                config_values[key] = value

    try:
        return TrainConfig(
            metadata_csv=_as_path(config_values["metadata_csv"]),
            images_dir=_as_path(config_values["images_dir"]),
            label_column=str(config_values["label_column"]),
            train_set=str(config_values["train_set"]),
            val_set=str(config_values["val_set"]),
            train_fraction=float(config_values["train_fraction"]),
            val_fraction=float(config_values["val_fraction"]),
            train_samples=_as_optional_int(config_values["train_samples"]),
            val_samples=_as_optional_int(config_values["val_samples"]),
            model_name=str(config_values["model_name"]),
            output_dir=_as_path(config_values["output_dir"]),
            epochs=int(config_values["epochs"]),
            batch_size=int(config_values["batch_size"]),
            image_size=int(config_values["image_size"]),
            lr=float(config_values["lr"]),
            weight_decay=float(config_values["weight_decay"]),
            num_workers=int(config_values["num_workers"]),
            seed=int(config_values["seed"]),
            device=str(config_values["device"]),
            gradient_accumulation_steps=int(
                config_values["gradient_accumulation_steps"]
            ),
            warmup_ratio=float(config_values["warmup_ratio"]),
            lr_scheduler_type=str(config_values["lr_scheduler_type"]),
            max_grad_norm=float(config_values["max_grad_norm"]),
            max_train_batches=_as_optional_int(
                config_values["max_train_batches"]
            ),
            max_val_batches=_as_optional_int(config_values["max_val_batches"]),
            resume_from_checkpoint=_as_optional_str(
                config_values["resume_from_checkpoint"]
            ),
            save_total_limit=_as_optional_int(
                config_values["save_total_limit"]
            ),
            freeze_backbone=_as_bool(
                config_values["freeze_backbone"], "freeze_backbone"
            ),
            mlflow_tracking_uri=str(config_values["mlflow_tracking_uri"]),
            mlflow_experiment_name=str(config_values["mlflow_experiment_name"]),
            mlflow_run_name=_as_optional_str(config_values["mlflow_run_name"]),
            mlflow_tags=config_values["mlflow_tags"],
            config=resolved_config_path,
        )
    except KeyError as error:
        missing_field = str(error.args[0])
        raise ValueError(
            f"Config file is missing required training setting: {missing_field}"
        ) from error


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Requested cuda but not available. Falling back to cpu.")
        return torch.device("cpu")

    if requested == "mps":
        if is_mps_available():
            return torch.device("mps")
        print("Requested mps but not available. Falling back to cpu.")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(
    mean: Sequence[float], std: Sequence[float], image_size: int
) -> tuple[transforms.Compose, transforms.Compose]:
    # HF-style augmentation for train/eval image classification pipelines.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    resize_size = max(image_size, int(image_size * 256 / 224))
    eval_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def choose_indices(
    base_indices: list[int],
    fraction: float,
    max_samples: int | None,
    seed: int,
) -> list[int]:
    indices = list(base_indices)
    random.Random(seed).shuffle(indices)

    keep = len(indices)
    if fraction < 1.0:
        keep = max(1, int(len(indices) * fraction))
    if max_samples is not None:
        keep = min(keep, max_samples)
    return indices[:keep]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_metadata_frame(metadata_csv: Path, label_column: str) -> pd.DataFrame:
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    metadata_df = pd.read_csv(metadata_csv)
    required_columns = REQUIRED_METADATA_COLUMNS | {label_column}
    missing_columns = sorted(required_columns - set(metadata_df.columns))
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing_str}"
        )
    if metadata_df.empty:
        raise ValueError(f"Metadata CSV is empty: {metadata_csv}")

    normalized_df = metadata_df.copy()
    for column in sorted(required_columns):
        if normalized_df[column].isna().any():
            raise ValueError(f"Metadata column contains null values: {column}")
        normalized_df[column] = normalized_df[column].astype(str).str.strip()
        if (normalized_df[column] == "").any():
            raise ValueError(f"Metadata column contains blank values: {column}")

    invalid_sets = sorted(set(normalized_df["set"]) - ALLOWED_METADATA_SETS)
    if invalid_sets:
        invalid_str = ", ".join(invalid_sets)
        raise ValueError(
            f"Metadata contains unsupported split values: {invalid_str}"
        )

    validate_lesion_split_consistency(normalized_df)
    return normalized_df


def validate_lesion_split_consistency(metadata_df: pd.DataFrame) -> None:
    lesion_split_counts = metadata_df.groupby("lesion_id")["set"].nunique()
    invalid = lesion_split_counts[lesion_split_counts != 1]
    if invalid.empty:
        return

    examples = ", ".join(
        f"{lesion_id}: {count}" for lesion_id, count in invalid.head(5).items()
    )
    raise ValueError(
        f"Each lesion_id must belong to exactly one set. Violations: {examples}"
    )


def build_class_names(
    metadata_df: pd.DataFrame, label_column: str
) -> list[str]:
    if label_column == MB_LABEL_COLUMN:
        invalid_labels = sorted(
            set(metadata_df[label_column]) - ALLOWED_MB_LABELS
        )
        if invalid_labels:
            invalid_str = ", ".join(invalid_labels)
            raise ValueError(
                f"Metadata contains unsupported '{label_column}' labels: {invalid_str}"
            )

    class_names = sorted(metadata_df[label_column].drop_duplicates().tolist())
    if len(class_names) < 2:
        raise ValueError(
            f"Need at least two unique labels in '{label_column}' for classification."
        )
    return class_names


def build_split_samples(
    metadata_df: pd.DataFrame,
    images_dir: Path,
    label_column: str,
    split_name: str,
    class_names: list[str],
    fraction: float,
    max_samples: int | None,
    seed: int,
) -> list[ImageRecord]:
    class_to_idx = {
        class_name: index for index, class_name in enumerate(class_names)
    }
    split_df = metadata_df[metadata_df["set"] == split_name].reset_index(
        drop=True
    )
    if split_df.empty:
        raise ValueError(f"No metadata rows found for split '{split_name}'.")

    selected_indices = choose_indices(
        list(range(len(split_df))), fraction, max_samples, seed
    )
    if not selected_indices:
        raise ValueError(
            f"Split '{split_name}' is empty after sampling settings."
        )

    records: list[ImageRecord] = []
    for row in split_df.iloc[selected_indices].itertuples(index=False):
        label_name = getattr(row, label_column)
        if label_name not in class_to_idx:
            raise ValueError(
                f"Unsupported label '{label_name}' in column '{label_column}'."
            )

        image_path = (
            images_dir / split_name / f"{row.image_id}{IMAGE_EXTENSION}"
        )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found for split '{split_name}': {image_path}"
            )

        records.append(
            ImageRecord(
                label_index=class_to_idx[label_name],
                image_path=image_path,
            )
        )

    return records


def compute_classification_metrics(
    labels: list[int],
    preds: list[int],
    probs: list[list[float]],
) -> tuple[float, float, float, float]:
    precision = precision_score(
        labels, preds, average="weighted", zero_division=0
    )
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        roc_auc = float("nan")
    else:
        try:
            if len(probs[0]) == 2:
                roc_auc = roc_auc_score(labels, [row[1] for row in probs])
            else:
                roc_auc = roc_auc_score(
                    labels,
                    probs,
                    multi_class="ovr",
                    average="weighted",
                    labels=list(range(len(probs[0]))),
                )
        except ValueError:
            roc_auc = float("nan")

    return float(precision), float(recall), float(f1), float(roc_auc)


def build_dataloaders(
    metadata_csv: Path,
    images_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    model_name: str,
    device: torch.device,
    seed: int,
    label_column: str,
    train_set: str,
    val_set: str,
    train_fraction: float,
    val_fraction: float,
    train_samples: int | None,
    val_samples: int | None,
) -> tuple[DataLoader, DataLoader, list[str], int, int]:
    if train_set == val_set:
        raise ValueError("--train-set and --val-set must be different.")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    metadata_df = read_metadata_frame(metadata_csv, label_column)
    class_names = build_class_names(metadata_df, label_column)
    mean, std = load_processor_mean_std(model_name)
    train_tf, eval_tf = build_transforms(mean, std, image_size=image_size)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    train_records = build_split_samples(
        metadata_df=metadata_df,
        images_dir=images_dir,
        label_column=label_column,
        split_name=train_set,
        class_names=class_names,
        fraction=train_fraction,
        max_samples=train_samples,
        seed=seed,
    )
    val_records = build_split_samples(
        metadata_df=metadata_df,
        images_dir=images_dir,
        label_column=label_column,
        split_name=val_set,
        class_names=class_names,
        fraction=val_fraction,
        max_samples=val_samples,
        seed=seed + 1,
    )

    train_ds = MetadataImageDataset(train_records, transform=train_tf)
    val_ds = MetadataImageDataset(val_records, transform=eval_tf)

    if len(train_ds) == 0:
        raise ValueError(
            "Train dataset is empty after applying split/fraction/sample settings."
        )
    if len(val_ds) == 0:
        raise ValueError(
            "Validation dataset is empty after applying split/fraction/sample settings."
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    return (
        train_loader,
        val_loader,
        class_names,
        len(train_records),
        len(val_records),
    )


def build_model(
    model_name: str, class_names: list[str], freeze_backbone: bool
) -> nn.Module:
    model = DinoV3Classifier(
        model_name=model_name,
        num_labels=len(class_names),
        freeze_backbone=freeze_backbone,
    )
    print("Model head mode: custom DINOv3 classifier")
    return model


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "layernorm.weight",
        "norm.weight",
        "norm.bias",
    )
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "weight_decay": weight_decay,
        })
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    return torch.optim.AdamW(param_groups, lr=lr)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, float, float, float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

            if max_batches is not None and batch_idx >= max_batches:
                break

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    precision, recall, f1, roc_auc = compute_classification_metrics(
        all_labels, all_preds, all_probs
    )
    return avg_loss, acc, precision, recall, f1, roc_auc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer_steps = 0

    planned_batches = len(loader)
    if max_batches is not None:
        planned_batches = min(planned_batches, max_batches)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(
        tqdm(loader, desc="train", leave=False), start=1
    ):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(pixel_values=images).logits
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        (loss / gradient_accumulation_steps).backward()

        should_step = (batch_idx % gradient_accumulation_steps == 0) or (
            batch_idx == planned_batches
        )
        if should_step:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if max_batches is not None and batch_idx >= max_batches:
            break

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, optimizer_steps


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def cleanup_old_checkpoints(
    checkpoint_dir: Path, keep_last: int | None
) -> None:
    if keep_last is None:
        return
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if len(checkpoints) <= keep_last:
        return
    for old_path in checkpoints[: len(checkpoints) - keep_last]:
        old_path.unlink(missing_ok=True)


def serialize_best_model_state(
    best_model_state: BestModelState,
) -> dict[str, object]:
    return {
        "metric_name": best_model_state.metric_name,
        "metric_value": best_model_state.metric_value,
        "epoch": best_model_state.epoch,
        "model_state_dict": best_model_state.model_state_dict,
        "class_names": best_model_state.class_names,
        "model_name": best_model_state.model_name,
        "image_size": best_model_state.image_size,
        "freeze_backbone": best_model_state.freeze_backbone,
    }


def deserialize_best_model_state(payload: dict[str, object]) -> BestModelState:
    return BestModelState(
        metric_name=str(payload["metric_name"]),
        metric_value=float(payload["metric_value"]),
        epoch=int(payload["epoch"]),
        model_state_dict=payload["model_state_dict"],
        class_names=list(payload["class_names"]),
        model_name=str(payload["model_name"]),
        image_size=int(payload["image_size"]),
        freeze_backbone=bool(payload["freeze_backbone"]),
    )


def promote_best_model_checkpoint(
    output_dir: Path,
    best_model_state: BestModelState | None,
) -> Path | None:
    if best_model_state is None:
        return None

    best_model_path = output_dir / BEST_MODEL_FILENAME
    torch.save(serialize_best_model_state(best_model_state), best_model_path)
    return best_model_path


def save_epoch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    best_model_state: BestModelState | None,
    history: list[Metrics],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": [asdict(item) for item in history],
    }
    if best_model_state is not None:
        payload["best_model_state"] = serialize_best_model_state(
            best_model_state
        )
    torch.save(payload, checkpoint_path)


def load_epoch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> tuple[int, BestModelState | None, list[Metrics]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    loaded_epoch = int(payload["epoch"])
    history_payload = payload.get("history", [])
    history: list[Metrics] = []
    if isinstance(history_payload, list):
        history = [Metrics(**item) for item in history_payload]
    best_payload = payload.get("best_model_state")
    best_model_state = None
    if isinstance(best_payload, dict):
        best_model_state = deserialize_best_model_state(best_payload)
    return loaded_epoch, best_model_state, history


def validate_config(config: TrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if config.image_size <= 0:
        raise ValueError("--image-size must be > 0.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0.")
    if config.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be >= 0.")
    if not (0.0 <= config.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1).")

    if not (0.0 < config.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in (0, 1].")
    if not (0.0 < config.val_fraction <= 1.0):
        raise ValueError("--val-fraction must be in (0, 1].")
    if config.train_set == config.val_set:
        raise ValueError("--train-set and --val-set must be different.")
    if config.train_samples is not None and config.train_samples <= 0:
        raise ValueError("--train-samples must be > 0 when provided.")
    if config.val_samples is not None and config.val_samples <= 0:
        raise ValueError("--val-samples must be > 0 when provided.")
    if config.max_train_batches is not None and config.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be > 0 when provided.")
    if config.max_val_batches is not None and config.max_val_batches <= 0:
        raise ValueError("--max-val-batches must be > 0 when provided.")
    if config.save_total_limit is not None and config.save_total_limit <= 0:
        raise ValueError("--save-total-limit must be > 0 when provided.")


def prepare_checkpoint_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def print_run_configuration(config: TrainConfig, device: torch.device) -> None:
    print(f"Config: {config.config}")
    print(f"Using device: {device.type}")
    print(f"Model: {config.model_name}")
    print(f"Metadata CSV: {config.metadata_csv}")
    print(f"Images dir: {config.images_dir}")
    print(
        f"Train split: {config.train_set} | "
        f"Validation split: {config.val_set} | "
        f"label_column={config.label_column} | "
        f"train_fraction={config.train_fraction} | "
        f"val_fraction={config.val_fraction}"
    )


def build_scheduler_for_training(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    epochs: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    lr_scheduler_type: str,
    max_train_batches: int | None,
) -> tuple[object, int, int, int]:
    train_batches_per_epoch = len(train_loader)
    if max_train_batches is not None:
        train_batches_per_epoch = min(
            train_batches_per_epoch, max_train_batches
        )

    updates_per_epoch = max(
        1, math.ceil(train_batches_per_epoch / gradient_accumulation_steps)
    )
    total_training_steps = updates_per_epoch * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    return scheduler, updates_per_epoch, total_training_steps, warmup_steps


def resolve_resume_checkpoint(
    checkpoint_dir: Path, resume_from_checkpoint: str | None
) -> Path | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"--resume-from-checkpoint=latest but no checkpoint found in {checkpoint_dir}"
            )
        return checkpoint_path

    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )
    return checkpoint_path


def maybe_resume_training(
    resume_from_checkpoint: str | None,
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> tuple[int, BestModelState | None, list[Metrics]]:
    checkpoint_path = resolve_resume_checkpoint(
        checkpoint_dir, resume_from_checkpoint
    )
    if checkpoint_path is None:
        return 1, None, []

    loaded_epoch, best_model_state, history = load_epoch_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    next_epoch = loaded_epoch + 1
    print(f"Resumed from: {checkpoint_path} (next epoch: {next_epoch})")
    return next_epoch, best_model_state, history


def log_epoch_summary(
    metrics: Metrics, epoch: int, total_epochs: int, optimizer_steps: int
) -> None:
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train_loss={metrics.train_loss:.4f} train_acc={metrics.train_acc:.4f} | "
        f"val_loss={metrics.val_loss:.4f} val_acc={metrics.val_acc:.4f} "
        f"val_precision={metrics.val_precision:.4f} val_recall={metrics.val_recall:.4f} "
        f"val_f1={metrics.val_f1:.4f} val_roc_auc={metrics.val_roc_auc:.4f} | "
        f"opt_steps={optimizer_steps}"
    )
    tracking.log_epoch_metrics(
        metrics=asdict(metrics),
        epoch=epoch,
        optimizer_steps=optimizer_steps,
    )


def is_better_metric(
    best_model_state: BestModelState | None, current_metric: float
) -> bool:
    previous_metric = (
        best_model_state.metric_value
        if best_model_state is not None
        else float("nan")
    )
    return (
        best_model_state is None
        or (math.isnan(previous_metric) and not math.isnan(current_metric))
        or (not math.isnan(current_metric) and current_metric > previous_metric)
    )


def capture_best_model_state(
    model: nn.Module,
    class_names: list[str],
    metric_name: str,
    metric_value: float,
    epoch: int,
    model_name: str,
    image_size: int,
    freeze_backbone: bool,
) -> BestModelState:
    return BestModelState(
        metric_name=metric_name,
        metric_value=metric_value,
        epoch=epoch,
        model_state_dict={
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        class_names=list(class_names),
        model_name=model_name,
        image_size=image_size,
        freeze_backbone=freeze_backbone,
    )


def finalize_training_run(
    model: nn.Module,
    history: list[Metrics],
    best_model_state: BestModelState | None,
    output_dir: Path,
    *,
    log_model_artifact: bool = True,
) -> Path | None:
    history_payload = [asdict(item) for item in history]
    best_model_for_mlflow = None
    promoted_model_path = None

    if best_model_state is not None:
        model.load_state_dict(best_model_state.model_state_dict)
        print(
            f"Loaded best model from epoch {best_model_state.epoch} "
            f"({best_model_state.metric_name}={best_model_state.metric_value:.4f})"
        )
        best_model_for_mlflow = model
        print(
            f"Best validation {best_model_state.metric_name}: {best_model_state.metric_value:.4f}"
        )
        tracking.log_summary_metrics({
            "best_val_roc_auc": best_model_state.metric_value,
            "best_epoch": float(best_model_state.epoch),
        })
        promoted_model_path = promote_best_model_checkpoint(
            output_dir=output_dir,
            best_model_state=best_model_state,
        )

    final_artifact_kwargs = {
        "best_model": best_model_for_mlflow,
        "history_payload": history_payload,
    }
    if not log_model_artifact:
        final_artifact_kwargs["log_model"] = False
    tracking.log_final_artifacts(**final_artifact_kwargs)
    return promoted_model_path


def _run_training_impl(
    config: TrainConfig,
    *,
    report_callback: Callable[[dict[str, float]], None] | None = None,
    nested_mlflow: bool = False,
    extra_mlflow_tags: dict[str, object] | None = None,
    log_model_artifact: bool = True,
) -> TrainingRunResult:
    validate_config(config)

    set_seed(config.seed)
    checkpoint_dir = prepare_checkpoint_dir(config.output_dir)
    mlflow_context_kwargs: dict[str, object] = {}
    if nested_mlflow:
        mlflow_context_kwargs["nested"] = True
    if extra_mlflow_tags:
        mlflow_context_kwargs["tags"] = extra_mlflow_tags

    with tracking.init_mlflow(config, **mlflow_context_kwargs):
        device = resolve_device(config.device)
        print_run_configuration(config, device)

        train_loader, val_loader, class_names, train_count, val_count = (
            build_dataloaders(
                metadata_csv=config.metadata_csv,
                images_dir=config.images_dir,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                image_size=config.image_size,
                model_name=config.model_name,
                device=device,
                seed=config.seed,
                label_column=config.label_column,
                train_set=config.train_set,
                val_set=config.val_set,
                train_fraction=config.train_fraction,
                val_fraction=config.val_fraction,
                train_samples=config.train_samples,
                val_samples=config.val_samples,
            )
        )
        print(f"Selected samples -> train: {train_count}, val: {val_count}")
        tracking.log_run_params(
            config=config,
            train_count=train_count,
            val_count=val_count,
            class_names=class_names,
        )

        model = build_model(
            model_name=config.model_name,
            class_names=class_names,
            freeze_backbone=config.freeze_backbone,
        ).to(device)
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params}/{all_params}")

        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(
            model=model, lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler, updates_per_epoch, total_training_steps, warmup_steps = (
            build_scheduler_for_training(
                optimizer=optimizer,
                train_loader=train_loader,
                epochs=config.epochs,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_ratio=config.warmup_ratio,
                lr_scheduler_type=config.lr_scheduler_type,
                max_train_batches=config.max_train_batches,
            )
        )
        print(
            f"Scheduler: {config.lr_scheduler_type} | "
            f"updates/epoch={updates_per_epoch} total_updates={total_training_steps} warmup_steps={warmup_steps}"
        )

        start_epoch, best_model_state, history = maybe_resume_training(
            resume_from_checkpoint=config.resume_from_checkpoint,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        for epoch in range(start_epoch, config.epochs + 1):
            train_loss, train_acc, optimizer_steps = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_grad_norm=config.max_grad_norm,
                max_batches=config.max_train_batches,
            )
            (
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
                val_roc_auc,
            ) = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=config.max_val_batches,
            )

            metrics = Metrics(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_precision=val_precision,
                val_recall=val_recall,
                val_f1=val_f1,
                val_roc_auc=val_roc_auc,
            )
            history.append(metrics)
            log_epoch_summary(
                metrics=metrics,
                epoch=epoch,
                total_epochs=config.epochs,
                optimizer_steps=optimizer_steps,
            )
            if report_callback is not None:
                report_callback({
                    "epoch": float(epoch),
                    "train_loss": metrics.train_loss,
                    "train_acc": metrics.train_acc,
                    "val_loss": metrics.val_loss,
                    "val_acc": metrics.val_acc,
                    "val_precision": metrics.val_precision,
                    "val_recall": metrics.val_recall,
                    "val_f1": metrics.val_f1,
                    "val_roc_auc": metrics.val_roc_auc,
                    "optimizer_steps": float(optimizer_steps),
                })

            if is_better_metric(best_model_state, metrics.val_roc_auc):
                best_model_state = capture_best_model_state(
                    model=model,
                    class_names=class_names,
                    metric_name="val_roc_auc",
                    metric_value=metrics.val_roc_auc,
                    epoch=epoch,
                    model_name=config.model_name,
                    image_size=config.image_size,
                    freeze_backbone=config.freeze_backbone,
                )

            epoch_ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_epoch_checkpoint(
                checkpoint_path=epoch_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_model_state=best_model_state,
                history=history,
            )
            cleanup_old_checkpoints(
                checkpoint_dir=checkpoint_dir, keep_last=config.save_total_limit
            )

        promoted_model_path = finalize_training_run(
            model=model,
            history=history,
            best_model_state=best_model_state,
            output_dir=config.output_dir,
            log_model_artifact=log_model_artifact,
        )

    return TrainingRunResult(
        config=config,
        history=history,
        best_model_state=best_model_state,
        promoted_model_path=promoted_model_path,
        checkpoint_dir=checkpoint_dir,
        train_count=train_count,
        val_count=val_count,
    )


def run_training(
    config: TrainConfig,
    *,
    report_callback: Callable[[dict[str, float]], None] | None = None,
    nested_mlflow: bool = False,
    extra_mlflow_tags: dict[str, object] | None = None,
    log_model_artifact: bool = True,
) -> TrainingRunResult:
    result = _run_training_impl(
        config,
        report_callback=report_callback,
        nested_mlflow=nested_mlflow,
        extra_mlflow_tags=extra_mlflow_tags,
        log_model_artifact=log_model_artifact,
    )

    print(f"Local epoch checkpoints saved to: {result.checkpoint_dir}")

    if result.promoted_model_path is not None and result.best_model_state is not None:
        print(
            "Training run complete. "
            f"Promoted best model to: {result.promoted_model_path} "
            f"(epoch {result.best_model_state.epoch}, "
            f"{result.best_model_state.metric_name}={result.best_model_state.metric_value:.4f})"
        )
        print(
            f"To share this model via DVC, run: dvc add {result.promoted_model_path}"
        )
    else:
        print("Training run complete. No promoted best model was written.")

    return result
