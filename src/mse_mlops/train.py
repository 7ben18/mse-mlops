from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, get_scheduler


DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DATA_DIR = "data/melanoma_cancer_dataset"
DEFAULT_OUTPUT_DIR = "outputs/dinov3_melanoma"
DEFAULT_CONFIG_PATH = "config/train.yaml"

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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def load_config_defaults(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    defaults: dict[str, object] = {}
    for section in ("model", "data", "training"):
        section_payload = data.get(section)
        if isinstance(section_payload, dict):
            defaults.update(section_payload)

    for key, value in data.items():
        if key not in {"model", "data", "training"}:
            defaults[key] = value

    return defaults


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_config_defaults(pre_args.config)

    freeze_default = bool(defaults.get("freeze_backbone", True))
    load_best_default = bool(defaults.get("load_best_model_at_end", True))

    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 on a local ImageFolder dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
        help=f"YAML config file path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(str(defaults.get("data_dir", DEFAULT_DATA_DIR))))
    parser.add_argument("--train-subdir", type=str, default=str(defaults.get("train_subdir", "train")))
    parser.add_argument("--val-subdir", type=str, default=str(defaults.get("val_subdir", "test")))
    parser.add_argument("--val-mode", choices=("test", "split", "train"), default=str(defaults.get("val_mode", "test")))
    parser.add_argument("--val-split", type=float, default=float(defaults.get("val_split", 0.2)))
    parser.add_argument("--train-fraction", type=float, default=float(defaults.get("train_fraction", 1.0)))
    parser.add_argument("--val-fraction", type=float, default=float(defaults.get("val_fraction", 1.0)))
    parser.add_argument("--train-samples", type=int, default=_optional_int(defaults.get("train_samples")))
    parser.add_argument("--val-samples", type=int, default=_optional_int(defaults.get("val_samples")))

    parser.add_argument("--model-name", type=str, default=str(defaults.get("model_name", DEFAULT_MODEL)))
    parser.add_argument("--output-dir", type=Path, default=Path(str(defaults.get("output_dir", DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 3)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 16)))
    parser.add_argument("--image-size", type=int, default=int(defaults.get("image_size", 224)))
    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 5e-5)))
    parser.add_argument("--weight-decay", type=float, default=float(defaults.get("weight_decay", 0.01)))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 0)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=str(defaults.get("device", "auto")))

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(defaults.get("gradient_accumulation_steps", 1)),
    )
    parser.add_argument("--warmup-ratio", type=float, default=float(defaults.get("warmup_ratio", 0.1)))
    parser.add_argument(
        "--lr-scheduler-type",
        choices=SCHEDULER_CHOICES,
        default=str(defaults.get("lr_scheduler_type", "linear")),
    )
    parser.add_argument("--max-grad-norm", type=float, default=float(defaults.get("max_grad_norm", 1.0)))

    parser.add_argument("--max-train-batches", type=int, default=_optional_int(defaults.get("max_train_batches")))
    parser.add_argument("--max-val-batches", type=int, default=_optional_int(defaults.get("max_val_batches")))

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=defaults.get("resume_from_checkpoint"),
        help="Checkpoint path or 'latest' to resume from the newest epoch checkpoint.",
    )
    parser.add_argument("--save-total-limit", type=int, default=_optional_int(defaults.get("save_total_limit")))

    parser.add_argument("--freeze-backbone", action="store_true", dest="freeze_backbone")
    parser.add_argument("--unfreeze-backbone", action="store_false", dest="freeze_backbone")
    parser.set_defaults(freeze_backbone=freeze_default)

    parser.add_argument("--load-best-model-at-end", action="store_true", dest="load_best_model_at_end")
    parser.add_argument("--no-load-best-model-at-end", action="store_false", dest="load_best_model_at_end")
    parser.set_defaults(load_best_model_at_end=load_best_default)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


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


def resolve_mean_std(processor: AutoImageProcessor) -> tuple[Sequence[float], Sequence[float]]:
    mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
    return mean, std


def build_transforms(mean: Sequence[float], std: Sequence[float], image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # HF-style augmentation for train/eval image classification pipelines.
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    resize_size = max(image_size, int(image_size * 256 / 224))
    eval_tf = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
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


class DinoV3Classifier(nn.Module):
    """Primary classifier wrapper for DINOv3 backbone checkpoints."""

    def __init__(self, model_name: str, num_labels: int, freeze_backbone: bool) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.backbone.config, "hidden_sizes"):
            hidden_size = self.backbone.config.hidden_sizes[-1]
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config.")

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, tuple) and outputs:
            pooled = outputs[0][:, 0]
        else:
            raise ValueError("Backbone output format is not supported.")

        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    model_name: str,
    device: torch.device,
    seed: int,
    train_subdir: str,
    val_subdir: str,
    val_mode: str,
    val_split: float,
    train_fraction: float,
    val_fraction: float,
    train_samples: int | None,
    val_samples: int | None,
) -> tuple[DataLoader, DataLoader, list[str], int, int]:
    train_dir = data_dir / train_subdir
    val_dir = data_dir / val_subdir
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    mean, std = resolve_mean_std(processor)
    train_tf, eval_tf = build_transforms(mean, std, image_size=image_size)

    train_full = datasets.ImageFolder(train_dir, transform=train_tf)
    train_full_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    class_names = list(train_full.classes)
    all_train_indices = list(range(len(train_full)))

    if val_mode == "test":
        if not val_dir.is_dir():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        val_full = datasets.ImageFolder(val_dir, transform=eval_tf)
        if class_names != list(val_full.classes):
            raise ValueError("Class folders under train and validation directories do not match.")

        train_indices = choose_indices(all_train_indices, train_fraction, train_samples, seed)
        val_indices = choose_indices(list(range(len(val_full))), val_fraction, val_samples, seed + 1)
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)

    elif val_mode == "split":
        if not (0.0 < val_split < 1.0):
            raise ValueError("--val-split must be between 0 and 1 when --val-mode split is used.")
        if len(all_train_indices) < 2:
            raise ValueError("Need at least 2 images in train directory for split mode.")

        shuffled = list(all_train_indices)
        random.Random(seed).shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_split))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1

        base_val_indices = shuffled[:val_count]
        base_train_indices = shuffled[val_count:]
        train_indices = choose_indices(base_train_indices, train_fraction, train_samples, seed + 2)
        val_indices = choose_indices(base_val_indices, val_fraction, val_samples, seed + 3)
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(train_full_eval, val_indices)

    else:
        train_indices = choose_indices(all_train_indices, train_fraction, train_samples, seed)
        val_indices = choose_indices(train_indices, val_fraction, val_samples, seed + 1)
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(train_full_eval, val_indices)

    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty after applying split/fraction/sample settings.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty after applying split/fraction/sample settings.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, class_names, len(train_ds), len(val_ds)


def build_model(model_name: str, class_names: list[str], freeze_backbone: bool) -> nn.Module:
    model = DinoV3Classifier(
        model_name=model_name,
        num_labels=len(class_names),
        freeze_backbone=freeze_backbone,
    )
    print("Model head mode: custom DINOv3 classifier")
    return model


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay_terms = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "norm.bias")
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
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
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
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if max_batches is not None and batch_idx >= max_batches:
                break

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


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

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(pixel_values=images).logits
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        (loss / gradient_accumulation_steps).backward()

        should_step = (batch_idx % gradient_accumulation_steps == 0) or (batch_idx == planned_batches)
        if should_step:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if len(checkpoints) <= keep_last:
        return
    for old_path in checkpoints[: len(checkpoints) - keep_last]:
        old_path.unlink(missing_ok=True)


def save_epoch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    best_acc: float,
    history: list[Metrics],
) -> None:
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": [asdict(item) for item in history],
    }
    torch.save(payload, checkpoint_path)


def load_epoch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> tuple[int, float, list[Metrics]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    loaded_epoch = int(payload["epoch"])
    best_acc = float(payload.get("best_acc", -1.0))
    history = [Metrics(**item) for item in payload.get("history", [])]
    return loaded_epoch, best_acc, history


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be > 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0.")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be >= 0.")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1).")

    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in (0, 1].")
    if not (0.0 < args.val_fraction <= 1.0):
        raise ValueError("--val-fraction must be in (0, 1].")
    if args.train_samples is not None and args.train_samples <= 0:
        raise ValueError("--train-samples must be > 0 when provided.")
    if args.val_samples is not None and args.val_samples <= 0:
        raise ValueError("--val-samples must be > 0 when provided.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be > 0 when provided.")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("--max-val-batches must be > 0 when provided.")
    if args.save_total_limit is not None and args.save_total_limit <= 0:
        raise ValueError("--save-total-limit must be > 0 when provided.")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Config: {args.config}")
    print(f"Using device: {device.type}")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_dir}")
    print(
        f"Validation mode: {args.val_mode} "
        f"(val_split={args.val_split}, train_fraction={args.train_fraction}, val_fraction={args.val_fraction})"
    )

    train_loader, val_loader, class_names, train_count, val_count = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        model_name=args.model_name,
        device=device,
        seed=args.seed,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        val_mode=args.val_mode,
        val_split=args.val_split,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )
    print(f"Selected samples -> train: {train_count}, val: {val_count}")

    model = build_model(
        model_name=args.model_name,
        class_names=class_names,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}/{all_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)

    train_batches_per_epoch = len(train_loader)
    if args.max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, args.max_train_batches)
    updates_per_epoch = max(1, math.ceil(train_batches_per_epoch / args.gradient_accumulation_steps))
    total_training_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(
        f"Scheduler: {args.lr_scheduler_type} | "
        f"updates/epoch={updates_per_epoch} total_updates={total_training_steps} warmup_steps={warmup_steps}"
    )

    start_epoch = 1
    best_acc = -1.0
    history: list[Metrics] = []

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"--resume-from-checkpoint=latest but no checkpoint found in {checkpoint_dir}"
                )
        else:
            checkpoint_path = Path(args.resume_from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        loaded_epoch, best_acc, history = load_epoch_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = loaded_epoch + 1
        print(f"Resumed from: {checkpoint_path} (next epoch: {start_epoch})")

    best_model_path = args.output_dir / "best_model.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, optimizer_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        metrics = Metrics(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"opt_steps={optimizer_steps}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                    "freeze_backbone": args.freeze_backbone,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_model_path,
            )

        epoch_ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_epoch_checkpoint(
            checkpoint_path=epoch_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_acc=best_acc,
            history=history,
        )
        cleanup_old_checkpoints(checkpoint_dir=checkpoint_dir, keep_last=args.save_total_limit)

        history_payload = [asdict(item) for item in history]
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)

    if args.load_best_model_at_end and best_model_path.exists():
        best_payload = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_payload["model_state_dict"])
        print(f"Loaded best model from epoch {best_payload.get('epoch')} (val_acc={best_payload.get('val_acc'):.4f})")

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
