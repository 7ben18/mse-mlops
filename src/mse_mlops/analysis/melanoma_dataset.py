from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from mse_mlops.paths import load_train_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetOverview:
    repo_root: Path
    config_path: Path
    data_dir: Path
    train_dir: Path
    eval_dir: Path
    eval_split_name: str
    validation_mode: str
    train_records: list[dict[str, Any]]
    eval_records: list[dict[str, Any]]
    classes: list[str]

    @property
    def all_records(self) -> list[dict[str, Any]]:
        return self.train_records + self.eval_records


def load_dataset_overview(start: Path | str | None = None) -> DatasetOverview:
    config_path, config = load_train_config(start)
    repo_root = config_path.parents[1]

    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError(f"'data' section must be a YAML mapping: {config_path}")

    data_dir = repo_root / str(data_config.get("data_dir", "data/raw/melanoma_cancer_dataset"))
    train_dir = data_dir / str(data_config.get("train_subdir", "train"))
    preferred_eval_name = str(data_config.get("val_subdir", "val"))
    eval_dir, eval_split_name = resolve_eval_split_dir(data_dir, preferred_eval_name)

    train_records = collect_split_records(train_dir, "train")
    eval_records = collect_split_records(eval_dir, eval_split_name)
    classes = sorted({record["class_name"] for record in train_records + eval_records})

    return DatasetOverview(
        repo_root=repo_root,
        config_path=config_path,
        data_dir=data_dir,
        train_dir=train_dir,
        eval_dir=eval_dir,
        eval_split_name=eval_split_name,
        validation_mode=str(data_config.get("val_mode", "split")),
        train_records=train_records,
        eval_records=eval_records,
        classes=classes,
    )


def resolve_eval_split_dir(data_dir: Path, preferred_name: str) -> tuple[Path, str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for name in (preferred_name, "test", "val"):
        if name not in seen:
            candidates.append(name)
            seen.add(name)

    for name in candidates:
        candidate = data_dir / name
        if candidate.exists():
            return candidate, name

    return data_dir / preferred_name, preferred_name


def collect_split_records(split_dir: Path, split_name: str) -> list[dict[str, Any]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    records: list[dict[str, Any]] = []
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                records.append({
                    "split": split_name,
                    "class_name": class_dir.name,
                    "path": image_path,
                    "width": width,
                    "height": height,
                    "mode": image.mode,
                })
    return records


def summarize_dimensions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty record collection.")

    widths = [int(record["width"]) for record in records]
    heights = [int(record["height"]) for record in records]
    shapes = Counter((record["width"], record["height"]) for record in records)
    most_common_shape, most_common_count = shapes.most_common(1)[0]
    return {
        "samples": len(records),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "median_width": int(statistics.median(widths)),
        "median_height": int(statistics.median(heights)),
        "most_common_shape": most_common_shape,
        "most_common_count": most_common_count,
    }


def build_overview_frame(overview: DatasetOverview) -> pd.DataFrame:
    train_summary = summarize_dimensions(overview.train_records)
    eval_summary = summarize_dimensions(overview.eval_records)

    return pd.DataFrame([
        {
            "split": "train",
            "samples": train_summary["samples"],
            "width_range": f"{train_summary['min_width']} - {train_summary['max_width']}",
            "height_range": f"{train_summary['min_height']} - {train_summary['max_height']}",
            "median_size": f"{train_summary['median_width']} x {train_summary['median_height']}",
            "most_common_size": (
                f"{train_summary['most_common_shape'][0]} x {train_summary['most_common_shape'][1]} "
                f"({train_summary['most_common_count']})"
            ),
        },
        {
            "split": overview.eval_split_name,
            "samples": eval_summary["samples"],
            "width_range": f"{eval_summary['min_width']} - {eval_summary['max_width']}",
            "height_range": f"{eval_summary['min_height']} - {eval_summary['max_height']}",
            "median_size": f"{eval_summary['median_width']} x {eval_summary['median_height']}",
            "most_common_size": (
                f"{eval_summary['most_common_shape'][0]} x {eval_summary['most_common_shape'][1]} "
                f"({eval_summary['most_common_count']})"
            ),
        },
    ])


def build_class_distribution_frame(overview: DatasetOverview) -> pd.DataFrame:
    train_class_counts = Counter(record["class_name"] for record in overview.train_records)
    eval_class_counts = Counter(record["class_name"] for record in overview.eval_records)

    return pd.DataFrame([
        {
            "class": class_name,
            "train": train_class_counts[class_name],
            overview.eval_split_name: eval_class_counts[class_name],
            "total": train_class_counts[class_name] + eval_class_counts[class_name],
        }
        for class_name in overview.classes
    ])


def select_class_records(overview: DatasetOverview, class_name: str, sample_size: int = 8) -> list[dict[str, Any]]:
    class_records = [record for record in overview.train_records if record["class_name"] == class_name]
    if len(class_records) < sample_size:
        class_records += [record for record in overview.eval_records if record["class_name"] == class_name]
    return class_records[:sample_size]


def make_contact_sheet(
    records: list[dict[str, Any]],
    thumb_size: tuple[int, int] = (180, 180),
    columns: int = 4,
) -> Image.Image:
    if not records:
        raise ValueError("Cannot build a contact sheet without records.")

    rows = math.ceil(len(records) / columns)
    margin = 12
    caption_height = 28
    canvas_width = columns * thumb_size[0] + (columns + 1) * margin
    canvas_height = rows * (thumb_size[1] + caption_height) + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)

    for index, record in enumerate(records):
        row = index // columns
        column = index % columns
        x = margin + column * thumb_size[0] + column * margin
        y = margin + row * (thumb_size[1] + caption_height) + row * margin

        with Image.open(record["path"]) as image:
            preview = ImageOps.contain(image.convert("RGB"), thumb_size)

        tile = Image.new("RGB", thumb_size, color="#f4f4f4")
        offset = ((thumb_size[0] - preview.width) // 2, (thumb_size[1] - preview.height) // 2)
        tile.paste(preview, offset)
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_size[0], y + thumb_size[1]), outline="#999999", width=1)
        draw.text((x, y + thumb_size[1] + 6), f"{record['width']}x{record['height']}", fill="black")

    return canvas


__all__ = [
    "IMAGE_EXTENSIONS",
    "DatasetOverview",
    "build_class_distribution_frame",
    "build_overview_frame",
    "collect_split_records",
    "load_dataset_overview",
    "make_contact_sheet",
    "resolve_eval_split_dir",
    "select_class_records",
    "summarize_dimensions",
]
