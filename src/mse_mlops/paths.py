from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/train.yaml")


def find_repo_root(start: Path | str | None = None) -> Path:
    candidate = Path.cwd() if start is None else Path(start)
    resolved = candidate.resolve()
    search_start = resolved.parent if resolved.is_file() else resolved

    for current in (search_start, *search_start.parents):
        if (current / DEFAULT_CONFIG_PATH).exists():
            return current

    raise FileNotFoundError(f"Could not find repo root from {candidate}")


def resolve_train_config_path(repo_root: Path | str | None = None) -> Path:
    root = (
        find_repo_root(repo_root) if repo_root is not None else find_repo_root()
    )
    return root / DEFAULT_CONFIG_PATH


def load_train_config(
    start: Path | str | None = None,
) -> tuple[Path, dict[str, Any]]:
    repo_root = find_repo_root(start)
    config_path = resolve_train_config_path(repo_root)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: {config_path}"
        )

    return config_path, config


REPO_ROOT = find_repo_root(Path(__file__))

RAW_DATA_DIR = Path(REPO_ROOT / "data" / "raw")

PROCESSED_DATA_DIR = Path(REPO_ROOT / "data" / "processed")

REPORTS_DIR = Path(REPO_ROOT / "reports")

CONFIG_DIR = Path(REPO_ROOT / "config")

HAM_DIR = Path("ham10000")

MAP_LESION_IMAGES = Path(PROCESSED_DATA_DIR / HAM_DIR / "all_lesion_images_mapping_HAM10000.csv")
EXT_METADATA = Path(PROCESSED_DATA_DIR / HAM_DIR / "extended_HAM10000_metadata.csv")

IMG_DIR = Path("HAM10000_images")
MASK_DIR = Path("HAM10000_segmentations_lesion_tschandl")
METADATA = "HAM10000_metadata.csv"

IMG_NAME_RE = r"^ISIC_\d{7}$"
MASK_NAME_RE = r"^ISIC_\d{7}_segmentation$"
LESION_NAME_RE = r"^HAM_\d{7}$"


__all__ = [
    "EXT_METADATA",
    "IMG_DIR",
    "MAP_LESION_IMAGES",
    "MASK_DIR",
    "METADATA",
    "REPORTS_DIR",
    "REPO_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "CONFIG_DIR",
    "HAM_DIR",
    "IMG_NAME_RE",
    "MASK_NAME_RE",
    "LESION_NAME_RE",
]