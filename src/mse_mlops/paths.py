from __future__ import annotations

from pathlib import Path

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
    root = find_repo_root(repo_root) if repo_root is not None else find_repo_root()
    return root / DEFAULT_CONFIG_PATH


REPO_ROOT = find_repo_root(Path(__file__))

RAW_DATA_DIR = Path(REPO_ROOT / "data" / "raw")

PROCESSED_DATA_DIR = Path(REPO_ROOT / "data" / "processed")
MODELS_DIR = Path(REPO_ROOT / "models")
PRETRAINED_MODELS_DIR = Path(MODELS_DIR / "pretrained")
FINETUNED_MODELS_DIR = Path(MODELS_DIR / "finetuned")

REPORTS_DIR = Path(REPO_ROOT / "reports")

CONFIG_DIR = Path(REPO_ROOT / "config")

HAM_DIR = Path("ham10000")

HAM_METADATA = Path(PROCESSED_DATA_DIR / HAM_DIR / "metadata.csv")
HAM_IMAGES_DIR = Path(PROCESSED_DATA_DIR / HAM_DIR / "HAM10000_images")
DEFAULT_PRETRAINED_MODEL_DIR = Path(PRETRAINED_MODELS_DIR / "dinov3-vits16-pretrain-lvd1689m")
DEFAULT_FINETUNED_MODEL_DIR = Path(FINETUNED_MODELS_DIR / "dinov3_ham10000")
EXT_METADATA = HAM_METADATA

IMG_DIR = Path("HAM10000_images")
MASK_DIR = Path("HAM10000_segmentations_lesion_tschandl")
METADATA = "HAM10000_metadata.csv"

IMG_NAME_RE = r"^ISIC_\d{7}$"
MASK_NAME_RE = r"^ISIC_\d{7}_segmentation$"
LESION_NAME_RE = r"^HAM_\d{7}$"


__all__ = [
    "CONFIG_DIR",
    "DEFAULT_FINETUNED_MODEL_DIR",
    "DEFAULT_PRETRAINED_MODEL_DIR",
    "EXT_METADATA",
    "FINETUNED_MODELS_DIR",
    "HAM_DIR",
    "HAM_IMAGES_DIR",
    "HAM_METADATA",
    "IMG_DIR",
    "IMG_NAME_RE",
    "LESION_NAME_RE",
    "MASK_DIR",
    "MASK_NAME_RE",
    "METADATA",
    "MODELS_DIR",
    "PRETRAINED_MODELS_DIR",
    "PROCESSED_DATA_DIR",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
    "REPO_ROOT",
]
