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
