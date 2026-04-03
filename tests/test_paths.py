from pathlib import Path

import yaml

from mse_mlops.paths import (
    DEFAULT_FINETUNED_MODEL_DIR,
    DEFAULT_PRETRAINED_MODEL_DIR,
    HAM_IMAGES_DIR,
    HAM_METADATA,
    REPO_ROOT,
    find_repo_root,
    resolve_train_config_path,
)


def write_config(repo_root: Path, payload: dict) -> None:
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "train.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_find_repo_root_from_nested_directory(tmp_path: Path):
    repo_root = tmp_path / "repo"
    write_config(repo_root, {"data": {"data_dir": "data/raw/example"}})

    nested_dir = repo_root / "notebooks" / "example"
    nested_dir.mkdir(parents=True)

    assert find_repo_root(nested_dir) == repo_root


def test_find_repo_root_from_file_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    write_config(repo_root, {"data": {"data_dir": "data/raw/example"}})

    notebook_path = repo_root / "notebooks" / "example" / "eda.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook_path.write_text("{}", encoding="utf-8")

    assert find_repo_root(notebook_path) == repo_root


def test_resolve_train_config_path_uses_repo_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    payload = {"data": {"data_dir": "data/raw/example"}, "training": {"epochs": 3}}
    write_config(repo_root, payload)

    config_path = resolve_train_config_path(repo_root)

    assert config_path == repo_root / "config" / "train.yaml"


def test_repo_train_config_matches_ham10000_models_contract():
    config_path = resolve_train_config_path(REPO_ROOT)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert isinstance(config, dict)
    assert config["model"]["model_name"] == str(DEFAULT_PRETRAINED_MODEL_DIR.relative_to(REPO_ROOT))
    assert config["data"]["metadata_csv"] == str(HAM_METADATA.relative_to(REPO_ROOT))
    assert config["data"]["images_dir"] == str(HAM_IMAGES_DIR.relative_to(REPO_ROOT))
    assert config["training"]["output_dir"] == str(DEFAULT_FINETUNED_MODEL_DIR.relative_to(REPO_ROOT))


def test_repo_train_config_is_yaml_mapping(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "train.yaml").write_text("- invalid\n", encoding="utf-8")
    config_path = resolve_train_config_path(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert not isinstance(config, dict)
