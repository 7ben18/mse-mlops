from pathlib import Path

import yaml
from PIL import Image

from mse_mlops.analysis.melanoma_dataset import (
    build_class_distribution_frame,
    build_overview_frame,
    load_dataset_overview,
    make_contact_sheet,
    resolve_eval_split_dir,
    summarize_dimensions,
)


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def write_repo_config(repo_root: Path) -> None:
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "train.yaml").write_text(
        yaml.safe_dump({
            "data": {
                "data_dir": "data/raw/melanoma_cancer_dataset",
                "train_subdir": "train",
                "val_subdir": "val",
                "val_mode": "split",
            }
        }),
        encoding="utf-8",
    )


def test_load_dataset_overview_prefers_existing_test_dir_when_val_dir_missing(tmp_path: Path):
    repo_root = tmp_path / "repo"
    write_repo_config(repo_root)

    write_image(repo_root / "data/raw/melanoma_cancer_dataset/train/benign/one.jpg", (20, 30))
    write_image(repo_root / "data/raw/melanoma_cancer_dataset/test/malignant/two.jpg", (40, 50))

    overview = load_dataset_overview(repo_root / "notebooks")

    assert overview.repo_root == repo_root
    assert overview.eval_split_name == "test"
    assert overview.classes == ["benign", "malignant"]


def test_build_frames_and_contact_sheet(tmp_path: Path):
    repo_root = tmp_path / "repo"
    write_repo_config(repo_root)

    write_image(repo_root / "data/raw/melanoma_cancer_dataset/train/benign/one.jpg", (20, 30))
    write_image(repo_root / "data/raw/melanoma_cancer_dataset/train/malignant/two.jpg", (40, 50))
    write_image(repo_root / "data/raw/melanoma_cancer_dataset/test/benign/three.jpg", (60, 70))

    overview = load_dataset_overview(repo_root / "notebooks")
    overview_df = build_overview_frame(overview)
    class_df = build_class_distribution_frame(overview)
    contact_sheet = make_contact_sheet(overview.train_records, columns=2)

    assert overview_df["split"].tolist() == ["train", "test"]
    assert class_df["class"].tolist() == ["benign", "malignant"]
    assert contact_sheet.size[0] > 0
    assert summarize_dimensions(overview.train_records)["samples"] == 2


def test_resolve_eval_split_dir_falls_back_to_preferred_name_when_nothing_exists(tmp_path: Path):
    data_dir = tmp_path / "data"
    resolved_dir, split_name = resolve_eval_split_dir(data_dir, "val")

    assert resolved_dir == data_dir / "val"
    assert split_name == "val"
