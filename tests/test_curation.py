from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from mse_mlops.curation import (
    PromotionConfig,
    get_training_batch_status,
    promote_feedback_to_train,
    set_training_batch_enabled,
)
from mse_mlops.serving.feedback_store import load_feedback_entries, write_feedback_entries


def write_rgb_image(path: Path, size: tuple[int, int] = (12, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def write_metadata_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_config(tmp_path: Path, *, min_items: int = 1) -> PromotionConfig:
    root = tmp_path / "project"
    return PromotionConfig(
        feedback_file=root / "reports" / "feedback" / "feedback.jsonl",
        feedback_images_dir=root / "reports" / "feedback" / "images",
        promotions_dir=root / "reports" / "feedback" / "promotions",
        metadata_csv=root / "data" / "processed" / "ham10000" / "metadata.csv",
        dataset_images_dir=root / "data" / "processed" / "ham10000" / "HAM10000_images",
        min_items=min_items,
    )


def test_promotion_stamps_one_batch_and_writes_one_manifest(tmp_path: Path):
    config = build_config(tmp_path)
    write_metadata_csv(
        config.metadata_csv,
        [
            {"lesion_id": "HAM_1", "image_id": "ISIC_1", "set": "val", "mb": "benign"},
        ],
    )
    write_rgb_image(config.feedback_images_dir / "feedback-a.png")
    write_rgb_image(config.feedback_images_dir / "feedback-b.png")
    write_feedback_entries(
        config.feedback_file,
        [
            {
                "image_id": "feedback-a",
                "filename": "doctor-a.png",
                "label": "benign",
                "source": "upload_labeled",
            },
            {
                "image_id": "feedback-b",
                "filename": "doctor-b.png",
                "label": "malignant",
                "source": "upload_labeled",
            },
        ],
    )

    result = promote_feedback_to_train(config, dry_run=False)

    metadata_df = pd.read_csv(config.metadata_csv)
    promoted_df = metadata_df[metadata_df["image_id"].str.startswith("upload_")]
    manifest_files = sorted(config.promotions_dir.glob("*.json"))
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))

    assert result.promoted_count == 2
    assert result.batch_id is not None
    assert result.manifest_path == manifest_files[0]
    assert len(manifest_files) == 1
    assert promoted_df["first_train_batch_id"].nunique() == 1
    assert promoted_df["first_train_batch_id"].iloc[0] == result.batch_id
    assert set(promoted_df["training_enabled"]) == {True}
    assert set(promoted_df["promotion_source"]) == {"feedback_upload"}
    assert manifest["batch_id"] == result.batch_id
    assert manifest["promoted_count"] == 2
    assert {item["promotion_source"] for item in manifest["items"]} == {"feedback_upload"}


def test_future_upload_gets_future_demo_promotion_source(tmp_path: Path):
    config = build_config(tmp_path)
    write_metadata_csv(
        config.metadata_csv,
        [
            {
                "lesion_id": "HAM_future",
                "image_id": "ISIC_future",
                "set": "future",
                "mb": "benign",
            },
        ],
    )
    write_rgb_image(config.dataset_images_dir / "future" / "ISIC_future.jpg")
    write_rgb_image(config.feedback_images_dir / "feedback-future.jpg")
    write_feedback_entries(
        config.feedback_file,
        [
            {
                "image_id": "feedback-future",
                "filename": "ISIC_future.jpg",
                "label": "malignant",
                "source": "upload_labeled",
            },
        ],
    )

    result = promote_feedback_to_train(config, dry_run=False)

    metadata_df = pd.read_csv(config.metadata_csv)
    promoted_row = metadata_df.loc[metadata_df["image_id"] == "ISIC_future"].iloc[0]
    feedback_entries = load_feedback_entries(config.feedback_file)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.promoted_count == 1
    assert promoted_row["set"] == "train"
    assert promoted_row["mb"] == "malignant"
    assert promoted_row["promotion_source"] == "future_demo"
    assert promoted_row["first_train_batch_id"] == result.batch_id
    assert config.dataset_images_dir.joinpath("train", "ISIC_future.jpg").is_file()
    assert feedback_entries[0]["promoted_image_id"] == "ISIC_future"
    assert manifest["items"][0]["promotion_source"] == "future_demo"


def test_permanent_batch_exclusion_disables_future_training(tmp_path: Path):
    config = build_config(tmp_path)
    write_metadata_csv(
        config.metadata_csv,
        [
            {
                "lesion_id": "upload_a",
                "image_id": "upload_a",
                "set": "train",
                "mb": "benign",
                "first_train_batch_id": "train_20260514123045",
                "first_train_at": "2026-05-14T12:30:45+00:00",
                "training_enabled": True,
                "promotion_source": "feedback_upload",
            },
            {
                "lesion_id": "upload_b",
                "image_id": "upload_b",
                "set": "train",
                "mb": "malignant",
                "first_train_batch_id": "train_20260514123045",
                "first_train_at": "2026-05-14T12:30:45+00:00",
                "training_enabled": True,
                "promotion_source": "feedback_upload",
            },
        ],
    )

    row = set_training_batch_enabled(
        "train_20260514123045",
        enabled=False,
        config=config,
    )

    metadata_df = pd.read_csv(config.metadata_csv)
    status = get_training_batch_status(config)

    assert row["enabled_rows"] == 0
    assert row["disabled_rows"] == 2
    assert metadata_df["training_enabled"].tolist() == [False, False]
    assert status[0]["batch_id"] == row["batch_id"]
    assert status[0]["enabled_rows"] == row["enabled_rows"]
    assert status[0]["disabled_rows"] == row["disabled_rows"]


def test_batch_exclusion_rejects_unknown_batch(tmp_path: Path):
    config = build_config(tmp_path)
    write_metadata_csv(
        config.metadata_csv,
        [
            {
                "lesion_id": "upload_a",
                "image_id": "upload_a",
                "set": "train",
                "mb": "benign",
            },
        ],
    )

    with pytest.raises(ValueError, match="Unknown training batch"):
        set_training_batch_enabled("train_missing", enabled=False, config=config)
