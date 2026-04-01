from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from mse_mlops.data_processing import split_data_csv, split_data_dir, split_data_full


def write_split_config(path: Path, split_sets: list[tuple[str, float]], seed: int = 13) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"seed: {seed}", "", "split_sets:"]

    for name, ratio in split_sets:
        lines.append(f'  - name: "{name}"')
        lines.append(f"    ratio: {ratio}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_rgb_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def write_mask_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_split_data_csv_builds_single_metadata_csv_with_lesion_consistent_splits(tmp_path: Path):
    raw_metadata_csv = tmp_path / "raw" / "ham10000" / "HAM10000_metadata.csv"
    config_file = tmp_path / "config" / "split.yaml"
    csv_output = tmp_path / "processed" / "ham10000"

    write_raw_metadata(
        raw_metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "dx": "mel"},
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0002", "dx": "mel"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0003", "dx": "nv"},
            {"lesion_id": "HAM_0003", "image_id": "ISIC_0004", "dx": "bcc"},
            {"lesion_id": "HAM_0003", "image_id": "ISIC_0005", "dx": "bcc"},
            {"lesion_id": "HAM_0004", "image_id": "ISIC_0006", "dx": "df"},
        ],
    )
    write_split_config(config_file, [("train", 0.5), ("val", 0.25), ("test", 0.25)], seed=7)

    metadata_csv = split_data_csv(
        config_file=config_file,
        raw_metadata_csv=raw_metadata_csv,
        csv_output=csv_output,
        verbose=False,
    )

    metadata_df = pd.read_csv(metadata_csv)

    assert metadata_csv.exists()
    assert metadata_csv.name == "metadata.csv"
    assert list(metadata_df[["lesion_id", "image_id", "dx", "mb", "set"]].columns) == [
        "lesion_id",
        "image_id",
        "dx",
        "mb",
        "set",
    ]
    assert metadata_df[["image_id", "mb"]].to_dict("records") == [
        {"image_id": "ISIC_0001", "mb": "malignant"},
        {"image_id": "ISIC_0002", "mb": "malignant"},
        {"image_id": "ISIC_0003", "mb": "benign"},
        {"image_id": "ISIC_0004", "mb": "malignant"},
        {"image_id": "ISIC_0005", "mb": "malignant"},
        {"image_id": "ISIC_0006", "mb": "benign"},
    ]
    assert metadata_df.groupby("lesion_id")["set"].nunique().to_dict() == {
        "HAM_0001": 1,
        "HAM_0002": 1,
        "HAM_0003": 1,
        "HAM_0004": 1,
    }
    assert metadata_df.groupby("set")["lesion_id"].nunique().to_dict() == {"train": 2, "val": 1, "test": 1}


def test_split_data_csv_is_deterministic_for_same_seed(tmp_path: Path):
    raw_metadata_csv = tmp_path / "raw" / "ham10000" / "HAM10000_metadata.csv"
    config_file = tmp_path / "config" / "split.yaml"

    write_raw_metadata(
        raw_metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "dx": "mel"},
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0002", "dx": "mel"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0003", "dx": "nv"},
            {"lesion_id": "HAM_0003", "image_id": "ISIC_0004", "dx": "bcc"},
            {"lesion_id": "HAM_0004", "image_id": "ISIC_0005", "dx": "df"},
            {"lesion_id": "HAM_0005", "image_id": "ISIC_0006", "dx": "vasc"},
        ],
    )
    write_split_config(config_file, [("train", 0.6), ("test", 0.4)], seed=13)

    metadata_csv_a = split_data_csv(
        config_file=config_file,
        raw_metadata_csv=raw_metadata_csv,
        csv_output=tmp_path / "processed_a" / "ham10000",
        verbose=False,
    )
    metadata_csv_b = split_data_csv(
        config_file=config_file,
        raw_metadata_csv=raw_metadata_csv,
        csv_output=tmp_path / "processed_b" / "ham10000",
        verbose=False,
    )

    pd.testing.assert_frame_equal(pd.read_csv(metadata_csv_a), pd.read_csv(metadata_csv_b))


def test_split_data_dir_copies_expected_files_and_keeps_only_metadata_csv(tmp_path: Path):
    data_input = tmp_path / "raw" / "ham10000"
    data_output = tmp_path / "processed" / "ham10000"
    metadata_csv = data_output / "metadata.csv"

    write_rgb_image(data_input / "HAM10000_images" / "ISIC_0001.jpg")
    write_rgb_image(data_input / "HAM10000_images" / "ISIC_0002.jpg")
    write_rgb_image(data_input / "HAM10000_images" / "ISIC_0003.jpg")
    write_mask_image(data_input / "HAM10000_segmentations_lesion_tschandl" / "ISIC_0001_segmentation.png")
    write_mask_image(data_input / "HAM10000_segmentations_lesion_tschandl" / "ISIC_0002_segmentation.png")
    write_mask_image(data_input / "HAM10000_segmentations_lesion_tschandl" / "ISIC_0003_segmentation.png")

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "set": "train"},
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0002", "set": "train"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0003", "set": "test"},
        ]
    ).to_csv(metadata_csv, index=False)

    stale_img = data_output / "HAM10000_images" / "legacy" / "stale.jpg"
    stale_mask = data_output / "HAM10000_segmentations_lesion_tschandl" / "legacy" / "stale.png"
    write_rgb_image(stale_img)
    write_mask_image(stale_mask)

    split_data_dir(split_csv=metadata_csv, data_input=data_input, data_output=data_output, verbose=False)

    assert metadata_csv.exists()
    assert sorted(path.name for path in data_output.glob("*.csv")) == ["metadata.csv"]
    assert not stale_img.exists()
    assert not stale_mask.exists()
    assert (data_output / "HAM10000_images" / "train" / "ISIC_0001.jpg").exists()
    assert (data_output / "HAM10000_images" / "train" / "ISIC_0002.jpg").exists()
    assert (data_output / "HAM10000_images" / "test" / "ISIC_0003.jpg").exists()
    assert (data_output / "HAM10000_segmentations_lesion_tschandl" / "train" / "ISIC_0001_segmentation.png").exists()
    assert (data_output / "HAM10000_segmentations_lesion_tschandl" / "train" / "ISIC_0002_segmentation.png").exists()
    assert (data_output / "HAM10000_segmentations_lesion_tschandl" / "test" / "ISIC_0003_segmentation.png").exists()


def test_split_data_dir_rejects_lesion_assigned_to_multiple_sets(tmp_path: Path):
    data_output = tmp_path / "processed" / "ham10000"
    metadata_csv = data_output / "metadata.csv"

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"lesion_id": "HAM_0000118", "image_id": "ISIC_0001", "set": "train"},
            {"lesion_id": "HAM_0000118", "image_id": "ISIC_0002", "set": "test"},
            {"lesion_id": "HAM_0007409", "image_id": "ISIC_0003", "set": "train"},
            {"lesion_id": "HAM_0007409", "image_id": "ISIC_0004", "set": "train"},
        ]
    ).to_csv(metadata_csv, index=False)

    with pytest.raises(ValueError, match=r"Each lesion_id must belong to exactly one set"):
        split_data_dir(
            split_csv=metadata_csv,
            data_input=tmp_path / "raw" / "ham10000",
            data_output=data_output,
            verbose=False,
        )


def test_split_data_full_rebuilds_processed_output_root_with_single_metadata_csv(tmp_path: Path):
    data_input = tmp_path / "raw" / "ham10000"
    data_output = tmp_path / "processed" / "ham10000"
    raw_metadata_csv = data_input / "HAM10000_metadata.csv"
    config_file = tmp_path / "config" / "split.yaml"

    write_raw_metadata(
        raw_metadata_csv,
        [
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "dx": "mel"},
            {"lesion_id": "HAM_0001", "image_id": "ISIC_0002", "dx": "mel"},
            {"lesion_id": "HAM_0002", "image_id": "ISIC_0003", "dx": "nv"},
            {"lesion_id": "HAM_0003", "image_id": "ISIC_0004", "dx": "bcc"},
        ],
    )
    write_split_config(config_file, [("train", 0.5), ("test", 0.5)], seed=5)

    for image_id in ["ISIC_0001", "ISIC_0002", "ISIC_0003", "ISIC_0004"]:
        write_rgb_image(data_input / "HAM10000_images" / f"{image_id}.jpg")
        write_mask_image(data_input / "HAM10000_segmentations_lesion_tschandl" / f"{image_id}_segmentation.png")

    stale_file = data_output / "obsolete.txt"
    stale_nested_file = data_output / "legacy" / "stale.txt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("old output", encoding="utf-8")
    stale_nested_file.parent.mkdir(parents=True, exist_ok=True)
    stale_nested_file.write_text("stale", encoding="utf-8")

    metadata_csv = split_data_full(
        config_file=config_file,
        data_input=data_input,
        data_output=data_output,
        raw_metadata_csv=raw_metadata_csv,
        verbose=False,
    )

    metadata_df = pd.read_csv(metadata_csv)

    assert metadata_csv == data_output / "metadata.csv"
    assert metadata_csv.exists()
    assert sorted(path.name for path in data_output.glob("*.csv")) == ["metadata.csv"]
    assert not stale_file.exists()
    assert not stale_nested_file.exists()
    assert metadata_df[["image_id", "mb"]].to_dict("records") == [
        {"image_id": "ISIC_0001", "mb": "malignant"},
        {"image_id": "ISIC_0002", "mb": "malignant"},
        {"image_id": "ISIC_0003", "mb": "benign"},
        {"image_id": "ISIC_0004", "mb": "malignant"},
    ]
    assert metadata_df.groupby("lesion_id")["set"].nunique().to_dict() == {
        "HAM_0001": 1,
        "HAM_0002": 1,
        "HAM_0003": 1,
    }

    for _, row in metadata_df.iterrows():
        assert (data_output / "HAM10000_images" / row["set"] / f"{row['image_id']}.jpg").exists()
        assert (
            data_output
            / "HAM10000_segmentations_lesion_tschandl"
            / row["set"]
            / f"{row['image_id']}_segmentation.png"
        ).exists()
