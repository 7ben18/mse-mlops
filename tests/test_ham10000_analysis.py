from pathlib import Path

import mse_mlops.analysis.ham10000 as ham
import pandas as pd
from PIL import Image

from mse_mlops.analysis.ham10000 import (
    find_image_paths_for_ids,
    get_ds,
    get_metadata,
    map_lesion_images,
    metadata_ext,
)


def write_rgb_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def write_mask_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_get_ds_matches_images_and_masks(tmp_path: Path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"

    write_rgb_image(img_dir / "ISIC_0001.jpg")
    write_rgb_image(img_dir / "ISIC_0002.jpg")
    write_mask_image(mask_dir / "ISIC_0001_segmentation.png")
    write_mask_image(mask_dir / "ISIC_0002.png")

    triplets = get_ds(img_dir=img_dir, mask_dir=mask_dir)

    assert [(img.name, mask.name, image_id) for img, mask, image_id in triplets] == [
        ("ISIC_0001.jpg", "ISIC_0001_segmentation.png", "ISIC_0001"),
        ("ISIC_0002.jpg", "ISIC_0002.png", "ISIC_0002"),
    ]


def test_metadata_ext_creates_mb_column_without_writing_output(tmp_path: Path):
    input_csv = tmp_path / "HAM10000_metadata.csv"
    pd.DataFrame([
        {"image_id": "ISIC_0001", "dx": "mel"},
        {"image_id": "ISIC_0002", "dx": "nv"},
    ]).to_csv(input_csv, index=False)

    extended = metadata_ext(input_file=input_csv)

    assert extended["mb"].tolist() == ["malignant", "benign"]


def test_get_metadata_preserves_requested_order(tmp_path: Path):
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame([
        {"image_id": "ISIC_0001", "dx": "mel"},
        {"image_id": "ISIC_0002", "dx": "nv"},
        {"image_id": "ISIC_0003", "dx": "bcc"},
    ]).to_csv(metadata_csv, index=False)

    selected = get_metadata(["ISIC_0003", "ISIC_0001"], csv_path=metadata_csv)

    assert selected["image_id"].tolist() == ["ISIC_0003", "ISIC_0001"]


def test_map_lesion_images_and_build_frame(tmp_path: Path):
    metadata = pd.DataFrame([
        {"lesion_id": "HAM_1", "image_id": "ISIC_0001"},
        {"lesion_id": "HAM_1", "image_id": "ISIC_0002"},
        {"lesion_id": "HAM_2", "image_id": "ISIC_0003"},
    ])

    counts, lesion_images = map_lesion_images(metadata, min_img_num=2, verbose=False)

    assert counts.index.tolist() == ["HAM_1"]
    assert lesion_images == {"HAM_1": ["ISIC_0001", "ISIC_0002"]}
    assert lesion_images["HAM_1"] == ["ISIC_0001", "ISIC_0002"]


def test_find_image_paths_for_ids(tmp_path: Path):
    img_dir = tmp_path / "images"
    write_rgb_image(img_dir / "ISIC_0001.jpg")
    write_rgb_image(img_dir / "ISIC_0002.jpeg")

    found = find_image_paths_for_ids(["ISIC_0001", "ISIC_0002"], img_dir=img_dir)

    assert found["ISIC_0001"] == img_dir / "ISIC_0001.jpg"
    assert found["ISIC_0002"] == img_dir / "ISIC_0002.jpeg"


def test_get_lesion_info_derives_image_ids_from_metadata_only(tmp_path: Path, monkeypatch):
    metadata_csv = tmp_path / "metadata.csv"
    img_dir = tmp_path / "images"
    pd.DataFrame([
        {"lesion_id": "HAM_1", "image_id": "ISIC_0001", "dx": "mel"},
        {"lesion_id": "HAM_1", "image_id": "ISIC_0002", "dx": "mel"},
        {"lesion_id": "HAM_2", "image_id": "ISIC_0003", "dx": "nv"},
    ]).to_csv(metadata_csv, index=False)
    write_rgb_image(img_dir / "ISIC_0001.jpg")
    write_rgb_image(img_dir / "ISIC_0002.jpg")
    write_rgb_image(img_dir / "ISIC_0003.jpg")

    monkeypatch.setattr(ham, "plot_images_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(ham, "_display_df", lambda df: None)

    lesion_id, img_ids, meta_rows = ham.get_lesion_info(
        metadata_csv_path=metadata_csv,
        img_dir=img_dir,
        lesion_id="HAM_1",
    )

    assert lesion_id == "HAM_1"
    assert img_ids == ["ISIC_0001", "ISIC_0002"]
    assert meta_rows["image_id"].tolist() == ["ISIC_0001", "ISIC_0002"]


def test_show_random_lesion_images_and_metadata_uses_metadata_only(tmp_path: Path, monkeypatch):
    metadata_csv = tmp_path / "metadata.csv"
    img_dir = tmp_path / "images"
    pd.DataFrame([
        {"lesion_id": "HAM_1", "image_id": "ISIC_0001", "dx": "mel"},
        {"lesion_id": "HAM_1", "image_id": "ISIC_0002", "dx": "mel"},
        {"lesion_id": "HAM_2", "image_id": "ISIC_0003", "dx": "nv"},
    ]).to_csv(metadata_csv, index=False)
    write_rgb_image(img_dir / "ISIC_0001.jpg")
    write_rgb_image(img_dir / "ISIC_0002.jpg")
    write_rgb_image(img_dir / "ISIC_0003.jpg")

    monkeypatch.setattr(ham, "_pick_random_lesion_id", lambda meta_df, lesion_col="lesion_id": "HAM_2")
    monkeypatch.setattr(ham, "plot_images_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(ham, "_display_df", lambda df: None)

    lesion_id, img_ids, meta_rows = ham.show_random_lesion_images_and_metadata(
        metadata_csv_path=metadata_csv,
        img_dir=img_dir,
    )

    assert lesion_id == "HAM_2"
    assert img_ids == ["ISIC_0003"]
    assert meta_rows["lesion_id"].unique().tolist() == ["HAM_2"]
