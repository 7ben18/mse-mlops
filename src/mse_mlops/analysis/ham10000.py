from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mse_mlops.paths import find_repo_root

REPO_ROOT = find_repo_root(Path(__file__))
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "ham10000"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed" / "ham10000"

MAP_LESION_IMAGES = (
    PROCESSED_DATA_DIR / "all_lesion_images_mapping_HAM10000.csv"
)
IMG_DIR = RAW_DATA_DIR / "HAM10000_images"
MASK_DIR = RAW_DATA_DIR / "HAM10000_segmentations_lesion_tschandl"
METADATA = RAW_DATA_DIR / "HAM10000_metadata.csv"
EXT_METADATA = PROCESSED_DATA_DIR / "extended_HAM10000_metadata.csv"

type Triplet = tuple[Path, Path, str]


def get_ds(
    get_sample: bool = False,
    sample_size: int = 6,
    sample_seed: int = 13,
    sample_show: bool = False,
    img_dir: Path | str = IMG_DIR,
    mask_dir: Path | str = MASK_DIR,
) -> list[Triplet]:
    triplets, missing = _build_img_mask_triplets(img_dir, mask_dir)

    if missing:
        print(f"Missing masks: {len(missing)}")

    if not triplets:
        img_paths = _collect_image_paths(img_dir)
        img_stems = [p.stem for p in img_paths[:5]]
        mask_stems = [p.stem for p in list(Path(mask_dir).glob("*.png"))[:5]]
        raise RuntimeError(
            "No (image, mask) matches found.\n"
            f"Sample image: {img_stems}\n"
            f"Sample mask:  {mask_stems}\n"
        )

    sampled: list[Triplet] | None = None
    if get_sample or sample_show:
        sampled = _sample_items(triplets, sample_size, sample_seed)
        if sample_show:
            _plot_triplets_image_mask_grid(sampled)

    return sampled if get_sample else triplets


def metadata_ext(
    input_file: Path | str = METADATA,
    output_file: Path | str = EXT_METADATA,
    save: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    malignant_set = {"akiec", "bcc", "mel"}
    benign_set = {"bkl", "df", "nv", "vasc"}

    df = _add_mb_column(
        df,
        malignant_set=malignant_set,
        benign_set=benign_set,
        malignant_label="malignant",
        benign_label="benign",
    )

    unmapped = sorted(set(df["dx"].dropna()) - (malignant_set | benign_set))
    print("unmapped dx:", unmapped)

    if save:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def get_metadata(
    ids: str | list[str],
    csv_path: Path | str = EXT_METADATA,
    id_col: str = "image_id",
) -> pd.DataFrame:
    normalized_ids = list(ids) if not isinstance(ids, (str, bytes)) else [ids]
    df = load_metadata_csv(csv_path)

    if id_col not in df.columns:
        raise KeyError(
            f"'{id_col}' not found in CSV columns: {list(df.columns)}"
        )

    out = df[df[id_col].isin(normalized_ids)].copy()
    order = {key: index for index, key in enumerate(normalized_ids)}
    out["_order"] = out[id_col].map(order)
    out = out.sort_values("_order", na_position="last").drop(columns=["_order"])
    return out


def map_ds_metadata(
    sample_size: int = 6,
    sample_seed: int = 13,
    sample_show: bool = False,
    get_sample: bool = False,
    img_dir: Path | str = IMG_DIR,
    mask_dir: Path | str = MASK_DIR,
    metadata_csv: Path | str = EXT_METADATA,
) -> tuple[list[Triplet], pd.DataFrame]:
    ds = get_ds(
        sample_size=sample_size,
        sample_seed=sample_seed,
        sample_show=sample_show,
        get_sample=get_sample,
        img_dir=img_dir,
        mask_dir=mask_dir,
    )

    ids = [img_id for _, _, img_id in ds]
    meta_df = get_metadata(ids, csv_path=metadata_csv)

    if sample_show:
        _display_df(meta_df)

    return ds, meta_df


def map_lesion_images(
    df: pd.DataFrame,
    min_img_num: int = 2,
    verbose: bool = True,
) -> tuple[pd.Series, dict[str, list[str]]]:
    counts, lesion_map = _compute_lesion_image_stats(df)

    _print_lesion_global_stats(df, counts)
    _print_lesion_distribution(counts)

    filt = counts[counts >= min_img_num]
    print(f"\nThreshold: >= {min_img_num} images/lesion")
    print(
        f"Matching lesions: {len(filt)} / {len(counts)} ({len(filt) / len(counts) * 100:.2f}%)"
    )

    if verbose:
        for lesion_id, count in filt.items():
            images = lesion_map[lesion_id]
            print(f"lesion {lesion_id} has {count} images: {', '.join(images)}")

    return filt, {lesion_id: lesion_map[lesion_id] for lesion_id in filt.index}


def build_lesion_images_frame(
    lesion_images: dict[str, list[str]],
) -> pd.DataFrame:
    return pd.DataFrame({
        "lesion_id": list(lesion_images.keys()),
        "images": [
            "{" + ", ".join(images) + "}" for images in lesion_images.values()
        ],
    }).sort_values("lesion_id")


def save_lesion_images_frame(
    lesion_images: dict[str, list[str]],
    output_file: Path | str = MAP_LESION_IMAGES,
) -> pd.DataFrame:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = build_lesion_images_frame(lesion_images)
    out_df.to_csv(output_path, index=False)
    return out_df


def parse_images_field(images_field: str) -> list[str]:
    parsed = str(images_field).strip()
    if not parsed:
        return []

    parsed = parsed.strip("{}")
    parsed = parsed.strip("[]")
    parsed = parsed.replace("'", "").replace('"', "")

    parts = [part.strip() for part in parsed.split(",")]
    return [part for part in parts if part]


def load_map_lesion_images(file_path: Path | str) -> pd.DataFrame:
    lesion_df = pd.read_csv(file_path)

    required = {"lesion_id", "images"}
    missing = required - set(lesion_df.columns)
    if missing:
        raise KeyError(
            f"lesion->images CSV missing columns: {missing}. Found: {list(lesion_df.columns)}"
        )

    if lesion_df.empty:
        raise RuntimeError("lesion->images CSV is empty.")

    return lesion_df


def pick_random_lesion(lesion_df: pd.DataFrame) -> tuple[str, list[str]]:
    rng = random.Random()
    row = lesion_df.iloc[rng.randrange(len(lesion_df))]

    lesion_id = row["lesion_id"]
    image_ids = parse_images_field(row["images"])

    if not image_ids:
        raise RuntimeError(
            f"Selected lesion {lesion_id} has no images listed in CSV."
        )

    return lesion_id, image_ids


def find_image_paths_for_ids(
    img_ids: list[str],
    img_dir: Path | str,
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".JPG", ".JPEG"),
) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    image_dir = Path(img_dir)
    for img_id in img_ids:
        found = None
        for ext in exts:
            candidate = image_dir / f"{img_id}{ext}"
            if candidate.exists():
                found = candidate
                break
        out[img_id] = found
    return out


def plot_images_grid(
    img_id_to_path: dict[str, Path | None],
    lesion_id: str | None = None,
    max_cols: int = 5,
) -> None:
    img_ids = list(img_id_to_path.keys())
    n_images = len(img_ids)

    cols = min(max_cols, max(1, n_images))
    rows = math.ceil(n_images / cols)
    _fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))

    flattened_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in flattened_axes:
        ax.axis("off")

    for index, img_id in enumerate(img_ids):
        ax = flattened_axes[index]
        path = img_id_to_path[img_id]

        if path is None:
            ax.set_title(f"{img_id}\n(MISSING)", fontsize=10)
            continue

        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
        ax.imshow(rgb_image)
        ax.set_title(img_id, fontsize=10)
        ax.axis("off")

    if lesion_id is not None:
        plt.suptitle(f"Lesion: {lesion_id}", y=1.02, fontsize=14)

    plt.tight_layout()
    plt.show()


def load_metadata_csv(metadata_csv_path: Path | str) -> pd.DataFrame:
    meta_df = pd.read_csv(metadata_csv_path)
    if meta_df.empty:
        raise RuntimeError("Metadata CSV is empty.")
    return meta_df


def get_metadata_for_lesion(
    meta_df: pd.DataFrame, lesion_id: str, lesion_col: str = "lesion_id"
) -> pd.DataFrame:
    if lesion_col not in meta_df.columns:
        raise KeyError(
            f"Metadata missing column '{lesion_col}'. Found: {list(meta_df.columns)}"
        )
    return meta_df[meta_df[lesion_col] == lesion_id].copy()


def show_random_lesion_images_and_metadata(
    lesion_images_csv_path: Path | str,
    metadata_csv_path: Path | str,
    img_dir: Path | str,
    max_cols: int = 5,
) -> tuple[str, list[str], pd.DataFrame]:
    lesion_df = load_map_lesion_images(lesion_images_csv_path)
    lesion_id, img_ids = pick_random_lesion(lesion_df)

    img_id_to_path = find_image_paths_for_ids(img_ids, img_dir=img_dir)
    plot_images_grid(img_id_to_path, lesion_id=lesion_id, max_cols=max_cols)

    meta_df = load_metadata_csv(metadata_csv_path)
    meta_rows = get_metadata_for_lesion(meta_df, lesion_id)
    _display_df(meta_rows)

    return lesion_id, img_ids, meta_rows


def get_image_ids_for_lesion(
    lesion_df: pd.DataFrame, lesion_id: str
) -> list[str]:
    hit = lesion_df[lesion_df["lesion_id"] == lesion_id]
    if hit.empty:
        return []
    return parse_images_field(hit.iloc[0]["images"])


def get_lesion_info(
    lesion_images_csv_path: Path | str,
    metadata_csv_path: Path | str,
    img_dir: Path | str,
    lesion_id: str | None = None,
    max_cols: int = 5,
    prefer_mapping_csv: bool = True,
) -> tuple[str, list[str], pd.DataFrame]:
    lesion_df = load_map_lesion_images(lesion_images_csv_path)

    if lesion_id is None:
        lesion_id, img_ids = pick_random_lesion(lesion_df)
    else:
        lesion_id = str(lesion_id).strip()
        img_ids = (
            get_image_ids_for_lesion(lesion_df, lesion_id)
            if prefer_mapping_csv
            else []
        )

    meta_df = load_metadata_csv(metadata_csv_path)
    meta_rows = get_metadata_for_lesion(meta_df, lesion_id)

    if meta_rows.empty:
        raise RuntimeError(f"No metadata rows found for lesion_id={lesion_id}")

    if not img_ids:
        if "image_id" not in meta_rows.columns:
            raise KeyError(
                "Metadata does not have 'image_id' column, can't derive images."
            )
        img_ids = sorted(meta_rows["image_id"].dropna().unique().tolist())

    if not img_ids:
        raise RuntimeError(
            f"Lesion {lesion_id} has no image_ids (mapping CSV + metadata both empty)."
        )

    img_id_to_path = find_image_paths_for_ids(img_ids, img_dir=img_dir)
    plot_images_grid(img_id_to_path, lesion_id=lesion_id, max_cols=max_cols)
    _display_df(meta_rows)
    return lesion_id, img_ids, meta_rows


def _collect_image_paths(
    img_dir: Path | str,
    patterns: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"),
) -> list[Path]:
    image_dir = Path(img_dir)
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(image_dir.glob(pattern))
    return sorted(image_paths)


def _build_mask_map(mask_dir: Path | str) -> dict[str, Path]:
    masks_dir = Path(mask_dir)
    mask_map: dict[str, Path] = {}

    for mask_path in masks_dir.glob("*.png"):
        stem = mask_path.stem
        keys = {stem}

        if stem.endswith("_segmentation"):
            keys.add(stem[: -len("_segmentation")])

        for key in keys:
            mask_map[key.lower()] = mask_path

    return mask_map


def _build_img_mask_triplets(
    img_dir: Path | str, mask_dir: Path | str
) -> tuple[list[Triplet], list[str]]:
    img_paths = _collect_image_paths(img_dir)
    mask_map = _build_mask_map(mask_dir)

    triplets: list[Triplet] = []
    missing: list[str] = []

    for img_path in img_paths:
        img_id = img_path.stem
        mask_path = mask_map.get(img_id.lower()) or mask_map.get(
            (img_id + "_segmentation").lower()
        )

        if mask_path is None:
            missing.append(img_path.name)
            continue

        triplets.append((img_path, mask_path, img_id))

    return triplets, missing


def _sample_items(
    items: list[Triplet], sample_size: int, sample_seed: int
) -> list[Triplet]:
    rng = random.Random(sample_seed)
    k = min(int(sample_size), len(items))
    return rng.sample(items, k)


def _plot_triplets_image_mask_grid(
    triplets: list[Triplet], title: str = "image vs mask"
) -> None:
    n_triplets = len(triplets)
    if n_triplets <= 0:
        return

    _fig, axes = plt.subplots(2, n_triplets, figsize=(4.5 * n_triplets, 9))

    if n_triplets == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for index, (img_path, mask_path, img_id) in enumerate(triplets):
        with Image.open(img_path) as image:
            rgb_image = image.convert("RGB")
        with Image.open(mask_path) as mask:
            mask_arr = np.array(mask)

        axes[0, index].imshow(rgb_image)
        axes[0, index].set_title(img_id, fontsize=18)
        axes[0, index].axis("off")

        if mask_arr.ndim == 2:
            axes[1, index].imshow(
                mask_arr, cmap="gray", interpolation="nearest"
            )
        else:
            axes[1, index].imshow(mask_arr, interpolation="nearest")
        axes[1, index].axis("off")

    plt.tight_layout()
    plt.suptitle(title, fontsize=20)
    plt.show()


def _add_mb_column(
    df: pd.DataFrame,
    malignant_set: set[str],
    benign_set: set[str],
    malignant_label: str = "malignant",
    benign_label: str = "benign",
) -> pd.DataFrame:
    labeled = df.copy()
    labeled["mb"] = pd.NA
    labeled.loc[labeled["dx"].isin(malignant_set), "mb"] = malignant_label
    labeled.loc[labeled["dx"].isin(benign_set), "mb"] = benign_label
    return labeled


def _compute_lesion_image_stats(
    df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, list[str]]]:
    grouped = df.groupby("lesion_id")["image_id"]
    lesion_map = grouped.apply(
        lambda values: sorted(values.dropna().unique())
    ).to_dict()
    counts = grouped.nunique(dropna=True).sort_values(ascending=False)
    return counts, lesion_map


def _print_lesion_global_stats(df: pd.DataFrame, counts: pd.Series) -> None:
    print(f"lesions: {counts.size}")
    print(f"total images (unique image_id): {df['image_id'].nunique()}")
    print(
        f"mean images/lesion: {counts.mean():.2f} | median: {counts.median():.0f} | max: {counts.max()}"
    )


def _print_lesion_distribution(counts: pd.Series) -> None:
    dist = counts.value_counts().sort_index()
    for image_count, lesion_count in dist.items():
        print(f"{lesion_count:5d} lesions have {image_count} images")


def _display_df(df: pd.DataFrame) -> None:
    try:
        from IPython.display import display

        display(df)
    except Exception:
        try:
            print(df.to_markdown(index=False))
        except Exception:
            print(df)


__all__ = [
    "EXT_METADATA",
    "IMG_DIR",
    "MAP_LESION_IMAGES",
    "MASK_DIR",
    "METADATA",
    "Triplet",
    "build_lesion_images_frame",
    "find_image_paths_for_ids",
    "get_ds",
    "get_image_ids_for_lesion",
    "get_lesion_info",
    "get_metadata",
    "get_metadata_for_lesion",
    "load_map_lesion_images",
    "load_metadata_csv",
    "map_ds_metadata",
    "map_lesion_images",
    "metadata_ext",
    "parse_images_field",
    "pick_random_lesion",
    "plot_images_grid",
    "save_lesion_images_frame",
    "show_random_lesion_images_and_metadata",
]
