from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import mse_mlops.paths as paths

type Triplet = tuple[Path, Path, str]


def get_ds(
    get_sample: bool = False,
    sample_size: int = 6,
    sample_seed: int = 13,
    sample_show: bool = False,
    img_dir: Path | str = paths.RAW_DATA_DIR / paths.HAM_DIR / paths.IMG_DIR,
    mask_dir: Path | str = paths.RAW_DATA_DIR / paths.HAM_DIR / paths.MASK_DIR,
) -> list[Triplet]:
    """
    Generating random (repeatable) sample of images and masks
    """
    triplets, missing = _build_img_mask_triplets(img_dir, mask_dir)

    if missing:
        print(f"Missing masks: {len(missing)}")

    if not triplets:
        img_paths = _collect_image_paths(img_dir)
        img_stems = [p.stem for p in img_paths[:5]]
        mask_stems = [p.stem for p in list(Path(mask_dir).glob("*.png"))[:5]]
        raise RuntimeError(f"No (image, mask) matches found.\nSample image: {img_stems}\nSample mask:  {mask_stems}\n")

    sampled: list[Triplet] | None = None
    if get_sample or sample_show:
        sampled = _sample_items(triplets, sample_size, sample_seed)
        if sample_show:
            _plot_triplets_image_mask_grid(sampled)

    return sampled if get_sample else triplets


def metadata_ext(
    input_file: Path | str = Path(paths.RAW_DATA_DIR / paths.HAM_DIR / paths.METADATA),
) -> pd.DataFrame:
    """
    Extending raw metadata with new labels (malignant and benign == mb)
    """
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
    return df


def get_metadata(
    ids: str | list[str],
    csv_path: Path | str = paths.EXT_METADATA,
    id_col: str = "image_id",
) -> pd.DataFrame:
    """
    Get metadata for given ids
    """
    normalized_ids = list(ids) if not isinstance(ids, (str, bytes)) else [ids]
    df = load_metadata_csv(csv_path)

    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found in CSV columns: {list(df.columns)}")

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
    img_dir: Path | str = Path(paths.RAW_DATA_DIR / paths.HAM_DIR / paths.IMG_DIR),
    mask_dir: Path | str = Path(paths.RAW_DATA_DIR / paths.HAM_DIR / paths.MASK_DIR),
    metadata_csv: Path | str = paths.EXT_METADATA,
) -> tuple[list[Triplet], pd.DataFrame]:
    """
    Generating random (repeatable) sample of images and masks. Map them to metadata.
    """
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
    """
    Map lesion to it's images.
    """
    counts, lesion_map = _compute_lesion_image_stats(df)

    _print_lesion_global_stats(df, counts)
    _print_lesion_distribution(counts)

    filt = counts[counts >= min_img_num]
    print(f"\nThreshold: >= {min_img_num} images/lesion")
    print(f"Matching lesions: {len(filt)} / {len(counts)} ({len(filt) / len(counts) * 100:.2f}%)")

    if verbose:
        for l_id, count in filt.items():
            images = lesion_map[l_id]
            print(f"lesion {l_id} has {count} images: {', '.join(images)}")

    return filt, {lesion_id: lesion_map[lesion_id] for lesion_id in filt.index}


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
    title: str = "Lesion Images",
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


def get_metadata_for_lesion(meta_df: pd.DataFrame, lesion_id: str, lesion_col: str = "lesion_id") -> pd.DataFrame:
    if lesion_col not in meta_df.columns:
        raise KeyError(f"Metadata missing column '{lesion_col}'. Found: {list(meta_df.columns)}")
    return meta_df[meta_df[lesion_col] == lesion_id].copy()


def show_random_lesion_images_and_metadata(
    metadata_csv_path: Path | str = paths.EXT_METADATA,
    img_dir: Path | str = Path(paths.RAW_DATA_DIR / paths.HAM_DIR / paths.IMG_DIR),
    max_cols: int = 5,
) -> tuple[str, list[str], pd.DataFrame]:
    return get_lesion_info(
        metadata_csv_path=metadata_csv_path,
        img_dir=img_dir,
        lesion_id=None,
        max_cols=max_cols,
    )


def get_image_ids_for_lesion(
    meta_df: pd.DataFrame,
    lesion_id: str,
    lesion_col: str = "lesion_id",
    image_col: str = "image_id",
) -> list[str]:
    if lesion_col not in meta_df.columns:
        raise KeyError(f"Metadata missing column '{lesion_col}'. Found: {list(meta_df.columns)}")
    if image_col not in meta_df.columns:
        raise KeyError(f"Metadata missing column '{image_col}'. Found: {list(meta_df.columns)}")

    hit = meta_df[meta_df[lesion_col] == lesion_id]
    if hit.empty:
        return []

    return sorted(hit[image_col].dropna().astype(str).unique().tolist())


def get_lesion_info(
    metadata_csv_path: Path | str = paths.EXT_METADATA,
    img_dir: Path | str = Path(paths.RAW_DATA_DIR / paths.HAM_DIR / paths.IMG_DIR),
    lesion_id: str | None = None,
    max_cols: int = 5,
) -> tuple[str, list[str], pd.DataFrame]:
    meta_df = load_metadata_csv(metadata_csv_path)

    if lesion_id is None:
        lesion_id = _pick_random_lesion_id(meta_df)
    else:
        lesion_id = str(lesion_id).strip()

    meta_rows = get_metadata_for_lesion(meta_df, lesion_id)

    if meta_rows.empty:
        raise RuntimeError(f"No metadata rows found for lesion_id={lesion_id}")

    img_ids = get_image_ids_for_lesion(meta_rows, lesion_id)
    if not img_ids:
        raise RuntimeError(f"Lesion {lesion_id} has no image_ids in metadata.")

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


def _build_img_mask_triplets(img_dir: Path | str, mask_dir: Path | str) -> tuple[list[Triplet], list[str]]:
    img_paths = _collect_image_paths(img_dir)
    mask_map = _build_mask_map(mask_dir)

    triplets: list[Triplet] = []
    missing: list[str] = []

    for img_path in img_paths:
        img_id = img_path.stem
        mask_path = mask_map.get(img_id.lower()) or mask_map.get((img_id + "_segmentation").lower())

        if mask_path is None:
            missing.append(img_path.name)
            continue

        triplets.append((img_path, mask_path, img_id))

    return triplets, missing


def _sample_items(items: list[Triplet], sample_size: int, sample_seed: int) -> list[Triplet]:
    rng = random.Random(sample_seed)
    k = min(int(sample_size), len(items))
    return rng.sample(items, k)


def _pick_random_lesion_id(meta_df: pd.DataFrame, lesion_col: str = "lesion_id") -> str:
    if lesion_col not in meta_df.columns:
        raise KeyError(f"Metadata missing column '{lesion_col}'. Found: {list(meta_df.columns)}")

    lesion_ids = meta_df[lesion_col].dropna().astype(str).str.strip()
    lesion_ids = lesion_ids[lesion_ids != ""].drop_duplicates().tolist()
    if not lesion_ids:
        raise RuntimeError("Metadata does not contain any lesion_id values.")

    rng = random.Random()
    return lesion_ids[rng.randrange(len(lesion_ids))]


def _plot_triplets_image_mask_grid(
    triplets: list[Triplet], title: str = "Image vs Mask"
) -> None:
    n_triplets = len(triplets)

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
            axes[1, index].imshow(mask_arr, cmap="gray", interpolation="nearest")
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
    lesion_map = grouped.apply(lambda values: sorted(values.dropna().unique())).to_dict()
    counts = grouped.nunique(dropna=True).sort_values(ascending=False)
    return counts, lesion_map


def _print_lesion_global_stats(df: pd.DataFrame, counts: pd.Series) -> None:
    print(f"lesions: {counts.size}")
    print(f"total images (unique image_id): {df['image_id'].nunique()}")
    print(f"mean images/lesion: {counts.mean():.2f} | median: {counts.median():.0f} | max: {counts.max()}")


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
    "Triplet",
    "find_image_paths_for_ids",
    "get_ds",
    "get_image_ids_for_lesion",
    "get_lesion_info",
    "get_metadata",
    "get_metadata_for_lesion",
    "load_metadata_csv",
    "map_ds_metadata",
    "map_lesion_images",
    "metadata_ext",
    "plot_images_grid",
    "show_random_lesion_images_and_metadata",
]
