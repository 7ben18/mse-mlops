"""
HAM10000 EDA helpers.

Assumptions
-----------
- Images live in IMG_DIR and are JPEG files: {image_id}.jpg/.jpeg
- Masks live in MASK_DIR and are PNG files:
    - either {image_id}.png
    - or {image_id}_segmentation.png (both are supported)
- Metadata CSV contains at least:
    - image_id: e.g. ISIC_0027419
    - lesion_id: e.g. HAM_0000118
    - dx: diagnosis label

Core data model
---------------
Triplet = (img_path, mask_path, img_id)

Quick start
----------------------
>>> ds = get_ds(get_sample=True, sample_size=4, sample_seed=42, sample_show=True)
>>> ds, meta = map_ds_metadata(get_sample=True, sample_size=8, sample_seed=0, sample_show=False)
>>> ext = metadata_ext(save=False)  # adds mb column
>>> counts, lesion_map = map_lesion_images(ext, min_img_num=2, verbose=False)
>>> get_lesion_info(MAP_LESION_IMAGES, METADATA, IMG_DIR, lesion_id="HAM_0006469")
"""

import math
import random
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent
MAP_LESION_IMAGES = BASE / "data/all_lesion_images_mapping_HAM10000.csv"
IMG_DIR = BASE / "data/HAM10000_images"
MASK_DIR = BASE / "data/HAM10000_segmentations_lesion_tschandl"
METADATA = BASE / "data/HAM10000_metadata.csv"
EXT_METADATA = BASE / "data/extended_HAM10000_metadata.csv"

Triplet: TypeAlias = tuple[Path, Path, str]


def get_ds(get_sample=False,
           sample_size=6,
           sample_seed=13,
           sample_show=False,
           img_dir=IMG_DIR,
           mask_dir=MASK_DIR):
    """
    Build dataset triplets (image, mask, image_id), optionally sampling and plotting.

    Args:
        get_sample: If True, return only a random sample of triplets
        sample_size: Sample size when sampling or plotting
        sample_seed: Seed controlling sampling (reproducible)
        sample_show: If True, plots a 2-row grid (images top, masks bottom)
        img_dir: Directory containing JPEG images
        mask_dir: Directory containing PNG masks

    Returns:
        If get_sample is True:
            List[Triplet] of length min(sample_size, dataset_size)
        Else:
            Full List[Triplet]

    Raises:
        RuntimeError: If no (image, mask) matches are found

    Notes:
        - Mask matching is done by stem:
            img_id == image file stem
            mask file stem can be either img_id or img_id + "_segmentation"
        - Prints a warning if masks are missing for some images

    Examples:
        >>> ds = get_ds(get_sample=True, sample_size=4, sample_seed=54, sample_show=True)
    """
    triplets, missing = _build_img_mask_triplets(img_dir, mask_dir)

    if len(missing) != 0: print(f"Missing masks: {len(missing)}")

    if not triplets:
        img_paths = _collect_image_paths(IMG_DIR)
        img_stems = [p.stem for p in img_paths[:5]]
        mask_stems = [p.stem for p in list(Path(mask_dir).glob("*.png"))[:5]]
        raise RuntimeError(
            "No (image, mask) matches found.\n"
            f"Sample image: {img_stems}\n"
            f"Sample mask:  {mask_stems}\n"
        )

    sampled = None
    if get_sample or sample_show:
        sampled = _sample_items(triplets, sample_size, sample_seed)

        if sample_show:
            _plot_triplets_image_mask_grid(sampled)

    return sampled if get_sample else triplets


def metadata_ext(input_file=METADATA, output_file=EXT_METADATA, save=True) -> pd.DataFrame:
    """
        Load metadata and add a binary label column 'mb' (malignant/benign).

        Args:
            input_file: Path to metadata CSV (must include 'dx').
            output_file: Save destination for extended CSV.
            save: If True, writes CSV to output_file.

        Returns:
            DataFrame with a new 'mb' column.

        Raises:
            KeyError: If 'dx' column is missing.

        Notes:
            Mapping (HAM10000):
                malignant: {'akiec', 'bcc', 'mel'}
                benign:    {'bkl', 'df', 'nv', 'vasc'}
            Prints any dx labels that are unmapped.

        Examples:
            >>> ext = metadata_ext(save=False)
            >>> ext['mb'].value_counts()
        """
    df = pd.read_csv(input_file)

    malignant_set = {"akiec", "bcc", "mel"}
    benign_set = {"bkl", "df", "nv", "vasc"}

    df = _add_mb_column(df, malignant_set=malignant_set, benign_set=benign_set,
                        malignant_label="malignant", benign_label="benign")

    unmapped = sorted(set(df["dx"].dropna()) - (malignant_set | benign_set))
    print("unmapped dx:", unmapped)

    if save: df.to_csv(output_file, index=False)
    return df


def get_metadata(ids, csv_path=EXT_METADATA, id_col="image_id") -> pd.DataFrame:
    """
        Return metadata rows for specific image IDs, preserving input order

        Args:
            ids: Iterable of image_id strings (e.g. ["ISIC_..."]).
            csv_path: Metadata CSV path.
            id_col: Column name containing image IDs.

        Returns:
            DataFrame containing all columns for the matching rows.
            Rows are ordered to match `ids` (where possible).

        Raises:
            KeyError: If id_col is missing.
        """
    ids = list(ids) if not isinstance(ids, (str, bytes)) else [ids]

    df = load_metadata_csv(csv_path)

    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found in CSV columns: {list(df.columns)}")

    out = df[df[id_col].isin(ids)].copy()

    order = {k: i for i, k in enumerate(ids)}
    out["_order"] = out[id_col].map(order)
    out = out.sort_values("_order", na_position="last").drop(columns=["_order"])

    return out


def map_ds_metadata(sample_size=6, sample_seed=13, sample_show=False, get_sample=False):
    """
    Build dataset triplets and fetch their corresponding metadata rows.

    Args:
        sample_size/sample_seed/sample_show/get_sample: forwarded to get_ds()

    Returns:
        (ds, meta_df)
            ds: List[Triplet] (sample or full, depending on get_sample)
            meta_df: DataFrame filtered to ds image_ids (same order as ds)
    """
    ds = get_ds(
        sample_size=sample_size,
        sample_seed=sample_seed,
        sample_show=sample_show,
        get_sample=get_sample,
    )

    ids = [img_id for _, _, img_id in ds]
    meta_df = get_metadata(ids)

    if sample_show:
        _display_df(meta_df)

    return ds, meta_df


def map_lesion_images(df, min_img_num=2, verbose=True) -> tuple[pd.Series, dict[str, list[str]]]:
    """
    Build lesion -> images mapping and compute image-count stats per lesion.

    Args:
        df: Metadata DataFrame containing 'lesion_id' and 'image_id'.
        min_img_num: Minimum number of images for lesion to be included in output.
        verbose: If True, prints each qualifying lesion and its images.

    Returns:
        filt_counts:
            pd.Series indexed by lesion_id, values = number of unique images.
        filt_map:
            dict[lesion_id] -> sorted list[image_id] for lesions passing threshold.

    Notes:
        Uses unique image_ids per lesion (nunique).
    """
    counts, lesion_map = _compute_lesion_image_stats(df)

    _print_lesion_global_stats(df, counts)
    _print_lesion_distribution(counts)

    filt = counts[counts >= min_img_num]
    print(f"\nThreshold: >= {min_img_num} images/lesion")
    print(f"Matching lesions: {len(filt)} / {len(counts)} ({len(filt) / len(counts) * 100:.2f}%)")

    if verbose:
        for lesion_id, k in filt.items():
            imgs = lesion_map[lesion_id]
            print(f"lesion {lesion_id} has {k} images: {', '.join(imgs)}")

    return filt, {lid: lesion_map[lid] for lid in filt.index}


def parse_images_field(images_field: str) -> list[str]:
    s = str(images_field).strip()
    if not s:
        return []

    s = s.strip()
    s = s.strip("{}")
    s = s.strip("[]")
    s = s.replace("'", "").replace('"', "")

    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _collect_image_paths(img_dir, patterns=("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")):
    img_dir = Path(img_dir)
    img_paths = []
    for pat in patterns:
        img_paths.extend(img_dir.glob(pat))
    return sorted(img_paths)


def _build_mask_map(mask_dir):
    mask_dir = Path(mask_dir)
    mask_map = {}

    for m in mask_dir.glob("*.png"):
        stem = m.stem
        keys = {stem}

        if stem.endswith("_segmentation"):
            keys.add(stem[: -len("_segmentation")])

        for k in keys:
            mask_map[k.lower()] = m

    return mask_map


def _build_img_mask_triplets(img_dir, mask_dir):
    img_paths = _collect_image_paths(img_dir)
    mask_map = _build_mask_map(mask_dir)

    triplets = []
    missing = []

    for img_p in img_paths:
        img_id = img_p.stem
        mask_p = mask_map.get(img_id.lower()) or mask_map.get((img_id + "_segmentation").lower())

        if mask_p is None:
            missing.append(img_p.name)
            continue

        triplets.append((img_p, mask_p, img_id))

    return triplets, missing


def _sample_items(items, sample_size, sample_seed):
    rng = random.Random(sample_seed)
    k = min(int(sample_size), len(items))
    return rng.sample(items, k)


def _plot_triplets_image_mask_grid(triplets, title="image vs mask"):
    k = len(triplets)
    if k <= 0:
        return

    fig, axes = plt.subplots(2, k, figsize=(4.5 * k, 9))

    # normalize to (2,1) when k==1 so the loop stays simple
    if k == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, (img_p, mask_p, img_id) in enumerate(triplets):
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)
        mask_arr = np.array(mask)

        axes[0, i].imshow(img)
        axes[0, i].set_title(img_id, fontsize=18)
        axes[0, i].axis("off")

        if mask_arr.ndim == 2:
            axes[1, i].imshow(mask_arr, cmap="gray", interpolation="nearest")
        else:
            axes[1, i].imshow(mask_arr, interpolation="nearest")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.suptitle(title, fontsize=20)
    plt.show()


def _add_mb_column(df, malignant_set, benign_set, malignant_label="malignant", benign_label="benign"):
    df = df.copy()
    df["mb"] = pd.NA
    df.loc[df["dx"].isin(malignant_set), "mb"] = malignant_label
    df.loc[df["dx"].isin(benign_set), "mb"] = benign_label
    return df


def _compute_lesion_image_stats(df):
    grp = df.groupby("lesion_id")["image_id"]

    lesion_map = grp.apply(lambda s: sorted(s.dropna().unique())).to_dict()
    counts = grp.nunique(dropna=True).sort_values(ascending=False)

    return counts, lesion_map


def _print_lesion_global_stats(df, counts):
    print(f"lesions: {counts.size}")
    print(f"total images (unique image_id): {df['image_id'].nunique()}")
    print(f"mean images/lesion: {counts.mean():.2f} | median: {counts.median():.0f} | max: {counts.max()}")


def _print_lesion_distribution(counts):
    dist = counts.value_counts().sort_index()
    for k, n_lesions in dist.items():
        print(f"{n_lesions:5d} lesions have {k} images")


def _print_lesion_listing(counts, lesion_map):
    for lesion_id, k in counts.items():
        imgs = lesion_map[lesion_id]
        print(f"lesion {lesion_id} has {k} images: {', '.join(imgs)}")


def _display_df(df):
    try:
        from IPython.display import display
        display(df)
    except Exception:
        try:
            print(df.to_markdown(index=False))
        except Exception:
            print(df)


def load_map_lesion_images(file_path):
    lesion_df = pd.read_csv(file_path)

    required = {"lesion_id", "images"}
    missing = required - set(lesion_df.columns)
    if missing:
        raise KeyError(f"lesion->images CSV missing columns: {missing}. Found: {list(lesion_df.columns)}")

    if lesion_df.empty:
        raise RuntimeError("lesion->images CSV is empty.")

    return lesion_df


def pick_random_lesion(lesion_df):
    """
    Pick a random lesion row from lesion->images mapping DataFrame.

    Notes:
        No seed argument.
    """
    rng = random.Random()
    row = lesion_df.iloc[rng.randrange(len(lesion_df))]

    lesion_id = row["lesion_id"]
    image_ids = parse_images_field(row["images"])

    if not image_ids:
        raise RuntimeError(f"Selected lesion {lesion_id} has no images listed in CSV.")

    return lesion_id, image_ids


def find_image_paths_for_ids(img_ids, img_dir, exts=(".jpg", ".jpeg", ".JPG", ".JPEG")) -> dict[str, Path | None]:
    out = {}
    for img_id in img_ids:
        found = None
        for ext in exts:
            candidate = Path(img_dir) / f"{img_id}{ext}"
            if candidate.exists():
                found = candidate
                break
        out[img_id] = found
    return out


def plot_images_grid(img_id_to_path, lesion_id=None, max_cols=5) -> None:
    img_ids = list(img_id_to_path.keys())
    n = len(img_ids)

    cols = min(max_cols, max(1, n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))

    if hasattr(axes, "ravel"):
        axes = axes.ravel()
    else:
        axes = [axes]

    for ax in axes:
        ax.axis("off")

    for i, img_id in enumerate(img_ids):
        ax = axes[i]
        path = img_id_to_path[img_id]

        if path is None:
            ax.set_title(f"{img_id}\n(MISSING)", fontsize=10)
            continue

        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(img_id, fontsize=10)
        ax.axis("off")

    if lesion_id is not None:
        plt.suptitle(f"Lesion: {lesion_id}", y=1.02, fontsize=14)

    plt.tight_layout()
    plt.show()


def load_metadata_csv(metadata_csv_path) -> pd.DataFrame:
    meta_df = pd.read_csv(metadata_csv_path)
    if meta_df.empty:
        raise RuntimeError("Metadata CSV is empty.")
    return meta_df


def get_metadata_for_lesion(meta_df, lesion_id, lesion_col="lesion_id"):
    if lesion_col not in meta_df.columns:
        raise KeyError(f"Metadata missing column '{lesion_col}'. Found: {list(meta_df.columns)}")

    out = meta_df[meta_df[lesion_col] == lesion_id].copy()
    return out


def show_random_lesion_images_and_metadata(
        lesion_images_csv_path,
        metadata_csv_path,
        img_dir,
        max_cols=5,
):
    # 1: read lesion->images mapping CSV
    lesion_df = load_map_lesion_images(lesion_images_csv_path)

    # 2: choose random lesion + parse its image list
    lesion_id, img_ids = pick_random_lesion(lesion_df)

    # 3: map each image_id to an actual .jpg path and plot them
    img_id_to_path = find_image_paths_for_ids(img_ids, img_dir=img_dir)
    plot_images_grid(img_id_to_path, lesion_id=lesion_id, max_cols=max_cols)

    # 4: load metadata and filter by lesion_id, then show it
    meta_df = load_metadata_csv(metadata_csv_path)
    meta_rows = get_metadata_for_lesion(meta_df, lesion_id)

    _display_df(meta_rows)

    return lesion_id, img_ids, meta_rows


def get_image_ids_for_lesion(lesion_df, lesion_id: str) -> list[str]:
    hit = lesion_df[lesion_df["lesion_id"] == lesion_id]
    if hit.empty:
        return []
    return parse_images_field(hit.iloc[0]["images"])


def get_lesion_info(
        lesion_images_csv_path,
        metadata_csv_path,
        img_dir,
        lesion_id=None,
        max_cols=5,
        prefer_mapping_csv=True,
):
    # 1: load mapping CSV (lesion_id -> images)
    lesion_df = load_map_lesion_images(lesion_images_csv_path)

    # 2: which lesion to show
    if lesion_id is None:
        lesion_id, img_ids = pick_random_lesion(lesion_df)
    else:
        lesion_id = str(lesion_id).strip()
        img_ids = get_image_ids_for_lesion(lesion_df, lesion_id) if prefer_mapping_csv else []

    # 3: load metadata and filter rows for this lesion
    meta_df = load_metadata_csv(metadata_csv_path)
    meta_rows = get_metadata_for_lesion(meta_df, lesion_id)

    if meta_rows.empty:
        raise RuntimeError(f"No metadata rows found for lesion_id={lesion_id}")

    # 4: if mapping CSV didn't give images, derive image_ids from metadata
    if not img_ids:
        if "image_id" not in meta_rows.columns:
            raise KeyError("Metadata does not have 'image_id' column, can't derive images.")
        img_ids = sorted(meta_rows["image_id"].dropna().unique().tolist())

    if not img_ids:
        raise RuntimeError(f"Lesion {lesion_id} has no image_ids (mapping CSV + metadata both empty).")

    # map image_ids -> actual file paths
    img_id_to_path = find_image_paths_for_ids(img_ids, img_dir=img_dir)

    # plot images
    plot_images_grid(img_id_to_path, lesion_id=lesion_id, max_cols=max_cols)

    # show metadata table
    _display_df(meta_rows)

    return lesion_id, img_ids, meta_rows
