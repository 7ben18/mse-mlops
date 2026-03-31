import ast
import pathlib
import shutil
import re
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from mse_mlops import paths



def apply_mask(
    image_id: str,
    data_dir: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR,
    to_show: bool = False,
) -> np.ndarray:
    """
    Apply mask to an image and return the masked image as a numpy array.
    """
    img = iio.imread(data_dir / paths.IMG_DIR / f"{image_id}.jpg")
    mask = iio.imread(data_dir / paths.MASK_DIR / f"{image_id}_segmentation.png")

    # if mask is not already grayscale, take one channel
    if mask.ndim == 3:
        mask = mask[..., 0]

    assert img.shape[:2] == mask.shape[:2], "Image and mask must match in size"

    # binary mask: True where object exists
    mask = mask > 0

    # apply mask to all image channels
    result_img = img * mask[..., None]

    if to_show:
        plt.imshow(result_img)
        plt.axis("off")
        plt.show()

    return result_img


def _helper_parse_split_sets(config: dict) -> list[dict]:
    """
    Parse split_sets from config into:
    [
        {"name": "train", "ratio": 0.6},
        {"name": "val", "ratio": 0.3},
        ...
    ]

    Supports:
    - direct numeric ratios
    - placeholders like "${train_ratio}"
    """

    raw_split_sets = config.get("split_sets")
    if not raw_split_sets:
        raise ValueError("Config must contain non-empty 'split_sets'")

    split_sets = []
    seen_names = set()

    for item in raw_split_sets:
        if not isinstance(item, dict):
            raise ValueError(f"Each item in split_sets must be a dict, got: {item}")

        name = item.get("name")
        raw_ratio = item.get("ratio")

        if not name:
            raise ValueError(f"Each split set must have a non-empty 'name', got: {item}")

        if name in seen_names:
            raise ValueError(f"Duplicate split set name: '{name}'")
        seen_names.add(name)

        # resolve ratio
        if isinstance(raw_ratio, (int, float)):
            ratio = float(raw_ratio)

        elif isinstance(raw_ratio, str):
            raw_ratio = raw_ratio.strip()

            # placeholder like "${train_ratio}"
            match = re.fullmatch(r"\$\{([^}]+)\}", raw_ratio)
            if match:
                ref_key = match.group(1)

                if ref_key not in config:
                    raise ValueError(
                        f"Split set '{name}' refers to missing config key '{ref_key}'"
                    )

                ref_value = config[ref_key]
                if not isinstance(ref_value, (int, float)):
                    raise ValueError(
                        f"Referenced config key '{ref_key}' must be numeric, "
                        f"got {type(ref_value).__name__}"
                    )

                ratio = float(ref_value)

            else:
                # also allow plain numeric strings like "0.2"
                try:
                    ratio = float(raw_ratio)
                except ValueError as e:
                    raise ValueError(
                        f"Cannot parse ratio for split '{name}': {raw_ratio}"
                    ) from e

        else:
            raise ValueError(
                f"Unsupported ratio type for split '{name}': {type(raw_ratio).__name__}"
            )

        if ratio < 0:
            raise ValueError(f"Ratio for split '{name}' must be >= 0, got {ratio}")

        split_sets.append({"name": name, "ratio": ratio})

    total_ratio = sum(item["ratio"] for item in split_sets)
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1, but got {total_ratio}")

    return split_sets

def _helper_allocate_split_counts(
    n_total: int,
    split_sets: list[dict],
) -> dict[str, int]:
    """
    Allocate exact integer counts so they sum to n_total.
    Uses largest remainder method.

    Example:
    ratios = [0.6, 0.3, 0.1], n_total = 101
    raw counts = [60.6, 30.3, 10.1]
    floor      = [60,   30,   10 ]  -> sum = 100
    remaining  = 1 -> assign to the largest remainder (0.6), so [61, 30, 10]
    """
    if n_total < 0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")

    raw_counts = np.array([n_total * item["ratio"] for item in split_sets], dtype=float)
    floor_counts = np.floor(raw_counts).astype(int)
    remainders = raw_counts - floor_counts

    remaining = n_total - floor_counts.sum()

    # stable tie-breaking by config order
    order = np.argsort(-remainders, kind="stable")
    for idx in order[:remaining]:
        floor_counts[idx] += 1

    return {
        split_sets[i]["name"]: int(floor_counts[i])
        for i in range(len(split_sets))
    }


def _helper_build_split_filename(split_sets: list[dict], seed: int) -> str:
    """
    Example:
    split_train-0.6_val-0.3_test-0.1_future-0_seed-65.csv
    """
    ratio_part = "_".join(f"{item['name']}-{item['ratio']:g}" for item in split_sets)
    return f"split_{ratio_part}_seed-{seed}.csv"


def split_data_csv(
    config_file: pathlib.Path = paths.CONFIG_DIR / "split.yaml",
    map_lesion_images: pathlib.Path = paths.MAP_LESION_IMAGES,
    csv_output: pathlib.Path = paths.PROCESSED_DATA_DIR / paths.HAM_DIR,
    verbose: bool = True,
) -> pathlib.Path:
    """
    Lesion-level CSV split (no image copying) based on given config file
    This avoids leakage when one lesion has multiple images.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed")
    if seed is None:
        raise ValueError("Config must contain 'seed'")

    split_sets = _helper_parse_split_sets(config)
    split_names = [item["name"] for item in split_sets]

    if verbose:
        print(f"Split config file: {config_file}")
        print("Split sets:")
        for item in split_sets:
            print(f"  {item['name']}: {item['ratio']}")
        print(f"Seed: {seed}")

    lesion_mapping = pd.read_csv(map_lesion_images, index_col=0)

    if verbose:
        print(
            f"Using lesion mapping from {map_lesion_images}, "
            f"total entries: {len(lesion_mapping)}"
        )
        print("Lesion mapping head:")
        print(lesion_mapping.head())

    if "images" not in lesion_mapping.columns:
        raise ValueError("Expected column 'images' in lesion mapping CSV")

    lesion_ids = lesion_mapping.index.to_list()
    if not lesion_ids:
        raise ValueError("No lesion_ids found in lesion mapping CSV")

    lesions_df = pd.DataFrame({"lesion_id": lesion_ids})
    lesions_df = lesions_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(lesions_df)
    split_counts = _helper_allocate_split_counts(n_total=n_total, split_sets=split_sets)

    start = 0
    lesions_df["set"] = None

    for split_name in split_names:
        count = split_counts[split_name]
        end = start + count
        lesions_df.loc[start:end - 1, "set"] = split_name
        start = end

    lesion_mapping = lesion_mapping.copy()
    lesion_mapping["n_images"] = lesion_mapping["images"].apply(_count_images)

    lesions_df = lesions_df.merge(
        lesion_mapping[["images", "n_images"]],
        left_on="lesion_id",
        right_index=True,
        how="left",
    )

    if verbose:
        lesion_stats = (
            lesions_df.groupby("set")["lesion_id"]
            .count()
            .reindex(split_names, fill_value=0)
        )
        image_stats = (
            lesions_df.groupby("set")["n_images"]
            .sum()
            .reindex(split_names, fill_value=0)
        )

        total_lesions = lesion_stats.sum()
        total_images = image_stats.sum()

        print("\nSplit stats:")
        for split_name in split_names:
            n_lesions = lesion_stats[split_name]
            n_images = image_stats[split_name]

            lesion_pct = 100 * n_lesions / total_lesions if total_lesions else 0.0
            image_pct = 100 * n_images / total_images if total_images else 0.0

            print(
                f"{split_name:>10}: "
                f"{n_lesions:4d} lesions ({lesion_pct:6.2f}%), "
                f"{n_images:4d} images ({image_pct:6.2f}%)"
            )

    csv_output.mkdir(parents=True, exist_ok=True)

    filename = _helper_build_split_filename(split_sets=split_sets, seed=seed)
    output_path = csv_output / filename

    lesions_df[["lesion_id", "images", "set"]].to_csv(output_path, index=False)

    if verbose:
        print(f"\nSaved split CSV to: {output_path}")

    return output_path


def _count_images(images_value: str) -> int:
    if pd.isna(images_value):
        return 0

    s = str(images_value).strip()
    if s == "{}" or s == "":
        return 0

    # expected format: "{a, b, c}"
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if inner == "":
            return 0
        return len([x.strip() for x in inner.split(",") if x.strip()])

    # fallback in case format changes
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return len(parsed)
    except Exception:
        pass
    return 1


def _helper_parse_images(images_value: str) -> list[str]:
    """
    Parse strings like:
    "{ISIC_0024579, ISIC_0025577, ISIC_0029638}"
    into:
    ["ISIC_0024579", "ISIC_0025577", "ISIC_0029638"]
    """
    if pd.isna(images_value):
        return []

    s = str(images_value).strip()
    if not s or s == "{}":
        return []

    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    if not s:
        return []

    return [img_id.strip() for img_id in s.split(",") if img_id.strip()]


def _helper_resolve_split_csv(split_csv: pathlib.Path, verbose: bool = False) -> pathlib.Path:
    """
    Resolve split_csv path.

    Normal case:
    - split_csv is already a full filepath to a CSV file.

    Fallback:
    - if split_csv is a directory, pick the newest split_*.csv from it.
    """
    split_csv = pathlib.Path(split_csv)

    if split_csv.is_file():
        return split_csv

    if split_csv.is_dir():
        candidates = sorted(
            split_csv.glob("split_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No split_*.csv files found in directory: {split_csv}"
            )

        resolved = candidates[0]
        if verbose:
            print(
                f"No explicit split CSV filepath was provided. "
                f"Using the newest split file found in directory:\n  {resolved}"
            )
        return resolved

    raise FileNotFoundError(f"split_csv path does not exist: {split_csv}")


def _helper_create_output_dirs(
    data_output: pathlib.Path,
    img_dir_name: str,
    mask_dir_name: str,
    set_names: list[str]
) -> None:
    valid_sets = set_names
    for split_name in valid_sets:
        (data_output / img_dir_name / split_name).mkdir(parents=True, exist_ok=True)
        (data_output / mask_dir_name / split_name).mkdir(parents=True, exist_ok=True)


def _helper_clear_split_dirs(
    data_output: pathlib.Path,
    img_dir_name: str,
    mask_dir_name: str,
    verbose: bool = False,
) -> None:
    """
    Delete all existing files inside all existing processed split subfolders
    for both images and masks.

    This also clears old split folders that are not present in the current
    config anymore, e.g. old 'val' after switching to only train/test.
    """
    deleted_files = 0
    deleted_dirs = 0

    for base_dir_name in [img_dir_name, mask_dir_name]:
        base_dir = data_output / base_dir_name

        if not base_dir.exists():
            continue

        # iterate over ALL existing split folders, not only current set_names
        for split_dir in base_dir.iterdir():
            if not split_dir.is_dir():
                continue

            for path in split_dir.iterdir():
                if path.is_file():
                    path.unlink()
                    deleted_files += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    deleted_dirs += 1

    if verbose:
        print(
            f"Deleted {deleted_files} files and {deleted_dirs} directories "
            f"from processed split folders."
        )

def _helper_print_split_summary(
    split_df: pd.DataFrame,
    verbose: bool = False,
) -> None:
    if not verbose:
        return
    set_names = split_df["set"].unique()
    split_counts = (
        split_df.assign(n_images=split_df["images"].apply(_helper_parse_images).apply(len))
        .groupby("set")["n_images"]
        .sum()
        .reindex(set_names, fill_value=0)
    )

    lesion_counts = (
        split_df.groupby("set")["lesion_id"]
        .count()
        .reindex(set_names, fill_value=0)
    )

    print("\nSplit summary:")
    split_names = set_names
    for split_name in split_names:
        print(
            f"  {split_name}: "
            f"{lesion_counts[split_name]} lesions, "
            f"{split_counts[split_name]} images"
        )


def split_data_dir(
    split_csv: pathlib.Path = paths.PROCESSED_DATA_DIR / paths.HAM_DIR,
    data_input: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR,
    data_output: pathlib.Path = paths.PROCESSED_DATA_DIR / paths.HAM_DIR,
    verbose: bool = True,
) -> None:
    """
    This is a lesion-level directory split (image copying), based on given CSV split.
    """
    data_input = pathlib.Path(data_input)
    data_output = pathlib.Path(data_output)

    split_csv = _helper_resolve_split_csv(split_csv, verbose=verbose)

    if verbose:
        print(f"Reading split CSV: {split_csv}")

    split_df = pd.read_csv(split_csv)

    input_img_dir = data_input / paths.IMG_DIR
    input_mask_dir = data_input / paths.MASK_DIR

    if not input_img_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {input_img_dir}")
    if not input_mask_dir.exists():
        raise FileNotFoundError(f"Input mask directory not found: {input_mask_dir}")


    _helper_create_output_dirs(
        data_output=data_output,
        img_dir_name=paths.IMG_DIR,
        mask_dir_name=paths.MASK_DIR,
        set_names=split_df["set"].unique().tolist(),
    )
    _helper_clear_split_dirs(
        data_output=data_output,
        img_dir_name=paths.IMG_DIR,
        mask_dir_name=paths.MASK_DIR,
        verbose=verbose,
    )
    copied_images = 0
    copied_masks = 0
    missing_images = []
    missing_masks = []

    for _, row in split_df.iterrows():
        split_name = row["set"]
        lesion_id = row["lesion_id"]
        image_ids = _helper_parse_images(row["images"])

        for image_id in image_ids:
            src_img = input_img_dir / f"{image_id}.jpg"
            dst_img = data_output / paths.IMG_DIR / split_name / f"{image_id}.jpg"

            src_mask = input_mask_dir / f"{image_id}_segmentation.png"
            dst_mask = data_output / paths.MASK_DIR / split_name / f"{image_id}_segmentation.png"

            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied_images += 1
            else:
                missing_images.append(src_img)

            if src_mask.exists():
                shutil.copy2(src_mask, dst_mask)
                copied_masks += 1
            else:
                missing_masks.append(src_mask)

    if verbose:
        print("\nDone.")
        print(f"Copied RGB images: {copied_images}")
        print(f"Copied masks:      {copied_masks}")

        _helper_print_split_summary(split_df, verbose=verbose)

        if missing_images:
            print(f"\nMissing RGB images: {len(missing_images)}")
            for p in missing_images[:10]:
                print(f"  {p}")
            if len(missing_images) > 10:
                print("  ...")

        if missing_masks:
            print(f"\nMissing masks: {len(missing_masks)}")
            for p in missing_masks[:10]:
                print(f"  {p}")
            if len(missing_masks) > 10:
                print("  ...")

def split_data_full(
    config_file: pathlib.Path = paths.CONFIG_DIR / "split.yaml",
    data_input: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR,
    data_output: pathlib.Path = paths.PROCESSED_DATA_DIR / paths.HAM_DIR,
    map_lesion_images: pathlib.Path = paths.MAP_LESION_IMAGES,
    verbose: bool = True,
) -> pathlib.Path:
    """
    This is a lesion-level CSV and directory split (image copying), based on given config file.
    """
    csv_split = split_data_csv(map_lesion_images=map_lesion_images,
                    config_file=config_file,
                    verbose=verbose)

    split_data_dir(data_input=data_input,
                   data_output=data_output,
                   split_csv=csv_split,
                   verbose=verbose)
    return csv_split







