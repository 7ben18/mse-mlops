import pathlib
import re
import shutil

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from mse_mlops import paths

MALIGNANT_DX = {"akiec", "bcc", "mel"}
BENIGN_DX = {"bkl", "df", "nv", "vasc"}


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
                    raise ValueError(f"Split set '{name}' refers to missing config key '{ref_key}'")

                ref_value = config[ref_key]
                if not isinstance(ref_value, (int, float)):
                    raise ValueError(
                        f"Referenced config key '{ref_key}' must be numeric, got {type(ref_value).__name__}"
                    )

                ratio = float(ref_value)

            else:
                # also allow plain numeric strings like "0.2"
                try:
                    ratio = float(raw_ratio)
                except ValueError as e:
                    raise ValueError(f"Cannot parse ratio for split '{name}': {raw_ratio}") from e

        else:
            raise ValueError(f"Unsupported ratio type for split '{name}': {type(raw_ratio).__name__}")

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

    return {split_sets[i]["name"]: int(floor_counts[i]) for i in range(len(split_sets))}


def _helper_add_mb_column(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if "dx" not in metadata_df.columns:
        raise ValueError("Raw metadata CSV must contain column 'dx'")

    labeled = metadata_df.copy()
    labeled["mb"] = pd.NA
    labeled.loc[labeled["dx"].isin(MALIGNANT_DX), "mb"] = "malignant"
    labeled.loc[labeled["dx"].isin(BENIGN_DX), "mb"] = "benign"
    return labeled


def _helper_load_raw_metadata(raw_metadata_csv: pathlib.Path) -> pd.DataFrame:
    raw_metadata_csv = pathlib.Path(raw_metadata_csv)
    metadata_df = pd.read_csv(raw_metadata_csv)

    required_columns = {"lesion_id", "image_id", "dx"}
    missing_columns = required_columns - set(metadata_df.columns)
    if missing_columns:
        raise ValueError(
            f"Raw metadata CSV must contain columns {sorted(required_columns)}, "
            f"missing {sorted(missing_columns)} in {raw_metadata_csv}"
        )

    metadata_df = metadata_df.dropna(subset=["lesion_id", "image_id"])
    metadata_df["lesion_id"] = metadata_df["lesion_id"].astype(str).str.strip()
    metadata_df["image_id"] = metadata_df["image_id"].astype(str).str.strip()
    metadata_df = metadata_df[(metadata_df["lesion_id"] != "") & (metadata_df["image_id"] != "")]
    metadata_df = metadata_df.drop_duplicates(subset=["lesion_id", "image_id"])

    if metadata_df.empty:
        raise ValueError(f"No valid lesion/image rows found in raw metadata CSV: {raw_metadata_csv}")

    return metadata_df


def _helper_assign_split_by_lesion(
    metadata_df: pd.DataFrame,
    split_sets: list[dict],
    seed: int,
) -> pd.DataFrame:
    lesion_ids = metadata_df["lesion_id"].drop_duplicates().tolist()
    if not lesion_ids:
        raise ValueError("No lesion_ids found in raw metadata CSV")

    lesion_splits = pd.DataFrame({"lesion_id": lesion_ids})
    lesion_splits = lesion_splits.sample(frac=1, random_state=seed).reset_index(drop=True)

    split_names = [item["name"] for item in split_sets]
    split_counts = _helper_allocate_split_counts(n_total=len(lesion_splits), split_sets=split_sets)

    start = 0
    lesion_splits["set"] = None

    for split_name in split_names:
        count = split_counts[split_name]
        end = start + count
        lesion_splits.loc[start : end - 1, "set"] = split_name
        start = end

    return metadata_df.merge(lesion_splits, on="lesion_id", how="left")


def _helper_resolve_output_csv(csv_output: pathlib.Path, default_name: str) -> pathlib.Path:
    csv_output = pathlib.Path(csv_output)
    if csv_output.suffix.lower() == ".csv":
        return csv_output
    return csv_output / default_name


def _helper_load_split_config(
    config_file: pathlib.Path,
    verbose: bool = False,
) -> tuple[int, list[dict], list[str]]:
    with open(config_file) as f:
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

    return seed, split_sets, split_names


def _helper_build_processed_metadata(
    raw_metadata_csv: pathlib.Path,
    split_sets: list[dict],
    split_names: list[str],
    seed: int,
    verbose: bool = False,
) -> pd.DataFrame:
    metadata_df = _helper_load_raw_metadata(raw_metadata_csv)
    metadata_df = _helper_add_mb_column(metadata_df)
    metadata_df = _helper_assign_split_by_lesion(metadata_df=metadata_df, split_sets=split_sets, seed=seed)

    if verbose:
        print(
            f"Using raw metadata from {raw_metadata_csv}, total rows: {len(metadata_df)}, "
            f"total lesions: {metadata_df['lesion_id'].nunique()}"
        )
        print("Metadata head:")
        print(metadata_df.head())

        lesion_stats = metadata_df.groupby("set")["lesion_id"].nunique().reindex(split_names, fill_value=0)
        image_stats = metadata_df.groupby("set")["image_id"].count().reindex(split_names, fill_value=0)

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

    return metadata_df


def _helper_write_processed_metadata(
    metadata_df: pd.DataFrame,
    csv_output: pathlib.Path,
    verbose: bool = False,
) -> pathlib.Path:
    output_path = _helper_resolve_output_csv(csv_output=csv_output, default_name=pathlib.Path(paths.EXT_METADATA).name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)

    if verbose:
        unmapped_dx = sorted(set(metadata_df["dx"].dropna()) - (MALIGNANT_DX | BENIGN_DX))
        print(f"\nSaved processed metadata to: {output_path}")
        print("Unmapped dx:", unmapped_dx)

    return output_path


def _helper_validate_lesion_split_consistency(
    split_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.Series:
    required_columns = {"lesion_id", "set"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata must contain columns {sorted(required_columns)}, "
            f"missing {sorted(missing_columns)}"
        )

    missing_set_lesions = split_df.loc[split_df["set"].isna(), "lesion_id"].dropna().drop_duplicates().tolist()
    if missing_set_lesions:
        raise ValueError(
            "Each lesion_id must belong to exactly one set. "
            f"Missing set values for lesions: {missing_set_lesions[:10]}"
        )

    lesion_set_counts = split_df.groupby("lesion_id")["set"].nunique()
    inconsistent = lesion_set_counts[lesion_set_counts != 1]
    if not inconsistent.empty:
        examples = ", ".join(f"{lesion_id}: {count}" for lesion_id, count in inconsistent.head(10).items())
        raise ValueError(
            "Each lesion_id must belong to exactly one set. "
            f"Found lesions assigned to multiple sets: {examples}"
        )

    if verbose:
        print(
            "Verified lesion split consistency: "
            f"{len(lesion_set_counts)} lesions assigned to exactly one set."
        )

    return lesion_set_counts


def split_data_csv(
    config_file: pathlib.Path = paths.CONFIG_DIR / "split.yaml",
    raw_metadata_csv: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR / paths.METADATA,
    csv_output: pathlib.Path = paths.EXT_METADATA,
    verbose: bool = True,
) -> pathlib.Path:
    """
    Build processed metadata with one image per row and a lesion-consistent split label.
    """
    seed, split_sets, split_names = _helper_load_split_config(config_file=config_file, verbose=verbose)
    metadata_df = _helper_build_processed_metadata(
        raw_metadata_csv=raw_metadata_csv,
        split_sets=split_sets,
        split_names=split_names,
        seed=seed,
        verbose=verbose,
    )
    _helper_validate_lesion_split_consistency(metadata_df, verbose=verbose)
    return _helper_write_processed_metadata(metadata_df=metadata_df, csv_output=csv_output, verbose=verbose)


def _helper_reset_output_root(data_output: pathlib.Path, verbose: bool = False) -> None:
    data_output = pathlib.Path(data_output)

    if data_output.exists():
        shutil.rmtree(data_output)
        if verbose:
            print(f"Deleted processed output root: {data_output}")

    data_output.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Created processed output root: {data_output}")


def _helper_resolve_split_csv(split_csv: pathlib.Path, verbose: bool = False) -> pathlib.Path:
    """
    Resolve split_csv path.

    Normal case:
    - split_csv is already a full filepath to a CSV file.

    Fallback:
    - if split_csv is a directory, use metadata.csv in that directory.
    """
    split_csv = pathlib.Path(split_csv)

    if split_csv.is_file():
        return split_csv

    if split_csv.is_dir():
        resolved = split_csv / pathlib.Path(paths.EXT_METADATA).name
        if not resolved.exists():
            raise FileNotFoundError(f"No metadata CSV found in directory: {resolved}")
        if verbose:
            print(
                f"No explicit metadata CSV filepath was provided. "
                f"Using the metadata file found in directory:\n  {resolved}"
            )
        return resolved

    raise FileNotFoundError(f"split_csv path does not exist: {split_csv}")


def _helper_create_output_dirs(
    data_output: pathlib.Path, img_dir_name: str, mask_dir_name: str, set_names: list[str]
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
        print(f"Deleted {deleted_files} files and {deleted_dirs} directories from processed split folders.")


def _helper_print_split_summary(
    split_df: pd.DataFrame,
    verbose: bool = False,
) -> None:
    if not verbose:
        return
    set_names = split_df["set"].dropna().drop_duplicates().tolist()
    lesion_counts = split_df.groupby("set")["lesion_id"].nunique().reindex(set_names, fill_value=0)
    split_counts = split_df.groupby("set")["image_id"].count().reindex(set_names, fill_value=0)

    print("\nSplit summary:")
    for split_name in set_names:
        print(f"  {split_name}: {lesion_counts[split_name]} lesions, {split_counts[split_name]} images")


def _helper_copy_split_data(
    split_df: pd.DataFrame,
    data_input: pathlib.Path,
    data_output: pathlib.Path,
    clear_existing: bool = True,
    verbose: bool = False,
) -> None:
    required_columns = {"lesion_id", "image_id", "set"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata CSV must contain columns {sorted(required_columns)}, "
            f"missing {sorted(missing_columns)}"
        )

    _helper_validate_lesion_split_consistency(split_df, verbose=verbose)

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
    if clear_existing:
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
        image_id = row["image_id"]
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


def split_data_dir(
    split_csv: pathlib.Path = paths.EXT_METADATA,
    data_input: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR,
    data_output: pathlib.Path = paths.PROCESSED_DATA_DIR / paths.HAM_DIR,
    clear_existing: bool = True,
    verbose: bool = True,
) -> None:
    """
    Copy images and masks into split directories based on processed metadata.
    By default, existing managed split folders are cleared first.
    """
    data_input = pathlib.Path(data_input)
    data_output = pathlib.Path(data_output)

    split_csv = _helper_resolve_split_csv(split_csv, verbose=verbose)

    if verbose:
        print(f"Reading metadata CSV: {split_csv}")

    split_df = pd.read_csv(split_csv)
    _helper_copy_split_data(
        split_df=split_df,
        data_input=data_input,
        data_output=data_output,
        clear_existing=clear_existing,
        verbose=verbose,
    )


def split_data_full(
    config_file: pathlib.Path = paths.CONFIG_DIR / "split.yaml",
    data_input: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR,
    data_output: pathlib.Path = paths.PROCESSED_DATA_DIR / paths.HAM_DIR,
    raw_metadata_csv: pathlib.Path = paths.RAW_DATA_DIR / paths.HAM_DIR / paths.METADATA,
    verbose: bool = True,
) -> pathlib.Path:
    """
    Rebuild the processed HAM10000 outputs from raw data only.
    """
    data_output = pathlib.Path(data_output)
    _helper_reset_output_root(data_output=data_output, verbose=verbose)
    seed, split_sets, split_names = _helper_load_split_config(config_file=config_file, verbose=verbose)
    metadata_df = _helper_build_processed_metadata(
        raw_metadata_csv=raw_metadata_csv,
        split_sets=split_sets,
        split_names=split_names,
        seed=seed,
        verbose=verbose,
    )
    metadata_csv = _helper_write_processed_metadata(
        metadata_df=metadata_df,
        csv_output=data_output / pathlib.Path(paths.EXT_METADATA).name,
        verbose=verbose,
    )
    _helper_copy_split_data(
        split_df=metadata_df,
        data_input=data_input,
        data_output=data_output,
        clear_existing=False,
        verbose=verbose,
    )
    return metadata_csv
