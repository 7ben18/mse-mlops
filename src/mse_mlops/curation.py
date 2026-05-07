# read reports/feedback/feedback.jsonl
# find unpromoted labeled uploads
# copy uploaded images into data/processed/ham10000/HAM10000_images/train/
# append rows to data/processed/ham10000/metadata.csv
# mark feedback entries as promoted_to_train=true

# convert every promoted image to .jpg using PIL, even if the doctor uploaded PNG/JPEG.
# That avoids extension weirdness.

from __future__ import annotations
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from mse_mlops.serving.feedback_store import (
    load_feedback_entries,
    write_feedback_entries,
)

VALID_LABELS = frozenset({"benign", "malignant"})
PROMOTABLE_SOURCES = frozenset({"upload_labeled"})

LABEL_COLUMN = "mb"
TARGET_SPLIT = "train"
IMAGE_EXTENSION = ".jpg"

REQUIRED_METADATA_COLUMNS = frozenset({"lesion_id", "image_id", "set", LABEL_COLUMN})

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FEEDBACK_DIR = PROJECT_ROOT / "reports" / "feedback"
DEFAULT_FEEDBACK_FILE = DEFAULT_FEEDBACK_DIR / "feedback.jsonl"
DEFAULT_FEEDBACK_IMAGES_DIR = DEFAULT_FEEDBACK_DIR / "images"

DEFAULT_HAM10000_DIR = PROJECT_ROOT / "data" / "processed" / "ham10000"
DEFAULT_METADATA_CSV = DEFAULT_HAM10000_DIR / "metadata.csv"
DEFAULT_DATASET_IMAGES_DIR = DEFAULT_HAM10000_DIR / "HAM10000_images"


@dataclass(frozen=True)
class PromotionConfig:
    feedback_file: Path = DEFAULT_FEEDBACK_FILE
    feedback_images_dir: Path = DEFAULT_FEEDBACK_IMAGES_DIR
    metadata_csv: Path = DEFAULT_METADATA_CSV
    dataset_images_dir: Path = DEFAULT_DATASET_IMAGES_DIR
    target_split: str = TARGET_SPLIT
    label_column: str = LABEL_COLUMN
    min_items: int = 10
    promotable_sources: frozenset[str] = PROMOTABLE_SOURCES
    overwrite_existing_images: bool = True


@dataclass(frozen=True)
class PromotionCandidate:
    feedback_index: int
    feedback_image_id: str
    target_image_id: str
    label: str
    source_image_path: Path
    target_image_path: Path
    original_filename: str | None
    metadata_row_index: int | None = None
    promotion_kind: str = "new_upload"
        # promotion_kind="future_existing"  → promote existing future row
        # promotion_kind="new_upload"       → append new upload_<uuid> row
        # promotion_kind="already_train"    → image is already in train

@dataclass(frozen=True)
class PromotionResult:
    dry_run: bool
    threshold_met: bool
    min_items: int
    candidate_count: int
    promoted_count: int
    already_present_count: int
    skipped_count: int
    label_counts: dict[str, int]
    promoted_image_ids: list[str] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)

    @property
    def should_trigger_training(self) -> bool:
        return self.threshold_met and not self.dry_run and self.promoted_count > 0


def count_promotable_feedback_entries(config: PromotionConfig | None = None) -> int:
    resolved_config = config or PromotionConfig()
    entries = load_feedback_entries(resolved_config.feedback_file)

    return sum(
        1
        for entry in entries
        if _is_promotable_entry(entry, resolved_config.promotable_sources)
    )


def get_promotion_status(config: PromotionConfig | None = None) -> dict[str, int | bool]:
    resolved_config = config or PromotionConfig()
    ready_count = count_promotable_feedback_entries(resolved_config)

    return {
        "ready_count": ready_count,
        "min_items": resolved_config.min_items,
        "threshold_met": ready_count >= resolved_config.min_items,
    }





def _write_metadata_atomic(updated_metadata_df: pd.DataFrame, metadata_csv: Path) -> None:
    """Write metadata DataFrame to CSV atomically.

    The CSV is first written to a temporary file in the same directory, then
    atomically replaces the old metadata file.
    """
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    old_mode: int | None = None
    if metadata_csv.exists():
        old_mode = metadata_csv.stat().st_mode

    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=metadata_csv.parent,
            delete=False,
            suffix=".csv",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            updated_metadata_df.to_csv(tmp_file, index=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        if old_mode is not None:
            tmp_path.chmod(old_mode)

        tmp_path.replace(metadata_csv)

    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise


def promote_feedback_to_train(
    config: PromotionConfig | None = None,
    *,
    dry_run: bool = False,
    require_threshold: bool = True,
) -> PromotionResult:
    """Promote doctor-uploaded labeled feedback images into the train split.

    This mutates two things when dry_run=False:
      1. data/processed/ham10000/metadata.csv
      2. reports/feedback/feedback.jsonl

    It also copies/converts uploaded images into:
      data/processed/ham10000/HAM10000_images/train/<image_id>.jpg

    The function intentionally does not call DVC, MLflow, or training.
    Those are separate MLOps steps.
    """
    resolved_config = config or PromotionConfig()

    entries = load_feedback_entries(resolved_config.feedback_file)
    metadata_df = _load_metadata(
        resolved_config.metadata_csv,
        label_column=resolved_config.label_column,
    )

    candidates, skipped_reasons = _collect_promotion_candidates(
        entries=entries,
        metadata_df=metadata_df,
        config=resolved_config,
    )

    label_counts = dict(Counter(candidate.label for candidate in candidates))
    threshold_met = len(candidates) >= resolved_config.min_items

    if require_threshold and not threshold_met:
        return PromotionResult(
            dry_run=dry_run,
            threshold_met=False,
            min_items=resolved_config.min_items,
            candidate_count=len(candidates),
            promoted_count=0,
            already_present_count=0,
            skipped_count=len(skipped_reasons),
            label_counts=label_counts,
            skipped_reasons=skipped_reasons,
        )

    if dry_run:
        return PromotionResult(
            dry_run=True,
            threshold_met=threshold_met,
            min_items=resolved_config.min_items,
            candidate_count=len(candidates),
            promoted_count=0,
            already_present_count=0,
            skipped_count=len(skipped_reasons),
            label_counts=label_counts,
            skipped_reasons=skipped_reasons,
        )

    metadata_rows_to_append: list[dict[str, Any]] = []
    promoted_image_ids: list[str] = []
    already_present_count = 0
    promoted_at = datetime.now(UTC).isoformat()

    updated_metadata_df = metadata_df.copy()
    metadata_changed = False

    metadata_rows_to_append: list[dict[str, Any]] = []
    promoted_image_ids: list[str] = []
    already_present_count = 0
    promoted_count = 0
    promoted_at = datetime.now(UTC).isoformat()

    for candidate in candidates:
        if candidate.promotion_kind == "future_existing":
            if candidate.metadata_row_index is None:
                raise ValueError(
                    f"Missing metadata row index for {candidate.target_image_id!r}"
                )

            group_indices = _get_lesion_group_indices(
                metadata_df=updated_metadata_df,
                metadata_row_index=candidate.metadata_row_index,
            )

            _validate_future_lesion_group(
                metadata_df=updated_metadata_df,
                group_indices=group_indices,
            )

            for row_index in group_indices:
                image_id = str(updated_metadata_df.loc[row_index, "image_id"])

                # For the actually uploaded image, use the uploaded file.
                # For sibling images of the same lesion, copy the existing future image.
                if image_id == candidate.target_image_id:
                    _copy_or_convert_image_to_jpg(
                        source_path=candidate.source_image_path,
                        target_path=candidate.target_image_path,
                        overwrite=resolved_config.overwrite_existing_images,
                    )
                else:
                    _copy_dataset_image_between_splits(
                        dataset_images_dir=resolved_config.dataset_images_dir,
                        source_split="future",
                        target_split=resolved_config.target_split,
                        image_id=image_id,
                        overwrite=resolved_config.overwrite_existing_images,
                    )

                updated_metadata_df.loc[row_index, "set"] = resolved_config.target_split
                updated_metadata_df.loc[row_index, resolved_config.label_column] = (
                    candidate.label
                )

            metadata_changed = True
            promoted_count += len(group_indices)
        elif candidate.promotion_kind == "already_train":
            existing_row = updated_metadata_df.loc[candidate.metadata_row_index]

            _validate_existing_metadata_row(
                existing_row=existing_row,
                candidate=candidate,
                label_column=resolved_config.label_column,
                target_split=resolved_config.target_split,
            )

            # If metadata already says train but the train image file is missing,
            # repair the physical dataset.
            if not candidate.target_image_path.exists():
                _copy_or_convert_image_to_jpg(
                    source_path=candidate.source_image_path,
                    target_path=candidate.target_image_path,
                    overwrite=resolved_config.overwrite_existing_images,
                )

            already_present_count += 1

        elif candidate.promotion_kind == "new_upload":
            existing_row = _find_existing_metadata_row(
                updated_metadata_df,
                candidate.target_image_id,
            )

            if existing_row is not None:
                _validate_existing_metadata_row(
                    existing_row=existing_row,
                    candidate=candidate,
                    label_column=resolved_config.label_column,
                    target_split=resolved_config.target_split,
                )
                already_present_count += 1
            else:
                _copy_or_convert_image_to_jpg(
                    source_path=candidate.source_image_path,
                    target_path=candidate.target_image_path,
                    overwrite=resolved_config.overwrite_existing_images,
                )
                metadata_rows_to_append.append(
                    _build_metadata_row(
                        metadata_columns=list(updated_metadata_df.columns),
                        candidate=candidate,
                        label_column=resolved_config.label_column,
                        target_split=resolved_config.target_split,
                    )
                )
                promoted_count += 1

        else:
            raise ValueError(f"Unknown promotion kind: {candidate.promotion_kind!r}")

        _mark_feedback_entry_as_promoted(
            entry=entries[candidate.feedback_index],
            candidate=candidate,
            promoted_at=promoted_at,
            target_split=resolved_config.target_split,
        )
        promoted_image_ids.append(candidate.target_image_id)

    for row in metadata_rows_to_append:
        updated_metadata_df.loc[len(updated_metadata_df)] = row
        metadata_changed = True

    if metadata_changed:
        _write_metadata_atomic(updated_metadata_df, resolved_config.metadata_csv)

    write_feedback_entries(resolved_config.feedback_file, entries)

    if metadata_rows_to_append:
        updated_metadata_df = metadata_df.copy()

        for row in metadata_rows_to_append:
            updated_metadata_df.loc[len(updated_metadata_df)] = row

        _write_metadata_atomic(updated_metadata_df, resolved_config.metadata_csv)

    write_feedback_entries(resolved_config.feedback_file, entries)

    return PromotionResult(
        dry_run=False,
        threshold_met=threshold_met,
        min_items=resolved_config.min_items,
        candidate_count=len(candidates),
        promoted_count=promoted_count,
        already_present_count=already_present_count,
        skipped_count=len(skipped_reasons),
        label_counts=label_counts,
        promoted_image_ids=promoted_image_ids,
        skipped_reasons=skipped_reasons,
    )


def _collect_promotion_candidates(
    *,
    entries: list[dict[str, Any]],
    metadata_df: pd.DataFrame,
    config: PromotionConfig,
) -> tuple[list[PromotionCandidate], list[str]]:
    candidates: list[PromotionCandidate] = []
    skipped_reasons: list[str] = []

    existing_image_ids = set(metadata_df["image_id"].astype(str))

    for index, entry in enumerate(entries):
        if not _is_promotable_entry(entry, config.promotable_sources):
            continue

        feedback_image_id = str(entry["image_id"]).strip()
        label = _normalize_label(entry["label"])
        original_filename = entry.get("filename")

        target_image_id, metadata_row_index, promotion_kind, skip_reason = (
            _resolve_target_from_metadata(
                metadata_df=metadata_df,
                feedback_image_id=feedback_image_id,
                original_filename=original_filename,
                target_split=config.target_split,
            )
        )

        if skip_reason is not None:
            skipped_reasons.append(skip_reason)
            continue

        if target_image_id is None or promotion_kind is None:
            skipped_reasons.append(
                f"Could not resolve target image_id for feedback image_id={feedback_image_id!r}"
            )
            continue

        target_image_path = (
                config.dataset_images_dir
                / config.target_split
                / f"{target_image_id}{IMAGE_EXTENSION}"
        )

        source_image_path = _resolve_feedback_image_path(
            feedback_images_dir=config.feedback_images_dir,
            feedback_image_id=feedback_image_id,
            original_filename=original_filename,
        )

        if source_image_path is None:
            skipped_reasons.append(
                f"Missing stored image for feedback image_id={feedback_image_id!r}"
            )
            continue

        if target_image_id in existing_image_ids:
            # Still keep it as candidate: this repairs a previous partial promotion
            # by marking the feedback JSONL entry as promoted after validating metadata.
            pass

        candidates.append(
            PromotionCandidate(
                feedback_index=index,
                feedback_image_id=feedback_image_id,
                target_image_id=target_image_id,
                label=label,
                source_image_path=source_image_path,
                target_image_path=target_image_path,
                original_filename=(
                    str(original_filename) if original_filename is not None else None
                ),
                metadata_row_index=metadata_row_index,
                promotion_kind=promotion_kind,
            )
        )

    return candidates, skipped_reasons


def _is_promotable_entry(
    entry: dict[str, Any],
    promotable_sources: frozenset[str],
) -> bool:
    if entry.get("promoted_to_train") is True:
        return False

    if entry.get("source") not in promotable_sources:
        return False

    image_id = entry.get("image_id")
    if image_id is None or str(image_id).strip() == "":
        return False

    label = entry.get("label")
    if label is None:
        return False

    return _normalize_label(label) in VALID_LABELS


def _normalize_label(label: object) -> str:
    return str(label).strip().lower()


def _make_target_image_id(feedback_image_id: str) -> str:
    # Hyphens are legal in filenames, but underscores make the generated IDs easier
    # to read in metadata.csv and avoid accidental CLI/shell weirdness.
    normalized = feedback_image_id.strip().replace("-", "_")
    return f"upload_{normalized}"


def _resolve_feedback_image_path(
    *,
    feedback_images_dir: Path,
    feedback_image_id: str,
    original_filename: object,
) -> Path | None:
    suffix = Path(str(original_filename)).suffix if original_filename else ""

    if suffix:
        direct_path = feedback_images_dir / f"{feedback_image_id}{suffix}"
        if direct_path.is_file():
            return direct_path

    matches = sorted(feedback_images_dir.glob(f"{feedback_image_id}.*"))
    if len(matches) == 1 and matches[0].is_file():
        return matches[0]

    return None


def _load_metadata(metadata_csv: Path, *, label_column: str) -> pd.DataFrame:
    if metadata_csv.is_file():
        metadata_df = pd.read_csv(metadata_csv)
    else:
        metadata_df = pd.DataFrame(
            columns=["lesion_id", "image_id", "set", label_column]
        )

    required_columns = {"lesion_id", "image_id", "set", label_column}
    missing_columns = sorted(required_columns - set(metadata_df.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    return metadata_df.copy()


def _find_existing_metadata_row(
    metadata_df: pd.DataFrame,
    image_id: str,
) -> pd.Series | None:
    matches = metadata_df[metadata_df["image_id"].astype(str) == image_id]

    if matches.empty:
        return None

    if len(matches) > 1:
        raise ValueError(f"Metadata contains duplicate image_id={image_id!r}")

    return matches.iloc[0]


def _validate_existing_metadata_row(
    *,
    existing_row: pd.Series,
    candidate: PromotionCandidate,
    label_column: str,
    target_split: str,
) -> None:
    existing_set = str(existing_row["set"]).strip()
    existing_label = str(existing_row[label_column]).strip().lower()

    if existing_set != target_split:
        raise ValueError(
            f"Existing metadata row for image_id={candidate.target_image_id!r} "
            f"is in split {existing_set!r}, expected {target_split!r}."
        )

    if existing_label != candidate.label:
        raise ValueError(
            f"Existing metadata row for image_id={candidate.target_image_id!r} "
            f"has label {existing_label!r}, expected {candidate.label!r}."
        )


def _build_metadata_row(
    *,
    metadata_columns: list[str],
    candidate: PromotionCandidate,
    label_column: str,
    target_split: str,
) -> dict[str, Any]:
    row = {column: pd.NA for column in metadata_columns}

    row["lesion_id"] = candidate.target_image_id
    row["image_id"] = candidate.target_image_id
    row["set"] = target_split
    row[label_column] = candidate.label

    # Fill optional provenance columns only if they already exist in metadata.csv.
    # This avoids changing the dataset schema unexpectedly.
    if "source" in row:
        row["source"] = "doctor_feedback"
    if "filename" in row:
        row["filename"] = candidate.original_filename
    if "original_filename" in row:
        row["original_filename"] = candidate.original_filename

    return row


def _copy_or_convert_image_to_jpg(
    *,
    source_path: Path,
    target_path: Path,
    overwrite: bool,
) -> None:
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target image already exists: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source_path, target_path)
        return

    try:
        with Image.open(source_path) as image:
            image.convert("RGB").save(target_path, format="JPEG", quality=95)
    except UnidentifiedImageError as error:
        raise ValueError(f"Could not read uploaded image: {source_path}") from error


def _mark_feedback_entry_as_promoted(
    *,
    entry: dict[str, Any],
    candidate: PromotionCandidate,
    promoted_at: str,
    target_split: str,
) -> None:
    entry["promoted_to_train"] = True
    entry["promoted_at"] = promoted_at
    entry["promoted_split"] = target_split
    entry["promoted_image_id"] = candidate.target_image_id
    entry["promoted_image_path"] = str(candidate.target_image_path)

def _resolve_target_from_metadata(
    *,
    metadata_df: pd.DataFrame,
    feedback_image_id: str,
    original_filename: object,
    target_split: str,
) -> tuple[str | None, int | None, str | None, str | None]:
    """Resolve whether feedback should append a new row or promote an existing one.

    Returns:
        target_image_id:
            Image ID to use in the canonical dataset.
        metadata_row_index:
            Existing metadata row index, if this upload corresponds to an
            existing dataset image.
        promotion_kind:
            One of: "new_upload", "future_existing", "already_train".
        skip_reason:
            Non-None when this feedback entry should be skipped.

    Why this exists:
        In real life, doctor uploads are new external images, so we create a
        new upload_<uuid> row.

        In the demo, however, uploads may come from the existing `future` split.
        In that case, we should promote the existing future row instead of
        duplicating the same pixels under a new upload_<uuid> ID.
    """
    if original_filename is None:
        return _make_target_image_id(feedback_image_id), None, "new_upload", None

    original_stem = Path(str(original_filename)).stem.strip()

    if not original_stem:
        return _make_target_image_id(feedback_image_id), None, "new_upload", None

    matches = metadata_df.index[
        metadata_df["image_id"].astype(str) == original_stem
    ].tolist()

    if not matches:
        return _make_target_image_id(feedback_image_id), None, "new_upload", None

    if len(matches) > 1:
        return (
            None,
            None,
            None,
            f"Metadata contains duplicate image_id={original_stem!r}",
        )

    metadata_row_index = matches[0]
    existing_split = str(metadata_df.loc[metadata_row_index, "set"]).strip()

    if existing_split == "future":
        return original_stem, metadata_row_index, "future_existing", None

    if existing_split == target_split:
        return original_stem, metadata_row_index, "already_train", None

    return (
        None,
        None,
        None,
        (
            f"Uploaded filename {original_filename!r} matches existing "
            f"image_id={original_stem!r} in split={existing_split!r}. "
            "Skipping to avoid train/val/test leakage."
        ),
    )

def _get_lesion_group_indices(
    *,
    metadata_df: pd.DataFrame,
    metadata_row_index: int,
) -> list[int]:
    """Return all metadata row indices belonging to the same lesion_id."""
    lesion_id = metadata_df.loc[metadata_row_index, "lesion_id"]

    return metadata_df.index[
        metadata_df["lesion_id"].astype(str) == str(lesion_id)
    ].tolist()

def _validate_future_lesion_group(
    *,
    metadata_df: pd.DataFrame,
    group_indices: list[int],
) -> None:
    """Ensure a future image promotion does not create split leakage.

    HAM10000 has multiple images per lesion. The train/val/test/future split
    must be lesion-level, not image-level. Therefore, if one image from a
    future lesion is promoted to train, all images for that lesion must be
    promoted together.
    """
    existing_sets = set(metadata_df.loc[group_indices, "set"].astype(str))

    if existing_sets != {"future"}:
        raise ValueError(
            "Cannot promote image because its lesion_id is not fully in the "
            f"future split. Existing sets: {sorted(existing_sets)}"
        )


def _copy_dataset_image_between_splits(
    *,
    dataset_images_dir: Path,
    source_split: str,
    target_split: str,
    image_id: str,
    overwrite: bool,
) -> None:
    """Copy an existing canonical dataset image from one split folder to another."""
    source_path = dataset_images_dir / source_split / f"{image_id}{IMAGE_EXTENSION}"
    target_path = dataset_images_dir / target_split / f"{image_id}{IMAGE_EXTENSION}"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source dataset image: {source_path}")

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target image already exists: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)