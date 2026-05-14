#This should call src/mse_mlops/curation.py.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mse_mlops.curation import (
    PromotionConfig,
    PromotionResult,
    get_training_batch_status,
    get_promotion_status,
    promote_feedback_to_train,
    set_training_batch_enabled,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feedback promotion script.

    This script is intentionally only a thin CLI wrapper around
    `mse_mlops.curation`.

    It should not contain the actual dataset mutation logic. That logic belongs
    in `src/mse_mlops/curation.py`, where it can be tested directly.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Promote labeled feedback images into the "
            "DVC-versioned HAM10000 train split."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually promote feedback into the training dataset. "
            "Without this flag, the script runs in dry-run mode."
        ),
    )

    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help=(
            "Allow promotion even when fewer than --min-items labeled images "
            "are ready. Useful for local smoke tests and demos."
        ),
    )

    parser.add_argument(
        "--min-items",
        type=int,
        default=None,
        help=(
            "Minimum number of unpromoted labeled feedback images required "
            "before promotion is allowed. Defaults to curation.PromotionConfig."
        ),
    )

    parser.add_argument(
        "--feedback-file",
        type=Path,
        default=None,
        help=(
            "Path to feedback JSONL file. "
            "Default: reports/feedback/feedback.jsonl from project root."
        ),
    )

    parser.add_argument(
        "--feedback-images-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing uploaded feedback images. "
            "Default: reports/feedback/images from project root."
        ),
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "Path to canonical HAM10000 metadata CSV. "
            "Default: data/processed/ham10000/metadata.csv from project root."
        ),
    )

    parser.add_argument(
        "--dataset-images-dir",
        type=Path,
        default=None,
        help=(
            "Path to canonical HAM10000 image directory. "
            "Default: data/processed/ham10000/HAM10000_images from project root."
        ),
    )

    parser.add_argument(
        "--promotions-dir",
        type=Path,
        default=None,
        help=(
            "Directory for git-managed promotion audit manifests. "
            "Default: reports/feedback/promotions from project root."
        ),
    )

    batch_actions = parser.add_mutually_exclusive_group()
    batch_actions.add_argument(
        "--batch-status",
        action="store_true",
        help="Show promoted training batches and whether they are enabled.",
    )

    batch_actions.add_argument(
        "--exclude-batch",
        type=str,
        default=None,
        help="Permanently disable one promoted training batch by batch ID.",
    )

    batch_actions.add_argument(
        "--include-batch",
        type=str,
        default=None,
        help="Re-enable one promoted training batch by batch ID.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final one-line summary.",
    )

    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help=(
            "Exit with a non-zero code if no new images were promoted. "
            "Useful when this script is used inside chained Makefile targets, "
            "where training should only start after an actual dataset update."
        ),
    )

    return parser.parse_args()




def build_config(args: argparse.Namespace) -> PromotionConfig:
    """Build a PromotionConfig from CLI overrides.

    We only pass values that the user explicitly supplied. Everything else
    falls back to the defaults defined in `mse_mlops.curation`.

    This keeps project-path logic centralized in the library module instead of
    duplicating it here.
    """
    overrides: dict[str, object] = {}

    if args.feedback_file is not None:
        overrides["feedback_file"] = args.feedback_file

    if args.feedback_images_dir is not None:
        overrides["feedback_images_dir"] = args.feedback_images_dir

    if args.metadata_csv is not None:
        overrides["metadata_csv"] = args.metadata_csv

    if args.dataset_images_dir is not None:
        overrides["dataset_images_dir"] = args.dataset_images_dir

    if args.promotions_dir is not None:
        overrides["promotions_dir"] = args.promotions_dir

    if args.min_items is not None:
        if args.min_items < 1:
            raise ValueError("--min-items must be >= 1")
        overrides["min_items"] = args.min_items

    return PromotionConfig(**overrides)


def print_batch_status(config: PromotionConfig, *, quiet: bool) -> None:
    """Print training batch enablement status."""
    rows = get_training_batch_status(config)

    if quiet:
        for row in rows:
            print(
                f"{row['batch_id']}: rows={row['rows']} "
                f"enabled={row['enabled_rows']} disabled={row['disabled_rows']}"
            )
        return

    print("Feedback training batch status:")
    if not rows:
        print("  no promoted batches found")
        return

    for row in rows:
        sources = ", ".join(row["promotion_sources"]) or "unknown"
        print(f"  {row['batch_id']}")
        print(f"    rows:          {row['rows']}")
        print(f"    enabled rows:  {row['enabled_rows']}")
        print(f"    disabled rows: {row['disabled_rows']}")
        print(f"    source:        {sources}")
        print(f"    first train:   {row['first_train_at']}")


def print_batch_update(row: dict[str, object], *, enabled: bool) -> None:
    action = "included" if enabled else "excluded"
    print(f"Batch {row['batch_id']} {action}.")
    print(f"  rows:          {row['rows']}")
    print(f"  enabled rows:  {row['enabled_rows']}")
    print(f"  disabled rows: {row['disabled_rows']}")


def print_status(config: PromotionConfig) -> None:
    """Print a short pre-flight status before promotion.

    This gives the user immediate feedback about whether the threshold has been
    reached before any mutation happens.
    """
    status = get_promotion_status(config)

    ready_count = status["ready_count"]
    min_items = status["min_items"]
    threshold_met = status["threshold_met"]

    print("Feedback promotion status:")
    print(f"  ready labeled images: {ready_count}/{min_items}")
    print(f"  threshold met:        {threshold_met}")
    print()


def print_result(result: PromotionResult, *, quiet: bool) -> None:
    """Print a human-readable summary of the promotion result."""
    mode = "DRY RUN" if result.dry_run else "APPLY"

    if quiet:
        print(
            f"{mode}: candidates={result.candidate_count}, "
            f"promoted={result.promoted_count}, "
            f"already_present={result.already_present_count}, "
            f"threshold_met={result.threshold_met}"
        )
        return

    print(f"Promotion result ({mode}):")
    print(f"  threshold met:        {result.threshold_met}")
    print(f"  required min items:   {result.min_items}")
    print(f"  candidate images:     {result.candidate_count}")
    print(f"  promoted new images:  {result.promoted_count}")
    print(f"  already present:      {result.already_present_count}")
    print(f"  skipped entries:      {result.skipped_count}")

    if result.label_counts:
        print("  label counts:")
        for label, count in sorted(result.label_counts.items()):
            print(f"    {label}: {count}")

    if result.promoted_image_ids:
        print("  promoted image ids:")
        for image_id in result.promoted_image_ids:
            print(f"    {image_id}")

    if result.batch_id:
        print(f"  batch id:           {result.batch_id}")

    if result.manifest_path:
        print(f"  manifest:           {result.manifest_path}")

    if result.skipped_reasons:
        print("  skipped reasons:")
        for reason in result.skipped_reasons:
            print(f"    - {reason}")

    print()

    if result.dry_run:
        print("No files were changed.")
        print("Run again with --apply to actually promote these images.")
        return

    if result.promoted_count > 0 or result.already_present_count > 0:
        print("Promotion completed.")

        if not result.threshold_met:
            print(
                "Note: the normal threshold was not met. "
                "Promotion was allowed because threshold checking was disabled."
            )

        print("Next manual steps:")
        print("  1. dvc status")
        print("  2. dvc add data")
        print('  3. git add data.dvc && git commit -m "Promote feedback batch to train"')
        print("  4. make train-docker")
        return

    if not result.threshold_met:
        print("Promotion did not run because the threshold was not met.")
        return

    print("No new images were promoted.")


def main() -> int:
    """Run feedback promotion CLI.

    Exit codes:
      0 = script ran successfully
      1 = unexpected failure
      2 = no promotion happened while --require-promotion was enabled

    Not meeting the threshold is normally not treated as an error. It is a
    valid flywheel state: for example, 7 labeled images when the configured
    threshold is 10.

    However, when this script is used in a chained command like:

        promote feedback -> dvc add data -> train model

    we need a stricter mode. In that case, --require-promotion prevents the
    pipeline from silently training without newly promoted data.
    """
    args = parse_args()

    config = build_config(args)

    if args.batch_status:
        print_batch_status(config, quiet=args.quiet)
        return 0

    if args.exclude_batch is not None:
        row = set_training_batch_enabled(
            args.exclude_batch,
            enabled=False,
            config=config,
        )
        print_batch_update(row, enabled=False)
        return 0

    if args.include_batch is not None:
        row = set_training_batch_enabled(
            args.include_batch,
            enabled=True,
            config=config,
        )
        print_batch_update(row, enabled=True)
        return 0

    if not args.quiet:
        print_status(config)

    result = promote_feedback_to_train(
        config,
        dry_run=not args.apply,
        require_threshold=not args.no_threshold,
    )

    print_result(result, quiet=args.quiet)

    if args.require_promotion and result.promoted_count == 0:
        print(
            "No new images were promoted; stopping because "
            "--require-promotion was set.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
