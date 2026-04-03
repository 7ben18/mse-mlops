from __future__ import annotations

import argparse
from pathlib import Path

from mse_mlops import train as train_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 on processed HAM10000 metadata.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config file path (default: auto-detected {train_lib.DEFAULT_CONFIG_PATH} from repo root)",
    )
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--label-column", type=str)
    parser.add_argument("--train-set", type=str)
    parser.add_argument("--val-set", type=str)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))

    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=train_lib.SCHEDULER_CHOICES,
        default=None,
    )
    parser.add_argument("--max-grad-norm", type=float)

    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or 'latest' to resume from the newest epoch checkpoint.",
    )
    parser.add_argument("--save-total-limit", type=int)

    parser.add_argument("--freeze-backbone", action="store_true", dest="freeze_backbone")
    parser.add_argument("--unfreeze-backbone", action="store_false", dest="freeze_backbone")
    parser.set_defaults(freeze_backbone=None)

    parser.add_argument("--mlflow-tracking-uri", type=str)
    parser.add_argument("--mlflow-experiment-name", type=str)
    parser.add_argument("--mlflow-run-name", type=str)
    parser.add_argument("--mlflow-tags", type=str)

    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "metadata_csv": args.metadata_csv,
            "images_dir": args.images_dir,
            "label_column": args.label_column,
            "train_set": args.train_set,
            "val_set": args.val_set,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "model_name": args.model_name,
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": args.device,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_grad_norm": args.max_grad_norm,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "save_total_limit": args.save_total_limit,
            "freeze_backbone": args.freeze_backbone,
            "mlflow_tracking_uri": args.mlflow_tracking_uri,
            "mlflow_experiment_name": args.mlflow_experiment_name,
            "mlflow_run_name": args.mlflow_run_name,
            "mlflow_tags": args.mlflow_tags,
        }.items()
        if value is not None
    }


def main() -> None:
    args = parse_args()
    config = train_lib.load_train_config(config_path=args.config, overrides=build_overrides(args))
    train_lib.run_training(config)


if __name__ == "__main__":
    main()
