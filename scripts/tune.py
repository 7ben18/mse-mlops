from __future__ import annotations

import argparse
from pathlib import Path

from mse_mlops import tune as tune_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ray Tune hyperparameter search against the existing training workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=tune_lib.DEFAULT_TUNE_CONFIG_PATH,
        help=f"YAML config file path (default: {tune_lib.DEFAULT_TUNE_CONFIG_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tune_lib.load_tune_config(args.config)
    tune_lib.run_tuning(config)


if __name__ == "__main__":
    main()
