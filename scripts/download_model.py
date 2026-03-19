from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_OUTPUT_DIR = Path("outputs/pretrained/dinov3-vits16-pretrain-lvd1689m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pretrained backbone and image processor locally.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id to download (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where model files are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    from transformers import AutoImageProcessor, AutoModel

    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading processor: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    processor.save_pretrained(output_dir)

    print(f"Downloading model: {args.model_id}")
    model = AutoModel.from_pretrained(args.model_id)
    model.save_pretrained(output_dir)

    print(f"Saved pretrained assets to: {output_dir}")


if __name__ == "__main__":
    main()
