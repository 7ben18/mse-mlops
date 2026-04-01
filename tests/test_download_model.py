from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_download_model_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "download_model_script", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_download_model_parse_args_defaults(monkeypatch):
    module = load_download_model_module()
    monkeypatch.setattr(sys, "argv", ["download_model.py"])

    args = module.parse_args()

    assert args.model_id == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert args.output_dir == Path(
        "models/pretrained/dinov3-vits16-pretrain-lvd1689m"
    )
