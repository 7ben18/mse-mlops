from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_train_script_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_script_parse_args_defaults(monkeypatch):
    module = load_train_script_module()
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = module.parse_args()

    assert args.config is None
    assert args.metadata_csv is None
    assert args.images_dir is None
    assert args.output_dir is None
    assert args.epochs is None


def test_train_script_main_delegates_config_loading_to_src(monkeypatch):
    module = load_train_script_module()
    fake_config = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "custom.yaml", "--epochs", "3", "--device", "cpu"],
    )

    def fake_load_train_config(config_path=None, overrides=None):
        captured["config_path"] = config_path
        captured["overrides"] = overrides
        return fake_config

    def fake_run_training(config):
        captured["run_training_config"] = config

    monkeypatch.setattr(module.train_lib, "load_train_config", fake_load_train_config)
    monkeypatch.setattr(module.train_lib, "run_training", fake_run_training)

    module.main()

    assert captured["config_path"] == Path("custom.yaml")
    assert captured["overrides"] == {"epochs": 3, "device": "cpu"}
    assert captured["run_training_config"] is fake_config
