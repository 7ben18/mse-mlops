from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_tune_script_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "tune.py"
    spec = importlib.util.spec_from_file_location("tune_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tune_script_parse_args_defaults(monkeypatch):
    module = load_tune_script_module()
    monkeypatch.setattr(sys, "argv", ["tune.py"])

    args = module.parse_args()

    assert args.config == Path("config/tune.yaml")


def test_tune_script_main_delegates_to_src(monkeypatch):
    module = load_tune_script_module()
    fake_config = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["tune.py", "--config", "custom-tune.yaml"])

    def fake_load_tune_config(config_path):
        captured["config_path"] = config_path
        return fake_config

    def fake_run_tuning(config):
        captured["run_tuning_config"] = config

    monkeypatch.setattr(module.tune_lib, "load_tune_config", fake_load_tune_config)
    monkeypatch.setattr(module.tune_lib, "run_tuning", fake_run_tuning)

    module.main()

    assert captured["config_path"] == Path("custom-tune.yaml")
    assert captured["run_tuning_config"] is fake_config
