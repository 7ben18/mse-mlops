from __future__ import annotations

from pathlib import Path

import pytest

from mse_mlops import tune


def test_load_tune_config_reads_expected_defaults():
    config = tune.load_tune_config(Path("config/tune.yaml"))

    assert config.config_path == Path("config/tune.yaml").resolve()
    assert config.base_run["tracking"]["mlflow_experiment_name"] == "mse-mlops-tuning"
    assert config.tune.metric == "val_roc_auc"
    assert config.tune.scheduler == "fifo"
    assert config.output.best_config_path == Path("reports/tuning/best_config.yaml")


def test_load_tune_config_rejects_invalid_distribution(tmp_path):
    config_path = tmp_path / "tune.yaml"
    config_path.write_text(
        """
base_run:
  model: {model_name: models/pretrained/dino}
  data:
    metadata_csv: data/processed/ham10000/metadata.csv
    images_dir: data/processed/ham10000/HAM10000_images
    label_column: mb
    train_set: train
    val_set: val
    train_fraction: 1.0
    val_fraction: 1.0
    train_samples: null
    val_samples: null
  training:
    output_dir: models/out
    epochs: 1
    batch_size: 8
    image_size: 224
    lr: 0.001
    weight_decay: 0.01
    num_workers: 0
    seed: 1
    device: cpu
    freeze_backbone: true
    gradient_accumulation_steps: 1
    warmup_ratio: 0.1
    lr_scheduler_type: linear
    max_grad_norm: 1.0
    max_train_batches: null
    max_val_batches: null
    resume_from_checkpoint: null
    save_total_limit: 1
  tracking:
    mlflow_tracking_uri: http://127.0.0.1:5001
    mlflow_experiment_name: mse-mlops-tuning
    mlflow_run_name: null
    mlflow_tags: {}
search_space:
  training:
    lr:
      type: badtype
tune:
  metric: val_roc_auc
  mode: max
  num_samples: 1
  scheduler: fifo
  search_alg: basic_variant
  resources: {cpu: 1, gpu: 0}
output:
  ray_results_dir: reports/ray_results
  best_config_path: reports/tuning/best.yaml
  leaderboard_path: reports/tuning/leaderboard.json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported distribution type"):
        tune.load_tune_config(config_path)


def test_build_ray_search_space_maps_supported_distributions(monkeypatch):
    class FakeTuneModule:
        @staticmethod
        def choice(values):
            return ("choice", values)

        @staticmethod
        def loguniform(lower, upper):
            return ("loguniform", lower, upper)

    monkeypatch.setattr(
        tune,
        "_require_ray",
        lambda: {"ray_tune": FakeTuneModule()},
    )

    built = tune.build_ray_search_space(
        {
            "training": {
                "lr": {"type": "loguniform", "lower": 1e-5, "upper": 1e-3},
                "batch_size": {"type": "choice", "values": [8, 16]},
            }
        }
    )

    assert built["training"]["lr"] == ("loguniform", 1e-5, 1e-3)
    assert built["training"]["batch_size"] == ("choice", [8, 16])


def test_build_trial_train_config_merges_trial_params(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_load_train_config_from_mapping(config_data, *, config_path, overrides=None):
        captured["config_data"] = config_data
        captured["config_path"] = config_path
        captured["overrides"] = overrides
        return "train-config"

    monkeypatch.setattr(
        tune.train_lib,
        "load_train_config_from_mapping",
        fake_load_train_config_from_mapping,
    )

    tune_config = tune.load_tune_config(Path("config/tune.yaml"))
    trial_config = tune.build_trial_train_config(
        tune_config,
        {"training": {"lr": 0.001, "batch_size": 16}},
        trial_id="trial-7",
        trial_root=tmp_path,
    )

    assert trial_config == "train-config"
    assert captured["config_data"]["training"]["lr"] == 0.001
    assert captured["config_data"]["training"]["batch_size"] == 16
    assert captured["config_path"] == Path("config/tune.yaml").resolve()
    assert captured["overrides"] == {
        "output_dir": tmp_path / "artifacts" / "trial-7",
        "mlflow_run_name": "trial-trial-7",
    }


def test_build_trial_train_config_resolves_repo_relative_paths(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    repo_root = Path("C:/repo-root")

    def fake_load_train_config_from_mapping(config_data, *, config_path, overrides=None):
        captured["config_data"] = config_data
        captured["config_path"] = config_path
        captured["overrides"] = overrides
        return "train-config"

    monkeypatch.setattr(
        tune.train_lib,
        "load_train_config_from_mapping",
        fake_load_train_config_from_mapping,
    )
    monkeypatch.setattr(tune.paths, "find_repo_root", lambda _start: repo_root)

    tune_config = tune.load_tune_config(Path("config/tune.yaml"))
    tune.build_trial_train_config(
        tune_config,
        {"training": {"lr": 0.001}},
        trial_id="trial-9",
        trial_root=tmp_path,
    )

    config_data = captured["config_data"]
    assert config_data["data"]["metadata_csv"] == repo_root / "data/processed/ham10000/metadata.csv"
    assert config_data["data"]["images_dir"] == repo_root / "data/processed/ham10000/HAM10000_images"
    assert config_data["model"]["model_name"] == str(repo_root / "models/pretrained/dinov3-vits16-pretrain-lvd1689m")
