from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mse_mlops import paths
from mse_mlops import tracking
from mse_mlops import train as train_lib

DEFAULT_TUNE_CONFIG_PATH = Path("config/tune.yaml")
TRAIN_CONFIG_SECTIONS = ("model", "data", "training", "tracking")
SEARCH_SPACE_TYPES = frozenset({"choice", "uniform", "loguniform", "randint"})
TUNE_MODES = frozenset({"min", "max"})
SEARCH_ALGORITHMS = frozenset({"basic_variant"})
SCHEDULERS = frozenset({"fifo", "asha"})


@dataclass(frozen=True)
class TuneSection:
    metric: str
    mode: str
    num_samples: int
    search_alg: str
    scheduler: str
    max_concurrent_trials: int | None
    resources: dict[str, float]
    seed: int | None
    ray_address: str | None


@dataclass(frozen=True)
class OutputSection:
    ray_results_dir: Path
    best_config_path: Path
    leaderboard_path: Path


@dataclass(frozen=True)
class TuneConfig:
    config_path: Path
    base_run: dict[str, Any]
    search_space: dict[str, Any]
    tune: TuneSection
    output: OutputSection


def _read_yaml_mapping(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def _as_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{field_name}' must be a YAML mapping.")
    return dict(value)


def _as_path(value: object, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"Config setting '{field_name}' must be set.")
    return Path(str(value))


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float_mapping(value: object, field_name: str) -> dict[str, float]:
    mapping = _as_mapping(value, field_name)
    resources = {str(key): float(raw_value) for key, raw_value in mapping.items()}
    if not resources:
        raise ValueError(f"Config section '{field_name}' must not be empty.")
    return resources


def _validate_base_run(base_run: dict[str, object]) -> None:
    for section_name in TRAIN_CONFIG_SECTIONS:
        if section_name not in base_run:
            raise ValueError(f"Tune config is missing required base_run section: {section_name}")
        if not isinstance(base_run[section_name], dict):
            raise ValueError(f"base_run.{section_name} must be a YAML mapping.")


def _validate_search_space_node(node: object, path: str) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"Search space entry '{path}' must be a YAML mapping.")

    if "type" in node:
        distribution_type = str(node["type"])
        if distribution_type not in SEARCH_SPACE_TYPES:
            allowed = ", ".join(sorted(SEARCH_SPACE_TYPES))
            raise ValueError(f"Unsupported distribution type '{distribution_type}' at '{path}'. Expected one of: {allowed}")
        return

    for key, value in node.items():
        child_path = f"{path}.{key}" if path else str(key)
        _validate_search_space_node(value, child_path)


def load_tune_config(config_path: Path | str = DEFAULT_TUNE_CONFIG_PATH) -> TuneConfig:
    resolved_path = Path(config_path).resolve()
    raw = _read_yaml_mapping(resolved_path)

    required_sections = ("base_run", "search_space", "tune", "output")
    missing = [name for name in required_sections if name not in raw]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Tune config is missing required sections: {missing_str}")

    base_run = _as_mapping(raw["base_run"], "base_run")
    _validate_base_run(base_run)

    search_space = _as_mapping(raw["search_space"], "search_space")
    _validate_search_space_node(search_space, "search_space")

    tune_mapping = _as_mapping(raw["tune"], "tune")
    tune_section = TuneSection(
        metric=str(tune_mapping["metric"]),
        mode=str(tune_mapping["mode"]),
        num_samples=int(tune_mapping["num_samples"]),
        search_alg=str(tune_mapping.get("search_alg", "basic_variant")),
        scheduler=str(tune_mapping.get("scheduler", "fifo")),
        max_concurrent_trials=_as_optional_int(tune_mapping.get("max_concurrent_trials")),
        resources=_as_float_mapping(tune_mapping.get("resources", {"cpu": 1}), "tune.resources"),
        seed=_as_optional_int(tune_mapping.get("seed")),
        ray_address=_as_optional_str(tune_mapping.get("ray_address")),
    )

    if tune_section.metric.strip() == "":
        raise ValueError("tune.metric must be set.")
    if tune_section.mode not in TUNE_MODES:
        raise ValueError("tune.mode must be 'min' or 'max'.")
    if tune_section.num_samples <= 0:
        raise ValueError("tune.num_samples must be > 0.")
    if tune_section.search_alg not in SEARCH_ALGORITHMS:
        raise ValueError("tune.search_alg must be 'basic_variant'.")
    if tune_section.scheduler not in SCHEDULERS:
        raise ValueError("tune.scheduler must be 'fifo' or 'asha'.")
    if tune_section.max_concurrent_trials is not None and tune_section.max_concurrent_trials <= 0:
        raise ValueError("tune.max_concurrent_trials must be > 0 when provided.")

    output_mapping = _as_mapping(raw["output"], "output")
    output = OutputSection(
        ray_results_dir=_as_path(output_mapping["ray_results_dir"], "output.ray_results_dir"),
        best_config_path=_as_path(output_mapping["best_config_path"], "output.best_config_path"),
        leaderboard_path=_as_path(output_mapping["leaderboard_path"], "output.leaderboard_path"),
    )

    return TuneConfig(
        config_path=resolved_path,
        base_run=base_run,
        search_space=search_space,
        tune=tune_section,
        output=output,
    )


def _require_ray():
    try:
        import ray
        from ray import tune as ray_tune
        from ray.air import RunConfig
        from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
        from ray.tune.search import BasicVariantGenerator
    except ImportError as error:
        raise RuntimeError(
            "Ray Tune is not installed. Run `uv sync` after adding the ray[tune] dependency."
        ) from error

    return {
        "ray": ray,
        "ray_tune": ray_tune,
        "RunConfig": RunConfig,
        "ASHAScheduler": ASHAScheduler,
        "FIFOScheduler": FIFOScheduler,
        "BasicVariantGenerator": BasicVariantGenerator,
    }


def build_ray_search_space(search_space: dict[str, object]) -> dict[str, object]:
    ray_modules = _require_ray()
    ray_tune = ray_modules["ray_tune"]
    return _build_ray_search_space_node(search_space, ray_tune=ray_tune, path="search_space")


def _build_ray_search_space_node(
    node: dict[str, object],
    *,
    ray_tune: Any,
    path: str,
) -> dict[str, object]:
    built: dict[str, object] = {}

    for key, value in node.items():
        current_path = f"{path}.{key}"
        if not isinstance(value, dict):
            raise ValueError(f"Search space entry '{current_path}' must be a YAML mapping.")

        if "type" in value:
            built[key] = _build_distribution(value, ray_tune=ray_tune, path=current_path)
        else:
            built[key] = _build_ray_search_space_node(value, ray_tune=ray_tune, path=current_path)

    return built


def _build_distribution(distribution: dict[str, object], *, ray_tune: Any, path: str) -> object:
    distribution_type = str(distribution["type"])
    if distribution_type == "choice":
        values = distribution.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Choice distribution at '{path}' requires a non-empty 'values' list.")
        return ray_tune.choice(values)

    if distribution_type == "uniform":
        return ray_tune.uniform(float(distribution["lower"]), float(distribution["upper"]))

    if distribution_type == "loguniform":
        return ray_tune.loguniform(float(distribution["lower"]), float(distribution["upper"]))

    if distribution_type == "randint":
        return ray_tune.randint(int(distribution["lower"]), int(distribution["upper"]))

    raise ValueError(f"Unsupported distribution type '{distribution_type}' at '{path}'.")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_trial_output_dir(trial_root: Path, trial_id: str) -> Path:
    return trial_root / "artifacts" / trial_id


def _resolve_repo_relative_path(value: object, repo_root: Path) -> object:
    if isinstance(value, Path):
        if value.is_absolute():
            return value
        return repo_root / value

    text = str(value)
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _resolve_trial_paths(merged_config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(merged_config)
    resolved["data"]["metadata_csv"] = _resolve_repo_relative_path(
        resolved["data"]["metadata_csv"], repo_root
    )
    resolved["data"]["images_dir"] = _resolve_repo_relative_path(
        resolved["data"]["images_dir"], repo_root
    )
    resolved["training"]["output_dir"] = _resolve_repo_relative_path(
        resolved["training"]["output_dir"], repo_root
    )

    model_name = str(resolved["model"]["model_name"])
    model_candidate = Path(model_name)
    if not model_candidate.is_absolute() and (
        "/" in model_name or "\\" in model_name or model_name.startswith(".")
    ):
        resolved["model"]["model_name"] = str(repo_root / model_candidate)

    return resolved


def build_trial_train_config(
    tune_config: TuneConfig,
    trial_params: dict[str, Any],
    *,
    trial_id: str,
    trial_root: Path,
) -> train_lib.TrainConfig:
    repo_root = paths.find_repo_root(tune_config.config_path)
    merged_config = _resolve_trial_paths(
        deep_merge(tune_config.base_run, trial_params),
        repo_root,
    )
    output_dir = _build_trial_output_dir(trial_root, trial_id)
    overrides = {
        "output_dir": output_dir,
        "mlflow_run_name": f"trial-{trial_id}",
    }
    return train_lib.load_train_config_from_mapping(
        merged_config,
        config_path=tune_config.config_path,
        overrides=overrides,
    )


def _make_trial_trainable(
    tune_config: TuneConfig,
    *,
    parent_run_id: str | None,
):
    ray_modules = _require_ray()
    ray_tune = ray_modules["ray_tune"]

    def trainable(trial_params: dict[str, Any]) -> dict[str, Any]:
        context = ray_tune.get_context()
        trial_id = context.get_trial_id()
        trial_root = tune_config.output.ray_results_dir.resolve()
        train_config = build_trial_train_config(
            tune_config,
            trial_params,
            trial_id=trial_id,
            trial_root=trial_root,
        )
        result = train_lib.run_training(
            train_config,
            nested_mlflow=False,
            extra_mlflow_tags={
                "ray_trial_id": trial_id,
                "mlflow.parentRunId": parent_run_id,
            },
            log_model_artifact=False,
            report_callback=ray_tune.report,
        )

        best_metric_name = result.best_metric_name or tune_config.tune.metric
        best_metric_value = result.best_metric_value
        metrics = {
            "trial_id": trial_id,
            "train_count": float(result.train_count),
            "val_count": float(result.val_count),
            "best_metric_name": best_metric_name,
            "best_metric_value": float(best_metric_value) if best_metric_value is not None else float("nan"),
            "promoted_model_path": str(result.promoted_model_path) if result.promoted_model_path is not None else "",
        }
        if result.history:
            final_metrics = result.history[-1]
            metrics.update({
                "train_loss": final_metrics.train_loss,
                "train_acc": final_metrics.train_acc,
                "val_loss": final_metrics.val_loss,
                "val_acc": final_metrics.val_acc,
                "val_precision": final_metrics.val_precision,
                "val_recall": final_metrics.val_recall,
                "val_f1": final_metrics.val_f1,
                "val_roc_auc": final_metrics.val_roc_auc,
            })
        return metrics

    return trainable


def _build_scheduler(tune_config: TuneConfig, ray_modules: dict[str, Any]) -> object:
    if tune_config.tune.scheduler == "asha":
        return ray_modules["ASHAScheduler"](
            metric=tune_config.tune.metric,
            mode=tune_config.tune.mode,
        )
    return ray_modules["FIFOScheduler"]()


def _build_search_alg(tune_config: TuneConfig, ray_modules: dict[str, Any]) -> object:
    return ray_modules["BasicVariantGenerator"](
        max_concurrent=tune_config.tune.max_concurrent_trials,
        random_state=tune_config.tune.seed,
    )


def _build_run_config(tune_config: TuneConfig, ray_modules: dict[str, Any]) -> object:
    ray_results_dir = tune_config.output.ray_results_dir.resolve()
    return ray_modules["RunConfig"](
        name=ray_results_dir.name,
        storage_path=str(ray_results_dir.parent),
    )


def _serialize_trial_entry(result: Any, metric_name: str) -> dict[str, Any]:
    metrics = dict(result.metrics)
    return {
        "trial_id": metrics.get("trial_id", ""),
        "metric": metrics.get(metric_name),
        "config": result.config,
        "metrics": metrics,
        "path": getattr(result, "path", None),
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def _write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _log_best_trial_artifacts(
    tune_config: TuneConfig,
    *,
    best_result: Any,
    leaderboard: list[dict[str, Any]],
) -> None:
    best_payload = {
        "trial_id": best_result.metrics.get("trial_id", ""),
        "metric": tune_config.tune.metric,
        "mode": tune_config.tune.mode,
        "metric_value": best_result.metrics.get(tune_config.tune.metric),
        "best_params": best_result.config,
        "resolved_run": deep_merge(tune_config.base_run, best_result.config),
    }
    _write_yaml(tune_config.output.best_config_path, best_payload)
    _write_json(tune_config.output.leaderboard_path, leaderboard)

    tracking.log_dict_artifact(
        best_payload,
        artifact_path="tuning",
        filename=tune_config.output.best_config_path.name,
    )
    tracking.log_dict_artifact(
        leaderboard,
        artifact_path="tuning",
        filename=tune_config.output.leaderboard_path.name,
    )

    promoted_model_path = str(best_result.metrics.get("promoted_model_path", "")).strip()
    if promoted_model_path:
        tracking.log_local_artifact(promoted_model_path, artifact_path="tuning/best_model")


def run_tuning(config: TuneConfig) -> dict[str, Any]:
    ray_modules = _require_ray()
    ray = ray_modules["ray"]
    ray_tune = ray_modules["ray_tune"]

    config.output.ray_results_dir.mkdir(parents=True, exist_ok=True)

    tracking_config = train_lib.load_train_config_from_mapping(
        config.base_run,
        config_path=config.config_path,
    )

    ray.init(address=config.tune.ray_address or None, ignore_reinit_error=True)
    try:
        with tracking.start_run(
            tracking_uri=str(tracking_config.mlflow_tracking_uri),
            experiment_name=str(tracking_config.mlflow_experiment_name),
            run_name="ray-tune-session",
            tags={"workflow": "tuning"},
        ):
            tracking.log_dict_artifact(
                {
                    "config_path": str(config.config_path),
                    "metric": config.tune.metric,
                    "mode": config.tune.mode,
                    "num_samples": config.tune.num_samples,
                    "search_alg": config.tune.search_alg,
                    "scheduler": config.tune.scheduler,
                    "ray_results_dir": str(config.output.ray_results_dir),
                },
                artifact_path="tuning",
                filename="session.json",
            )
            parent_run_id = tracking.get_active_run_id()

            tuner = ray_tune.Tuner(
                ray_tune.with_resources(
                    _make_trial_trainable(config, parent_run_id=parent_run_id),
                    resources=config.tune.resources,
                ),
                param_space=build_ray_search_space(config.search_space),
                tune_config=ray_tune.TuneConfig(
                    metric=config.tune.metric,
                    mode=config.tune.mode,
                    num_samples=config.tune.num_samples,
                    scheduler=_build_scheduler(config, ray_modules),
                    search_alg=_build_search_alg(config, ray_modules),
                    max_concurrent_trials=config.tune.max_concurrent_trials,
                ),
                run_config=_build_run_config(config, ray_modules),
            )
            result_grid = tuner.fit()
            best_result = result_grid.get_best_result(
                metric=config.tune.metric,
                mode=config.tune.mode,
            )
            leaderboard = [
                _serialize_trial_entry(result, config.tune.metric)
                for result in result_grid
            ]
            _log_best_trial_artifacts(
                config,
                best_result=best_result,
                leaderboard=leaderboard,
            )
            tracking.log_summary_metrics({
                "best_trial_metric": float(best_result.metrics[config.tune.metric]),
            })
    finally:
        ray.shutdown()

    return {
        "best_trial_id": best_result.metrics.get("trial_id", ""),
        "best_metric": best_result.metrics.get(config.tune.metric),
        "best_config": best_result.config,
        "best_config_path": config.output.best_config_path,
        "leaderboard_path": config.output.leaderboard_path,
    }
