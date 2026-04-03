from pathlib import Path

import yaml

REPO_FILES = [
    "README.md",
    "compose.yaml",
    "config/train.yaml",
    "docs/data.md",
    "docs/index.md",
    "docs/mlflow-tracking.md",
    "docs/modules.md",
    "docs/pipeline.md",
    "docs/serving.md",
    "docs/serving-architecture.md",
    "mkdocs.yml",
    "scripts/download_model.py",
    "scripts/train.py",
    "src/mse_mlops/modeling.py",
    "src/mse_mlops/serving/inference.py",
    "src/mse_mlops/train.py",
]

FORBIDDEN_SNIPPETS = (
    "outputs/",
    "data/raw/melanoma_cancer_dataset",
    "ImageFolder",
    "load_best_model_at_end",
    "melanoma_dataset",
)


def test_repo_training_contract_has_no_legacy_paths():
    repo_root = Path(__file__).resolve().parents[1]

    for relative_path in REPO_FILES:
        content = (repo_root / relative_path).read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            assert snippet not in content, f"Found legacy snippet {snippet!r} in {relative_path}"


def test_src_package_has_no_cli_entrypoints():
    repo_root = Path(__file__).resolve().parents[1]
    forbidden_snippets = (
        'if __name__ == "__main__":',
        "import argparse",
        "from argparse import",
    )

    for path in (repo_root / "src").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in content, f"Found CLI snippet {snippet!r} in {path.relative_to(repo_root)}"


def test_pyproject_has_no_console_script_entrypoints():
    repo_root = Path(__file__).resolve().parents[1]
    content = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" not in content


def test_compose_has_default_mlflow_service_and_docker_training_uses_it():
    repo_root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((repo_root / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    train_config = yaml.safe_load((repo_root / "config" / "train.yaml").read_text(encoding="utf-8"))

    mlflow_service = services["mlflow"]
    assert "5001:5001" in mlflow_service["ports"]
    assert "./:/repo" in mlflow_service["volumes"]
    assert "sqlite:////repo/mlflow.db" in mlflow_service["command"]
    assert "file:///repo/mlartifacts" in mlflow_service["command"]
    assert "--allowed-hosts" in mlflow_service["command"]
    assert "localhost:5001,127.0.0.1:5001,mlflow:5001" in mlflow_service["command"]

    train_service = services["train"]
    assert train_service["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert train_service["command"] == [
        "uv",
        "run",
        "--no-sync",
        "--frozen",
        "--no-dev",
        "python",
        "scripts/train.py",
    ]
    assert train_config["tracking"]["mlflow_tracking_uri"] == "http://mlflow:5001"

    api_service = services["api"]
    ui_service = services["ui"]
    assert api_service["profiles"] == ["ui"]
    assert ui_service["profiles"] == ["ui"]
