from pathlib import Path

REPO_FILES = [
    "README.md",
    "compose.yaml",
    "config/train.yaml",
    "docs/data.md",
    "docs/index.md",
    "docs/mlflow-tracking.md",
    "docs/pipeline.md",
    "docs/serving.md",
    "docs/serving-architecture.md",
    "scripts/download_model.py",
    "scripts/train.py",
    "src/mse_mlops/serving/inference.py",
    "src/mse_mlops/train.py",
]

FORBIDDEN_SNIPPETS = (
    "outputs/",
    "data/raw/melanoma_cancer_dataset",
    "ImageFolder",
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
