# mse-mlops

[![Build status](https://img.shields.io/github/actions/workflow/status/7ben18/mse-mlops/main.yml?branch=main)](https://github.com/7ben18/mse-mlops/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/7ben18/mse-mlops)](https://github.com/7ben18/mse-mlops/commits/main)
[![License](https://img.shields.io/github/license/7ben18/mse-mlops)](https://github.com/7ben18/mse-mlops/blob/main/LICENSE)

DINOv3 fine-tuning setup for melanoma skin cancer classification.

## Resources

- 🚀 [Repository](https://github.com/7ben18/mse-mlops)
- 📖 [Documentation](https://7ben18.github.io/mse-mlops/)

## DINOv3 Fine-Tuning

Fine-tuning DINOv3 for melanoma skin cancer classification.

Model:

`facebook/dinov3-vits16-pretrain-lvd1689m`

Dataset download (manual):

https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

Extract into `data/` so this path exists:

`data/melanoma_cancer_dataset`

Expected dataset structure:

`data/melanoma_cancer_dataset/train/{benign,malignant}`

`data/melanoma_cancer_dataset/test/{benign,malignant}`

Run training:

`uv run train-dinov3`

Training settings:

`config/train.yaml`

## TODO

- Move the dataset from `data/melanoma_cancer_dataset` into either `data/raw` or `data/processed`.
- Update and align all dataset path documentation after that move.

## Structure

    ├── .github
    │   ├── actions        <- Github Actions configuration.
    │   └── workflows      <- Github Actions workflows.
    │
    ├── src/mse_mlops      <- Source code for this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- MkDocs documentation for the project.
    ├── models             <- Model checkpoints, predictions, metrics, and summaries.
    ├── notebooks          <- Jupyter notebooks or Quarto Markdown Notebooks.
    │                         Naming convention is a number (for ordering) and a short `-`
    │                         delimited description, e.g. `00-example.qmd`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    ├── tests              <- Unit tests for the project.
    ├── .gitignore         <- Files to be ignored by git.
    ├── Dockerfile         <- Dockerfile for the Docker image.
    ├── LICENSE            <- MIT License.
    ├── Makefile           <- Makefile with commands like `make install` or `make test`.
    ├── mkdocs.yml         <- MkDocs configuration.
    ├── pyproject.toml     <- Package build configuration.
    ├── README.md          <- The top-level README for this project.
    └── uv.lock            <- Lock file for uv.
