# mse-mlops-project

[![Build status](https://img.shields.io/github/actions/workflow/status/github_handle/mse-mlops-project/main.yml?branch=main)](https://github.com/github_handle/mse-mlops-project/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/github_handle/mse-mlops-project)](https://img.shields.io/github/commit-activity/m/github_handle/mse-mlops-project)
[![License](https://img.shields.io/github/license/github_handle/mse-mlops-project)](https://img.shields.io/github/license/github_handle/mse-mlops-project)

A short description of the project

## Resources

- 🚀 [Repository](https://github.com/github_handle/mse-mlops-project/)
- 📖 [Documentation](https://github_handle.github.io/mse-mlops-project/)

## Getting started with your project

Follow the instructions under https://gabrieltorresgamez.github.io/gabocutter/tutorial/ to get started.

## Structure

    ├── .github
    │   ├── actions        <- Github Actions configuration.
    │   └── workflows      <- Github Actions workflows.
    │   
    ├── mse-mlops-project <- Source code for use in this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- MkDocs documentation for the project.
    ├── models             <- Modelcheckpoints, model predictions, metrics, and model summaries.
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