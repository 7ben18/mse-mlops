# docs/

This folder is the **primary source directory for the project's MkDocs documentation site**.

All Markdown pages, Jupyter notebooks, and other content served by the documentation site live here. The site is built and configured via [`mkdocs.yml`](../mkdocs.yml) at the root of the repository.

## Structure

| File / Folder | Purpose                                           |
| ------------- | ------------------------------------------------- |
| `index.md`    | Site home page                                    |
| `pipeline.md` | Pipeline overview                                 |
| `modules.md`  | Auto-generated API reference (via `mkdocstrings`) |
| `notebooks/`  | Jupyter notebooks rendered as documentation pages |

## Local development

The following commands use the [`Makefile`](../Makefile) at the root of the repository. Run them from there.

Serve the docs locally with live-reload:

```bash
make docs
```

Build and validate the site (fails on warnings or errors):

```bash
make docs-test
```

## Deployment

The documentation is automatically deployed to GitHub Pages via the `.github/workflows/deploy-docs.yml` workflow on every push to `main`.

## Theme & plugins

The site uses the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme and the following plugins:

- **`mkdocs-jupyter`** — renders `.ipynb` notebooks as pages
- **`mkdocstrings`** — generates API reference from Python docstrings in `src/`
- **`search`** — built-in full-text search
