# docs/

This folder is the **primary source directory for the project's MkDocs documentation site**.

Markdown pages live here directly. Notebook pages are sourced from the top-level [`notebooks/`](../notebooks/README.md) tree through the tracked `docs/notebooks -> ../notebooks` symlink and are referenced from [`mkdocs.yml`](../mkdocs.yml).

## Structure

| File / Folder | Purpose                                           |
| ------------- | ------------------------------------------------- |
| `index.md`    | Site home page                                          |
| `pipeline.md` | Pipeline overview                                       |
| `modules.md`  | Auto-generated API reference (via `mkdocstrings`)       |
| `notebooks/`  | Symlink to top-level notebooks rendered as doc pages    |

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
