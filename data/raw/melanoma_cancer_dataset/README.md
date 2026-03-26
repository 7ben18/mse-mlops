# Melanoma Dataset Provenance

## Source

This local dataset was sourced from Kaggle:

`https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images`

Keep this file updated if the upstream source, curation process, or local folder layout changes.

## Local Acquisition Notes

- Record the local acquisition or refresh date in `DATE.txt` using `YYYYMMDD`.
- Keep raw data out of git.
- Manage dataset files with DVC or the agreed data-management workflow for this project.

## Expected Local Layout

The repo expects a curated image-folder layout under this directory:

```text
data/raw/melanoma_cancer_dataset/
  train/
    benign/
    malignant/
  val/                # optional dedicated validation split
    benign/
    malignant/
  future/             # optional production-collected pool kept separate until curated
  test/
    benign/
    malignant/
```

Current default training behavior uses `train/` plus a stratified validation split from `train/`, leaving `test/` untouched for the final pre-production evaluation.
