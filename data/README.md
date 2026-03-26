# Data

This file is the single tracked data overview for the repository.

## Policy

- Keep dataset contents out of git.
- Put raw source data under `data/raw/`.
- Put derived tables and intermediate analysis outputs under `data/processed/`.
- Keep acquisition dates in dataset-specific `DATE.txt` files where needed.

## HAM10000

### Source

- Harvard Dataverse: `doi:10.7910/DVN/DBW86T`
- License: CC0 / Public Domain
- Download script: `scripts/download_ham10000.sh`

## Melanoma Cancer Dataset

### Source

- Kaggle: `https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images`
- Latest recorded local acquisition date: `20260226`

### Expected local layout

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
