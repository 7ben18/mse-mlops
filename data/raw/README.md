# Raw Data

This folder contains raw datasets for the MLOps project.

## HAM10000 Dataset

**Dermatoscopic images of skin lesions** for melanoma detection.

### Data Source & Download

This data was downloaded using the script in `scripts/` folder:

- **Script:** `scripts/download_ham10000.sh`
- **Quick command:** `make data-download`
- **Source:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **License:** CC0 (Public Domain)

For download instructions, see `scripts/README.md`

### Folder Structure

```text
raw/
├── dataverse_files.zip                          ← Downloaded ZIP archive
└── dataverse_files/
    ├── HAM10000_metadata.csv                    ← Metadata (10,016 rows)
    ├── ISIC_*.jpg                               ← Images (10,015 files)
    └── HAM10000_segmentations_lesion_tschandl/  ← Segmentation masks
        └── ISIC_*_segmentation.png              ← Masks (10,015 files)
```

### Dataset Information

- **Name:** HAM10000 (Human Against Machine with 10,000 training images)
- **Size:** ~3.4 GB (10,015 images + masks)
- **Diagnosis:** 7 types of skin lesions (malignant & benign)
- **Metadata:** Age, sex, body location, diagnosis type

### Metadata Columns

- `image_id` - Unique image ID (ISIC_0027419)
- `lesion_id` - Lesion identifier (multiple images per lesion)
- `dx` - Diagnosis (mel, bcc, akiec, nv, bkl, df, vasc)
- `age` - Patient age
- `sex` - Patient gender
- `localization` - Body part location

### References

- Paper: [Nature Scientific Data (2018)](https://www.nature.com/articles/sdata2018161)
- GitHub: [ptschandl/HAM10000_dataset](https://github.com/ptschandl/HAM10000_dataset)
