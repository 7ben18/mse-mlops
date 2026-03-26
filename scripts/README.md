# Scripts

Utility scripts for the MLOps project.

## Data Download

### `download_ham10000.sh`

Download and normalize the **HAM10000 skin lesion dataset** from Harvard Dataverse.

**Dataset:** ~10,000 dermatoscopic images + segmentation masks (~3.4 GB)

#### Quick Start

```bash
# Test extraction logic first (no download)
bash scripts/download_ham10000.sh --test-extract

# Check URL reachability
bash scripts/download_ham10000.sh --check-url

# Download and normalize full dataset
bash scripts/download_ham10000.sh
```

Or use the Makefile:

```bash
make data-download
```

#### Output Structure

```text
data/raw/ham10000/
├── HAM10000_metadata.csv                        ← Metadata (10,000+ rows)
├── HAM10000_images/                             ← Images (10,015 files)
└── HAM10000_segmentations_lesion_tschandl/      ← Segmentation masks
    └── ISIC_*_segmentation.png                  ← Masks (10,015 files)
```

#### Options

| Option | Description |
|--------|-------------|
| `--test-extract` | Test extraction with dummy ZIPs (no download) |
| `--check-url` | Check if URL is reachable and show file size |
| `--help` | Show help message |
| (no args) | Download and extract full dataset |

#### Features

- ✅ Handles nested ZIP files (ZIP inside ZIP)
- ✅ Automatic extraction of all nested ZIPs
- ✅ Normalizes extracted files into the canonical repo layout
- ✅ Deletes the temporary top-level ZIP after a successful normalization
- ✅ Cleans up extra files (ISIC2018, __MACOSX)
- ✅ Validates extracted files (metadata, images, masks)
- ✅ Progress bar during download
- ✅ Test mode before committing to 3GB download
- ✅ Renames metadata file for consistency

#### Requirements

- `curl` - for downloading
- `unzip` - for extracting
- `bash` - for running

#### Source

**Dataset:** HAM10000 (Human Against Machine with 10,000 training images)
**From:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
**License:** CC0 (Public Domain)

---

### `download_melanoma.sh`

Download and extract the **Melanoma Skin Cancer Dataset** from Kaggle.

**Dataset:** ~10,600 dermoscopy images in train/test splits (~104 MB)

#### Prerequisites

A Kaggle API credentials file at `~/.kaggle/kaggle.json`:

```json
{"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
```

Get your API key at [kaggle.com/settings/account](https://www.kaggle.com/settings/account), then:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

#### Quick Start

```bash
# Check URL reachability
bash scripts/download_melanoma.sh --check-url

# Download and extract full dataset
bash scripts/download_melanoma.sh
```

Or use the Makefile:

```bash
make data-download-kaggle
```

#### Output Structure

```text
data/raw/
├── archive.zip                   ← Downloaded archive (kept)
└── melanoma_cancer_dataset/
    ├── train/
    │   ├── benign/               ← 5,000 images
    │   └── malignant/            ← 4,605 images
    └── test/
        ├── benign/               ← 500 images
        └── malignant/            ← 500 images
```

#### Options

| Option | Description |
|--------|-------------|
| `--check-url` | Check if URL is reachable and show file size |
| `--help` | Show help message |
| (no args) | Download and extract full dataset |

#### Features

- ✅ Reads credentials from `~/.kaggle/kaggle.json` automatically
- ✅ Idempotent — skips download if data already exists
- ✅ Progress bar during download
- ✅ Validates all four split/class directories after extraction
- ✅ Cleans up `__MACOSX` metadata artifacts

#### Requirements

- `curl` - for downloading
- `unzip` - for extracting
- `python3` - for parsing `kaggle.json`
- `bash` - for running

#### Source

**Dataset:** Melanoma Skin Cancer Dataset of 10,000 Images
**From:** [Kaggle](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
**License:** Data files © Original Authors