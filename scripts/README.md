# Scripts

Utility scripts for the MLOps project.

## Data Download

### `download_ham10000.sh`

Download and extract the **HAM10000 skin lesion dataset** from Harvard Dataverse.

**Dataset:** ~10,000 dermatoscopic images + segmentation masks (~3.4 GB)

#### Quick Start

```bash
# Test extraction logic first (no download)
bash scripts/download_ham10000.sh --test-extract

# Check URL reachability
bash scripts/download_ham10000.sh --check-url

# Download and extract full dataset
bash scripts/download_ham10000.sh
```

Or use the Makefile:

```bash
make data-download
```

#### Output Structure

```text
data/raw/
├── dataverse_files.zip                          ← Downloaded archive (kept)
└── dataverse_files/
    ├── HAM10000_metadata.csv                    ← Metadata (10,000+ rows)
    ├── ISIC_*.jpg                               ← Images (10,015 files)
    └── HAM10000_segmentations_lesion_tschandl/  ← Segmentation masks
        └── ISIC_*_segmentation.png              ← Masks (10,015 files)
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