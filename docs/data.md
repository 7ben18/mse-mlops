# Data Acquisition

This project uses the HAM10000 dataset plus one pretrained DINOv3 backbone model.
All acquisition scripts live under `scripts/` and can also be triggered through
the project `Makefile`.

---

## HAM10000 Dataset

**Source:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
**License:** CC0 (Public Domain)
**Size:** ~3.4 GB (10,015 dermatoscopic images + segmentation masks)

### Quick Start

```bash
# Test extraction logic first (no download)
bash scripts/download_ham10000.sh --test-extract

# Check URL reachability
bash scripts/download_ham10000.sh --check-url

# Download and normalize full dataset (~3.4 GB)
bash scripts/download_ham10000.sh
```

### Output Structure

```text
data/raw/ham10000/
├── HAM10000_metadata.csv                             ← Metadata (10,000+ rows)
├── HAM10000_images/                                  ← Dermoscopic images (10,015 files)
└── HAM10000_segmentations_lesion_tschandl/           ← Segmentation masks (10,015 files)
```

### CLI Options

| Option           | Description                                   |
| ---------------- | --------------------------------------------- |
| `--test-extract` | Test extraction with dummy ZIPs (no download) |
| `--check-url`    | Check if URL is reachable and show file size  |
| `--help`         | Show help message                             |
| *(no args)*      | Download and extract full dataset             |

### Requirements

- `curl` — for downloading
- `unzip` — for extracting
- `bash` — for running

## DINOv3 Pretrained Backbone

**Source:** [Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
**Default model ID:** `facebook/dinov3-vits16-pretrain-lvd1689m`

The training pipeline expects the pretrained backbone to be cached locally.
You must request access on Hugging Face before downloading. The project stores the
backbone under `models/pretrained/`.

### Quick Start

```bash
# Log in to Hugging Face
hf auth login

# Download the default model
uv run python scripts/download_model.py

# Download a different model or change output location
uv run python scripts/download_model.py \
    --model-id facebook/dinov3-vits16-pretrain-lvd1689m \
    --output-dir models/pretrained/my-model
```

### CLI Options

| Option         | Default                                              | Description                           |
| -------------- | ---------------------------------------------------- | ------------------------------------- |
| `--model-id`   | `facebook/dinov3-vits16-pretrain-lvd1689m`           | Hugging Face model ID to download     |
| `--output-dir` | `models/pretrained/dinov3-vits16-pretrain-lvd1689m` | Directory where model files are saved |

### Output Structure

```text
models/pretrained/dinov3-vits16-pretrain-lvd1689m/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

The default config (`config/train.yaml`) expects the model at this exact path. After the
download finishes, add it to DVC yourself if you want to track it:

`dvc add models/pretrained/dinov3-vits16-pretrain-lvd1689m`
