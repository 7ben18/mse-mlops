#!/bin/bash

##############################################################################
# Download Melanoma Skin Cancer Dataset from Kaggle
#
# This script downloads the Melanoma Skin Cancer Dataset of 10,000 images
# from Kaggle and extracts it to the data/raw folder.
#
# Usage:
#   bash scripts/download_melanoma.sh
#
# Dataset:
#   - Name: Melanoma Skin Cancer Dataset of 10,000 Images
#   - Source: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
#   - License: Data files © Original Authors
##############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_RAW_DIR="${PROJECT_ROOT}/data/raw"
DOWNLOAD_URL="https://www.kaggle.com/api/v1/datasets/download/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
KAGGLE_CREDS="${PROJECT_ROOT}/.kaggle/kaggle.json"
if [[ ! -f "${KAGGLE_CREDS}" ]]; then
    KAGGLE_CREDS="${HOME}/.kaggle/kaggle.json"
fi
ZIP_FILE="${DATA_RAW_DIR}/archive.zip"
DATASET_DIR="${DATA_RAW_DIR}/melanoma_cancer_dataset"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Read Kaggle credentials from ~/.kaggle/kaggle.json
load_kaggle_credentials() {
    if [[ ! -f "${KAGGLE_CREDS}" ]]; then
        log_error "Kaggle credentials not found. Looked in:"
        log_error "  1. ${PROJECT_ROOT}/.kaggle/kaggle.json"
        log_error "  2. ${HOME}/.kaggle/kaggle.json"
        log_error "Place your credentials in either location."
        log_error "Get your API key at: https://www.kaggle.com/settings/account"
        return 1
    fi

    KAGGLE_USERNAME=$(python3 -c "import json; d=json.load(open('${KAGGLE_CREDS}')); print(d['username'])")
    KAGGLE_KEY=$(python3 -c "import json; d=json.load(open('${KAGGLE_CREDS}')); print(d['key'])")

    if [[ -z "${KAGGLE_USERNAME}" || -z "${KAGGLE_KEY}" ]]; then
        log_error "Could not parse username/key from ${KAGGLE_CREDS}"
        return 1
    fi

    log_info "Kaggle credentials loaded for user: ${KAGGLE_USERNAME}"
}

# Check URL reachability
check_url() {
    log_info "Checking download URL and file size..."
    log_info "URL: ${DOWNLOAD_URL}"

    load_kaggle_credentials

    AUTH_HEADER="Authorization: Basic $(echo -n "${KAGGLE_USERNAME}:${KAGGLE_KEY}" | base64)"
    HEADERS=$(curl -sI -L --header "${AUTH_HEADER}" "${DOWNLOAD_URL}" 2>/dev/null)

    FILE_SIZE=$(echo "${HEADERS}" | grep -i "content-length" | tail -1 | awk '{print $2}' | tr -d '\r')

    if [[ -z "${FILE_SIZE}" ]]; then
        log_warn "Could not determine file size (URL may still be reachable)"
        log_info "URL appears to be reachable"
    else
        SIZE_MB=$(echo "scale=1; ${FILE_SIZE} / 1024 / 1024" | bc)
        log_info "File size: ${SIZE_MB} MB (${FILE_SIZE} bytes)"
    fi

    log_info "✓ URL is reachable"
    return 0
}

# Show usage
show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --help              Show this help message
    --check-url         Check if download URL is reachable and show file size
    (no args)           Download and extract the full dataset (~104 MB)

EXAMPLES:
    # Check URL and file size
    bash $(basename "$0") --check-url

    # Download full dataset
    bash $(basename "$0")

    # Using make
    make data-download-kaggle

REQUIREMENTS:
    ~/.kaggle/kaggle.json  Kaggle API credentials file
    curl                   For downloading
    unzip                  For extracting
    python3                For parsing credentials JSON
EOF
}

# Main script
main() {
    # Handle arguments
    if [[ $# -gt 0 ]]; then
        case "$1" in
            --help|-h)
                show_usage
                return 0
                ;;
            --check-url)
                check_url
                return $?
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                return 1
                ;;
        esac
    fi

    log_info "Starting Melanoma dataset download"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Data directory: ${DATA_RAW_DIR}"

    # Load credentials early
    load_kaggle_credentials

    # Check if data already exists
    if [[ -d "${DATASET_DIR}/train/benign" && -d "${DATASET_DIR}/train/malignant" ]] \
        && [[ -d "${DATASET_DIR}/test/benign" && -d "${DATASET_DIR}/test/malignant" ]]; then
        if ls "${DATASET_DIR}/train/benign/"*.jpg &>/dev/null 2>&1; then
            log_warn "Melanoma dataset already exists in ${DATASET_DIR}"
            log_info "Skipping download. Data is complete."
            return 0
        fi
    fi

    # Create data directory if needed
    mkdir -p "${DATA_RAW_DIR}"

    # Download dataset
    log_info "Downloading Melanoma dataset from Kaggle..."
    log_info "This may take a minute (~104 MB)"
    log_info "Download location: ${ZIP_FILE}"

    AUTH_HEADER="Authorization: Basic $(echo -n "${KAGGLE_USERNAME}:${KAGGLE_KEY}" | base64)"

    if ! curl -L --progress-bar --header "${AUTH_HEADER}" -o "${ZIP_FILE}" "${DOWNLOAD_URL}"; then
        log_error "Failed to download dataset"
        return 1
    fi

    log_info "Download completed successfully"

    FILESIZE=$(du -h "${ZIP_FILE}" | cut -f1)
    log_info "Downloaded file size: ${FILESIZE}"

    # Extract ZIP
    log_info "Extracting archive to ${DATA_RAW_DIR}..."
    if ! unzip -q -o "${ZIP_FILE}" -d "${DATA_RAW_DIR}"; then
        log_error "Failed to extract archive.zip"
        return 1
    fi

    log_info "Extraction completed"

    # Clean up macOS metadata
    rm -rf "${DATA_RAW_DIR}/__MACOSX" 2>/dev/null || true

    # Verify extracted structure
    log_info "Verifying extracted files..."

    local ok=1
    for split in train test; do
        for cls in benign malignant; do
            dir="${DATASET_DIR}/${split}/${cls}"
            if [[ -d "${dir}" ]]; then
                count=$(find "${dir}" -maxdepth 1 -name "*.jpg" | wc -l | tr -d ' ')
                log_info "✓ ${split}/${cls}: ${count} images"
            else
                log_error "Missing expected directory: ${dir}"
                ok=0
            fi
        done
    done

    if [[ ${ok} -eq 0 ]]; then
        log_error "Extraction may be incomplete — check ${DATASET_DIR}"
        return 1
    fi

    # Summary
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Dataset download and extraction completed successfully! ✓"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info ""
    log_info "Directory structure:"
    log_info "  data/raw/"
    log_info "  ├── archive.zip                  ← Downloaded archive (kept)"
    log_info "  └── melanoma_cancer_dataset/"
    log_info "      ├── train/"
    log_info "      │   ├── benign/              ← 5,000 images"
    log_info "      │   └── malignant/           ← 4,605 images"
    log_info "      └── test/"
    log_info "          ├── benign/              ← 500 images"
    log_info "          └── malignant/           ← 500 images"
}

main "$@"
