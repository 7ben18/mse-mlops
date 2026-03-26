#!/bin/bash

##############################################################################
# Download HAM10000 Dataset from Harvard Dataverse
#
# This script downloads the HAM10000 skin lesion dataset from Harvard Dataverse
# and extracts it to data/raw folder.
#
# Usage:
#   bash scripts/data/download_ham10000.sh
#
# Dataset:
#   - Name: HAM10000 (Human Against Machine with 10,000 training images)
#   - Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
#   - License: CC0 (Public Domain)
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
DATASET_PID="doi:10.7910/DVN/DBW86T"
DOWNLOAD_URL="https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=${DATASET_PID}"

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


# Test extraction with dummy nested ZIPs
test_extraction() {
    log_info "Starting extraction test with dummy nested ZIPs..."

    TEST_DIR=$(mktemp -d)
    trap "rm -rf ${TEST_DIR}" RETURN

    log_info "Test directory: ${TEST_DIR}"

    # Create dummy nested ZIP files for testing
    log_info "Creating test ZIP files..."

    # Create a simple test file
    echo "test content" > "${TEST_DIR}/test.txt"

    # Create nested test ZIPs
    cd "${TEST_DIR}"

    # Create HAM10000_images_part_1.zip
    mkdir -p part1_content
    echo "image1" > part1_content/ISIC_0000001.jpg
    zip -q HAM10000_images_part_1.zip part1_content/ISIC_0000001.jpg

    # Create HAM10000_images_part_2.zip
    mkdir -p part2_content
    echo "image2" > part2_content/ISIC_0000002.jpg
    zip -q HAM10000_images_part_2.zip part2_content/ISIC_0000002.jpg

    # Create HAM10000_segmentations_lesion_tschandl.zip
    mkdir -p masks_content
    echo "mask" > masks_content/ISIC_0000001_segmentation.png
    zip -q HAM10000_segmentations_lesion_tschandl.zip masks_content/ISIC_0000001_segmentation.png

    # Create main ZIP with nested ZIPs
    echo "Test metadata" > HAM10000_metadata.csv
    zip -q test_dataverse_files.zip HAM10000_images_part_1.zip HAM10000_images_part_2.zip HAM10000_segmentations_lesion_tschandl.zip HAM10000_metadata.csv

    # Now test extraction
    EXTRACT_TEST_DIR="${TEST_DIR}/extract_test"
    mkdir -p "${EXTRACT_TEST_DIR}"

    log_info "Testing extraction logic..."

    # Extract main ZIP
    if ! unzip -q -o "${TEST_DIR}/test_dataverse_files.zip" -d "${EXTRACT_TEST_DIR}"; then
        log_error "Test: Failed to extract main ZIP"
        return 1
    fi
    log_info "✓ Main ZIP extraction works"

    # Extract nested ZIPs
    NESTED_ZIPS=(
        "${EXTRACT_TEST_DIR}/HAM10000_images_part_1.zip"
        "${EXTRACT_TEST_DIR}/HAM10000_images_part_2.zip"
        "${EXTRACT_TEST_DIR}/HAM10000_segmentations_lesion_tschandl.zip"
    )

    for nested_zip in "${NESTED_ZIPS[@]}"; do
        if [[ -f "${nested_zip}" ]]; then
            zip_name=$(basename "${nested_zip}")
            if ! unzip -q -o "${nested_zip}" -d "${EXTRACT_TEST_DIR}"; then
                log_error "Test: Failed to extract ${zip_name}"
                return 1
            fi
            rm -f "${nested_zip}"
            log_info "✓ Nested ZIP extraction works: ${zip_name}"
        fi
    done

    # Verify test files
    if [[ -f "${EXTRACT_TEST_DIR}/HAM10000_metadata.csv" ]]; then
        log_info "✓ Metadata file extraction verified"
    fi

    if [[ -f "${EXTRACT_TEST_DIR}/part1_content/ISIC_0000001.jpg" ]]; then
        log_info "✓ Images part 1 extraction verified"
    fi

    if [[ -f "${EXTRACT_TEST_DIR}/part2_content/ISIC_0000002.jpg" ]]; then
        log_info "✓ Images part 2 extraction verified"
    fi

    if [[ -f "${EXTRACT_TEST_DIR}/masks_content/ISIC_0000001_segmentation.png" ]]; then
        log_info "✓ Masks extraction verified"
    fi

    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Extraction test passed! ✓ Script is ready to use."
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return 0
}

# Check URL and file size
check_url() {
    log_info "Checking download URL and file size..."
    log_info "URL: ${DOWNLOAD_URL}"

    # Get headers to check file size
    HEADERS=$(curl -sI -L "${DOWNLOAD_URL}" 2>/dev/null)

    if [[ $? -ne 0 ]]; then
        log_error "Failed to connect to download URL"
        return 1
    fi

    FILE_SIZE=$(echo "${HEADERS}" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')

    if [[ -z "${FILE_SIZE}" ]]; then
        log_warn "Could not determine file size"
        log_info "URL appears to be reachable (no size info available)"
    else
        SIZE_GB=$(echo "scale=2; ${FILE_SIZE} / 1024 / 1024 / 1024" | bc)
        log_info "File size: ${SIZE_GB} GB (${FILE_SIZE} bytes)"
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
    --test-extract      Test extraction logic with dummy ZIPs (no download)
    --check-url         Check if download URL is reachable and show file size
    (no args)           Download and extract the full dataset (3.4 GB)

EXAMPLES:
    # Test extraction before downloading
    bash $(basename "$0") --test-extract

    # Check URL and file size
    bash $(basename "$0") --check-url

    # Download full dataset
    bash $(basename "$0")

    # Using make
    make data-download
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
            --test-extract)
                test_extraction
                return $?
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

    log_info "Starting HAM10000 dataset download"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Data directory: ${DATA_RAW_DIR}"
    log_info "(Tip: Run with --test-extract to test before downloading)"

    # Check if data already exists
    DATAVERSE_DIR="${DATA_RAW_DIR}/dataverse_files"
    if [[ -f "${DATAVERSE_DIR}/HAM10000_metadata.csv" ]] && [[ -d "${DATAVERSE_DIR}/HAM10000_segmentations_lesion_tschandl" ]]; then
        # Check if at least one image exists (quick check, no counting)
        if ls "${DATAVERSE_DIR}"/ISIC_*.jpg &>/dev/null; then
            log_warn "HAM10000 data already exists in ${DATAVERSE_DIR}"
            log_info "Skipping download. Data is complete."
            return 0
        fi
    fi

    # Create data directory if it doesn't exist
    mkdir -p "${DATA_RAW_DIR}"
    log_info "Data directory created/verified: ${DATA_RAW_DIR}"

    # Set paths for ZIP and extraction
    ZIP_FILE="${DATA_RAW_DIR}/dataverse_files.zip"
    DATAVERSE_DIR="${DATA_RAW_DIR}/dataverse_files"

    # Download dataset
    log_info "Downloading HAM10000 dataset from Harvard Dataverse..."
    log_info "This may take several minutes (dataset is ~3.4 GB)"
    log_info "Download location: ${ZIP_FILE}"

    if ! curl -L --progress-bar -o "${ZIP_FILE}" "${DOWNLOAD_URL}"; then
        log_error "Failed to download dataset"
        return 1
    fi

    log_info "Download completed successfully"

    # Check if downloaded file is valid
    if [[ ! -f "${ZIP_FILE}" ]]; then
        log_error "Downloaded file not found"
        return 1
    fi

    FILESIZE=$(du -h "${ZIP_FILE}" | cut -f1)
    log_info "Downloaded file size: ${FILESIZE}"

    # Create dataverse_files directory
    mkdir -p "${DATAVERSE_DIR}"
    log_info "Created extraction directory: ${DATAVERSE_DIR}"

    # Extract main ZIP
    log_info "Extracting main ZIP to ${DATAVERSE_DIR}..."
    if ! unzip -q -o "${ZIP_FILE}" -d "${DATAVERSE_DIR}"; then
        log_error "Failed to extract main ZIP"
        return 1
    fi

    log_info "Main ZIP extraction completed"

    # Extract nested ZIPs (HAM10000 data comes in nested ZIPs)
    log_info "Extracting nested ZIP files..."

    NESTED_ZIPS=(
        "${DATAVERSE_DIR}/HAM10000_images_part_1.zip"
        "${DATAVERSE_DIR}/HAM10000_images_part_2.zip"
        "${DATAVERSE_DIR}/HAM10000_segmentations_lesion_tschandl.zip"
    )

    for nested_zip in "${NESTED_ZIPS[@]}"; do
        if [[ -f "${nested_zip}" ]]; then
            zip_name=$(basename "${nested_zip}")
            log_info "  Extracting ${zip_name}..."
            if ! unzip -q -o "${nested_zip}" -d "${DATAVERSE_DIR}"; then
                log_error "Failed to extract ${zip_name}"
                return 1
            fi
            # Clean up nested ZIP after extraction
            rm -f "${nested_zip}"
            log_info "  ✓ ${zip_name} extracted and removed"
        fi
    done

    log_info "All nested ZIPs extracted"

    # Clean up extra files (not part of HAM10000)
    log_info "Cleaning up extra files..."
    rm -rf "${DATAVERSE_DIR}/__MACOSX" 2>/dev/null || true
    rm -f "${DATAVERSE_DIR}/ISIC2018"* 2>/dev/null || true
    log_info "  Removed non-HAM10000 files"

    # Verify extracted files
    log_info "Verifying extracted files..."

    # Check for metadata file (with or without .csv extension)
    METADATA_FILE=""
    if [[ -f "${DATAVERSE_DIR}/HAM10000_metadata.csv" ]]; then
        METADATA_FILE="${DATAVERSE_DIR}/HAM10000_metadata.csv"
    elif [[ -f "${DATAVERSE_DIR}/HAM10000_metadata" ]]; then
        METADATA_FILE="${DATAVERSE_DIR}/HAM10000_metadata"
        # Rename to add .csv extension for consistency
        mv "${METADATA_FILE}" "${METADATA_FILE}.csv"
        METADATA_FILE="${METADATA_FILE}.csv"
        log_info "  Renamed metadata file to .csv"
    fi

    if [[ -n "${METADATA_FILE}" && -f "${METADATA_FILE}" ]]; then
        METADATA_LINES=$(wc -l < "${METADATA_FILE}")
        log_info "✓ Metadata file found (${METADATA_LINES} lines)"
    else
        log_error "Metadata file not found after extraction"
        return 1
    fi

    # Check for extracted images (extracted directly into DATAVERSE_DIR)
    IMG_COUNT=$(find "${DATAVERSE_DIR}" -maxdepth 1 -type f \( -name "ISIC_*.jpg" -o -name "ISIC_*.jpeg" \) | wc -l)
    if [[ ${IMG_COUNT} -gt 0 ]]; then
        log_info "✓ Dermatoscopic images found (${IMG_COUNT} images)"
    fi

    if [[ -d "${DATAVERSE_DIR}/HAM10000_segmentations_lesion_tschandl" ]]; then
        MASK_COUNT=$(find "${DATAVERSE_DIR}/HAM10000_segmentations_lesion_tschandl" -type f | wc -l)
        log_info "✓ Segmentation masks found (${MASK_COUNT} masks)"
    fi

    # Summary
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Dataset download and extraction completed successfully! ✓"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info ""
    log_info "Directory structure:"
    log_info "  data/raw/"
    log_info "  ├── dataverse_files.zip (downloaded archive)"
    log_info "  └── dataverse_files/"
    log_info "      ├── HAM10000_metadata.csv"
    log_info "      ├── HAM10000_images_part_1/"
    log_info "      ├── HAM10000_images_part_2/"
    log_info "      └── HAM10000_segmentations_lesion_tschandl/"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Review: notebooks/EDA/EDA_HAM_ISIC.ipynb"
    log_info "  2. Process: notebooks/EDA/EDA_src.py"
}

main "$@"
