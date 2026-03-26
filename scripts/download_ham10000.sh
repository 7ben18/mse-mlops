#!/usr/bin/env bash

##############################################################################
# Download HAM10000 Dataset from Harvard Dataverse
#
# This script downloads the HAM10000 skin lesion dataset from Harvard Dataverse
# and normalizes it into the canonical data/raw/ham10000 layout.
#
# Usage:
#   bash scripts/download_ham10000.sh
#
# Dataset:
#   - Name: HAM10000 (Human Against Machine with 10,000 training images)
#   - Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
#   - License: CC0 (Public Domain)
##############################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_RAW_DIR="${PROJECT_ROOT}/data/raw"
HAM10000_DIR="${DATA_RAW_DIR}/ham10000"
HAM10000_IMAGES_DIR="${HAM10000_DIR}/HAM10000_images"
HAM10000_MASKS_DIR="${HAM10000_DIR}/HAM10000_segmentations_lesion_tschandl"
HAM10000_METADATA_FILE="${HAM10000_DIR}/HAM10000_metadata.csv"
HAM10000_ARCHIVE="${HAM10000_DIR}/dataverse_files.zip"
LEGACY_DATAVERSE_DIR="${DATA_RAW_DIR}/dataverse_files"
LEGACY_ARCHIVE="${DATA_RAW_DIR}/dataverse_files.zip"
DATASET_PID="doi:10.7910/DVN/DBW86T"
DOWNLOAD_URL="https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=${DATASET_PID}"


log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}


log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}


log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


cleanup_archives() {
    local removed=0

    for archive in "${HAM10000_ARCHIVE}" "${LEGACY_ARCHIVE}"; do
        if [[ -f "${archive}" ]]; then
            rm -f "${archive}"
            log_info "Removed temporary archive ${archive}"
            removed=1
        fi
    done

    if [[ ${removed} -eq 0 ]]; then
        log_info "No leftover HAM10000 ZIP archive to remove"
    fi
}


is_canonical_dataset_complete() {
    [[ -f "${HAM10000_METADATA_FILE}" ]] &&
    [[ -d "${HAM10000_IMAGES_DIR}" ]] &&
    [[ -d "${HAM10000_MASKS_DIR}" ]] &&
    find "${HAM10000_IMAGES_DIR}" -maxdepth 1 -type f \( -name "ISIC_*.jpg" -o -name "ISIC_*.jpeg" \) | grep -q . &&
    find "${HAM10000_MASKS_DIR}" -maxdepth 1 -type f -name "ISIC_*_segmentation.png" | grep -q .
}


has_legacy_dataset() {
    [[ -d "${LEGACY_DATAVERSE_DIR}" ]] &&
    [[ -f "${LEGACY_DATAVERSE_DIR}/HAM10000_metadata.csv" || -f "${LEGACY_DATAVERSE_DIR}/HAM10000_metadata" ]] &&
    find "${LEGACY_DATAVERSE_DIR}" -maxdepth 2 -type f \( -name "ISIC_*.jpg" -o -name "ISIC_*.jpeg" \) | grep -q . &&
    find "${LEGACY_DATAVERSE_DIR}" -maxdepth 3 -type f -name "ISIC_*_segmentation.png" | grep -q .
}


extract_nested_zips() {
    local extract_dir="$1"
    local nested_zips=(
        "${extract_dir}/HAM10000_images_part_1.zip"
        "${extract_dir}/HAM10000_images_part_2.zip"
        "${extract_dir}/HAM10000_segmentations_lesion_tschandl.zip"
    )

    for nested_zip in "${nested_zips[@]}"; do
        if [[ -f "${nested_zip}" ]]; then
            local zip_name
            zip_name=$(basename "${nested_zip}")
            log_info "  Extracting ${zip_name}..."
            unzip -q -o "${nested_zip}" -d "${extract_dir}"
            rm -f "${nested_zip}"
            log_info "  ✓ ${zip_name} extracted and removed"
        fi
    done
}


normalize_ham10000_layout() {
    local source_dir="$1"
    local target_dir="$2"
    local images_dir="${target_dir}/HAM10000_images"
    local masks_dir="${target_dir}/HAM10000_segmentations_lesion_tschandl"
    local metadata_target="${target_dir}/HAM10000_metadata.csv"
    local metadata_source=""

    mkdir -p "${images_dir}" "${masks_dir}"

    if [[ -f "${source_dir}/HAM10000_metadata.csv" ]]; then
        metadata_source="${source_dir}/HAM10000_metadata.csv"
    elif [[ -f "${source_dir}/HAM10000_metadata" ]]; then
        metadata_source="${source_dir}/HAM10000_metadata"
    else
        metadata_source=$(find "${source_dir}" -maxdepth 2 -type f \( -name "HAM10000_metadata.csv" -o -name "HAM10000_metadata" \) | head -n 1)
    fi

    if [[ -z "${metadata_source}" || ! -f "${metadata_source}" ]]; then
        log_error "Metadata file not found in ${source_dir}"
        return 1
    fi

    cp -f "${metadata_source}" "${metadata_target}"
    log_info "✓ Metadata normalized to ${metadata_target}"

    local image_count=0
    while IFS= read -r -d '' image_path; do
        mv -f "${image_path}" "${images_dir}/"
        image_count=$((image_count + 1))
    done < <(find "${source_dir}" -type f \( -name "ISIC_*.jpg" -o -name "ISIC_*.jpeg" -o -name "ISIC_*.JPG" -o -name "ISIC_*.JPEG" \) -print0)

    local mask_count=0
    while IFS= read -r -d '' mask_path; do
        mv -f "${mask_path}" "${masks_dir}/"
        mask_count=$((mask_count + 1))
    done < <(find "${source_dir}" -type f -name "ISIC_*_segmentation.png" -print0)

    if [[ ${image_count} -eq 0 ]]; then
        log_error "No HAM10000 images found while normalizing ${source_dir}"
        return 1
    fi

    if [[ ${mask_count} -eq 0 ]]; then
        log_error "No HAM10000 masks found while normalizing ${source_dir}"
        return 1
    fi

    log_info "✓ Dermatoscopic images normalized (${image_count} images)"
    log_info "✓ Segmentation masks normalized (${mask_count} masks)"
}


test_extraction() {
    log_info "Starting extraction and normalization test with dummy nested ZIPs..."

    local test_dir
    test_dir=$(mktemp -d)
    trap "rm -rf ${test_dir}" RETURN

    log_info "Test directory: ${test_dir}"
    log_info "Creating test ZIP files..."

    echo "test content" > "${test_dir}/test.txt"
    cd "${test_dir}"

    mkdir -p part1_content
    echo "image1" > part1_content/ISIC_0000001.jpg
    zip -q HAM10000_images_part_1.zip part1_content/ISIC_0000001.jpg

    mkdir -p part2_content
    echo "image2" > part2_content/ISIC_0000002.jpg
    zip -q HAM10000_images_part_2.zip part2_content/ISIC_0000002.jpg

    mkdir -p masks_content
    echo "mask" > masks_content/ISIC_0000001_segmentation.png
    zip -q HAM10000_segmentations_lesion_tschandl.zip masks_content/ISIC_0000001_segmentation.png

    echo "Test metadata" > HAM10000_metadata.csv
    zip -q test_dataverse_files.zip HAM10000_images_part_1.zip HAM10000_images_part_2.zip HAM10000_segmentations_lesion_tschandl.zip HAM10000_metadata.csv

    local extract_test_dir="${test_dir}/extract_test"
    local canonical_test_dir="${test_dir}/ham10000"
    mkdir -p "${extract_test_dir}"

    log_info "Testing extraction logic..."
    unzip -q -o "${test_dir}/test_dataverse_files.zip" -d "${extract_test_dir}"
    log_info "✓ Main ZIP extraction works"

    extract_nested_zips "${extract_test_dir}"
    log_info "✓ Nested ZIP extraction works"

    normalize_ham10000_layout "${extract_test_dir}" "${canonical_test_dir}"

    if [[ -f "${canonical_test_dir}/HAM10000_metadata.csv" ]]; then
        log_info "✓ Metadata normalization verified"
    fi

    if [[ -f "${canonical_test_dir}/HAM10000_images/ISIC_0000001.jpg" ]] && [[ -f "${canonical_test_dir}/HAM10000_images/ISIC_0000002.jpg" ]]; then
        log_info "✓ Image normalization verified"
    fi

    if [[ -f "${canonical_test_dir}/HAM10000_segmentations_lesion_tschandl/ISIC_0000001_segmentation.png" ]]; then
        log_info "✓ Mask normalization verified"
    fi

    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Extraction test passed! ✓ Canonical layout is ready to use."
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}


check_url() {
    log_info "Checking download URL and file size..."
    log_info "URL: ${DOWNLOAD_URL}"

    local headers
    headers=$(curl -sI -L "${DOWNLOAD_URL}" 2>/dev/null)

    if [[ $? -ne 0 ]]; then
        log_error "Failed to connect to download URL"
        return 1
    fi

    local file_size
    file_size=$(echo "${headers}" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')

    if [[ -z "${file_size}" ]]; then
        log_warn "Could not determine file size"
        log_info "URL appears to be reachable (no size info available)"
    else
        local size_gb
        size_gb=$(echo "scale=2; ${file_size} / 1024 / 1024 / 1024" | bc)
        log_info "File size: ${size_gb} GB (${file_size} bytes)"
    fi

    log_info "✓ URL is reachable"
}


show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --help              Show this help message
    --test-extract      Test extraction and normalization logic with dummy ZIPs
    --check-url         Check if download URL is reachable and show file size
    (no args)           Download and normalize the full dataset (3.4 GB)

EXAMPLES:
    bash $(basename "$0") --test-extract
    bash $(basename "$0") --check-url
    bash $(basename "$0")
    make data-download
EOF
}


main() {
    local extract_dir

    if [[ $# -gt 0 ]]; then
        case "$1" in
            --help|-h)
                show_usage
                return 0
                ;;
            --test-extract)
                test_extraction
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

    log_info "Starting HAM10000 dataset download"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Canonical dataset directory: ${HAM10000_DIR}"
    log_info "(Tip: Run with --test-extract to test extraction and normalization before downloading)"

    if is_canonical_dataset_complete; then
        log_warn "HAM10000 data already exists in canonical layout at ${HAM10000_DIR}"
        cleanup_archives
        log_info "Skipping download. Data is complete."
        return 0
    fi

    mkdir -p "${HAM10000_DIR}"
    log_info "Canonical dataset directory created/verified: ${HAM10000_DIR}"

    if has_legacy_dataset; then
        log_warn "Found legacy HAM10000 extraction at ${LEGACY_DATAVERSE_DIR}"
        log_info "Normalizing existing files into ${HAM10000_DIR}..."
        normalize_ham10000_layout "${LEGACY_DATAVERSE_DIR}" "${HAM10000_DIR}"
        log_info "Legacy extraction normalized successfully"
        cleanup_archives
        return 0
    fi

    extract_dir=$(mktemp -d "${HAM10000_DIR}/extract.XXXXXX")
    trap "rm -rf ${extract_dir}" EXIT

    log_info "Downloading HAM10000 dataset from Harvard Dataverse..."
    log_info "This may take several minutes (dataset is ~3.4 GB)"
    log_info "Download location: ${HAM10000_ARCHIVE}"

    if ! curl -L --progress-bar -o "${HAM10000_ARCHIVE}" "${DOWNLOAD_URL}"; then
        log_error "Failed to download dataset"
        return 1
    fi

    if [[ ! -f "${HAM10000_ARCHIVE}" ]]; then
        log_error "Downloaded file not found"
        return 1
    fi

    local filesize
    filesize=$(du -h "${HAM10000_ARCHIVE}" | cut -f1)
    log_info "Downloaded file size: ${filesize}"

    log_info "Extracting main ZIP to temporary directory ${extract_dir}..."
    unzip -q -o "${HAM10000_ARCHIVE}" -d "${extract_dir}"
    log_info "Main ZIP extraction completed"

    log_info "Extracting nested ZIP files..."
    extract_nested_zips "${extract_dir}"
    log_info "All nested ZIPs extracted"

    log_info "Cleaning up extra files..."
    rm -rf "${extract_dir}/__MACOSX" 2>/dev/null || true
    rm -f "${extract_dir}/ISIC2018"* 2>/dev/null || true
    log_info "  Removed non-HAM10000 files"

    log_info "Normalizing extracted files into canonical layout..."
    normalize_ham10000_layout "${extract_dir}" "${HAM10000_DIR}"
    cleanup_archives

    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Dataset download and normalization completed successfully! ✓"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Directory structure:"
    log_info "  data/raw/ham10000/"
    log_info "  ├── HAM10000_metadata.csv"
    log_info "  ├── HAM10000_images/"
    log_info "  └── HAM10000_segmentations_lesion_tschandl/"
    log_info "Next steps:"
    log_info "  1. Review: notebooks/ham10000/eda.ipynb"
    log_info "  2. Process: src/mse_mlops/analysis/ham10000.py"
}


main "$@"
