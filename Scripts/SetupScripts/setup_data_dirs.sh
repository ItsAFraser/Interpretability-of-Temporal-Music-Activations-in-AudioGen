#!/usr/bin/env bash
set -euo pipefail

# Create standard project data/output directories.
# Usage:
#   Scripts/SetupScripts/setup_data_dirs.sh [base_dir]
#
# Examples:
#   Scripts/SetupScripts/setup_data_dirs.sh
#   Scripts/SetupScripts/setup_data_dirs.sh /Volumes/MySSD/temporal-music

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${1:-$ROOT_DIR}"

if [[ "$BASE_DIR" != /* ]]; then
  BASE_DIR="$ROOT_DIR/$BASE_DIR"
fi

mkdir -p "$BASE_DIR/Data/raw"
mkdir -p "$BASE_DIR/Data/metadata"
mkdir -p "$BASE_DIR/Data/interim"
mkdir -p "$BASE_DIR/Data/processed"
mkdir -p "$BASE_DIR/Data/Models/features"
mkdir -p "$BASE_DIR/Output/sae"
mkdir -p "$BASE_DIR/Output/logs"

cat <<EOF
Created/verified directories under: $BASE_DIR
  Data/raw
  Data/metadata
  Data/interim
  Data/processed
  Data/Models/features
  Output/sae
  Output/logs
EOF
