#!/usr/bin/env bash
set -euo pipefail

# Download and optionally extract a dataset archive.
# Usage:
#   Scripts/SetupScripts/download_dataset_archive.sh <url> [target_dir]
#
# Examples:
#   Scripts/SetupScripts/download_dataset_archive.sh https://example.com/dataset.zip Data/raw
#   Scripts/SetupScripts/download_dataset_archive.sh https://example.com/dataset.zip /Volumes/MySSD/mtg/raw
#   Scripts/SetupScripts/download_dataset_archive.sh https://example.com/dataset.tar.gz

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  cat <<'EOF'
Usage:
  Scripts/SetupScripts/download_dataset_archive.sh <url> [target_dir]

Arguments:
  <url>         URL to archive file (.zip, .tar, .tar.gz, .tgz)
  [target_dir]  Destination directory (default: Data/raw)
                Accepts absolute paths, e.g. /Volumes/MySSD/mtg/raw
EOF
  exit 0
fi

URL="$1"
TARGET_DIR="${2:-Data/raw}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$TARGET_DIR" == /* ]]; then
  DEST_DIR="$TARGET_DIR"
else
  DEST_DIR="$ROOT_DIR/$TARGET_DIR"
fi

mkdir -p "$DEST_DIR"

FILENAME="$(basename "$URL")"
ARCHIVE_PATH="$DEST_DIR/$FILENAME"

if command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$ARCHIVE_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE_PATH" "$URL"
else
  echo "Error: neither curl nor wget is installed." >&2
  exit 1
fi

case "$ARCHIVE_PATH" in
  *.zip)
    if ! command -v unzip >/dev/null 2>&1; then
      echo "Error: unzip is required to extract .zip archives." >&2
      exit 1
    fi
    unzip -o "$ARCHIVE_PATH" -d "$DEST_DIR"
    ;;
  *.tar|*.tar.gz|*.tgz)
    tar -xf "$ARCHIVE_PATH" -C "$DEST_DIR"
    ;;
  *)
    echo "Downloaded file is not a recognized archive; kept as-is at: $ARCHIVE_PATH"
    ;;
esac

echo "Done. Dataset artifact available at: $DEST_DIR"
