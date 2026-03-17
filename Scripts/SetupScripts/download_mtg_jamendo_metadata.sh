#!/usr/bin/env bash
set -euo pipefail

# Download MTG-Jamendo metadata/split files from user-provided URLs.
# This script only downloads metadata files, not audio tracks.
#
# Usage:
#   Scripts/SetupScripts/download_mtg_jamendo_metadata.sh <urls_file> [target_dir]
#
# urls_file format:
#   One URL per line
#   Lines starting with # are ignored

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  cat <<'EOF'
Usage:
  Scripts/SetupScripts/download_mtg_jamendo_metadata.sh <urls_file> [target_dir]

Arguments:
  <urls_file>   Text file with one metadata URL per line
  [target_dir]  Destination directory (default: Data/metadata/mtg-jamendo)
                Accepts absolute paths, e.g. /Volumes/MySSD/mtg/metadata

Example:
  Scripts/SetupScripts/download_mtg_jamendo_metadata.sh Docs/mtg_jamendo_urls.txt
EOF
  exit 0
fi

URLS_FILE="$1"
TARGET_DIR="${2:-Data/metadata/mtg-jamendo}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$TARGET_DIR" == /* ]]; then
  DEST_DIR="$TARGET_DIR"
else
  DEST_DIR="$ROOT_DIR/$TARGET_DIR"
fi

if [[ "$URLS_FILE" == /* ]]; then
  URLS_FILE_PATH="$URLS_FILE"
else
  URLS_FILE_PATH="$ROOT_DIR/$URLS_FILE"
fi

if [[ ! -f "$URLS_FILE_PATH" ]]; then
  echo "Error: urls_file not found: $URLS_FILE_PATH" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "Error: neither curl nor wget is installed." >&2
  exit 1
fi

while IFS= read -r line; do
  url="$(echo "$line" | sed 's/[[:space:]]*$//')"
  [[ -z "$url" ]] && continue
  [[ "$url" =~ ^# ]] && continue

  filename="$(basename "$url")"
  out="$DEST_DIR/$filename"

  echo "Downloading: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  else
    wget -O "$out" "$url"
  fi
done < "$URLS_FILE_PATH"

echo "Done. Metadata saved in: $DEST_DIR"
