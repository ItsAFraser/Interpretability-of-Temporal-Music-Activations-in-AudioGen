#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 4 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/stage_selected_features_to_local.sh <source_dir> <staged_dir> <manifest_path> <metadata_path>

Environment:
  TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON or TEMPORAL_MUSIC_ACTIVATIONS_PYTHON
  TRAIN_MAX_FILES
  TRAIN_RANDOM_SUBSET_FILES
  TRAIN_SUBSET_SEED
  TRAIN_STAGE_MAX_MB
  TRAIN_STAGE_TIMEOUT_SEC

Behavior:
  Resolves the deterministic selected-file subset from <source_dir>, writes a
  relative-path manifest, checks local scratch capacity, and copies only those
  files into <staged_dir> while preserving relative paths.
EOF
  exit 0
fi

SOURCE_DIR="$1"
STAGED_DIR="$2"
MANIFEST_PATH="$3"
METADATA_PATH="$4"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESOLVER_SCRIPT="$ROOT_DIR/Temporal-Music-Activations/ResolveFeatureSubset.py"
PYTHON_BIN="${TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON:-${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-python3}}"
TRAIN_MAX_FILES="${TRAIN_MAX_FILES:-0}"
TRAIN_RANDOM_SUBSET_FILES="${TRAIN_RANDOM_SUBSET_FILES:-0}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
TRAIN_STAGE_MAX_MB="${TRAIN_STAGE_MAX_MB:-0}"
TRAIN_STAGE_TIMEOUT_SEC="${TRAIN_STAGE_TIMEOUT_SEC:-0}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: Source feature directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -f "$RESOLVER_SCRIPT" ]]; then
  echo "Error: Missing subset resolver script at $RESOLVER_SCRIPT" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required for local staging but was not found." >&2
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST_PATH")"
mkdir -p "$(dirname "$METADATA_PATH")"

echo "Resolving selected feature subset from source layer"
echo "  source dir: $SOURCE_DIR"
echo "  manifest path: $MANIFEST_PATH"
echo "  metadata path: $METADATA_PATH"
echo "  max_files: $TRAIN_MAX_FILES"
echo "  random_subset_files: $TRAIN_RANDOM_SUBSET_FILES"
echo "  subset_seed: $TRAIN_SUBSET_SEED"

"$PYTHON_BIN" "$RESOLVER_SCRIPT" \
  --data_dir "$SOURCE_DIR" \
  --manifest_path "$MANIFEST_PATH" \
  --metadata_path "$METADATA_PATH" \
  --max_files "$TRAIN_MAX_FILES" \
  --random_subset_files "$TRAIN_RANDOM_SUBSET_FILES" \
  --subset_seed "$TRAIN_SUBSET_SEED"

selected_file_count="$(awk 'NF{count++} END{print count+0}' "$MANIFEST_PATH")"
if [[ "$selected_file_count" -le 0 ]]; then
  echo "Error: Selected-file manifest is empty: $MANIFEST_PATH" >&2
  exit 1
fi

subset_bytes=0
while IFS= read -r relative_path; do
  if [[ -z "$relative_path" ]]; then
    continue
  fi
  source_path="$SOURCE_DIR/$relative_path"
  if [[ ! -f "$source_path" ]]; then
    echo "Error: Manifest entry is missing from source layer: $source_path" >&2
    exit 1
  fi
  file_size_bytes="$(stat -Lc '%s' "$source_path")"
  subset_bytes=$((subset_bytes + file_size_bytes))
done < "$MANIFEST_PATH"

if [[ "$TRAIN_STAGE_MAX_MB" -gt 0 ]]; then
  stage_limit_bytes=$((TRAIN_STAGE_MAX_MB * 1024 * 1024))
  if [[ "$subset_bytes" -gt "$stage_limit_bytes" ]]; then
    echo "Error: Selected subset size ${subset_bytes} bytes exceeds TRAIN_STAGE_MAX_MB=${TRAIN_STAGE_MAX_MB}." >&2
    exit 1
  fi
fi

staged_parent_dir="$(dirname "$STAGED_DIR")"
mkdir -p "$staged_parent_dir"
available_bytes="$(df -PB1 "$staged_parent_dir" | awk 'NR==2 {print $4}')"
required_bytes=$((subset_bytes + subset_bytes / 10 + 1048576))
if [[ "$required_bytes" -gt "$available_bytes" ]]; then
  echo "Error: Not enough local scratch to stage selected subset." >&2
  echo "  staged parent dir: $staged_parent_dir" >&2
  echo "  subset bytes: $subset_bytes" >&2
  echo "  required bytes with headroom: $required_bytes" >&2
  echo "  available bytes: $available_bytes" >&2
  exit 1
fi

rm -rf "$STAGED_DIR"
mkdir -p "$STAGED_DIR"

echo "Staging selected feature subset to local scratch"
echo "  staged dir: $STAGED_DIR"
echo "  selected files: $selected_file_count"
echo "  selected bytes: $subset_bytes"
echo "  available local bytes: $available_bytes"

rsync_command=(rsync -a --files-from="$MANIFEST_PATH" "$SOURCE_DIR/" "$STAGED_DIR/")
if [[ "$TRAIN_STAGE_TIMEOUT_SEC" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
  timeout "${TRAIN_STAGE_TIMEOUT_SEC}s" "${rsync_command[@]}"
else
  "${rsync_command[@]}"
fi

staged_file_count="$(find "$STAGED_DIR" -type f \( -name '*.pt' -o -name '*.npy' \) | wc -l | tr -d '[:space:]')"
if [[ "$staged_file_count" != "$selected_file_count" ]]; then
  echo "Error: Staged file count mismatch after copy." >&2
  echo "  expected: $selected_file_count" >&2
  echo "  actual: $staged_file_count" >&2
  exit 1
fi

echo "Local staging completed successfully"
echo "  staged dir: $STAGED_DIR"
echo "  staged files: $staged_file_count"