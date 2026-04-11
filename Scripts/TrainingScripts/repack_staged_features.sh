#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 3 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/repack_staged_features.sh <staged_dir> <manifest_path> <repacked_dir>

Environment:
  TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON or TEMPORAL_MUSIC_ACTIVATIONS_PYTHON
  TRAIN_REPACK_SHARD_SIZE_MB

Behavior:
  Repack a staged deterministic subset into sharded .npy files plus
  repacked_metadata.json so training can read from a smaller number of files.
EOF
  exit 0
fi

STAGED_DIR="$1"
MANIFEST_PATH="$2"
REPACKED_DIR="$3"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
REPACK_SCRIPT="$ROOT_DIR/Temporal-Music-Activations/RepackFeatureSubset.py"
PYTHON_BIN="${TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON:-${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-python3}}"
TRAIN_REPACK_SHARD_SIZE_MB="${TRAIN_REPACK_SHARD_SIZE_MB:-512}"

if [[ ! -d "$STAGED_DIR" ]]; then
  echo "Error: Staged feature directory does not exist: $STAGED_DIR" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Error: Selected-file manifest does not exist: $MANIFEST_PATH" >&2
  exit 1
fi

if [[ ! -f "$REPACK_SCRIPT" ]]; then
  echo "Error: Missing repack script at $REPACK_SCRIPT" >&2
  exit 1
fi

rm -rf "$REPACKED_DIR"
mkdir -p "$REPACKED_DIR"

echo "Repacking staged feature subset"
echo "  staged dir: $STAGED_DIR"
echo "  manifest path: $MANIFEST_PATH"
echo "  repacked dir: $REPACKED_DIR"
echo "  target shard size mb: $TRAIN_REPACK_SHARD_SIZE_MB"

"$PYTHON_BIN" "$REPACK_SCRIPT" \
  --data_dir "$STAGED_DIR" \
  --manifest_path "$MANIFEST_PATH" \
  --output_dir "$REPACKED_DIR" \
  --target_shard_size_mb "$TRAIN_REPACK_SHARD_SIZE_MB"