#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for Temporal-Music-Activations/TrainNewSAE.py
#
# Usage examples:
#   Scripts/TrainingScripts/run_train_sae.sh Data/Models/features Output/sae
#   Scripts/TrainingScripts/run_train_sae.sh Data/Models/features Output/sae --epochs 20 --batch_size 16 --device cpu

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/run_train_sae.sh <data_dir> <output_dir> [extra training args]

Required:
  <data_dir>    Folder containing .pt or .npy feature tensors
  <output_dir>  Folder where checkpoints/final model are saved

Optional extra args are forwarded directly to TrainNewSAE.py.
Examples:
  Scripts/TrainingScripts/run_train_sae.sh Data/Models/features Output/sae
  Scripts/TrainingScripts/run_train_sae.sh Data/Models/features Output/sae --epochs 20 --batch_size 16 --device cpu
  Scripts/TrainingScripts/run_train_sae.sh Data/Models/features Output/sae --val_split 0.1 --num_workers 4
EOF
  exit 0
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN_SCRIPT="$ROOT_DIR/Temporal-Music-Activations/TrainNewSAE.py"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: Training script not found at: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ -n "${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}" ]]; then
  PYTHON_CMD=("$TEMPORAL_MUSIC_ACTIVATIONS_PYTHON")
elif conda env list | grep -q '^temporal-music-activations-apple '; then
  PYTHON_CMD=(conda run -n temporal-music-activations-apple python)
else
  PYTHON_CMD=(python)
fi

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
