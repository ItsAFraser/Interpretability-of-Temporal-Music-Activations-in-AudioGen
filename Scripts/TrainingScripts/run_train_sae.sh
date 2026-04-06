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
CONDA_UTILS="$ROOT_DIR/Scripts/TrainingScripts/conda_utils.sh"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: Training script not found at: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$CONDA_UTILS" ]]; then
  echo "Error: Missing helper script at: $CONDA_UTILS" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_UTILS"
temporal_setup_python_environment temporal-music-activations-cuda temporal-music-activations-apple
temporal_print_python_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch numpy

PYTHON_CMD=("$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON")

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
