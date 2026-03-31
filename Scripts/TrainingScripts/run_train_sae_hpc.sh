#!/usr/bin/env bash
set -euo pipefail

# HPC-oriented preset for SAE training on larger datasets.
#
# Usage:
#   Scripts/TrainingScripts/run_train_sae_hpc.sh <data_dir> <output_dir> [extra args]
#
# Typical usage:
#   Scripts/TrainingScripts/run_train_sae_hpc.sh Data/Models/features/layer_final Output/sae-hpc-small
#   Scripts/TrainingScripts/run_train_sae_hpc.sh Data/Models/features/layer_final Output/sae-hpc-medium --latent_dim 512 --batch_size 64

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/run_train_sae_hpc.sh <data_dir> <output_dir> [extra training args]

Defaults optimized for multi-GPU or high-memory CUDA nodes:
  --device cuda
  --epochs 100
  --batch_size 128
  --learning_rate 3e-4
  --latent_dim 256
  --checkpoint_interval 2
  --val_split 0.1
  --num_workers 16

Environment:
  Set TEMPORAL_MUSIC_ACTIVATIONS_PYTHON to override the Python command.

Examples:
  Scripts/TrainingScripts/run_train_sae_hpc.sh Data/Models/features/layer_final Output/sae-hpc
  Scripts/TrainingScripts/run_train_sae_hpc.sh Data/Models/features/layer_08 Output/sae-layer08 --epochs 150 --latent_dim 512
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
elif conda env list | grep -q '^temporal-music-activations-cuda '; then
  PYTHON_CMD=(conda run -n temporal-music-activations-cuda python)
else
  PYTHON_CMD=(python)
fi

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --epochs 100 \
  --batch_size 128 \
  --learning_rate 3e-4 \
  --latent_dim 256 \
  --checkpoint_interval 2 \
  --val_split 0.1 \
  --num_workers 16 \
  "$@"
