#!/usr/bin/env bash
set -euo pipefail

# CUDA preset for regular training runs.
#
# Usage:
#   Scripts/TrainingScripts/run_train_sae_cuda.sh <data_dir> <output_dir> [extra args]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/run_train_sae_cuda.sh <data_dir> <output_dir> [extra training args]

Defaults optimized for CUDA training:
  --device cuda
  --epochs 50
  --batch_size 32
  --latent_dim 128
  --checkpoint_interval 5

Examples:
  Scripts/TrainingScripts/run_train_sae_cuda.sh Data/Models/features Output/sae-cuda
  Scripts/TrainingScripts/run_train_sae_cuda.sh Data/Models/features Output/sae-cuda --epochs 100 --learning_rate 5e-4
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
  --epochs 50 \
  --batch_size 32 \
  --latent_dim 128 \
  --checkpoint_interval 5 \
  "$@"
