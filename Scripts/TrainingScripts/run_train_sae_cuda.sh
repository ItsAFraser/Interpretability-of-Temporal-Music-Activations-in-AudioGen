#!/usr/bin/env bash
set -euo pipefail

# CUDA preset for regular training runs.
#
# Usage:
#   Scripts/run_train_sae_cuda.sh <data_dir> <output_dir> [extra args]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/run_train_sae_cuda.sh <data_dir> <output_dir> [extra training args]

Defaults optimized for CUDA training:
  --device cuda
  --epochs 50
  --batch_size 32
  --latent_dim 128
  --checkpoint_interval 5

Examples:
  Scripts/run_train_sae_cuda.sh Data/Models/features Output/sae-cuda
  Scripts/run_train_sae_cuda.sh Data/Models/features Output/sae-cuda --epochs 100 --learning_rate 5e-4
EOF
  exit 0
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$ROOT_DIR/Temporal-Music-Activations/TrainNewSAE.py"

python "$TRAIN_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --epochs 50 \
  --batch_size 32 \
  --latent_dim 128 \
  --checkpoint_interval 5 \
  "$@"
