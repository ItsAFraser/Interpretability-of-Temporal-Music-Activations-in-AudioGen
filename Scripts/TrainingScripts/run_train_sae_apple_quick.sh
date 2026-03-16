#!/usr/bin/env bash
set -euo pipefail

# Apple Silicon quick test preset for end-to-end validation.
# Intentionally uses a small subset and short training schedule.
#
# Usage:
#   Scripts/run_train_sae_apple_quick.sh <data_dir> <output_dir> [extra args]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/run_train_sae_apple_quick.sh <data_dir> <output_dir> [extra training args]

Defaults optimized for quick Apple Silicon testing:
  --device mps
  --epochs 2
  --batch_size 8
  --latent_dim 64
  --max_samples 128
  --checkpoint_interval 1
  --log_interval 5

Examples:
  Scripts/run_train_sae_apple_quick.sh Data/Models/features Output/sae-quick
  Scripts/run_train_sae_apple_quick.sh Data/Models/features Output/sae-quick --max_samples 256 --epochs 3
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
  --device mps \
  --epochs 2 \
  --batch_size 8 \
  --latent_dim 64 \
  --max_samples 128 \
  --checkpoint_interval 1 \
  --log_interval 5 \
  --verbose \
  "$@"
