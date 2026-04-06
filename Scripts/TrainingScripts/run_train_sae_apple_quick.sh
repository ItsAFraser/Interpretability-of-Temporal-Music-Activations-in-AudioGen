#!/usr/bin/env bash
set -euo pipefail

# Apple Silicon quick test preset for end-to-end validation.
# Intentionally uses a small subset and short training schedule.
#
# Usage:
#   Scripts/TrainingScripts/run_train_sae_apple_quick.sh <data_dir> <output_dir> [extra args]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/run_train_sae_apple_quick.sh <data_dir> <output_dir> [extra training args]

Defaults optimized for quick Apple Silicon testing:
  --device mps
  --epochs 2
  --batch_size 8
  --latent_dim 64
  --max_files 128
  --val_split 0.1
  --num_workers 0
  --checkpoint_interval 1
  --log_interval 5

Examples:
  Scripts/TrainingScripts/run_train_sae_apple_quick.sh Data/Models/features Output/sae-quick
  Scripts/TrainingScripts/run_train_sae_apple_quick.sh Data/Models/features Output/sae-quick --max_files 256 --epochs 3
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
temporal_setup_python_environment temporal-music-activations-apple
temporal_print_python_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch numpy

PYTHON_CMD=("$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON")

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device mps \
  --epochs 2 \
  --batch_size 8 \
  --latent_dim 64 \
  --max_files 128 \
  --val_split 0.1 \
  --num_workers 0 \
  --checkpoint_interval 1 \
  --log_interval 5 \
  --verbose \
  "$@"
