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
  --epochs 50
  --batch_size 64
  --learning_rate 3e-4
  --latent_dim 256
  --checkpoint_interval 5
  --val_split 0.1
  --num_workers 8
  --sample_mode frames
  --frame_stride 4

Recommended for temporally faithful reduced-corpus runs:
  --sample_mode frames
  --frame_stride 1
  --random_subset_files 2048
  --subset_seed 42

Environment:
  Set TEMPORAL_MUSIC_ACTIVATIONS_PYTHON to override the Python command.

Examples:
  Scripts/TrainingScripts/run_train_sae_hpc.sh Data/Models/features/layer_final Output/sae-hpc
  Scripts/TrainingScripts/run_train_sae_hpc.sh Data/Models/features/layer_08 Output/sae-layer08 --epochs 150 --latent_dim 512
  Scripts/TrainingScripts/run_train_sae_hpc.sh /scratch/general/vast/$USER/sae_output/features/layer_16 /scratch/general/vast/$USER/sae_output/models/layer_16 --frame_stride 2 --max_frames 250000
  Scripts/TrainingScripts/run_train_sae_hpc.sh /scratch/general/vast/$USER/sae_output/features-all-layers/layer_final /scratch/general/vast/$USER/sae_output/models-all-layers/layer_final-subset --frame_stride 1 --random_subset_files 2048 --subset_seed 42
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
temporal_setup_python_environment temporal-music-activations-cuda
temporal_print_python_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch numpy

PYTHON_CMD=("$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON")

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --epochs 50 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --latent_dim 256 \
  --checkpoint_interval 5 \
  --sample_mode frames \
  --frame_stride 4 \
  --val_split 0.1 \
  --num_workers 8 \
  "$@"
