#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/rebuild_chpc_cuda_env.sh

Purpose:
  Recreate the CHPC CUDA Conda environment in a deterministic order:
  1. Create a clean Python 3.11 env.
  2. Install GPU-enabled PyTorch from the pytorch and nvidia channels with strict priority.
  3. Install the remaining Conda dependencies.
  4. Install the Hugging Face stack with pip.
  5. Verify torch CUDA build metadata.

Optional environment overrides:
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME=temporal-music-activations-cuda
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX=/scratch/general/vast/$USER/conda-envs/temporal-music-activations-cuda
  TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE=cuda/12.6.3
  TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE=miniforge3
  CHPC_SCRATCH_BASE=/scratch/general/vast/$USER
  CONDA_PKGS_DIRS=/scratch/general/vast/$USER/conda-pkgs
  PIP_CACHE_DIR=/scratch/general/vast/$USER/pip-cache
EOF
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONDA_UTILS="$ROOT_DIR/Scripts/TrainingScripts/conda_utils.sh"

if [[ ! -f "$CONDA_UTILS" ]]; then
  echo "Error: Missing helper script at $CONDA_UTILS" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_UTILS"

ENV_NAME="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME:-temporal-music-activations-cuda}"
CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"
ENV_PREFIX="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX:-${CHPC_SCRATCH_BASE}/conda-envs/${ENV_NAME}}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CHPC_SCRATCH_BASE}/conda-pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CHPC_SCRATCH_BASE}/pip-cache}"
export CONDA_PKGS_DIRS PIP_CACHE_DIR
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$(dirname "$ENV_PREFIX")"

temporal_load_chpc_cuda_modules
temporal_initialize_conda

echo "Using environment name: $ENV_NAME"
echo "Using environment prefix: $ENV_PREFIX"
echo "Using CONDA_PKGS_DIRS: $CONDA_PKGS_DIRS"
echo "Using PIP_CACHE_DIR: $PIP_CACHE_DIR"

if [[ -d "$ENV_PREFIX" ]]; then
  echo "Removing existing Conda environment prefix: $ENV_PREFIX"
  conda env remove -p "$ENV_PREFIX" -y || rm -rf "$ENV_PREFIX"
fi

echo "Creating clean base environment..."
conda create -p "$ENV_PREFIX" -y python=3.11 pip

echo "Installing GPU-enabled PyTorch with strict channel priority..."
conda install -p "$ENV_PREFIX" -y \
  --strict-channel-priority \
  -c pytorch \
  -c nvidia \
  pytorch=2.5.* \
  torchaudio=2.5.* \
  pytorch-cuda=12.1

echo "Installing remaining Conda dependencies..."
conda install -p "$ENV_PREFIX" -y -c conda-forge \
  'numpy<2' \
  tqdm \
  pyyaml \
  librosa \
  pysoundfile \
  ffmpeg

echo "Installing pip-only ML dependencies..."
conda run -p "$ENV_PREFIX" pip install \
  transformers==4.46.3 \
  tokenizers \
  sentencepiece \
  safetensors \
  huggingface_hub

echo "Verifying rebuilt environment..."
conda run -p "$ENV_PREFIX" python - <<'PY'
import torch

print(f"Torch version: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
PY

echo "Environment rebuild complete."
echo "Set this before CHPC runs: export TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX=$ENV_PREFIX"
echo "Next: run bash Scripts/TrainingScripts/check_chpc_cuda_env.sh on a GPU node or submit Scripts/TrainingScripts/chpc_check_cuda_env.slurm"