#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/check_chpc_cuda_env.sh

Purpose:
  Verify that the active CHPC GPU node, loaded CUDA module, and Python environment
  resolve to a CUDA-enabled PyTorch build suitable for this repository.

Optional environment overrides:
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=/path/to/python
  TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH=/path/to/conda.sh
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME=temporal-music-activations-cuda
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX=/scratch/general/vast/$USER/conda-envs/temporal-music-activations-cuda
  TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE=cuda/12.6.3
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

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi is unavailable. Run this on a CHPC GPU node, not a login node." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_UTILS"

temporal_load_chpc_cuda_modules
temporal_setup_python_environment temporal-music-activations-cuda
temporal_print_python_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch transformers

echo "GPU detected by nvidia-smi:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "Running torch CUDA diagnostics..."
"$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" - <<'PY'
import sys
import torch

print(f"Torch version: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if not torch.backends.cuda.is_built():
    raise SystemExit(
        "Failure: active environment contains a CPU-only PyTorch build. "
        "Recreate the CUDA environment before submitting CHPC jobs."
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "Failure: CUDA-enabled PyTorch is installed, but no usable CUDA device is available. "
        "Check that you are on a GPU node and that CUDA modules loaded correctly."
    )

for index in range(torch.cuda.device_count()):
    print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")

print("Environment check passed.")
PY