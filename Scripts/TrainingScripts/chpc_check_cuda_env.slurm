#!/bin/bash

#=======================================================
# Slurm Settings for CHPC CUDA Environment Check
#=======================================================
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=CheckCudaEnv
#SBATCH --output=CheckCudaEnv_%j.out
#SBATCH --error=CheckCudaEnv_%j.err

set -euo pipefail

resolve_repo_root() {
    local candidate="${SLURM_SUBMIT_DIR:-$PWD}"

    while [[ "$candidate" != "/" ]]; do
        if [[ -f "$candidate/Scripts/TrainingScripts/check_chpc_cuda_env.sh" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
        candidate="$(dirname "$candidate")"
    done

    return 1
}

ROOT_DIR="$(resolve_repo_root)" || {
    echo "Error: Unable to locate repository root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
}

CHECK_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/check_chpc_cuda_env.sh"
if [[ ! -f "$CHECK_SCRIPT" ]]; then
    echo "Error: Missing verification script at $CHECK_SCRIPT" >&2
    exit 1
fi

if [[ -z "${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}" ]]; then
    NAMED_ENV_PYTHON="$HOME/.conda/envs/temporal-music-activations-cuda/bin/python"
    SCRATCH_ENV_PYTHON="/scratch/general/vast/$USER/conda-envs/temporal-music-activations-cuda/bin/python"

    if [[ -x "$SCRATCH_ENV_PYTHON" ]]; then
        export TEMPORAL_MUSIC_ACTIVATIONS_PYTHON="$SCRATCH_ENV_PYTHON"
        echo "Using scratch environment python: $TEMPORAL_MUSIC_ACTIVATIONS_PYTHON"
    elif [[ -x "$NAMED_ENV_PYTHON" ]]; then
        export TEMPORAL_MUSIC_ACTIVATIONS_PYTHON="$NAMED_ENV_PYTHON"
        echo "Using named environment python: $TEMPORAL_MUSIC_ACTIVATIONS_PYTHON"
    fi
fi

cd "$ROOT_DIR"
bash "$CHECK_SCRIPT"