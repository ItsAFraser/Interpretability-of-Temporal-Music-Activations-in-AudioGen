#!/bin/bash

#=======================================================
# Slurm Settings for Feature Extraction (MusicGen)
#=======================================================
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=ExtractFeatures
#SBATCH --output=Logs/Slurm/ExtractFeatures_%A_%a.out
#SBATCH --error=Logs/Slurm/ExtractFeatures_%A_%a.err

set -euo pipefail

#=======================================================
# Directory Setup & Environment
#=======================================================
resolve_repo_root() {
    local candidate="${SLURM_SUBMIT_DIR:-$PWD}"

    while [[ "$candidate" != "/" ]]; do
        if [[ -f "$candidate/Scripts/TrainingScripts/conda_utils.sh" ]]; then
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
cd "$ROOT_DIR"

CONDA_UTILS="$ROOT_DIR/Scripts/TrainingScripts/conda_utils.sh"
if [[ ! -f "$CONDA_UTILS" ]]; then
    echo "Error: Missing helper script at $CONDA_UTILS" >&2
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

# shellcheck disable=SC1090
source "$CONDA_UTILS"

temporal_load_chpc_cuda_modules
temporal_setup_python_environment temporal-music-activations-cuda
temporal_print_python_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch transformers
temporal_print_torch_device_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"

# Defaults target CHPC VAST scratch for the current user.
CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"

HF_HOME="${HF_HOME:-${CHPC_SCRATCH_BASE}/hf-cache}"
TORCH_HOME="${TORCH_HOME:-${CHPC_SCRATCH_BASE}/torch-cache}"
TMPDIR="${TMPDIR:-${CHPC_SCRATCH_BASE}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CHPC_SCRATCH_BASE}/xdg-cache}"
export HF_HOME TORCH_HOME TMPDIR XDG_CACHE_HOME
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$XDG_CACHE_HOME"

RAW_AUDIO_DIR="${RAW_AUDIO_DIR:-${CHPC_SCRATCH_BASE}/mtg-jamendo/raw_30s/audio}"
SCRATCH_FEATURES_DIR="${SCRATCH_FEATURES_DIR:-${CHPC_SCRATCH_BASE}/sae_output/features}"
mkdir -p "$SCRATCH_FEATURES_DIR"

RUN_EXTRACT_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/run_extract_musicgen_features.sh"
if [[ ! -f "$RUN_EXTRACT_SCRIPT" ]]; then
    echo "Error: Missing extraction launcher at $RUN_EXTRACT_SCRIPT" >&2
    exit 1
fi

EXTRACT_LAYERS="${EXTRACT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,-1}"
EXTRACT_MAX_DURATION_SEC="${EXTRACT_MAX_DURATION_SEC:-30}"
EXTRACT_CHUNK_SIZE="${EXTRACT_CHUNK_SIZE:-4096}"
EXTRACT_FILE_START="${EXTRACT_FILE_START:-}"
EXTRACT_MAX_FILES="${EXTRACT_MAX_FILES:-0}"
EXTRACT_MANIFEST_PATH="${EXTRACT_MANIFEST_PATH:-}"
EXTRACT_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
EXTRACT_DEVICE="${EXTRACT_DEVICE:-}"

# Slurm uses 4294967294 as a sentinel when the job is not part of an array.
if [[ "$EXTRACT_ARRAY_TASK_ID" == "4294967294" ]]; then
    EXTRACT_ARRAY_TASK_ID=0
fi

if [[ -z "$EXTRACT_DEVICE" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" || -n "${SLURM_JOB_GPUS:-}" || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        EXTRACT_DEVICE="cuda"
    else
        EXTRACT_DEVICE="auto"
    fi
fi

if [[ "$EXTRACT_DEVICE" == "cuda" ]]; then
    "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    details = [
        f"torch.__version__={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
        f"torch.backends.cuda.is_built()={torch.backends.cuda.is_built()}",
        f"torch.cuda.device_count()={torch.cuda.device_count()}",
    ]
    if not torch.backends.cuda.is_built():
        details.append(
            "The active environment appears to contain a CPU-only PyTorch build. "
            "Rebuild or reinstall the CUDA-enabled pytorch package before rerunning."
        )
    raise SystemExit(
        "CUDA was requested for extraction, but torch.cuda.is_available() is False. "
        "Check the active conda environment, loaded CUDA modules, and GPU allocation. "
        + " ".join(details)
    )
PY
fi

if [[ -z "$EXTRACT_FILE_START" ]]; then
    EXTRACT_FILE_START=$((EXTRACT_ARRAY_TASK_ID * EXTRACT_CHUNK_SIZE))
fi

EXTRACT_ARGS=(
    --file_start "$EXTRACT_FILE_START"
    --file_count "$EXTRACT_CHUNK_SIZE"
    --max_duration_sec "$EXTRACT_MAX_DURATION_SEC"
    --decoder_layers "$EXTRACT_LAYERS"
    --device "$EXTRACT_DEVICE"
    --metadata_json
)

if [[ "$EXTRACT_MAX_FILES" != "0" ]]; then
    EXTRACT_ARGS+=(--max_files "$EXTRACT_MAX_FILES")
fi

if [[ -n "$EXTRACT_MANIFEST_PATH" ]]; then
    EXTRACT_ARGS+=(--manifest_path "$EXTRACT_MANIFEST_PATH")
fi

echo "Starting Feature Extraction on CHPC..."
echo "  slurm account: ${SLURM_JOB_ACCOUNT:-soc-gpu-np}"
echo "  slurm partition: ${SLURM_JOB_PARTITION:-soc-gpu-np}"
echo "  raw audio dir: $RAW_AUDIO_DIR"
echo "  output dir: $SCRATCH_FEATURES_DIR"
echo "  decoder layers: $EXTRACT_LAYERS"
echo "  device: $EXTRACT_DEVICE"
echo "  array task id: $EXTRACT_ARRAY_TASK_ID"
echo "  file_start: $EXTRACT_FILE_START"
echo "  file_count: $EXTRACT_CHUNK_SIZE"
echo "  manifest: ${EXTRACT_MANIFEST_PATH:-<filesystem scan>}"
echo "  python: $TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
echo "  HF_HOME: $HF_HOME"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  TMPDIR: $TMPDIR"

# Run the repository extraction launcher.
bash "$RUN_EXTRACT_SCRIPT" \
    "$RAW_AUDIO_DIR" \
    "$SCRATCH_FEATURES_DIR" \
    "${EXTRACT_ARGS[@]}"

echo "Extraction finished."
