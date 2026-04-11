#!/usr/bin/env bash
set -euo pipefail

# Submit a CHPC SAE training job against one extracted feature layer.
#
# Defaults align with the extraction run-label workflow so training can follow
# extraction without manually rebuilding the feature/output paths.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/submit_train_sae_hpc.sh

Environment overrides:
  TRAIN_FEATURES_RUN_NAME     Optional label appended to the default feature root
                              Example: layers20-22-final -> .../features-layers20-22-final
  SCRATCH_FEATURES_DIR        Explicit feature root override
  TRAIN_LAYER_NAME            Layer subdirectory to train on (default: layer_final)
  TRAIN_MODEL_RUN_NAME        Optional label appended to the default model root
                              Defaults to TRAIN_FEATURES_RUN_NAME when set
  TRAIN_OUTPUT_BASE           Explicit model output root override
  TRAIN_DATA_DIR              Explicit full training data directory override
  TRAIN_OUTPUT_DIR            Explicit full training output directory override
  TRAIN_BATCH_SIZE            Defaults to 64
  TRAIN_EPOCHS                Defaults to 50
  TRAIN_FRAME_STRIDE          Defaults to 4
  TRAIN_MAX_FRAMES            Defaults to 0
  TRAIN_MAX_FILES             Defaults to 0
  TRAIN_RANDOM_SUBSET_FILES   Randomly choose this many feature files before frame expansion
                              Defaults to 0 (use all files)
  TRAIN_SUBSET_SEED           Seed for deterministic random file subset selection (default: 42)
  TRAIN_SAMPLER_MODE          Batch construction mode: random or grouped (default: grouped)
  TRAIN_STAGE_TO_LOCAL        Stage the selected file subset into node-local scratch before training (default: 0)
  TRAIN_STAGE_MAX_MB          Abort staging when the selected subset exceeds this many MB (default: 0 = disabled)
  TRAIN_STAGE_TIMEOUT_SEC     Timeout in seconds for staged rsync copy (default: 0 = disabled)
  TRAIN_REPACK_STAGED         Repack staged local files into shard arrays before training (default: 0)
  TRAIN_REPACK_SHARD_SIZE_MB  Approximate target shard size in MB for local repack (default: 512)
  TRAIN_NUM_WORKERS           Defaults to 8
  SBATCH_ACCOUNT              Defaults to soc-gpu-np
  SBATCH_PARTITION            Defaults to soc-gpu-np
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON   Optional explicit env python passed through to SLURM
  TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH Optional explicit conda.sh path passed through to SLURM
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME Optional explicit conda env name passed through to SLURM
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX Optional explicit conda env prefix passed through to SLURM

Examples:
  Scripts/TrainingScripts/submit_train_sae_hpc.sh
  TRAIN_FEATURES_RUN_NAME=layers20-22-final Scripts/TrainingScripts/submit_train_sae_hpc.sh
  TRAIN_FEATURES_RUN_NAME=layers20-22-final TRAIN_LAYER_NAME=layer_20 Scripts/TrainingScripts/submit_train_sae_hpc.sh
  TRAIN_FEATURES_RUN_NAME=layers20-22-final TRAIN_MODEL_RUN_NAME=layer-comparison TRAIN_LAYER_NAME=layer_21 Scripts/TrainingScripts/submit_train_sae_hpc.sh
  TRAIN_DATA_DIR=/scratch/general/vast/$USER/sae_output/features-all-layers/layer_final TRAIN_OUTPUT_DIR=/scratch/general/vast/$USER/sae_output/models-all-layers/layer_final-subset TRAIN_FRAME_STRIDE=1 TRAIN_RANDOM_SUBSET_FILES=2048 TRAIN_SUBSET_SEED=42 TRAIN_SAMPLER_MODE=grouped Scripts/TrainingScripts/submit_train_sae_hpc.sh
  TRAIN_DATA_DIR=/scratch/general/vast/$USER/sae_output/features-all-layers/layer_final TRAIN_OUTPUT_DIR=/scratch/general/vast/$USER/sae_output/models-all-layers/layer_final-repacked TRAIN_FRAME_STRIDE=1 TRAIN_RANDOM_SUBSET_FILES=8192 TRAIN_SUBSET_SEED=42 TRAIN_SAMPLER_MODE=grouped TRAIN_STAGE_TO_LOCAL=1 TRAIN_REPACK_STAGED=1 Scripts/TrainingScripts/submit_train_sae_hpc.sh
  TRAIN_DATA_DIR=/scratch/general/vast/$USER/sae_output/features-custom/layer_final TRAIN_OUTPUT_DIR=/scratch/general/vast/$USER/sae_output/models-custom/layer_final Scripts/TrainingScripts/submit_train_sae_hpc.sh
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SLURM_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/chpc_submit.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: SLURM training script not found at: $SLURM_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"

CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"
TRAIN_FEATURES_RUN_NAME="${TRAIN_FEATURES_RUN_NAME:-${EXTRACT_RUN_NAME:-}}"
TRAIN_LAYER_NAME="${TRAIN_LAYER_NAME:-layer_final}"

DEFAULT_FEATURES_DIR="${CHPC_SCRATCH_BASE}/sae_output/features"
if [[ -n "$TRAIN_FEATURES_RUN_NAME" ]]; then
  DEFAULT_FEATURES_DIR="${CHPC_SCRATCH_BASE}/sae_output/features-${TRAIN_FEATURES_RUN_NAME}"
fi
SCRATCH_FEATURES_DIR="${SCRATCH_FEATURES_DIR:-${DEFAULT_FEATURES_DIR}}"

SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-${CHPC_SCRATCH_BASE}/sae_output}"
TRAIN_MODEL_RUN_NAME="${TRAIN_MODEL_RUN_NAME:-${TRAIN_FEATURES_RUN_NAME}}"
DEFAULT_TRAIN_OUTPUT_BASE="${SCRATCH_OUTPUT_DIR}/models"
if [[ -n "$TRAIN_MODEL_RUN_NAME" ]]; then
  DEFAULT_TRAIN_OUTPUT_BASE="${SCRATCH_OUTPUT_DIR}/models-${TRAIN_MODEL_RUN_NAME}"
fi
TRAIN_OUTPUT_BASE="${TRAIN_OUTPUT_BASE:-${DEFAULT_TRAIN_OUTPUT_BASE}}"

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH_FEATURES_DIR}/${TRAIN_LAYER_NAME}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${TRAIN_OUTPUT_BASE}/${TRAIN_LAYER_NAME}}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-soc-gpu-np}"
SBATCH_PARTITION="${SBATCH_PARTITION:-soc-gpu-np}"
TEMPORAL_MUSIC_ACTIVATIONS_PYTHON="${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH="${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX:-}"
TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE:-}"
TRAIN_STAGE_TO_LOCAL="${TRAIN_STAGE_TO_LOCAL:-0}"
TRAIN_STAGE_MAX_MB="${TRAIN_STAGE_MAX_MB:-0}"
TRAIN_STAGE_TIMEOUT_SEC="${TRAIN_STAGE_TIMEOUT_SEC:-0}"
TRAIN_REPACK_STAGED="${TRAIN_REPACK_STAGED:-0}"
TRAIN_REPACK_SHARD_SIZE_MB="${TRAIN_REPACK_SHARD_SIZE_MB:-512}"

mkdir -p Logs/Slurm

echo "Submitting SAE training job"
echo "  sbatch account: $SBATCH_ACCOUNT"
echo "  sbatch partition: $SBATCH_PARTITION"
echo "  feature run label: ${TRAIN_FEATURES_RUN_NAME:-<default>}"
echo "  feature root: $SCRATCH_FEATURES_DIR"
echo "  training layer: $TRAIN_LAYER_NAME"
echo "  data dir: $TRAIN_DATA_DIR"
echo "  model run label: ${TRAIN_MODEL_RUN_NAME:-<default>}"
echo "  model output root: $TRAIN_OUTPUT_BASE"
echo "  output dir: $TRAIN_OUTPUT_DIR"
echo "  epochs: ${TRAIN_EPOCHS:-50}"
echo "  batch_size: ${TRAIN_BATCH_SIZE:-64}"
echo "  frame_stride: ${TRAIN_FRAME_STRIDE:-4}"
echo "  random_subset_files: ${TRAIN_RANDOM_SUBSET_FILES:-0}"
echo "  subset_seed: ${TRAIN_SUBSET_SEED:-42}"
echo "  sampler_mode: ${TRAIN_SAMPLER_MODE:-grouped}"
echo "  stage_to_local: $TRAIN_STAGE_TO_LOCAL"
echo "  stage_max_mb: $TRAIN_STAGE_MAX_MB"
echo "  stage_timeout_sec: $TRAIN_STAGE_TIMEOUT_SEC"
echo "  repack_staged: $TRAIN_REPACK_STAGED"
echo "  repack_shard_size_mb: $TRAIN_REPACK_SHARD_SIZE_MB"

export CHPC_UID CHPC_SCRATCH_BASE SCRATCH_OUTPUT_DIR SCRATCH_FEATURES_DIR
export TRAIN_FEATURES_RUN_NAME TRAIN_MODEL_RUN_NAME TRAIN_LAYER_NAME
export TRAIN_DATA_DIR TRAIN_OUTPUT_BASE TRAIN_OUTPUT_DIR
export TRAIN_BATCH_SIZE TRAIN_EPOCHS TRAIN_FRAME_STRIDE TRAIN_MAX_FRAMES TRAIN_MAX_FILES
export TRAIN_RANDOM_SUBSET_FILES TRAIN_SUBSET_SEED TRAIN_SAMPLER_MODE TRAIN_NUM_WORKERS
export TRAIN_STAGE_TO_LOCAL TRAIN_STAGE_MAX_MB TRAIN_STAGE_TIMEOUT_SEC
export TRAIN_REPACK_STAGED TRAIN_REPACK_SHARD_SIZE_MB
export TEMPORAL_MUSIC_ACTIVATIONS_PYTHON TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH
export TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX
export TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE

sbatch \
  --account="$SBATCH_ACCOUNT" \
  --partition="$SBATCH_PARTITION" \
  --export=ALL \
  "$SLURM_SCRIPT"