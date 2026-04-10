#!/usr/bin/env bash
set -euo pipefail

# Submit a CHPC SAE training array with one layer per SLURM task.
#
# Defaults align with the single-layer wrapper while auto-discovering all
# extracted layer directories under the resolved feature root.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/submit_train_sae_array_hpc.sh

Environment overrides:
  TRAIN_FEATURES_RUN_NAME     Optional label appended to the default feature root
                              Example: all-layers -> .../features-all-layers
  SCRATCH_FEATURES_DIR        Explicit feature root override
  TRAIN_LAYER_NAMES           Optional comma-separated layer directory names
                              Example: layer_08,layer_16,layer_final
                              Defaults to all discovered layer_* directories
  TRAIN_ARRAY_MAX_CONCURRENT  Max concurrent array tasks (default: 4)
  TRAIN_LAYER_MANIFEST_PATH   Optional manifest path written before submission
  TRAIN_MODEL_RUN_NAME        Optional label appended to the default model root
                              Defaults to TRAIN_FEATURES_RUN_NAME when set
  TRAIN_OUTPUT_BASE           Explicit model output root override
  TRAIN_BATCH_SIZE            Defaults to 64
  TRAIN_EPOCHS                Defaults to 50
  TRAIN_FRAME_STRIDE          Defaults to 4
  TRAIN_MAX_FRAMES            Defaults to 0
  TRAIN_MAX_FILES             Defaults to 0
  TRAIN_RANDOM_SUBSET_FILES   Randomly choose this many feature files before frame expansion
                              Defaults to 0 (use all files)
  TRAIN_SUBSET_SEED           Seed for deterministic random file subset selection (default: 42)
  TRAIN_NUM_WORKERS           Defaults to 8
  SBATCH_ACCOUNT              Defaults to soc-gpu-np
  SBATCH_PARTITION            Defaults to soc-gpu-np
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON   Optional explicit env python passed through to SLURM
  TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH Optional explicit conda.sh path passed through to SLURM
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME Optional explicit conda env name passed through to SLURM
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX Optional explicit conda env prefix passed through to SLURM

Examples:
  Scripts/TrainingScripts/submit_train_sae_array_hpc.sh
  TRAIN_FEATURES_RUN_NAME=all-layers Scripts/TrainingScripts/submit_train_sae_array_hpc.sh
  TRAIN_FEATURES_RUN_NAME=all-layers TRAIN_ARRAY_MAX_CONCURRENT=2 Scripts/TrainingScripts/submit_train_sae_array_hpc.sh
  TRAIN_FEATURES_RUN_NAME=all-layers TRAIN_LAYER_NAMES=layer_08,layer_16,layer_final Scripts/TrainingScripts/submit_train_sae_array_hpc.sh
  TRAIN_FEATURES_RUN_NAME=all-layers TRAIN_FRAME_STRIDE=1 TRAIN_RANDOM_SUBSET_FILES=2048 TRAIN_SUBSET_SEED=42 TRAIN_ARRAY_MAX_CONCURRENT=2 Scripts/TrainingScripts/submit_train_sae_array_hpc.sh
  SCRATCH_FEATURES_DIR=/scratch/general/vast/$USER/sae_output/features-all-layers TRAIN_OUTPUT_BASE=/scratch/general/vast/$USER/sae_output/models-all-layers Scripts/TrainingScripts/submit_train_sae_array_hpc.sh
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SLURM_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/chpc_submit_train_array.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: SLURM training array script not found at: $SLURM_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"

CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"
TRAIN_FEATURES_RUN_NAME="${TRAIN_FEATURES_RUN_NAME:-${EXTRACT_RUN_NAME:-}}"

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

TRAIN_LAYER_NAMES="${TRAIN_LAYER_NAMES:-}"
TRAIN_ARRAY_MAX_CONCURRENT="${TRAIN_ARRAY_MAX_CONCURRENT:-4}"
TRAIN_LAYER_MANIFEST_PATH="${TRAIN_LAYER_MANIFEST_PATH:-${TRAIN_OUTPUT_BASE}/train_layer_manifest.txt}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-soc-gpu-np}"
SBATCH_PARTITION="${SBATCH_PARTITION:-soc-gpu-np}"
TEMPORAL_MUSIC_ACTIVATIONS_PYTHON="${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH="${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX:-}"
TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE:-}"

if [[ ! -d "$SCRATCH_FEATURES_DIR" ]]; then
  echo "Error: Feature root does not exist: $SCRATCH_FEATURES_DIR" >&2
  exit 1
fi

if ! [[ "$TRAIN_ARRAY_MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$TRAIN_ARRAY_MAX_CONCURRENT" -le 0 ]]; then
  echo "Error: TRAIN_ARRAY_MAX_CONCURRENT must be a positive integer, got '$TRAIN_ARRAY_MAX_CONCURRENT'" >&2
  exit 1
fi

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "$value"
}

is_valid_layer_name() {
  local layer_name="$1"
  [[ "$layer_name" =~ ^layer_([0-9]{2}|final)$ ]]
}

discover_layers() {
  find "$SCRATCH_FEATURES_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | LC_ALL=C sort \
    | while IFS= read -r layer_name; do
        if is_valid_layer_name "$layer_name"; then
          printf '%s\n' "$layer_name"
        fi
      done
}

has_feature_files() {
  local layer_dir="$1"
  local sample_file
  sample_file="$(find "$layer_dir" -type f \( -name '*.pt' -o -name '*.npy' \) -print -quit)"
  [[ -n "$sample_file" ]]
}

declare -a requested_layers=()
declare -A seen_layers=()

if [[ -n "$TRAIN_LAYER_NAMES" ]]; then
  IFS=',' read -r -a raw_layers <<< "$TRAIN_LAYER_NAMES"
  for raw_layer in "${raw_layers[@]}"; do
    layer_name="$(trim_whitespace "$raw_layer")"
    if [[ -z "$layer_name" ]]; then
      continue
    fi
    if ! is_valid_layer_name "$layer_name"; then
      echo "Error: Invalid layer name '$layer_name'. Expected layer_00..layer_99 or layer_final." >&2
      exit 1
    fi
    if [[ -z "${seen_layers[$layer_name]:-}" ]]; then
      requested_layers+=("$layer_name")
      seen_layers["$layer_name"]=1
    fi
  done
else
  while IFS= read -r layer_name; do
    if [[ -z "$layer_name" ]]; then
      continue
    fi
    requested_layers+=("$layer_name")
  done < <(discover_layers)
fi

if [[ "${#requested_layers[@]}" -eq 0 ]]; then
  echo "Error: No training layers resolved under $SCRATCH_FEATURES_DIR" >&2
  exit 1
fi

declare -a resolved_layers=()
for layer_name in "${requested_layers[@]}"; do
  layer_dir="$SCRATCH_FEATURES_DIR/$layer_name"
  if [[ ! -d "$layer_dir" ]]; then
    echo "Error: Requested layer directory does not exist: $layer_dir" >&2
    exit 1
  fi
  if ! has_feature_files "$layer_dir"; then
    echo "Error: Requested layer directory contains no .pt or .npy files: $layer_dir" >&2
    exit 1
  fi
  resolved_layers+=("$layer_name")
done

mkdir -p "$(dirname "$TRAIN_LAYER_MANIFEST_PATH")"
mkdir -p "$TRAIN_OUTPUT_BASE"
mkdir -p Logs/Slurm

printf '%s\n' "${resolved_layers[@]}" > "$TRAIN_LAYER_MANIFEST_PATH"

ARRAY_END=$(( ${#resolved_layers[@]} - 1 ))
ARRAY_SPEC="0-$ARRAY_END%$TRAIN_ARRAY_MAX_CONCURRENT"

# Avoid conflicting per-layer overrides when submitting the array workflow.
unset TRAIN_DATA_DIR || true
unset TRAIN_OUTPUT_DIR || true
unset TRAIN_LAYER_NAME || true

echo "Submitting SAE training array"
echo "  sbatch account: $SBATCH_ACCOUNT"
echo "  sbatch partition: $SBATCH_PARTITION"
echo "  feature run label: ${TRAIN_FEATURES_RUN_NAME:-<default>}"
echo "  feature root: $SCRATCH_FEATURES_DIR"
echo "  model run label: ${TRAIN_MODEL_RUN_NAME:-<default>}"
echo "  model output root: $TRAIN_OUTPUT_BASE"
echo "  manifest path: $TRAIN_LAYER_MANIFEST_PATH"
echo "  array spec: $ARRAY_SPEC"
echo "  max concurrent tasks: $TRAIN_ARRAY_MAX_CONCURRENT"
echo "  epochs: ${TRAIN_EPOCHS:-50}"
echo "  batch_size: ${TRAIN_BATCH_SIZE:-64}"
echo "  frame_stride: ${TRAIN_FRAME_STRIDE:-4}"
echo "  random_subset_files: ${TRAIN_RANDOM_SUBSET_FILES:-0}"
echo "  subset_seed: ${TRAIN_SUBSET_SEED:-42}"
echo "  layers:"
for layer_name in "${resolved_layers[@]}"; do
  echo "    - $layer_name"
done

export CHPC_UID CHPC_SCRATCH_BASE SCRATCH_OUTPUT_DIR SCRATCH_FEATURES_DIR
export TRAIN_FEATURES_RUN_NAME TRAIN_MODEL_RUN_NAME TRAIN_OUTPUT_BASE
export TRAIN_BATCH_SIZE TRAIN_EPOCHS TRAIN_FRAME_STRIDE TRAIN_MAX_FRAMES TRAIN_MAX_FILES
export TRAIN_RANDOM_SUBSET_FILES TRAIN_SUBSET_SEED TRAIN_NUM_WORKERS
export TRAIN_LAYER_MANIFEST_PATH
export TEMPORAL_MUSIC_ACTIVATIONS_PYTHON TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH
export TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX
export TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE

sbatch \
  --account="$SBATCH_ACCOUNT" \
  --partition="$SBATCH_PARTITION" \
  --array="$ARRAY_SPEC" \
  --export=ALL \
  "$SLURM_SCRIPT"