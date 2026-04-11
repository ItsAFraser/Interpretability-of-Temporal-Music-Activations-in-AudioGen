#!/bin/bash

#=======================================================
# Slurm Settings for CHPC Granite (CS6966 Class)
#=======================================================
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=TrainSAE
#SBATCH --output=Logs/Slurm/TrainSAE_%j.out
#SBATCH --error=Logs/Slurm/TrainSAE_%j.err

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
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch numpy

# Defaults target CHPC VAST scratch for the current user.
CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"

HF_HOME="${HF_HOME:-${CHPC_SCRATCH_BASE}/hf-cache}"
TORCH_HOME="${TORCH_HOME:-${CHPC_SCRATCH_BASE}/torch-cache}"
TMPDIR="${TMPDIR:-${CHPC_SCRATCH_BASE}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CHPC_SCRATCH_BASE}/xdg-cache}"
export HF_HOME TORCH_HOME TMPDIR XDG_CACHE_HOME
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$XDG_CACHE_HOME"

SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-${CHPC_SCRATCH_BASE}/sae_output}"
mkdir -p "$SCRATCH_OUTPUT_DIR"

TRAIN_FEATURES_RUN_NAME="${TRAIN_FEATURES_RUN_NAME:-${EXTRACT_RUN_NAME:-}}"
DEFAULT_FEATURES_DIR="${CHPC_SCRATCH_BASE}/sae_output/features"
if [[ -n "$TRAIN_FEATURES_RUN_NAME" ]]; then
	DEFAULT_FEATURES_DIR="${CHPC_SCRATCH_BASE}/sae_output/features-${TRAIN_FEATURES_RUN_NAME}"
fi
SCRATCH_FEATURES_DIR="${SCRATCH_FEATURES_DIR:-${DEFAULT_FEATURES_DIR}}"
TRAIN_LAYER_NAME="${TRAIN_LAYER_NAME:-layer_final}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH_FEATURES_DIR}/${TRAIN_LAYER_NAME}}"
TRAIN_MODEL_RUN_NAME="${TRAIN_MODEL_RUN_NAME:-${TRAIN_FEATURES_RUN_NAME}}"
DEFAULT_TRAIN_OUTPUT_BASE="${SCRATCH_OUTPUT_DIR}/models"
if [[ -n "$TRAIN_MODEL_RUN_NAME" ]]; then
	DEFAULT_TRAIN_OUTPUT_BASE="${SCRATCH_OUTPUT_DIR}/models-${TRAIN_MODEL_RUN_NAME}"
fi
TRAIN_OUTPUT_BASE="${TRAIN_OUTPUT_BASE:-${DEFAULT_TRAIN_OUTPUT_BASE}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${TRAIN_OUTPUT_BASE}/${TRAIN_LAYER_NAME}}"
mkdir -p "$TRAIN_OUTPUT_DIR"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TRAIN_FRAME_STRIDE="${TRAIN_FRAME_STRIDE:-4}"
TRAIN_MAX_FRAMES="${TRAIN_MAX_FRAMES:-0}"
TRAIN_MAX_FILES="${TRAIN_MAX_FILES:-0}"
TRAIN_RANDOM_SUBSET_FILES="${TRAIN_RANDOM_SUBSET_FILES:-0}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_SAMPLER_MODE="${TRAIN_SAMPLER_MODE:-grouped}"
TRAIN_STAGE_TO_LOCAL="${TRAIN_STAGE_TO_LOCAL:-0}"
TRAIN_STAGE_MAX_MB="${TRAIN_STAGE_MAX_MB:-0}"
TRAIN_STAGE_TIMEOUT_SEC="${TRAIN_STAGE_TIMEOUT_SEC:-0}"
TRAIN_REPACK_STAGED="${TRAIN_REPACK_STAGED:-0}"
TRAIN_REPACK_SHARD_SIZE_MB="${TRAIN_REPACK_SHARD_SIZE_MB:-512}"

SOURCE_TRAIN_DATA_DIR="${SCRATCH_FEATURES_DIR}/${TRAIN_LAYER_NAME}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SOURCE_TRAIN_DATA_DIR}}"
TRAIN_FILE_MANIFEST_PATH="${TRAIN_FILE_MANIFEST_PATH:-}"
TRAIN_FILE_MANIFEST_METADATA_PATH="${TRAIN_FILE_MANIFEST_METADATA_PATH:-}"
TRAIN_STAGED_DATA_DIR="${TRAIN_STAGED_DATA_DIR:-${TMPDIR}/sae-staged/${TRAIN_LAYER_NAME}}"
TRAIN_REPACKED_DATA_DIR="${TRAIN_REPACKED_DATA_DIR:-${TMPDIR}/sae-repacked/${TRAIN_LAYER_NAME}}"

RUN_TRAIN_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/run_train_sae_hpc.sh"
if [[ ! -f "$RUN_TRAIN_SCRIPT" ]]; then
	echo "Error: Missing training launcher at $RUN_TRAIN_SCRIPT" >&2
	exit 1
fi

STAGE_HELPER_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/stage_selected_features_to_local.sh"
REPACK_HELPER_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/repack_staged_features.sh"
if [[ "$TRAIN_STAGE_TO_LOCAL" == "1" ]]; then
	if [[ ! -f "$STAGE_HELPER_SCRIPT" ]]; then
		echo "Error: Missing staging helper at $STAGE_HELPER_SCRIPT" >&2
		exit 1
	fi
	if [[ -z "$TRAIN_FILE_MANIFEST_PATH" ]]; then
		TRAIN_FILE_MANIFEST_PATH="$TRAIN_OUTPUT_DIR/selected_feature_files.txt"
	fi
	if [[ -z "$TRAIN_FILE_MANIFEST_METADATA_PATH" ]]; then
		TRAIN_FILE_MANIFEST_METADATA_PATH="$TRAIN_OUTPUT_DIR/selected_feature_files.json"
	fi
	if bash "$STAGE_HELPER_SCRIPT" \
		"$SOURCE_TRAIN_DATA_DIR" \
		"$TRAIN_STAGED_DATA_DIR" \
		"$TRAIN_FILE_MANIFEST_PATH" \
		"$TRAIN_FILE_MANIFEST_METADATA_PATH"; then
		TRAIN_DATA_DIR="$TRAIN_STAGED_DATA_DIR"
		TRAIN_DATA_SOURCE_LABEL="local-staged"
		if [[ "$TRAIN_REPACK_STAGED" == "1" ]]; then
			if [[ ! -f "$REPACK_HELPER_SCRIPT" ]]; then
				echo "Error: Missing repack helper at $REPACK_HELPER_SCRIPT" >&2
				exit 1
			fi
			if bash "$REPACK_HELPER_SCRIPT" \
				"$TRAIN_STAGED_DATA_DIR" \
				"$TRAIN_FILE_MANIFEST_PATH" \
				"$TRAIN_REPACKED_DATA_DIR"; then
				TRAIN_DATA_DIR="$TRAIN_REPACKED_DATA_DIR"
				TRAIN_DATA_SOURCE_LABEL="local-repacked"
			else
				echo "Warning: Local repack failed; falling back to staged individual files." >&2
			fi
		fi
	else
		echo "Warning: Local staging failed; falling back to source features on VAST." >&2
		TRAIN_DATA_DIR="$SOURCE_TRAIN_DATA_DIR"
		TRAIN_FILE_MANIFEST_PATH=""
		TRAIN_FILE_MANIFEST_METADATA_PATH=""
		TRAIN_DATA_SOURCE_LABEL="vast-source"
	fi
else
	TRAIN_DATA_SOURCE_LABEL="vast-source"
fi

echo "Starting SAE Training on CHPC!"
echo "  slurm account: ${SLURM_JOB_ACCOUNT:-soc-gpu-np}"
echo "  slurm partition: ${SLURM_JOB_PARTITION:-soc-gpu-np}"
echo "  feature run label: ${TRAIN_FEATURES_RUN_NAME:-<default>}"
echo "  source data dir: $SOURCE_TRAIN_DATA_DIR"
echo "  data dir: $TRAIN_DATA_DIR"
echo "  data source: $TRAIN_DATA_SOURCE_LABEL"
echo "  model run label: ${TRAIN_MODEL_RUN_NAME:-<default>}"
echo "  output dir: $TRAIN_OUTPUT_DIR"
echo "  layer: $TRAIN_LAYER_NAME"
echo "  batch_size: $TRAIN_BATCH_SIZE"
echo "  epochs: $TRAIN_EPOCHS"
echo "  frame_stride: $TRAIN_FRAME_STRIDE"
echo "  random_subset_files: $TRAIN_RANDOM_SUBSET_FILES"
echo "  subset_seed: $TRAIN_SUBSET_SEED"
echo "  sampler_mode: $TRAIN_SAMPLER_MODE"
echo "  stage_to_local: $TRAIN_STAGE_TO_LOCAL"
echo "  stage_max_mb: $TRAIN_STAGE_MAX_MB"
echo "  stage_timeout_sec: $TRAIN_STAGE_TIMEOUT_SEC"
echo "  repack_staged: $TRAIN_REPACK_STAGED"
echo "  repack_shard_size_mb: $TRAIN_REPACK_SHARD_SIZE_MB"
echo "  file_manifest_path: ${TRAIN_FILE_MANIFEST_PATH:-<none>}"
echo "  python: $TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
echo "  HF_HOME: $HF_HOME"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  TMPDIR: $TMPDIR"

train_cmd=(
	bash "$RUN_TRAIN_SCRIPT" "$TRAIN_DATA_DIR" "$TRAIN_OUTPUT_DIR"
	--batch_size "$TRAIN_BATCH_SIZE"
	--epochs "$TRAIN_EPOCHS"
	--frame_stride "$TRAIN_FRAME_STRIDE"
	--max_frames "$TRAIN_MAX_FRAMES"
	--max_files "$TRAIN_MAX_FILES"
	--random_subset_files "$TRAIN_RANDOM_SUBSET_FILES"
	--subset_seed "$TRAIN_SUBSET_SEED"
	--sampler_mode "$TRAIN_SAMPLER_MODE"
	--num_workers "$TRAIN_NUM_WORKERS"
	--source_data_dir "$SOURCE_TRAIN_DATA_DIR"
)

if [[ -n "$TRAIN_FILE_MANIFEST_PATH" ]]; then
	train_cmd+=(--file_manifest_path "$TRAIN_FILE_MANIFEST_PATH")
fi

"${train_cmd[@]}"

echo "Job finished."
