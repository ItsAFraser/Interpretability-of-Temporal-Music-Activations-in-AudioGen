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
#SBATCH --job-name=TrainSAEArray
#SBATCH --output=Logs/Slurm/TrainSAEArray_%A_%a.out
#SBATCH --error=Logs/Slurm/TrainSAEArray_%A_%a.err

set -euo pipefail

resolve_repo_root() {
	local candidate="${SLURM_SUBMIT_DIR:-$PWD}"

	while [[ "$candidate" != "/" ]]; do
		if [[ -f "$candidate/Scripts/TrainingScripts/chpc_submit.slurm" ]]; then
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

TRAIN_WRAPPER_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/chpc_submit.slurm"
if [[ ! -f "$TRAIN_WRAPPER_SCRIPT" ]]; then
	echo "Error: Missing training wrapper script at $TRAIN_WRAPPER_SCRIPT" >&2
	exit 1
fi

TRAIN_LAYER_MANIFEST_PATH="${TRAIN_LAYER_MANIFEST_PATH:-}"
if [[ -z "$TRAIN_LAYER_MANIFEST_PATH" ]]; then
	echo "Error: TRAIN_LAYER_MANIFEST_PATH must be set for array submission." >&2
	exit 1
fi

if [[ ! -f "$TRAIN_LAYER_MANIFEST_PATH" ]]; then
	echo "Error: Layer manifest not found at $TRAIN_LAYER_MANIFEST_PATH" >&2
	exit 1
fi

TRAIN_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$TRAIN_ARRAY_TASK_ID" == "4294967294" ]]; then
	TRAIN_ARRAY_TASK_ID=0
fi

if ! [[ "$TRAIN_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
	echo "Error: Invalid SLURM_ARRAY_TASK_ID='$TRAIN_ARRAY_TASK_ID'" >&2
	exit 1
fi

TOTAL_LAYERS="$(awk 'END{print NR+0}' "$TRAIN_LAYER_MANIFEST_PATH")"
if [[ "$TOTAL_LAYERS" -le 0 ]]; then
	echo "Error: Layer manifest is empty: $TRAIN_LAYER_MANIFEST_PATH" >&2
	exit 1
fi

if [[ "$TRAIN_ARRAY_TASK_ID" -ge "$TOTAL_LAYERS" ]]; then
	echo "Error: Array task id $TRAIN_ARRAY_TASK_ID exceeds manifest size $TOTAL_LAYERS" >&2
	exit 1
fi

TRAIN_LAYER_NAME="$(sed -n "$((TRAIN_ARRAY_TASK_ID + 1))p" "$TRAIN_LAYER_MANIFEST_PATH")"
if [[ -z "$TRAIN_LAYER_NAME" ]]; then
	echo "Error: Failed to resolve layer name from manifest for task $TRAIN_ARRAY_TASK_ID" >&2
	exit 1
fi

# Force per-layer path derivation inside the standard single-layer wrapper.
unset TRAIN_DATA_DIR || true
unset TRAIN_OUTPUT_DIR || true
export TRAIN_LAYER_NAME

echo "Starting SAE layer-array task"
echo "  slurm account: ${SLURM_JOB_ACCOUNT:-soc-gpu-np}"
echo "  slurm partition: ${SLURM_JOB_PARTITION:-soc-gpu-np}"
echo "  array task id: $TRAIN_ARRAY_TASK_ID"
echo "  layer manifest: $TRAIN_LAYER_MANIFEST_PATH"
echo "  selected layer: $TRAIN_LAYER_NAME"
echo "  feature root: ${SCRATCH_FEATURES_DIR:-<unset>}"
echo "  model output root: ${TRAIN_OUTPUT_BASE:-<unset>}"
echo "  stage_to_local: ${TRAIN_STAGE_TO_LOCAL:-0}"

bash "$TRAIN_WRAPPER_SCRIPT"