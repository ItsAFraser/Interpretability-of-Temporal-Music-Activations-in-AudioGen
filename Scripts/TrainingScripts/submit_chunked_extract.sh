#!/usr/bin/env bash
set -euo pipefail

# Submit a chunked CHPC extraction array for all MusicGen decoder layers.
#
# Defaults target a meaningful first run without turning back into one long,
# queue-heavy job: 1,024 files in 8 jobs of 128 files each.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/submit_chunked_extract.sh

Environment overrides:
  TOTAL_FILES               Total number of sorted audio files to extract (default: 1024)
  CHUNK_SIZE                Files per SLURM array task (default: 128)
  RAW_AUDIO_DIR             Defaults to /scratch/general/vast/$USER/mtg-jamendo/raw_30s/audio
  SCRATCH_FEATURES_DIR      Defaults to /scratch/general/vast/$USER/sae_output/features
  EXTRACT_LAYERS            Defaults to all MusicGen-small decoder layers plus final
  EXTRACT_MAX_DURATION_SEC  Defaults to 30
  EXTRACT_DEVICE            Optional override passed to the extractor (default: wrapper picks cuda on GPU jobs)
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON   Optional explicit env python passed through to SLURM jobs
  TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH Optional explicit conda.sh path passed through to SLURM jobs
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME Optional explicit conda env name passed through to SLURM jobs
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX Optional explicit conda env prefix passed through to SLURM jobs

Examples:
  Scripts/TrainingScripts/submit_chunked_extract.sh
  TOTAL_FILES=2048 CHUNK_SIZE=128 Scripts/TrainingScripts/submit_chunked_extract.sh
  TOTAL_FILES=4096 CHUNK_SIZE=64 SCRATCH_FEATURES_DIR=/scratch/general/vast/$USER/sae_output/features-all Scripts/TrainingScripts/submit_chunked_extract.sh
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=$HOME/miniforge3/envs/temporal-music-activations-cuda/bin/python Scripts/TrainingScripts/submit_chunked_extract.sh
EOF
  exit 0
fi

TOTAL_FILES="${TOTAL_FILES:-1024}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"

if ! [[ "$TOTAL_FILES" =~ ^[0-9]+$ ]] || [[ "$TOTAL_FILES" -le 0 ]]; then
  echo "Error: TOTAL_FILES must be a positive integer, got '$TOTAL_FILES'" >&2
  exit 1
fi

if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || [[ "$CHUNK_SIZE" -le 0 ]]; then
  echo "Error: CHUNK_SIZE must be a positive integer, got '$CHUNK_SIZE'" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SLURM_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/chpc_extract.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: SLURM extraction script not found at: $SLURM_SCRIPT" >&2
  exit 1
fi

CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"
RAW_AUDIO_DIR="${RAW_AUDIO_DIR:-${CHPC_SCRATCH_BASE}/mtg-jamendo/raw_30s/audio}"
SCRATCH_FEATURES_DIR="${SCRATCH_FEATURES_DIR:-${CHPC_SCRATCH_BASE}/sae_output/features}"
EXTRACT_LAYERS="${EXTRACT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1}"
EXTRACT_MAX_DURATION_SEC="${EXTRACT_MAX_DURATION_SEC:-30}"
EXTRACT_DEVICE="${EXTRACT_DEVICE:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-soc-gpu-np}"
SBATCH_PARTITION="${SBATCH_PARTITION:-soc-gpu-np}"
TEMPORAL_MUSIC_ACTIVATIONS_PYTHON="${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH="${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX:-}"
TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE:-}"

NUM_CHUNKS=$(( (TOTAL_FILES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
ARRAY_END=$(( NUM_CHUNKS - 1 ))

echo "Submitting extraction array with total_files=$TOTAL_FILES and chunk_size=$CHUNK_SIZE"
echo "  array range: 0-$ARRAY_END"
echo "  sbatch account: $SBATCH_ACCOUNT"
echo "  sbatch partition: $SBATCH_PARTITION"
echo "  raw audio dir: $RAW_AUDIO_DIR"
echo "  output dir: $SCRATCH_FEATURES_DIR"
echo "  decoder layers: $EXTRACT_LAYERS"
echo "  device override: ${EXTRACT_DEVICE:-auto}"

sbatch \
  --account="$SBATCH_ACCOUNT" \
  --partition="$SBATCH_PARTITION" \
  --array="0-$ARRAY_END" \
  --export="ALL,CHPC_UID=$CHPC_UID,CHPC_SCRATCH_BASE=$CHPC_SCRATCH_BASE,RAW_AUDIO_DIR=$RAW_AUDIO_DIR,SCRATCH_FEATURES_DIR=$SCRATCH_FEATURES_DIR,EXTRACT_LAYERS=$EXTRACT_LAYERS,EXTRACT_MAX_DURATION_SEC=$EXTRACT_MAX_DURATION_SEC,EXTRACT_CHUNK_SIZE=$CHUNK_SIZE,EXTRACT_MAX_FILES=$TOTAL_FILES,EXTRACT_DEVICE=$EXTRACT_DEVICE,TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=$TEMPORAL_MUSIC_ACTIVATIONS_PYTHON,TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH=$TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH,TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME=$TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME,TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX=$TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX,TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE=$TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE,TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE=$TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE" \
  "$SLURM_SCRIPT"