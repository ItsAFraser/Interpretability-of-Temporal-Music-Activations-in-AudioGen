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
  TOTAL_FILES               Total number of sorted audio files to extract (default: 1024, or 'all')
  CHUNK_SIZE                Files per SLURM array task (default: 128)
  RAW_AUDIO_DIR             Defaults to /scratch/general/vast/$USER/mtg-jamendo/raw_30s/audio
  SCRATCH_FEATURES_DIR      Defaults to /scratch/general/vast/$USER/sae_output/features
  EXTRACT_RUN_NAME          Optional label appended to the default feature output path
  EXTRACT_MANIFEST_PATH     Optional cached sorted file manifest used by array tasks
  REBUILD_EXTRACT_MANIFEST  Set to 1 to rebuild the manifest before submission
  EXTRACT_LAYERS            Defaults to 20,21,22,-1
  EXTRACT_MAX_DURATION_SEC  Defaults to 0 (full track; no truncation)
  EXTRACT_DEVICE            Optional override passed to the extractor (default: wrapper picks cuda on GPU jobs)
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON   Optional explicit env python passed through to SLURM jobs
  TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH Optional explicit conda.sh path passed through to SLURM jobs
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME Optional explicit conda env name passed through to SLURM jobs
  TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX Optional explicit conda env prefix passed through to SLURM jobs

Examples:
  Scripts/TrainingScripts/submit_chunked_extract.sh
  TOTAL_FILES=2048 CHUNK_SIZE=128 Scripts/TrainingScripts/submit_chunked_extract.sh
  TOTAL_FILES=4096 CHUNK_SIZE=64 EXTRACT_RUN_NAME=layers20-22-final Scripts/TrainingScripts/submit_chunked_extract.sh
  TOTAL_FILES=all CHUNK_SIZE=2048 EXTRACT_RUN_NAME=layers20-22-final Scripts/TrainingScripts/submit_chunked_extract.sh
  TOTAL_FILES=4096 CHUNK_SIZE=64 SCRATCH_FEATURES_DIR=/scratch/general/vast/$USER/sae_output/features-all Scripts/TrainingScripts/submit_chunked_extract.sh
  TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=$HOME/miniforge3/envs/temporal-music-activations-cuda/bin/python Scripts/TrainingScripts/submit_chunked_extract.sh
EOF
  exit 0
fi

TOTAL_FILES="${TOTAL_FILES:-1024}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
EXTRACT_CHUNK_SIZE="${EXTRACT_CHUNK_SIZE:-$CHUNK_SIZE}"
EXTRACT_MAX_FILES="${EXTRACT_MAX_FILES:-0}"

if [[ "$TOTAL_FILES" != "all" ]] && { ! [[ "$TOTAL_FILES" =~ ^[0-9]+$ ]] || [[ "$TOTAL_FILES" -le 0 ]]; }; then
  echo "Error: TOTAL_FILES must be a positive integer, got '$TOTAL_FILES'" >&2
  exit 1
fi

if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || [[ "$CHUNK_SIZE" -le 0 ]]; then
  echo "Error: CHUNK_SIZE must be a positive integer, got '$CHUNK_SIZE'" >&2
  exit 1
fi

if ! [[ "$EXTRACT_CHUNK_SIZE" =~ ^[0-9]+$ ]] || [[ "$EXTRACT_CHUNK_SIZE" -le 0 ]]; then
  echo "Error: EXTRACT_CHUNK_SIZE must be a positive integer, got '$EXTRACT_CHUNK_SIZE'" >&2
  exit 1
fi

if ! [[ "$EXTRACT_MAX_FILES" =~ ^[0-9]+$ ]] || [[ "$EXTRACT_MAX_FILES" -lt 0 ]]; then
  echo "Error: EXTRACT_MAX_FILES must be a non-negative integer, got '$EXTRACT_MAX_FILES'" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SLURM_SCRIPT="$ROOT_DIR/Scripts/TrainingScripts/chpc_extract.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: SLURM extraction script not found at: $SLURM_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"

CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"
RAW_AUDIO_DIR="${RAW_AUDIO_DIR:-${CHPC_SCRATCH_BASE}/mtg-jamendo/raw_30s/audio}"
EXTRACT_RUN_NAME="${EXTRACT_RUN_NAME:-}"
DEFAULT_FEATURES_DIR="${CHPC_SCRATCH_BASE}/sae_output/features"
if [[ -n "$EXTRACT_RUN_NAME" ]]; then
  DEFAULT_FEATURES_DIR="${CHPC_SCRATCH_BASE}/sae_output/features-${EXTRACT_RUN_NAME}"
fi
SCRATCH_FEATURES_DIR="${SCRATCH_FEATURES_DIR:-${DEFAULT_FEATURES_DIR}}"
EXTRACT_LAYERS="${EXTRACT_LAYERS:-20,21,22,-1}"
EXTRACT_MAX_DURATION_SEC="${EXTRACT_MAX_DURATION_SEC:-0}"
EXTRACT_DEVICE="${EXTRACT_DEVICE:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-soc-gpu-np}"
SBATCH_PARTITION="${SBATCH_PARTITION:-soc-gpu-np}"
TEMPORAL_MUSIC_ACTIVATIONS_PYTHON="${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH="${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME:-}"
TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX:-}"
TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE:-}"
TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE="${TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE:-}"
RAW_AUDIO_DIR_SLUG="$(printf '%s' "$RAW_AUDIO_DIR" | sed 's#[^A-Za-z0-9._-]#_#g')"
EXTRACT_MANIFEST_PATH="${EXTRACT_MANIFEST_PATH:-${CHPC_SCRATCH_BASE}/sae_output/manifests/${RAW_AUDIO_DIR_SLUG}.txt}"
REBUILD_EXTRACT_MANIFEST="${REBUILD_EXTRACT_MANIFEST:-0}"

build_audio_manifest() {
  local manifest_path="$1"

  mkdir -p "$(dirname "$manifest_path")"

  python3 - "$RAW_AUDIO_DIR" "$manifest_path" <<'PY'
import os
import sys
from pathlib import Path

audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
input_dir = Path(sys.argv[1]).expanduser().resolve()
manifest_path = Path(sys.argv[2]).expanduser().resolve()

paths = []
for root, dirnames, filenames in os.walk(input_dir):
    dirnames.sort()
    filenames.sort()
    root_path = Path(root)
    for filename in filenames:
        path = root_path / filename
        if path.suffix.lower() in audio_extensions:
            paths.append(path.relative_to(input_dir).as_posix())

manifest_path.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
print(len(paths))
PY
}

if [[ "$REBUILD_EXTRACT_MANIFEST" == "1" || ! -f "$EXTRACT_MANIFEST_PATH" ]]; then
  echo "Building audio manifest at $EXTRACT_MANIFEST_PATH"
  MANIFEST_TOTAL_FILES="$(build_audio_manifest "$EXTRACT_MANIFEST_PATH")"
else
  MANIFEST_TOTAL_FILES="$(awk 'END{print NR+0}' "$EXTRACT_MANIFEST_PATH")"
fi

if [[ "$TOTAL_FILES" == "all" ]]; then
  TOTAL_FILES="$MANIFEST_TOTAL_FILES"
fi

if [[ "$TOTAL_FILES" -gt "$MANIFEST_TOTAL_FILES" ]]; then
  echo "Requested TOTAL_FILES=$TOTAL_FILES exceeds manifest size $MANIFEST_TOTAL_FILES; clamping."
  TOTAL_FILES="$MANIFEST_TOTAL_FILES"
fi

mkdir -p Logs/Slurm

NUM_CHUNKS=$(( (TOTAL_FILES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
ARRAY_END=$(( NUM_CHUNKS - 1 ))

echo "Submitting extraction array with total_files=$TOTAL_FILES and chunk_size=$CHUNK_SIZE"
echo "  array range: 0-$ARRAY_END"
echo "  sbatch account: $SBATCH_ACCOUNT"
echo "  sbatch partition: $SBATCH_PARTITION"
echo "  raw audio dir: $RAW_AUDIO_DIR"
echo "  manifest path: $EXTRACT_MANIFEST_PATH"
echo "  output dir: $SCRATCH_FEATURES_DIR"
if [[ -n "$EXTRACT_RUN_NAME" ]]; then
  echo "  run label: $EXTRACT_RUN_NAME"
fi
echo "  decoder layers: $EXTRACT_LAYERS"
echo "  max duration sec: ${EXTRACT_MAX_DURATION_SEC:-0} (0 means full track)"
echo "  extract chunk size passed to jobs: $EXTRACT_CHUNK_SIZE"
echo "  extract max files passed to jobs: $EXTRACT_MAX_FILES"
echo "  device override: ${EXTRACT_DEVICE:-auto}"

export CHPC_UID CHPC_SCRATCH_BASE RAW_AUDIO_DIR SCRATCH_FEATURES_DIR
export EXTRACT_LAYERS EXTRACT_MAX_DURATION_SEC EXTRACT_CHUNK_SIZE EXTRACT_MAX_FILES EXTRACT_MANIFEST_PATH
export TEMPORAL_MUSIC_ACTIVATIONS_PYTHON TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH
export TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX
export TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE
export EXTRACT_DEVICE

sbatch \
  --account="$SBATCH_ACCOUNT" \
  --partition="$SBATCH_PARTITION" \
  --array="0-$ARRAY_END" \
  --export=ALL \
  "$SLURM_SCRIPT"
