#!/usr/bin/env bash
set -euo pipefail

# Launcher for MusicGen residual feature extraction.
#
# Usage:
#   Scripts/TrainingScripts/run_extract_musicgen_features.sh <input_dir> <output_dir> [extra args]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  cat <<'EOF'
Usage:
  Scripts/TrainingScripts/run_extract_musicgen_features.sh <input_dir> <output_dir> [extra extraction args]

Examples:
  Scripts/TrainingScripts/run_extract_musicgen_features.sh /Volumes/SSD_ALF/DataSets/MTG-Jamendo/Data/raw/raw_30s_audio_low Data/Models/features
  Scripts/TrainingScripts/run_extract_musicgen_features.sh /Volumes/SSD_ALF/DataSets/MTG-Jamendo/Data/raw/raw_30s_audio_low Data/Models/features --max_files 16 --max_duration_sec 30 --decoder_layers 0,8,16,-1
  Scripts/TrainingScripts/run_extract_musicgen_features.sh /scratch/general/vast/$USER/mtg-jamendo/raw_30s/audio /scratch/general/vast/$USER/sae_output/features --file_start 512 --file_count 256 --decoder_layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1
EOF
  exit 0
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
EXTRACT_SCRIPT="$ROOT_DIR/Temporal-Music-Activations/ExtractMusicGenFeatures.py"
CONDA_UTILS="$ROOT_DIR/Scripts/TrainingScripts/conda_utils.sh"

if [[ ! -f "$EXTRACT_SCRIPT" ]]; then
  echo "Error: Extraction script not found at: $EXTRACT_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$CONDA_UTILS" ]]; then
  echo "Error: Missing helper script at: $CONDA_UTILS" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_UTILS"
temporal_setup_python_environment temporal-music-activations-cuda temporal-music-activations-apple
temporal_print_python_diagnostics "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON"
temporal_validate_python_imports "$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON" torch transformers

PYTHON_CMD=("$TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON")

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$EXTRACT_SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "$@"