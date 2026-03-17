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
  Scripts/TrainingScripts/run_extract_musicgen_features.sh /Volumes/SSD_ALF/DataSets/MTG-Jamendo/Data/raw/raw_30s_audio_low Data/Models/features --max_files 16 --max_duration_sec 30 --decoder_layer -1
EOF
  exit 0
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXTRACT_SCRIPT="$ROOT_DIR/Temporal-Music-Activations/ExtractMusicGenFeatures.py"

if [[ ! -f "$EXTRACT_SCRIPT" ]]; then
  echo "Error: Extraction script not found at: $EXTRACT_SCRIPT" >&2
  exit 1
fi

if [[ -n "${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}" ]]; then
  PYTHON_CMD=("$TEMPORAL_MUSIC_ACTIVATIONS_PYTHON")
elif conda env list | grep -q '^temporal-music-activations-apple '; then
  PYTHON_CMD=(conda run -n temporal-music-activations-apple python)
else
  PYTHON_CMD=(python)
fi

cd "$ROOT_DIR"

"${PYTHON_CMD[@]}" "$EXTRACT_SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "$@"