#!/usr/bin/env bash
set -euo pipefail

# CHPC-focused MTG-Jamendo downloader.
#
# This script is intended for use on CHPC Data Transfer Nodes (dtn05-08)
# and writes data to VAST scratch by default.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  Scripts/SetupScripts/chpc_download_mtg_jamendo.sh [options]

Options:
  --dataset <name>        Dataset subset (default: raw_30s)
                          Choices: raw_30s, autotagging_moodtheme
  --type <name>           Payload type (default: audio)
                          Choices: audio, audio-low, melspecs, acousticbrainz
  --source <name>         Download source (default: mtg-fast)
                          Choices: mtg-fast, mtg
  --target-dir <path>     Final output directory
                          (default: /scratch/general/vast/$USER/mtg-jamendo)
  --repo-dir <path>       Local clone location for mtg-jamendo-dataset repo
                          (default: <target-dir>/mtg-jamendo-dataset)
  --yes                   Skip interactive confirmation prompt

Examples:
  Scripts/SetupScripts/chpc_download_mtg_jamendo.sh --yes

  Scripts/SetupScripts/chpc_download_mtg_jamendo.sh \
    --dataset raw_30s \
    --type audio \
    --source mtg-fast \
    --target-dir /scratch/general/vast/u1406806/mtg-jamendo \
    --yes
EOF
  exit 0
fi

DATASET="raw_30s"
DATA_TYPE="audio"
SOURCE="mtg-fast"
TARGET_DIR="/scratch/general/vast/${USER}/mtg-jamendo"
REPO_DIR=""
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --type)
      DATA_TYPE="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "$DATASET" in
  raw_30s|autotagging_moodtheme) ;;
  *)
    echo "Invalid --dataset: $DATASET" >&2
    exit 1
    ;;
esac

case "$DATA_TYPE" in
  audio|audio-low|melspecs|acousticbrainz) ;;
  *)
    echo "Invalid --type: $DATA_TYPE" >&2
    exit 1
    ;;
esac

case "$SOURCE" in
  mtg|mtg-fast) ;;
  *)
    echo "Invalid --source: $SOURCE" >&2
    exit 1
    ;;
esac

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$TARGET_DIR/mtg-jamendo-dataset"
fi

HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOST_SHORT" != dtn* ]]; then
  cat >&2 <<EOF
Warning: current host is '$HOST_SHORT', not a CHPC DTN host.
For large off-campus transfers, CHPC recommends running on dtn05-08.
EOF
fi

if [[ "$TARGET_DIR" != /scratch/general/vast/* ]]; then
  cat >&2 <<EOF
Warning: target directory is outside /scratch/general/vast:
  $TARGET_DIR
Make sure this destination is approved and has enough capacity.
EOF
fi

mkdir -p "$TARGET_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required but was not found in PATH." >&2
  exit 1
fi

AVAILABLE_KB="$(df -Pk "$TARGET_DIR" | awk 'NR==2 {print $4}')"

# Approximate free space thresholds for safer unpack/remove behavior.
REQUIRED_GB=120
if [[ "$DATASET" == "raw_30s" && "$DATA_TYPE" == "audio" ]]; then
  REQUIRED_GB=650
elif [[ "$DATASET" == "raw_30s" && "$DATA_TYPE" == "audio-low" ]]; then
  REQUIRED_GB=220
elif [[ "$DATASET" == "raw_30s" && "$DATA_TYPE" == "melspecs" ]]; then
  REQUIRED_GB=320
elif [[ "$DATASET" == "autotagging_moodtheme" && "$DATA_TYPE" == "audio" ]]; then
  REQUIRED_GB=220
elif [[ "$DATASET" == "autotagging_moodtheme" && "$DATA_TYPE" == "audio-low" ]]; then
  REQUIRED_GB=90
elif [[ "$DATASET" == "autotagging_moodtheme" && "$DATA_TYPE" == "melspecs" ]]; then
  REQUIRED_GB=120
fi
REQUIRED_KB="$((REQUIRED_GB * 1024 * 1024))"

echo "Host: $HOST_SHORT"
echo "Target directory: $TARGET_DIR"
echo "Dataset selection: dataset=$DATASET type=$DATA_TYPE source=$SOURCE"
echo "Free space: $((AVAILABLE_KB / 1024 / 1024)) GiB"
echo "Recommended free space floor: ${REQUIRED_GB} GiB"

if (( AVAILABLE_KB < REQUIRED_KB )); then
  cat >&2 <<EOF
Error: available space appears below the recommended floor for this selection.
Please free space or choose another target before continuing.
EOF
  exit 1
fi

if (( ASSUME_YES == 0 )); then
  cat <<'EOF'
License reminder:
- MTG-Jamendo metadata is CC BY-NC-SA 4.0.
- Audio is under Creative Commons licenses listed per track.
- Dataset is for non-commercial research and academic use.
EOF
  read -r -p "Proceed with download? [y/N]: " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Updating existing mtg-jamendo-dataset clone..."
  git -C "$REPO_DIR" pull --ff-only
else
  echo "Cloning mtg-jamendo-dataset into $REPO_DIR ..."
  git clone https://github.com/MTG/mtg-jamendo-dataset.git "$REPO_DIR"
fi

VENV_DIR="$REPO_DIR/.venv-download"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$REPO_DIR/scripts/requirements.txt"

mkdir -p "$TARGET_DIR/logs"
LOG_FILE="$TARGET_DIR/logs/mtg_download_$(date +%Y%m%d_%H%M%S).log"

echo "Starting download. Log: $LOG_FILE"
set -o pipefail
python3 "$REPO_DIR/scripts/download/download.py" \
  --dataset "$DATASET" \
  --type "$DATA_TYPE" \
  --from "$SOURCE" \
  "$TARGET_DIR" \
  --unpack \
  --remove | tee "$LOG_FILE"

echo
echo "Download command completed."
echo "If interrupted earlier, rerun this script with the same options to resume."
