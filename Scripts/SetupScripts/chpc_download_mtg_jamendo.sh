#!/usr/bin/env bash
set -euo pipefail

# CHPC-focused MTG-Jamendo downloader.
#
# Modes:
#   1) full_subset    - existing MTG bulk download via the official downloader
#   2) sampled_tracks - select a diverse subset of full-length tracks from MTG
#                       metadata and download them one by one via the Jamendo API
#
# This script is intended for use on CHPC Data Transfer Nodes (dtn05-08)
# and writes data to VAST scratch by default.

print_help() {
  cat <<'EOF'
Usage:
  Scripts/SetupScripts/chpc_download_mtg_jamendo.sh [options]

Modes:
  --mode <name>           Download mode (default: full_subset)
                          Choices: full_subset, sampled_tracks

Shared options:
  --dataset <name>        Dataset subset / metadata source (default: raw_30s)
                          Choices: raw_30s, autotagging_moodtheme
  --type <name>           Payload type (default: audio)
                          full_subset choices: audio, audio-low, melspecs, acousticbrainz
                          sampled_tracks choices: audio, audio-low
  --source <name>         Download source for full_subset mode (default: mtg-fast)
                          Choices: mtg-fast, mtg
  --target-dir <path>     Final output directory
                          (default: /scratch/general/vast/$USER/mtg-jamendo)
  --repo-dir <path>       Local clone location for mtg-jamendo-dataset repo
                          (default: <target-dir>/mtg-jamendo-dataset)
  --yes                   Skip interactive confirmation prompt

Sampled-track mode options:
  --max-tracks <n>        Number of tracks to download (default: 1000)
  --min-duration-sec <n>  Minimum duration filter for candidate tracks
                          (default: 0, meaning no extra duration filter)
  --max-tracks-per-artist Maximum tracks per artist before relaxing the cap (default: 2)
  --seed <n>              RNG seed for reproducible selection (default: 42)
  --sample-name <name>    Output folder name under target-dir
                          (default: full_length_diverse_<max_tracks>)
  --jamendo-client-id <id>
                          Jamendo API client_id required for sampled_tracks mode
                          Can also be passed via JAMENDO_CLIENT_ID env var
  --overwrite             Re-download sampled tracks even if output files exist

Notes:
  - full_subset mode uses the official MTG downloader and preserves the existing behavior.
  - sampled_tracks mode downloads individual full-length tracks via the Jamendo API.
    For this mode, --type audio maps to Jamendo's mp32 format, and --type audio-low
    maps to mp31. This is not the same packaging as MTG's bulk TAR archives.

Examples:
  # Existing full-subset bulk download
  Scripts/SetupScripts/chpc_download_mtg_jamendo.sh \
    --mode full_subset \
    --dataset raw_30s \
    --type audio \
    --source mtg-fast \
    --target-dir /scratch/general/vast/$USER/mtg-jamendo \
    --yes

  # Sample 1000 diverse full-length tracks from raw_30s metadata
  Scripts/SetupScripts/chpc_download_mtg_jamendo.sh \
    --mode sampled_tracks \
    --dataset raw_30s \
    --type audio \
    --max-tracks 1000 \
    --max-tracks-per-artist 2 \
    --jamendo-client-id YOUR_CLIENT_ID \
    --target-dir /scratch/general/vast/$USER/mtg-jamendo \
    --yes
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

MODE="full_subset"
DATASET="raw_30s"
DATA_TYPE="audio"
SOURCE="mtg-fast"
TARGET_DIR="/scratch/general/vast/${USER}/mtg-jamendo"
REPO_DIR=""
ASSUME_YES=0
MAX_TRACKS=1000
MIN_DURATION_SEC=0
MAX_TRACKS_PER_ARTIST=2
SEED=42
SAMPLE_NAME=""
JAMENDO_CLIENT_ID="${JAMENDO_CLIENT_ID:-}"
OVERWRITE_EXISTING=0

require_value() {
  local option_name="$1"
  local option_value="${2:-}"
  if [[ -z "$option_value" || "$option_value" == --* ]]; then
    echo "Error: $option_name requires a value." >&2
    print_help >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      require_value "$1" "${2:-}"
      MODE="$2"
      shift 2
      ;;
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="$2"
      shift 2
      ;;
    --type)
      require_value "$1" "${2:-}"
      DATA_TYPE="$2"
      shift 2
      ;;
    --source)
      require_value "$1" "${2:-}"
      SOURCE="$2"
      shift 2
      ;;
    --target-dir)
      require_value "$1" "${2:-}"
      TARGET_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      require_value "$1" "${2:-}"
      REPO_DIR="$2"
      shift 2
      ;;
    --max-tracks)
      require_value "$1" "${2:-}"
      MAX_TRACKS="$2"
      shift 2
      ;;
    --min-duration-sec)
      require_value "$1" "${2:-}"
      MIN_DURATION_SEC="$2"
      shift 2
      ;;
    --max-tracks-per-artist)
      require_value "$1" "${2:-}"
      MAX_TRACKS_PER_ARTIST="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --sample-name)
      require_value "$1" "${2:-}"
      SAMPLE_NAME="$2"
      shift 2
      ;;
    --jamendo-client-id)
      require_value "$1" "${2:-}"
      JAMENDO_CLIENT_ID="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE_EXISTING=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help >&2
      exit 1
      ;;
  esac
done

case "$MODE" in
  full_subset|sampled_tracks) ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

case "$DATASET" in
  raw_30s|autotagging_moodtheme) ;;
  *)
    echo "Invalid --dataset: $DATASET" >&2
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

if [[ "$MODE" == "full_subset" ]]; then
  case "$DATA_TYPE" in
    audio|audio-low|melspecs|acousticbrainz) ;;
    *)
      echo "Invalid --type for full_subset: $DATA_TYPE" >&2
      exit 1
      ;;
  esac
else
  case "$DATA_TYPE" in
    audio|audio-low) ;;
    *)
      echo "Invalid --type for sampled_tracks: $DATA_TYPE" >&2
      echo "Use --type audio or --type audio-low when downloading individual tracks." >&2
      exit 1
      ;;
  esac
fi

if ! [[ "$MAX_TRACKS" =~ ^[0-9]+$ ]] || [[ "$MAX_TRACKS" -le 0 ]]; then
  echo "Error: --max-tracks must be a positive integer, got '$MAX_TRACKS'" >&2
  exit 1
fi

if ! [[ "$MIN_DURATION_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: --min-duration-sec must be numeric, got '$MIN_DURATION_SEC'" >&2
  exit 1
fi

if ! [[ "$MAX_TRACKS_PER_ARTIST" =~ ^[0-9]+$ ]] || [[ "$MAX_TRACKS_PER_ARTIST" -le 0 ]]; then
  echo "Error: --max-tracks-per-artist must be a positive integer, got '$MAX_TRACKS_PER_ARTIST'" >&2
  exit 1
fi

if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "Error: --seed must be an integer, got '$SEED'" >&2
  exit 1
fi

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$TARGET_DIR/mtg-jamendo-dataset"
fi

if [[ "$MODE" == "sampled_tracks" ]]; then
  if [[ -z "$JAMENDO_CLIENT_ID" ]]; then
    echo "Error: sampled_tracks mode requires --jamendo-client-id or JAMENDO_CLIENT_ID." >&2
    exit 1
  fi
  if [[ -z "$SAMPLE_NAME" ]]; then
    SAMPLE_NAME="full_length_diverse_${MAX_TRACKS}"
  fi
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
if [[ "$MODE" == "full_subset" ]]; then
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
else
  if [[ "$DATA_TYPE" == "audio" ]]; then
    REQUIRED_GB=25
  else
    REQUIRED_GB=10
  fi
fi
REQUIRED_KB="$((REQUIRED_GB * 1024 * 1024))"

echo "Host: $HOST_SHORT"
echo "Mode: $MODE"
echo "Target directory: $TARGET_DIR"
echo "Dataset selection: dataset=$DATASET type=$DATA_TYPE source=$SOURCE"
echo "Free space: $((AVAILABLE_KB / 1024 / 1024)) GiB"
echo "Recommended free space floor: ${REQUIRED_GB} GiB"
if [[ "$MODE" == "sampled_tracks" ]]; then
  echo "Max tracks: $MAX_TRACKS"
  echo "Min duration filter: ${MIN_DURATION_SEC}s (0 means no extra filter)"
  echo "Max tracks per artist: $MAX_TRACKS_PER_ARTIST"
  echo "Seed: $SEED"
  echo "Sample name: $SAMPLE_NAME"
fi

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

prepare_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Updating existing mtg-jamendo-dataset clone..."
    git -C "$REPO_DIR" pull --ff-only
  else
    echo "Cloning mtg-jamendo-dataset into $REPO_DIR ..."
    git clone https://github.com/MTG/mtg-jamendo-dataset.git "$REPO_DIR"
  fi
}

install_bulk_downloader_env() {
  VENV_DIR="$REPO_DIR/.venv-download"
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "$REPO_DIR/scripts/requirements.txt"
}

run_full_subset_download() {
  mkdir -p "$TARGET_DIR/logs"
  LOG_FILE="$TARGET_DIR/logs/mtg_download_$(date +%Y%m%d_%H%M%S).log"

  echo "Starting bulk download. Log: $LOG_FILE"
  echo "Bulk output root: $TARGET_DIR"
  set -o pipefail
  python3 "$REPO_DIR/scripts/download/download.py" \
    --dataset "$DATASET" \
    --type "$DATA_TYPE" \
    --from "$SOURCE" \
    "$TARGET_DIR" \
    --unpack \
    --remove | tee "$LOG_FILE"

  echo
  echo "Bulk download command completed."
  echo "If interrupted earlier, rerun this script with the same options to resume."
}

select_sampled_tracks() {
  local manifest_path="$1"
  local summary_path="$2"
  python3 - "$REPO_DIR" "$DATASET" "$MAX_TRACKS" "$MIN_DURATION_SEC" "$MAX_TRACKS_PER_ARTIST" "$SEED" "$manifest_path" "$summary_path" <<'PY'
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

repo_dir = Path(sys.argv[1]).resolve()
dataset_name = sys.argv[2]
max_tracks = int(sys.argv[3])
min_duration = float(sys.argv[4])
max_tracks_per_artist = int(sys.argv[5])
seed = int(sys.argv[6])
manifest_path = Path(sys.argv[7]).resolve()
summary_path = Path(sys.argv[8]).resolve()

sys.path.insert(0, str(repo_dir / "scripts"))
import commons  # type: ignore

metadata_file = repo_dir / "data" / f"{dataset_name}.tsv"
if not metadata_file.exists():
    raise SystemExit(f"Metadata file not found: {metadata_file}")

tracks, _, _ = commons.read_file(str(metadata_file))
rng = random.Random(seed)

tag_frequency = Counter()
candidates = []
for track_id, track in tracks.items():
    duration = float(track.get("duration", 0.0))
    if duration < min_duration:
        continue

    rel_path = str(track.get("path", "")).strip()
    if not rel_path:
        continue

    genres = set(track.get("genre", set()))
    moods = set(track.get("mood/theme", set()))
    instruments = set(track.get("instrument", set()))
    all_tags = set(track.get("tags", []))
    for tag in all_tags:
        tag_frequency[tag] += 1

    candidates.append(
        {
            "track_id": int(track_id),
            "artist_id": int(track["artist_id"]),
            "album_id": int(track["album_id"]),
            "path": rel_path,
            "duration": duration,
            "genres": genres,
            "moods": moods,
            "instruments": instruments,
            "all_tags": all_tags,
        }
    )

if not candidates:
    raise SystemExit("No metadata candidates matched the requested sampling constraints.")

rng.shuffle(candidates)
for item in candidates:
    item["rare_weight"] = sum(1.0 / tag_frequency[tag] for tag in item["all_tags"] if tag_frequency[tag] > 0)

selected = []
selected_ids = set()
artist_counts = defaultdict(int)
seen_genres = set()
seen_moods = set()
seen_instruments = set()
artist_cap = max_tracks_per_artist

while len(selected) < max_tracks:
    best_index = None
    best_score = None

    for index, item in enumerate(candidates):
        if item["track_id"] in selected_ids:
            continue
        if artist_counts[item["artist_id"]] >= artist_cap:
            continue

        new_genres = item["genres"] - seen_genres
        new_moods = item["moods"] - seen_moods
        new_instruments = item["instruments"] - seen_instruments

        score = 0.0
        score += 9.0 * len(new_genres)
        score += 7.0 * len(new_moods)
        score += 5.0 * len(new_instruments)
        score += 1.5 * item["rare_weight"]
        score += 0.25 * len(item["all_tags"])
        score += 0.05 * math.log1p(item["duration"])
        score += rng.random() * 1e-4

        if best_score is None or score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
      if artist_cap >= max_tracks:
          break
      artist_cap += 1
      continue

    best_item = candidates[best_index]
    selected.append(best_item)
    selected_ids.add(best_item["track_id"])
    artist_counts[best_item["artist_id"]] += 1
    seen_genres.update(best_item["genres"])
    seen_moods.update(best_item["moods"])
    seen_instruments.update(best_item["instruments"])

if len(selected) < max_tracks:
    print(
        f"[warn] Selected {len(selected)} tracks, fewer than requested {max_tracks}. "
        f"Final artist cap relaxation reached {artist_cap}.",
        flush=True,
    )

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "track_id",
        "artist_id",
        "album_id",
        "path",
        "duration",
        "genre",
        "mood_theme",
        "instrument",
        "tags",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for item in selected:
        writer.writerow(
            {
                "track_id": item["track_id"],
                "artist_id": item["artist_id"],
                "album_id": item["album_id"],
                "path": item["path"],
                "duration": f"{item['duration']:.3f}",
                "genre": "|".join(sorted(item["genres"])),
                "mood_theme": "|".join(sorted(item["moods"])),
                "instrument": "|".join(sorted(item["instruments"])),
                "tags": "|".join(sorted(item["all_tags"])),
            }
        )

summary = {
    "dataset": dataset_name,
    "requested_tracks": max_tracks,
    "selected_tracks": len(selected),
    "min_duration_sec": min_duration,
    "max_tracks_per_artist_initial": max_tracks_per_artist,
    "artist_cap_final": artist_cap,
    "seed": seed,
    "unique_artists": len({item["artist_id"] for item in selected}),
    "unique_genres": len({tag for item in selected for tag in item["genres"]}),
    "unique_moodthemes": len({tag for item in selected for tag in item["moods"]}),
    "unique_instruments": len({tag for item in selected for tag in item["instruments"]}),
    "manifest_path": str(manifest_path),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

print(f"Selected {len(selected)} diverse tracks from {metadata_file.name}", flush=True)
print(
    "Coverage: "
    f"{summary['unique_artists']} artists, "
    f"{summary['unique_genres']} genres, "
    f"{summary['unique_moodthemes']} mood/theme tags, "
    f"{summary['unique_instruments']} instruments",
    flush=True,
)
print(f"Manifest: {manifest_path}", flush=True)
print(f"Summary:  {summary_path}", flush=True)
PY
}

download_sampled_tracks() {
  local manifest_path="$1"
  local output_dir="$2"
  local failed_path="$3"
  local api_audio_format="$4"
  python3 - "$manifest_path" "$output_dir" "$failed_path" "$JAMENDO_CLIENT_ID" "$api_audio_format" "$OVERWRITE_EXISTING" <<'PY'
import csv
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
failed_path = Path(sys.argv[3]).resolve()
client_id = sys.argv[4]
audio_format = sys.argv[5]
overwrite_existing = bool(int(sys.argv[6]))

output_dir.mkdir(parents=True, exist_ok=True)
failed_path.parent.mkdir(parents=True, exist_ok=True)

base_url = "https://api.jamendo.com/v3.0/tracks/file/"
request_headers = {"User-Agent": "Interpretability-of-Temporal-Music-Activations-in-AudioGen/1.0"}

downloaded = 0
skipped = 0
failed_rows = []
start_time = time.time()

with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)

total_rows = len(rows)
for index, row in enumerate(rows, start=1):
    track_id = row["track_id"]
    rel_path = row["path"].strip() or f"{track_id}.mp3"
    out_path = output_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite_existing:
        skipped += 1
        print(f"[skip {index}/{total_rows}] track={track_id} exists -> {out_path}", flush=True)
        continue

    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "id": track_id,
            "action": "download",
            "audioformat": audio_format,
        }
    )
    url = f"{base_url}?{query}"
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    try:
        request = urllib.request.Request(url, headers=request_headers)
        with urllib.request.urlopen(request, timeout=180) as response, tmp_path.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        os.replace(tmp_path, out_path)
        downloaded += 1
        print(f"[ok   {index}/{total_rows}] track={track_id} -> {out_path}", flush=True)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        failed_rows.append(
            {
                "track_id": track_id,
                "artist_id": row.get("artist_id", ""),
                "path": rel_path,
                "error": str(exc),
            }
        )
        print(f"[fail {index}/{total_rows}] track={track_id}: {exc}", flush=True)

if failed_rows:
    with failed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "artist_id", "path", "error"], delimiter="\t")
        writer.writeheader()
        writer.writerows(failed_rows)
else:
    failed_path.unlink(missing_ok=True)

elapsed = time.time() - start_time
print(
    f"Finished sampled download in {elapsed / 60:.1f} min: "
    f"{downloaded} downloaded, {skipped} skipped, {len(failed_rows)} failed.",
    flush=True,
)
if failed_rows:
    print(f"Failed manifest: {failed_path}", flush=True)
PY
}

run_sampled_track_download() {
  local sample_dir="$TARGET_DIR/$SAMPLE_NAME"
  local manifest_dir="$sample_dir/manifests"
  local audio_dir="$sample_dir/audio"
  local log_dir="$sample_dir/logs"
  local selected_tracks_path="$manifest_dir/selected_tracks.tsv"
  local summary_path="$manifest_dir/selection_summary.json"
  local failed_tracks_path="$manifest_dir/failed_tracks.tsv"
  local log_path="$log_dir/mtg_sampled_download_$(date +%Y%m%d_%H%M%S).log"
  local api_audio_format="mp32"

  if [[ "$DATA_TYPE" == "audio-low" ]]; then
    api_audio_format="mp31"
  fi

  mkdir -p "$manifest_dir" "$audio_dir" "$log_dir"

  echo "Preparing diverse sampled full-length download under: $sample_dir"
  echo "Sampled audio output: $audio_dir"
  echo "Selection manifest:   $selected_tracks_path"
  echo "Selection summary:    $summary_path"
  echo "Download log:         $log_path"
  echo "Jamendo API format:   $api_audio_format"

  select_sampled_tracks "$selected_tracks_path" "$summary_path"

  set -o pipefail
  download_sampled_tracks "$selected_tracks_path" "$audio_dir" "$failed_tracks_path" "$api_audio_format" | tee "$log_path"

  echo
  echo "Sampled full-length download completed."
  echo "Audio directory: $audio_dir"
  echo "Manifest:        $selected_tracks_path"
  echo "Summary:         $summary_path"
  if [[ -f "$failed_tracks_path" ]]; then
    echo "Failed tracks:   $failed_tracks_path"
  fi
}

prepare_repo

if [[ "$MODE" == "full_subset" ]]; then
  install_bulk_downloader_env
  run_full_subset_download
else
  run_sampled_track_download
fi
