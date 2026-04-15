#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --job-name=render-new-world-sync
#SBATCH --partition=notchpeak-freecycle
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/video_%j.log
#SBATCH --error=/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/video_%j.err

set -euo pipefail

FEATURES_DIR="/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/features"
OUTPUT_DIR="/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/video"
AUDIO_PATH="/scratch/general/vast/u1406806/sae_output/songs-of-interest/Antonin_Dvorak_-_symphony_no._9_in_e_minor_'from_the_new_world',_op._95_-_i._adagio_-_allegro_molto.ogg.mp3"
PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Interpretability-of-Temporal-Music-Activations-in-AudioGen"
RENDER_SCRIPT="$PROJECT_ROOT/Analysis/render_layer_video.py"
OUTPUT_PATH="$OUTPUT_DIR/new_world_synchronized_layers_playhead.mp4"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "New World Symphony - Synchronized Layer Video Render"
echo "========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Features dir:  $FEATURES_DIR"
echo "Output path:   $OUTPUT_PATH"
echo "Audio path:    $AUDIO_PATH"
echo "Start:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

if [[ ! -d "$FEATURES_DIR" ]]; then
    echo "ERROR: Features directory not found: $FEATURES_DIR"
    exit 1
fi

if [[ ! -f "$AUDIO_PATH" ]]; then
    echo "WARNING: Audio file not found. Rendering silent video."
fi

module load miniconda3
cd "$PROJECT_ROOT"

conda run -n temporal-music-activations-cuda python "$RENDER_SCRIPT" \
    --features_root "$FEATURES_DIR" \
    --output_path "$OUTPUT_PATH" \
    --audio_path "$AUDIO_PATH" \
    --track_index 0 \
    --video_bins 14400 \
    --fps 24 \
    --dpi 120 \
    --device cpu \
    --include_audio

echo ""
echo "========================================="
echo "Video render complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
ls -lh "$OUTPUT_DIR"