#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=extract-new-world
#SBATCH --partition=soc-gpu-np
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/extract_%j.log
#SBATCH --error=/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/extract_%j.err

set -euo pipefail

SONG_DIR="/scratch/general/vast/u1406806/sae_output/songs-of-interest"
FEATURES_OUT="$SONG_DIR/New-World-Symphony-Output/features"
PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Interpretability-of-Temporal-Music-Activations-in-AudioGen"

mkdir -p "$FEATURES_OUT"

echo "========================================="
echo "New World Symphony - Feature Extraction"
echo "========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Input dir:  $SONG_DIR"
echo "Output dir: $FEATURES_OUT"
echo "Start:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo "Loading conda environment..."
module load miniconda3

echo "Extracting MusicGen features for all 24 decoder layers..."
cd "$PROJECT_ROOT"
conda run -n temporal-music-activations-cuda python \
    Temporal-Music-Activations/ExtractMusicGenFeatures.py \
    --input_dir "$SONG_DIR" \
    --output_dir "$FEATURES_OUT" \
    --decoder_layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,-1" \
    --device auto 2>&1 | tee -a "$FEATURES_OUT/../extract.log"

echo ""
echo "========================================="
echo "Extraction complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
ls -lh "$FEATURES_OUT/"
