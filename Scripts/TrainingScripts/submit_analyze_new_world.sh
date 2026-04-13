#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --job-name=analyze-new-world
#SBATCH --partition=notchpeak-freecycle
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/analyze_%j.log
#SBATCH --error=/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/analyze_%j.err

set -e

FEATURES_DIR="/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/features"
ANALYSIS_OUT="/scratch/general/vast/u1406806/sae_output/songs-of-interest/New-World-Symphony-Output/analysis"
PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Interpretability-of-Temporal-Music-Activations-in-AudioGen"
ANALYSIS_SCRIPT="$PROJECT_ROOT/Analysis/temporal_feature_analysis.py"

mkdir -p "$ANALYSIS_OUT"

echo "========================================="
echo "New World Symphony - SAE Analysis"
echo "========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Features dir:  $FEATURES_DIR"
echo "Output dir:    $ANALYSIS_OUT"
echo "Start:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

if [[ ! -d "$FEATURES_DIR" ]]; then
    echo "ERROR: Features directory not found: $FEATURES_DIR"
    echo "Run submit_extract_new_world.sh first."
    exit 1
fi

echo "Loading conda environment..."
module load miniconda3
echo ""

echo "Running temporal feature analysis..."
cd "$PROJECT_ROOT"
conda run -n temporal-music-activations-cuda python "$ANALYSIS_SCRIPT" \
    --features_root "$FEATURES_DIR" \
    --output_dir    "$ANALYSIS_OUT" \
    --max_tracks    0 \
    2>&1 | tee -a "$ANALYSIS_OUT/analysis.log"

echo ""
echo "========================================="
echo "Analysis complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
ls -lh "$ANALYSIS_OUT/"*.png 2>/dev/null || echo "No PNG files found."
