#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --job-name=temporal-feature-analysis
#SBATCH --partition=notchpeak-freecycle
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/general/vast/u1406806/sae_output/analysis-output/job_%j.log
#SBATCH --error=/scratch/general/vast/u1406806/sae_output/analysis-output/job_%j.err

set -e

echo "========================================="
echo "Temporal Feature Analysis - CHPC Job"
echo "========================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $(hostname)"
echo "CPUs allocated:   $SLURM_CPUS_PER_TASK"
echo "Memory:           $SLURM_MEM_PER_NODE MB"
echo "Start time:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Interpretability-of-Temporal-Music-Activations-in-AudioGen"
ANALYSIS_SCRIPT="$PROJECT_ROOT/Analysis/temporal_feature_analysis.py"
OUTPUT_DIR="/scratch/general/vast/u1406806/sae_output/analysis-output"

mkdir -p "$OUTPUT_DIR"

echo "Project root:     $PROJECT_ROOT"
echo "Analysis script:  $ANALYSIS_SCRIPT"
echo "Output dir:       $OUTPUT_DIR"
echo ""

if [[ ! -f "$ANALYSIS_SCRIPT" ]]; then
    echo "ERROR: Analysis script not found at $ANALYSIS_SCRIPT"
    exit 1
fi

echo "Loading conda environment..."
module load miniconda3
echo "Environment ready."
echo ""

echo "Starting temporal feature analysis..."
cd "$PROJECT_ROOT"
conda run -n temporal-music-activations-cuda python "$ANALYSIS_SCRIPT" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"

EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "========================================="
echo "Temporal Feature Analysis - Job Complete"
echo "========================================="
echo "Exit code:        $EXIT_CODE"
echo "End time:         $END_TIME"
echo "Output location:  $OUTPUT_DIR"
echo "========================================="

if [[ -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "Generated outputs:"
    ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  (No PNG files yet)"
fi

exit $EXIT_CODE
