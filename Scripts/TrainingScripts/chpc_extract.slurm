#!/bin/bash

#=======================================================
# Slurm Settings for Feature Extraction (MusicGen)
#=======================================================
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=ExtractFeatures
#SBATCH --output=ExtractFeatures_%j.out
#SBATCH --error=ExtractFeatures_%j.err

#=======================================================
# Environment Setup
#=======================================================
echo "Loading modern CUDA environment..."
module purge
module load gcc
module load cuda/12.6.3


echo "Activating Conda Environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate temporal-music-activations-cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

#=======================================================
# Directory Setup & Execution
#=======================================================
cd $SLURM_SUBMIT_DIR

# Defaults target CHPC VAST scratch for the current user.
CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"

RAW_AUDIO_DIR="${RAW_AUDIO_DIR:-${CHPC_SCRATCH_BASE}/mtg-jamendo/raw_30s/audio}"
SCRATCH_FEATURES_DIR="${SCRATCH_FEATURES_DIR:-${CHPC_SCRATCH_BASE}/sae_output/features}"
mkdir -p "$SCRATCH_FEATURES_DIR"

echo "Starting Feature Extraction on CHPC..."

# Run the repository extraction launcher.
bash Scripts/TrainingScripts/run_extract_musicgen_features.sh \
    "$RAW_AUDIO_DIR" \
    "$SCRATCH_FEATURES_DIR" \
    --max_files 16 \
    --max_duration_sec 30 \
    --decoder_layers 0,8,16,-1 \
    --metadata_json

echo "Extraction finished."
