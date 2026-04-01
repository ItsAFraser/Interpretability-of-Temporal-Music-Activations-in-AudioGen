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
echo "Loading CUDA 11.8.0..."
module purge
module load cuda/11.8.0

echo "Activating Conda Environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate temporal-music-activations-cuda

#=======================================================
# Directory Setup & Execution
#=======================================================
cd $SLURM_SUBMIT_DIR

# 将原始生肉数据和提出来的特征全部放进大容量专用的 scratch 区
RAW_AUDIO_DIR="/scratch/general/vast/u1499410/raw_30s_audio_low"
SCRATCH_FEATURES_DIR="/scratch/general/vast/u1499410/sae_output/features"
mkdir -p "$SCRATCH_FEATURES_DIR"

echo "Starting Feature Extraction on CHPC..."

# 运行库中自带的提取脚本
bash Scripts/TrainingScripts/run_extract_musicgen_features.sh \
    "$RAW_AUDIO_DIR" \
    "$SCRATCH_FEATURES_DIR" \
    --max_files 16 \
    --max_duration_sec 30 \
    --decoder_layers 0,8,16,-1 \
    --metadata_json

echo "Extraction finished."
