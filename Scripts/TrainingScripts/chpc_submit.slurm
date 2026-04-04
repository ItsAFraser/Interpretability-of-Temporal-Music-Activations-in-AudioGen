#!/bin/bash

#=======================================================
# Slurm Settings for CHPC Granite (CS6966 Class)
#=======================================================
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --gres=gpu:rtxpr6000bl:1  # Option: rtxpr6000bl or rtxpr4000bl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=TrainSAE
#SBATCH --output=TrainSAE_%j.out
#SBATCH --error=TrainSAE_%j.err

#=======================================================
# Environment Setup (Based on CHPC Instructions)
#=======================================================
echo "Loading modern CUDA environment..."
module purge
module load cuda/12.6.3
module list

echo "Activating Conda Environment..."
# Adjust the miniconda path to match your installation
source ~/miniconda3/etc/profile.d/conda.sh
# Assuming you will create this environment on CHPC
conda activate temporal-music-activations-cuda

#=======================================================
# Directory Setup & Execution
#=======================================================
# Go to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

# Defaults target CHPC VAST scratch for the current user.
CHPC_UID="${CHPC_UID:-$USER}"
CHPC_SCRATCH_BASE="${CHPC_SCRATCH_BASE:-/scratch/general/vast/${CHPC_UID}}"

SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-${CHPC_SCRATCH_BASE}/sae_output}"
mkdir -p "$SCRATCH_OUTPUT_DIR"

echo "Starting SAE Training on CHPC!"
# Using the recently pulled HPC training script
# Note: You may need to replace 'Data/Models/features/layer_final' 
# with your actual data directory on CHPC if it is located elsewhere.
bash Scripts/TrainingScripts/run_train_sae_hpc.sh "Data/Models/features/layer_final" "$SCRATCH_OUTPUT_DIR"

echo "Job finished."
