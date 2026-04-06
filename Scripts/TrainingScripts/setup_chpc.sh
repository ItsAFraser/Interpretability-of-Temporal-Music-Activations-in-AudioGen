#!/bin/bash
# CHPC Setup Script for GPU, example from another class
# Run this ONCE to set up the Python environment
#
# Usage: ./setup_chpc.sh

set -e

echo "=================================================="
echo "CHPC Setup for GPU Assignment 01"
echo "=================================================="
echo ""

# Create venv directory
VENV_DIR=~/gpu-assignment-1-venv

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, run: rm -rf $VENV_DIR && ./setup_chpc.sh"
    exit 0
fi

# Load required modules
echo "Loading modules..."
module load cuda/13.1
module load python3

# Create virtual environment
echo "Creating Python virtual environment at $VENV_DIR..."
python -m venv $VENV_DIR

# Activate and install packages
echo "Activating environment and installing packages..."
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core packages first
# Note: Need PyTorch NIGHTLY for sm_120 (Blackwell Max-Q) support
# Stable PyTorch only supports up to sm_90
echo "Installing PyTorch NIGHTLY with CUDA 12.8 support (required for sm_120)..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install "numpy<2.0"  # Some packages require numpy 1.x

echo "Installing Triton (nightly for sm_120 support)..."
pip install --pre triton --index-url https://download.pytorch.org/whl/nightly/cu128 || \
pip install "triton>=2.1" || echo "Warning: triton install failed"

echo "Installing utilities..."
pip install tabulate pandas


echo -n "  helion: "
pip install helion 2>/dev/null && echo "OK" || echo "FAILED (optional)"

echo -n "  cuda-tile: "
pip install cuda-tile 2>/dev/null && echo "OK" || echo "FAILED (optional)"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To run the assignment:"
echo "  1. Get interactive GPU session: ./run_interactive.sh"
echo "  2. Once on GPU node, run: ./run_chpc.sh"
echo ""
