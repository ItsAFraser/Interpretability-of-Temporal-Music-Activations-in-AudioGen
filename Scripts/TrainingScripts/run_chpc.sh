#!/bin/bash
# CHPC Run Script for GPU Assignment 01
# Run this AFTER getting an interactive GPU session
#
# Usage: ./run_chpc.sh [q1|q2|all]
#   q1  - Run only Question 1 (CUDA kernels)
#   q2  - Run only Question 2 (Python benchmarks)
#   all - Run both (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR=~/gpu-assignment-1-venv
RUN_Q1=false
RUN_Q2=false

# Parse arguments
case "${1:-all}" in
    q1)
        RUN_Q1=true
        ;;
    q2)
        RUN_Q2=true
        ;;
    all|"")
        RUN_Q1=true
        RUN_Q2=true
        ;;
    *)
        echo "Usage: $0 [q1|q2|all]"
        exit 1
        ;;
esac

echo "=================================================="
echo "GPU Assignment 01 - CHPC Runner"
echo "=================================================="
echo ""

# Check we're on a GPU node
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Run ./run_interactive.sh first to get a GPU node."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Load required modules
echo "Loading modules..."
module load cuda/13.1
module load python3

# ==========================================
# Question 1: CUDA Kernels
# ==========================================
if $RUN_Q1; then
    echo ""
    echo "=================================================="
    echo "Question 1: Array Shift (CUDA)"
    echo "=================================================="
    echo ""

    cd "$SCRIPT_DIR/question1"

    echo "Building CUDA kernels..."
    make clean && make all
    echo ""

    echo "Running all parts..."
    ./run_all.sh

    cd "$SCRIPT_DIR"
fi

# ==========================================
# Question 2: Python Matrix Multiplication
# ==========================================
if $RUN_Q2; then
    echo ""
    echo "=================================================="
    echo "Question 2: Matrix Multiplication (Python)"
    echo "=================================================="
    echo ""

    # Activate virtual environment
    if [ -d "$VENV_DIR" ]; then
        echo "Activating Python virtual environment..."
        source $VENV_DIR/bin/activate
    else
        echo "ERROR: Virtual environment not found at $VENV_DIR"
        echo "Run ./setup_chpc.sh first to create it."
        exit 1
    fi

    cd "$SCRIPT_DIR/question2"

    echo "Running benchmarks..."
    echo "(First run may take longer due to autotuning)"
    echo ""

    python run_all_benchmarks.py --sizes 256,512,1024,2048,4096

    cd "$SCRIPT_DIR"
fi

echo ""
echo "=================================================="
echo "Done!"
echo "=================================================="
