#!/bin/bash

# Wrapper script to run instantngp_zapbench_temporal.py with correct environment
# Usage: ./run_temporal.sh [start_frame] [n_iterations]

set -e  # Exit on any error

# Set correct directory
SCRIPT_DIR="/groups/saalfeld/home/allierc/Py/NeuralGraph/src/NeuralGraph/instantNGP"
CONDA_ENV="neural-graph-linux"

# Default parameters
START_FRAME=${1:-0}  # Default to frame 0 if not provided
N_ITERATIONS=${2:-1000}  # Default to 1000 iterations if not provided

echo "=========================================="
echo "InstantNGP Temporal ZapBench Runner"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Conda environment: $CONDA_ENV"
echo "Start frame: $START_FRAME"
echo "Iterations: $N_ITERATIONS"
echo "=========================================="

# Check if directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Directory $SCRIPT_DIR does not exist!"
    exit 1
fi

# Change to correct directory
cd "$SCRIPT_DIR"
echo "Changed to directory: $(pwd)"

# Check if script exists
if [ ! -f "instantngp_zapbench_temporal.py" ]; then
    echo "ERROR: Script instantngp_zapbench_temporal.py not found in $(pwd)"
    exit 1
fi

# Check if config exists
if [ ! -f "config_hash_4d.json" ]; then
    echo "ERROR: Config file config_hash_4d.json not found in $(pwd)"
    exit 1
fi

# Activate conda environment and run script
echo "Activating conda environment: $CONDA_ENV"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Verify we're in the right environment
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$CURRENT_ENV" != "$CONDA_ENV" ]; then
    echo "ERROR: Failed to activate $CONDA_ENV environment. Currently in: $CURRENT_ENV"
    exit 1
fi

echo "Successfully activated environment: $CURRENT_ENV"
echo "Python path: $(which python)"

# Run the script
echo "Running temporal script..."
echo "Command: python instantngp_zapbench_temporal.py config_hash_4d.json $START_FRAME $N_ITERATIONS"
echo "=========================================="

python instantngp_zapbench_temporal.py config_hash_4d.json "$START_FRAME" "$N_ITERATIONS"

echo "=========================================="
echo "Script completed successfully!"
echo "=========================================="