#!/bin/bash
#
# Cluster training script for GNN_LLM code modifications
# Called via: ssh login1 'bsub -n 8 -gpu "num=1" -q gpu_h100 "Graph/NeuralGraph/cluster/training_cluster.sh"'
#
# This script reads parameters from a job file and runs training on the cluster

set -e  # Exit on error

# Configuration - adjust these paths for your environment
NEURALGRAPH_DIR="/groups/bhatti/bhattilabb/allierc/Graph/NeuralGraph"
CONDA_ENV="pyg"
RESULTS_DIR="${NEURALGRAPH_DIR}/cluster/results"

# Job file contains the training parameters (written by GNN_LLM.py before submission)
JOB_FILE="${NEURALGRAPH_DIR}/cluster/current_job.txt"

# Check job file exists
if [ ! -f "$JOB_FILE" ]; then
    echo "Error: Job file not found: $JOB_FILE"
    exit 1
fi

# Read parameters from job file
CONFIG_PATH=$(grep "^CONFIG_PATH=" "$JOB_FILE" | cut -d'=' -f2-)
DEVICE=$(grep "^DEVICE=" "$JOB_FILE" | cut -d'=' -f2-)
LOG_FILE=$(grep "^LOG_FILE=" "$JOB_FILE" | cut -d'=' -f2-)
CONFIG_FILE=$(grep "^CONFIG_FILE=" "$JOB_FILE" | cut -d'=' -f2-)
ERROR_LOG=$(grep "^ERROR_LOG=" "$JOB_FILE" | cut -d'=' -f2-)
ERASE=$(grep "^ERASE=" "$JOB_FILE" | cut -d'=' -f2-)
ITERATION=$(grep "^ITERATION=" "$JOB_FILE" | cut -d'=' -f2-)

echo "========================================"
echo "Cluster Training Job - Iteration $ITERATION"
echo "========================================"
echo "Config: $CONFIG_PATH"
echo "Device: $DEVICE"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "========================================"

# Change to NeuralGraph directory
cd "$NEURALGRAPH_DIR"

# Build command
CMD="python train_signal_subprocess.py --config '$CONFIG_PATH' --device '$DEVICE'"

if [ -n "$LOG_FILE" ]; then
    CMD="$CMD --log_file '$LOG_FILE'"
fi

if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config_file '$CONFIG_FILE'"
fi

if [ -n "$ERROR_LOG" ]; then
    CMD="$CMD --error_log '$ERROR_LOG'"
fi

if [ "$ERASE" = "true" ]; then
    CMD="$CMD --erase"
fi

echo "Running: $CMD"
echo ""

# Activate conda and run
source /groups/bhatti/bhattilabb/allierc/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Run training
eval $CMD
EXIT_CODE=$?

# Write completion status
DONE_FILE="${NEURALGRAPH_DIR}/cluster/job_done.txt"
echo "ITERATION=$ITERATION" > "$DONE_FILE"
echo "EXIT_CODE=$EXIT_CODE" >> "$DONE_FILE"
echo "COMPLETED_AT=$(date)" >> "$DONE_FILE"

echo ""
echo "========================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "========================================"

exit $EXIT_CODE
