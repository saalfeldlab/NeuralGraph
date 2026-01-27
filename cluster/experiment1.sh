#!/bin/bash

cd /groups/saalfeld/home/allierc/Graph/NeuralGraph

# Activate conda environment
source /groups/saalfeld/home/allierc/miniforge3/etc/profile.d/conda.sh
conda activate neural-graph

python GNN_LLM.py

echo "Experiment completed"

