#!/bin/bash
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# Script to run VMAS model survey experiment
# This will test all VMAS environments with multiple model configurations

echo "Starting VMAS Model Survey Experiment"
echo "======================================"
echo ""
echo "This will run experiments on all VMAS environments with:"
echo "  - MLP baseline"
echo "  - GNN (GraphConv)"
echo "  - GNN (GATv2)"
echo "  - DeepSets"
echo "  - MLP+GNN sequence"
echo ""
echo "Configuration:"
echo "  - 1 seed"
echo "  - 500 iterations"
echo "  - CUDA device (if available)"
echo "  - WandB project: benchmarl-10-30"
echo ""
echo "======================================"
echo ""

# Navigate to the BenchMARL root directory
cd "$(dirname "$0")/.."

# Run the experiment
python experiments/vmas_model_survey.py

echo ""
echo "======================================"
echo "Experiment completed!"
echo "Check WandB project 'benchmarl-10-30' for results"
echo "======================================"
