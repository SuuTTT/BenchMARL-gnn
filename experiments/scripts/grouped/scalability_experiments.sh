#!/bin/bash
# Scalability Experiments - Variable Number of Agents
# Tests how well models scale with increasing number of agents
# Based on VMAS_Experiment_Design.md - Group 1: Scalability

echo "=========================================="
echo "Scalability Experiments"
echo "=========================================="
echo "Hypothesis: GNNs, DeepSets, and AttentionGNNs will show better"
echo "            performance and scalability than MLPs with large"
echo "            and variable number of agents."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:1}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# Models to compare
MODELS="mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets"

# Scalability Tasks
# Low-Coordination: dispersion, navigation
# High-Coordination: flocking, discovery, wind_flocking
TASKS="vmas/dispersion,vmas/navigation,vmas/flocking,vmas/discovery,vmas/wind_flocking"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Tasks (Scalability):"
echo "    Low-Coordination: dispersion, navigation"
echo "    High-Coordination: flocking, discovery, wind_flocking"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $TASKS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS))

echo "Total runs: $TOTAL_RUNS (${NUM_MODELS} models × ${NUM_TASKS} tasks)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run experiments
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$TASKS \
    model=$MODELS \
    seed=0,1,2 \
    experiment.max_n_iters=$MAX_ITERS \
    experiment.sampling_device=$DEVICE \
    experiment.train_device=$DEVICE \
    experiment.buffer_device=$DEVICE \
    experiment.on_policy_collected_frames_per_batch=6000 \
    experiment.on_policy_n_envs_per_worker=10 \
    experiment.loggers="[wandb,csv]" \
    experiment.project_name=$WANDB_PROJECT \
    experiment.evaluation=true \
    experiment.evaluation_interval=120000 \
    experiment.evaluation_episodes=5 \
    experiment.checkpoint_interval=0 \
    experiment.create_json=true

echo ""
echo "=========================================="
echo "Scalability Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
