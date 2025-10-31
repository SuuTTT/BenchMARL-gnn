#!/bin/bash
# Simple Tasks Baseline Experiments
# Tests models on simple tasks with fixed, small number of agents
# Based on VMAS_Experiment_Design.md - Experiment 3: Simple Tasks Baseline

echo "=========================================="
echo "Simple Tasks Baseline Experiments"
echo "=========================================="
echo "Hypothesis: For simple tasks with fixed, small number of agents,"
echo "            performance differences will be minimal, and MLPs may"
echo "            offer the best trade-off between performance and speed."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:1}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# Models to compare
MODELS="mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets"

# Simple tasks with fixed, small number of agents
TASKS="vmas/balance,vmas/give_way"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Tasks (Simple Baseline): balance, give_way"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $TASKS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS * 5))

echo "Total runs: $TOTAL_RUNS (${NUM_MODELS} models × ${NUM_TASKS} tasks × 1 seeds)"
echo ""
echo "Note: Using 5 seeds for reproducibility analysis"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run experiments
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$TASKS \
    model=$MODELS \
    seed=0 \
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
echo "Simple Tasks Baseline Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
