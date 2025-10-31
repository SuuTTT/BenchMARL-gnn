#!/bin/bash
# Task Type Experiments
# Tests models grouped by high-level objective
# Based on VMAS_Experiment_Design.md - Group 3: Task Type

echo "=========================================="
echo "Task Type Experiments"
echo "=========================================="
echo "Hypothesis: Certain architectures are better suited for"
echo "            specific task types (e.g., GNN for transport)."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:1}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# Models to compare
MODELS="mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets"

# Task Types
# Navigation & Dispersion
TASKS_NAV="vmas/navigation,vmas/dispersion,vmas/discovery"

# Transport & Manipulation
TASKS_TRANSPORT="vmas/transport,vmas/reverse_transport,vmas/balance,vmas/wheel,vmas/ball_passage"

# Flocking & Formation
TASKS_FLOCKING="vmas/flocking,vmas/wind_flocking"

# Competitive/Mixed
TASKS_COMPETITIVE="vmas/football"

# All tasks combined
ALL_TASKS="$TASKS_NAV,$TASKS_TRANSPORT,$TASKS_FLOCKING,$TASKS_COMPETITIVE"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Task Types:"
echo "    Navigation & Dispersion: navigation, dispersion, discovery"
echo "    Transport & Manipulation: transport, reverse_transport, balance, wheel, ball_passage"
echo "    Flocking & Formation: flocking, wind_flocking"
echo "    Competitive/Mixed: football"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $ALL_TASKS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS * 3))

echo "Total runs: $TOTAL_RUNS (${NUM_MODELS} models × ${NUM_TASKS} tasks × 1 seeds)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run experiments
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$ALL_TASKS \
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
echo "Task Type Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
