#!/bin/bash
# GNN-Focused Experiments
# Tests GNN models on tasks that benefit from spatial relationships and coordination
# Based on task analysis: tasks requiring graph-structured reasoning

echo "=========================================="
echo "GNN-Focused Experiments"
echo "=========================================="
echo "Hypothesis: GNN models excel at tasks requiring spatial"
echo "            relationships, explicit communication, and"
echo "            coordinated positioning among agents."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:0}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# GNN models to compare
MODELS="mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2"

# Tasks that benefit from GNN
# High Coordination & Spatial Reasoning
TASKS_HIGH_COORD="vmas/transport,vmas/reverse_transport,vmas/football"

# Joint Manipulation & Formation
TASKS_FORMATION="vmas/joint_passage,vmas/ball_passage,vmas/ball_trajectory,vmas/wheel"

# Graph-Structured Relationships
TASKS_GRAPH="vmas/flocking,vmas/wind_flocking,vmas/discovery,vmas/passage"

# Communication & Teamwork
TASKS_COMM="vmas/simple_spread,vmas/simple_reference,vmas/simple_speaker_listener"

# All GNN-favorable tasks combined
ALL_TASKS="$TASKS_HIGH_COORD,$TASKS_FORMATION,$TASKS_GRAPH,$TASKS_COMM"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Task Categories:"
echo "    High Coordination: transport, reverse_transport, football"
echo "    Formation Control: joint_passage, ball_passage, ball_trajectory, wheel"
echo "    Graph Structure: flocking, wind_flocking, discovery, passage"
echo "    Communication: simple_spread, simple_reference, simple_speaker_listener"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $ALL_TASKS | tr ',' '\n' | wc -l)
SEEDS="0,1,2"
NUM_SEEDS=$(echo $SEEDS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS * NUM_SEEDS))

echo "Total runs: $TOTAL_RUNS (${NUM_MODELS} models × ${NUM_TASKS} tasks × ${NUM_SEEDS} seeds)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run experiments
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$ALL_TASKS \
    model=$MODELS \
    seed=$SEEDS \
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
echo "GNN-Focused Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
