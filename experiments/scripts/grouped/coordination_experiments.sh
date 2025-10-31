#!/bin/bash
# Coordination Complexity Experiments
# Tests models on tasks requiring different levels of coordination
# Based on VMAS_Experiment_Design.md - Group 2: Coordination Complexity

echo "=========================================="
echo "Coordination Complexity Experiments"
echo "=========================================="
echo "Hypothesis: GNN and AttentionGNN will outperform MLP and DeepSets"
echo "            in tasks requiring complex, spatially-aware coordination."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:1}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# Models to compare (including combination architectures)
MODELS="mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets"

# Coordination Complexity Tasks
# Simple Coordination: navigation, give_way
# Medium Coordination: balance, wheel, passage
# Complex Coordination: transport, reverse_transport, football, joint_passage
TASKS_SIMPLE="vmas/navigation,vmas/give_way"
TASKS_MEDIUM="vmas/balance,vmas/wheel,vmas/passage"
TASKS_COMPLEX="vmas/transport,vmas/reverse_transport,vmas/football,vmas/joint_passage"

# All tasks combined
ALL_TASKS="$TASKS_SIMPLE,$TASKS_MEDIUM,$TASKS_COMPLEX"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Tasks (Coordination Complexity):"
echo "    Simple: navigation, give_way"
echo "    Medium: balance, wheel, passage"
echo "    Complex: transport, reverse_transport, football, joint_passage"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $ALL_TASKS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS * 3))

echo "Total runs: $TOTAL_RUNS (${NUM_MODELS} models × ${NUM_TASKS} tasks × 3 seeds)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run experiments
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$ALL_TASKS \
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
echo "Coordination Complexity Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
