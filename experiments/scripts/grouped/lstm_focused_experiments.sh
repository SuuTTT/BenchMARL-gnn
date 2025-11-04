#!/bin/bash
# LSTM-Focused Experiments
# Tests LSTM/GRU models on tasks that benefit from temporal reasoning and memory
# Based on task analysis: tasks requiring sequential patterns and trajectory prediction

echo "=========================================="
echo "LSTM-Focused Experiments"
echo "=========================================="
echo "Hypothesis: LSTM/GRU models excel at tasks requiring"
echo "            temporal patterns, memory of past states,"
echo "            and trajectory prediction."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:1}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# LSTM/GRU models to compare with MLP baseline
MODELS="mlp_balanced,gru_balanced,lstm_balanced"

# Tasks that benefit from LSTM
# Temporal Coordination
TASKS_TEMPORAL="vmas/ball_trajectory,vmas/buzz_wire,vmas/wheel,vmas/balance"

# Sequential Decision Making
TASKS_SEQUENTIAL="vmas/dropout,vmas/give_way,vmas/multi_give_way,vmas/passage"

# Partial Observability & Memory
TASKS_MEMORY="vmas/navigation,vmas/sampling,vmas/simple_adversary,vmas/simple_crypto,vmas/simple_tag"

# Trajectory Prediction
TASKS_TRAJECTORY="vmas/football,vmas/ball_passage"

# All LSTM-favorable tasks combined
ALL_TASKS="$TASKS_TEMPORAL,$TASKS_SEQUENTIAL,$TASKS_MEMORY,$TASKS_TRAJECTORY"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Task Categories:"
echo "    Temporal Coordination: ball_trajectory, buzz_wire, wheel, balance"
echo "    Sequential Decisions: dropout, give_way, multi_give_way, passage"
echo "    Memory & Observation: navigation, sampling, simple_adversary, simple_crypto, simple_tag"
echo "    Trajectory Prediction: football, ball_passage"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $ALL_TASKS | tr ',' '\n' | wc -l)
SEEDS="1,2,3"
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
echo "LSTM-Focused Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
