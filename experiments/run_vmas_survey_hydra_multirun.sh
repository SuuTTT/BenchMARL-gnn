#!/bin/bash
# VMAS Model Survey - Full Multi-Run with Hydra
# This uses Hydra's -m flag for efficient multi-run execution

echo "=========================================="
echo "VMAS Model Survey - Full Multi-Run"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo ""

# Configuration
ALGORITHM="mappo"
SEED="1"
MAX_ITERS="500"
DEVICE="cuda:1"
WANDB_PROJECT="benchmarl-2025-10-31"

# Model configurations (balanced parameter counts)
MODELS="mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets"

# VMAS tasks - start with a subset for testing, uncomment for full run
TASKS="vmas/navigation,vmas/transport,vmas/sampling"

# Full list of VMAS tasks (uncomment to run all 26 tasks):
# TASKS="vmas/ball_passage,vmas/ball_trajectory,vmas/buzz_wire,vmas/dispersion,vmas/dropout,vmas/flocking,vmas/football,vmas/give_way,vmas/joint_passage,vmas/joint_passage_size,vmas/multi_give_way,vmas/navigation,vmas/passage,vmas/reverse_transport,vmas/sampling,vmas/simple_adversary,vmas/simple_crypto,vmas/simple_push,vmas/simple_reference,vmas/simple_speaker_listener,vmas/simple_spread,vmas/simple_tag,vmas/simple_world_comm,vmas/transport,vmas/wheel,vmas/wind_flocking"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Models: $MODELS"
echo "  Tasks: $TASKS"
echo "  Seed: $SEED"
echo "  Max iterations: $MAX_ITERS (calculated as total_frames / frames_per_batch)"
echo "  Frames per batch: 6000"
echo "  Total frames: ~3M (500 iters × 6000 frames/batch)"
echo "  Device: $DEVICE"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate total runs
NUM_MODELS=$(echo $MODELS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $TASKS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS))

echo "Total runs: $TOTAL_RUNS (${NUM_MODELS} models × ${NUM_TASKS} tasks)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run Hydra multi-run
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$TASKS \
    model=$MODELS \
    seed=$SEED \
    experiment.max_n_iters=$MAX_ITERS \
    experiment.sampling_device=$DEVICE \
    experiment.train_device=$DEVICE \
    experiment.on_policy_collected_frames_per_batch=6000 \
    experiment.on_policy_n_envs_per_worker=10 \
    experiment.loggers="[wandb,csv]" \
    experiment.project_name=$WANDB_PROJECT \
    experiment.evaluation=true \
    experiment.evaluation_interval=120000 \
    experiment.evaluation_episodes=5 \
    experiment.checkpoint_interval=0 \
    experiment.create_json=true

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Survey completed successfully!"
    echo ""
    echo "Results:"
    echo "  - WandB: https://wandb.ai (project: $WANDB_PROJECT)"
    echo "  - Local: ./outputs/"
else
    echo "✗ Survey failed with exit code $EXIT_CODE"
    echo "Check the error messages above"
fi
