#!/bin/bash
# VMAS Model Survey using Hydra Multi-Run
# This script runs all VMAS tasks with different model configurations using Hydra's -m flag

echo "=========================================="
echo "VMAS Model Survey - Hydra Multi-Run"
echo "=========================================="
echo "Starting experiment at: $(date)"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
echo ""

# Configuration
ALGORITHM="mappo"
SEED="1"
MAX_ITERS="500"
DEVICE="cuda:1"

# All VMAS tasks (commented tasks can be uncommented as needed)
TASKS="vmas/navigation,vmas/transport,vmas/wheel,vmas/balance,vmas/sampling"
# Full list (uncomment to run all):
# TASKS="vmas/ball_passage,vmas/ball_trajectory,vmas/buzz_wire,vmas/dispersion,vmas/dropout,vmas/flocking,vmas/football,vmas/give_way,vmas/joint_passage,vmas/joint_passage_size,vmas/multi_give_way,vmas/navigation,vmas/passage,vmas/reverse_transport,vmas/sampling,vmas/simple_adversary,vmas/simple_crypto,vmas/simple_push,vmas/simple_reference,vmas/simple_speaker_listener,vmas/simple_spread,vmas/simple_tag,vmas/simple_world_comm,vmas/transport,vmas/wheel,vmas/wind_flocking"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Tasks: $TASKS"
echo "  Seed: $SEED"
echo "  Max iterations: $MAX_ITERS"
echo "  Device: $DEVICE"
echo ""

# We need to create custom model configs for the balanced models
# For now, let's test with the standard MLP first
echo "Running MLP baseline..."
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$TASKS \
    seed=$SEED \
    model=layers/mlp \
    experiment.max_n_iters=$MAX_ITERS \
    experiment.sampling_device=$DEVICE \
    experiment.train_device=$DEVICE \
    experiment.loggers="[wandb,csv]" \
    experiment.wandb_extra_kwargs.project=benchmarl-10-30 \
    experiment.wandb_extra_kwargs.tags="[vmas_survey,mlp]" \
    experiment.evaluation=true \
    experiment.evaluation_interval=120000

echo ""
echo "=========================================="
echo "Experiment completed at: $(date)"
echo "=========================================="
echo ""
echo "Check WandB project 'benchmarl-10-30' for results"
echo "Or view local logs in ./outputs/"
