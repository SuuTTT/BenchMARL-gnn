#!/bin/bash
# Test single run with Hydra configuration
# Use this to verify setup before running full survey

echo "=========================================="
echo "Single Task Test Run (Hydra)"
echo "=========================================="
echo ""

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo ""

# Test configuration
ALGORITHM="mappo"
TASK="vmas/navigation"
SEED="1"
MAX_ITERS="50"
DEVICE="cuda:1"
MODEL="mlp_balanced"
WANDB_PROJECT="benchmarl-2025-10-31"

echo "Testing with:"
echo "  Algorithm: $ALGORITHM"
echo "  Task: $TASK"
echo "  Model: $MODEL"
echo "  Iterations: $MAX_ITERS (calculated as total_frames / frames_per_batch)"
echo "  Frames per batch: 6000 (default)"
echo "  Total frames: ~300K (50 iters × 6000 frames/batch)"
echo "  Device: $DEVICE"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

python benchmarl/run.py \
    algorithm=$ALGORITHM \
    task=$TASK \
    seed=$SEED \
    model=$MODEL \
    experiment.max_n_iters=$MAX_ITERS \
    experiment.sampling_device=$DEVICE \
    experiment.train_device=$DEVICE \
    experiment.loggers="[csv]" \
    experiment.project_name=$WANDB_PROJECT \
    experiment.evaluation=false \
    experiment.checkpoint_interval=0 \
    experiment.create_json=true

echo ""
if [ $? -eq 0 ]; then
    echo "✓ Test run completed successfully!"
    echo ""
    echo "You can now run the full survey:"
    echo "  bash experiments/run_vmas_survey_hydra_multirun.sh"
else
    echo "✗ Test run failed. Check the error messages above."
fi
