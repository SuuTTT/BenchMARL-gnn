#!/bin/bash
# Test a single combination architecture
# Usage: bash test_single_combination.sh <config_name> <task_name> [device] [iterations]

CONFIG=${1:-gnn_lstm_combo}
TASK=${2:-vmas/navigation}
DEVICE=${3:-cuda:1}
ITERS=${4:-500}

echo "Testing combination: $CONFIG"
echo "Task: $TASK"
echo "Device: $DEVICE"
echo "Iterations: $ITERS"

cd /date/nfsShare/sudingli/gnn-rl/BenchMARL-gnn

python benchmarl/run.py \
  algorithm=mappo \
  task=$TASK \
  model=$CONFIG \
  seed=0 \
  experiment.max_n_iters=$ITERS \
  experiment.on_policy_collected_frames_per_batch=6000 \
  experiment.checkpoint_interval=0 \
  experiment.evaluation=true \
  experiment.render=false \
  experiment.loggers="[wandb,csv]" \
  experiment.project_name=benchmarl-2025-10-31 \
  experiment.sampling_device=$DEVICE \
  experiment.train_device=$DEVICE \
  experiment.buffer_device=$DEVICE

echo "Test completed: $CONFIG on $TASK"
