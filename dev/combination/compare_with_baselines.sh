#!/bin/bash
# Full benchmark: Compare combinations vs baselines across 8 representative tasks
# Uses Hydra multi-run for parallel execution

cd /date/nfsShare/sudingli/gnn-rl/BenchMARL-gnn

TASKS=(
  "vmas/navigation"
  "vmas/sampling"
  "vmas/transport"
  "vmas/passage"
)

MODELS=(
  # Baselines
  "mlp_balanced"
  "gnn_balanced_graphconv"
  "gnn_balanced_gatv2"
  "lstm"
  "gru"
  "deepsets"
  # Combinations
  "dev/combination/gnn_lstm_combo"
  "dev/combination/deepsets_gnn_combo"
  "dev/combination/lstm_gnn_lstm_combo"
  "dev/combination/mlp_gnn_gru_combo"
  "dev/combination/multi_gnn_stack"
  "dev/combination/gru_deepsets_combo"
)

DEVICE="cuda:1"
ITERS=500
SEEDS="0,1,2"

echo "====================================="
echo "Full Benchmark: Combinations vs Baselines"
echo "Tasks: ${#TASKS[@]} tasks"
echo "Models: ${#MODELS[@]} models"
echo "Seeds: $SEEDS"
echo "Iterations: $ITERS"
echo "====================================="

# Build task list for Hydra
TASK_LIST=$(IFS=,; echo "${TASKS[*]}")

# Build model list for Hydra  
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")

echo "Running multi-run experiment..."
echo "This will launch ${#TASKS[@]} x ${#MODELS[@]} x 3 = $((${#TASKS[@]} * ${#MODELS[@]} * 3)) runs"

python benchmarl/run.py -m \
  algorithm=mappo \
  task=$TASK_LIST \
  model=$MODEL_LIST \
  seed=$SEEDS \
  experiment.max_n_iters=$ITERS \
  experiment.on_policy_collected_frames_per_batch=6000 \
  experiment.checkpoint_interval=50000 \
  experiment.evaluation=true \
  experiment.render=false \
  experiment.loggers.wandb.project_name=benchmarl-2025-10-31 \
  experiment.loggers.wandb.mode=online \
  experiment.loggers.csv.file_name=training_results \
  experiment.sampling_device=$DEVICE \
  experiment.train_device=$DEVICE \
  experiment.buffer_device=$DEVICE

echo ""
echo "====================================="
echo "Benchmark complete!"
echo "Check WandB: benchmarl-2025-10-31"
echo "====================================="
