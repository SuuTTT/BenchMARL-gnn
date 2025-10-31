#!/bin/bash
# Extended benchmark: Test all combinations on diverse tasks (8 scenarios)

cd /date/nfsShare/sudingli/gnn-rl/BenchMARL-gnn

# 8 representative tasks across all categories
TASKS=(
  "vmas/navigation"       # Navigation
  "vmas/sampling"         # Coordination
  "vmas/transport"        # Coordination
  "vmas/passage"          # Adversarial
  "vmas/balance"          # Communication
  "vmas/wheel"            # Manipulation
  "vmas/give_way"         # Coordination
  "vmas/discovery"        # Communication
)

COMBOS=(
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
echo "Extended Benchmark: 8 Tasks x 6 Combos"
echo "Total runs: 8 x 6 x 3 seeds = 144"
echo "====================================="

# Build lists for Hydra
TASK_LIST=$(IFS=,; echo "${TASKS[*]}")
COMBO_LIST=$(IFS=,; echo "${COMBOS[*]}")

nohup python benchmarl/run.py -m \
  algorithm=mappo \
  task=$TASK_LIST \
  model=$COMBO_LIST \
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
  experiment.buffer_device=$DEVICE \
  > combination_benchmark.log 2>&1 &

echo "Background job started: PID $!"
echo "Log file: combination_benchmark.log"
echo "Monitor: tail -f combination_benchmark.log"
