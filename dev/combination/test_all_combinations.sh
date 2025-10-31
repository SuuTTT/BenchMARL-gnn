#!/bin/bash
# Quick validation of all combination architectures on navigation task
# Runs each combo for 100 iterations to verify configs work

cd /date/nfsShare/sudingli/gnn-rl/BenchMARL-gnn

COMBOS=(
  "gnn_lstm_combo"
  "deepsets_gnn_combo"
  "lstm_gnn_lstm_combo"
  "mlp_gnn_gru_combo"
  "multi_gnn_stack"
  "gru_deepsets_combo"
)

TASK="vmas/navigation"
DEVICE="cuda:1"
ITERS=100

echo "====================================="
echo "Quick validation: All combinations"
echo "Task: $TASK"
echo "Iterations: $ITERS (validation only)"
echo "====================================="

for combo in "${COMBOS[@]}"; do
  echo ""
  echo "Testing: $combo"
  echo "-------------------------------------"
  
  python benchmarl/run.py \
    algorithm=mappo \
    task=$TASK \
    model=$combo \
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
  
  if [ $? -eq 0 ]; then
    echo "✓ $combo passed validation"
  else
    echo "✗ $combo failed validation"
  fi
done

echo ""
echo "====================================="
echo "Validation complete!"
echo "====================================="
