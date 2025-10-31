#!/bin/bash
# Combination Architectures Experiments
# Tests promising model combinations on grouped tasks
# Based on VMAS_Experiment_Design.md + dev/combination architectures

echo "=========================================="
echo "Combination Architectures Experiments"
echo "=========================================="
echo "Tests promising model combinations on representative tasks"
echo "from each category to identify best hybrid architectures."
echo ""

# Configuration
ALGORITHM="mappo"
DEVICE="${1:-cuda:1}"
MAX_ITERS="${2:-500}"
WANDB_PROJECT="benchmarl-2025-10-31"

# Combination architectures from dev/combination/
# Available combos: gnn_lstm_combo, deepsets_gnn_combo, gru_deepsets_combo, 
#                   lstm_gnn_lstm_combo, mlp_gnn_gru_combo, multi_gnn_stack
COMBOS="gnn_lstm_combo,deepsets_gnn_combo,gru_deepsets_combo,mlp_gnn_gru_combo"

# Representative tasks from each category
# Scalability: navigation (low), flocking (high)
# Coordination: give_way (simple), wheel (medium), transport (complex)
# Task Type: dispersion (nav), balance (transport), football (competitive)
TASKS="vmas/navigation,vmas/flocking,vmas/give_way,vmas/wheel,vmas/transport,vmas/dispersion,vmas/balance,vmas/football"

echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Combination Architectures:"
echo "    - GNN-LSTM: MLP → GNN → LSTM → MLP"
echo "    - DeepSets-GNN: DeepSets → GNN hybrid"
echo "    - GRU-DeepSets: GRU → DeepSets hybrid"
echo "    - MLP-GNN-GRU: MLP → GNN → GRU hybrid"
echo "  Representative Tasks:"
echo "    Scalability: navigation, flocking"
echo "    Coordination: give_way, wheel, transport"
echo "    Task Types: dispersion, balance, football"
echo "  Device: $DEVICE"
echo "  Max Iterations: $MAX_ITERS"
echo "  WandB Project: $WANDB_PROJECT"
echo ""

# Calculate runs
NUM_COMBOS=$(echo $COMBOS | tr ',' '\n' | wc -l)
NUM_TASKS=$(echo $TASKS | tr ',' '\n' | wc -l)
TOTAL_RUNS=$((NUM_COMBOS * NUM_TASKS * 3))

echo "Total runs: $TOTAL_RUNS (${NUM_COMBOS} combos × ${NUM_TASKS} tasks × 3 seeds)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run experiments
python benchmarl/run.py -m \
    algorithm=$ALGORITHM \
    task=$TASKS \
    model=$COMBOS \
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
echo "Combination Architectures Experiments Complete!"
echo "Check WandB: $WANDB_PROJECT"
echo "=========================================="
