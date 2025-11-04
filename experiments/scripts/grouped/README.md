# Grouped Experiments for VMAS

This directory contains experiment scripts organized by task characteristics as defined in `experiments/docs/VMAS_Experiment_Design.md`.

## Overview

The experiments are grouped to test specific hypotheses about model performance on different types of multi-agent tasks.

## Experiment Groups

### 1. Scalability Experiments (`scalability_experiments.sh`)

**Hypothesis**: GNNs, DeepSets, and AttentionGNNs will show better performance and scalability than MLPs in environments with a large and variable number of agents.

**Tasks**:
- **Low-Coordination**: `dispersion`, `navigation`
- **High-Coordination**: `flocking`, `discovery`, `wind_flocking`

**Usage**:
```bash
bash experiments/scripts/grouped/scalability_experiments.sh [device] [iterations]
# Example: bash experiments/scripts/grouped/scalability_experiments.sh cuda:1 500

nohup bash experiments/scripts/grouped/scalability_experiments.sh cuda:1 500 > 2025-10-31-c225-cuda1-scalability_experiments.log &
```

### 2. Coordination Complexity Experiments (`coordination_experiments.sh`)

**Hypothesis**: GNN and AttentionGNN will outperform MLP and DeepSets in tasks requiring complex, spatially-aware coordination.

**Tasks**:
- **Simple Coordination**: `navigation`, `give_way`
- **Medium Coordination**: `balance`, `wheel`, `passage`
- **Complex Coordination**: `transport`, `reverse_transport`, `football`, `joint_passage`

**Usage**:
```bash
bash experiments/scripts/grouped/coordination_experiments.sh [device] [iterations]
# Example: bash experiments/scripts/grouped/coordination_experiments.sh cuda:0 500

nohup bash experiments/scripts/grouped/coordination_experiments.sh cuda:0 500 > 2025-10-31-c225-cuda0-coordination_experiments.log &
```

### 3. Task Type Experiments (`task_type_experiments.sh`)

**Hypothesis**: Certain architectures are better suited for specific task types (e.g., GNN for transport, DeepSets for flocking).

**Task Categories**:
- **Navigation & Dispersion**: `navigation`, `dispersion`, `discovery`
- **Transport & Manipulation**: `transport`, `reverse_transport`, `balance`, `wheel`, `ball_passage`
- **Flocking & Formation**: `flocking`, `wind_flocking`
- **Competitive/Mixed**: `football`

**Usage**:
```bash
bash experiments/scripts/grouped/task_type_experiments.sh [device] [iterations]

nohup bash experiments/scripts/grouped/task_type_experiments.sh cuda:0 500 > 2025-10-31-c223-cuda0-task_type_experiments.log &
```

### 4. Simple Tasks Baseline (`simple_baseline_experiments.sh`)

**Hypothesis**: For simple tasks with a fixed, small number of agents, the performance difference between models will be minimal, and MLPs may offer the best trade-off between performance and training speed.

**Tasks**: `balance`, `give_way`

**Note**: Uses 5 random seeds for robust statistical analysis.

**Usage**:
```bash
bash experiments/scripts/grouped/simple_baseline_experiments.sh [device] [iterations]

nohup bash experiments/scripts/grouped/simple_baseline_experiments.sh cuda:0 500 > 2025-10-31-c223-cuda1-simple_baseline_experiments.log &

nohup bash experiments/scripts/grouped/simple_baseline_experiments.sh cuda:1 500 > 2025-11-02-c223-cuda1-simple_baseline_experiments.log &
nohup bash experiments/scripts/grouped/simple_baseline_experiments.sh cuda:1 500 > 2025-11-02-c223-cuda1-simple_baseline_experiments.log &
nohup bash experiments/scripts/grouped/simple_baseline_experiments.sh cuda:1 500 > 2025-11-02-c225-cuda1-simple_baseline_experiments-seed1.log 
```

### 5. Combination Architectures (`combination_experiments.sh`)

Tests promising model combinations on representative tasks from each category.

**Architectures**:
- **Attention-GNN**: Attention[4] → GNN → Attention[2]
- **Hierarchical DeepSets-GNN**: DeepSets → GNN → MLP
- **Recurrent-GNN**: GRU[64] → GNN → GRU[32]
- **Multi-Scale GNN**: Parallel GNN branches (add/mean/max)

**Representative Tasks**: Selected from scalability, coordination, and task type categories.

**Usage**:
```bash
bash experiments/scripts/grouped/combination_experiments.sh [device] [iterations]

nohup bash experiments/scripts/grouped/combination_experiments.sh cuda:0 500 > 2025-10-31-c224-cuda0-combination_experiments.log &
#update archi name
nohup bash experiments/scripts/grouped/combination_experiments.sh cuda:1 500 > 2025-11-01-c224-cuda1-combination_experiments.log &
nohup bash experiments/scripts/grouped/combination_experiments.sh cuda:0 500 > 2025-11-01-c224-cuda0-seed1-combination_experiments.log &
```

### 6. GNN-Focused Experiments (`gnn_focused_experiments.sh`)

**Hypothesis**: GNN models excel at tasks requiring spatial relationships, explicit communication, and coordinated positioning among agents.

**Task Categories**:
- **High Coordination & Spatial Reasoning**: `transport`, `reverse_transport`, `football`
- **Formation Control**: `joint_passage`, `ball_passage`, `ball_trajectory`, `wheel`
- **Graph-Structured Relationships**: `flocking`, `wind_flocking`, `discovery`, `passage`
- **Communication & Teamwork**: `simple_spread`, `simple_reference`, `simple_speaker_listener`

**Models**: MLP baseline, GNN (GraphConv), GNN (GATv2)

**Usage**:
```bash
bash experiments/scripts/grouped/gnn_focused_experiments.sh [device] [iterations]

# Run on c223
nohup bash experiments/scripts/grouped/gnn_focused_experiments.sh cuda:0 500 > 2025-11-04-c223-cuda0-gnn_focused_experiments.log &
```

### 7. LSTM-Focused Experiments (`lstm_focused_experiments.sh`)

**Hypothesis**: LSTM/GRU models excel at tasks requiring temporal patterns, memory of past states, and trajectory prediction.

**Task Categories**:
- **Temporal Coordination**: `ball_trajectory`, `buzz_wire`, `wheel`, `balance`
- **Sequential Decision Making**: `dropout`, `give_way`, `multi_give_way`, `passage`
- **Partial Observability & Memory**: `navigation`, `sampling`, `simple_adversary`, `simple_crypto`, `simple_tag`
- **Trajectory Prediction**: `football`, `ball_passage`

**Models**: MLP baseline, GRU, LSTM

**Usage**:
```bash
bash experiments/scripts/grouped/lstm_focused_experiments.sh [device] [iterations]

# Run on c223
nohup bash experiments/scripts/grouped/lstm_focused_experiments.sh cuda:1 500 > 2025-11-04-c223-cuda1-lstm_focused_experiments.log &
```

## Models Evaluated

All experiments use balanced parameter counts (~13-19K parameters):
- `mlp_balanced`: Baseline MLP [128, 128]
- `gnn_balanced_graphconv`: MLP[64] → GraphConv GNN → MLP[64]
- `gnn_balanced_gatv2`: MLP[64] → GATv2 GNN → MLP[64]
- `gru_balanced`: GRU[128] + MLP[256, 256] recurrent architecture
- `lstm_balanced`: LSTM[128] + MLP[256, 256] recurrent architecture
- `layers/deepsets`: DeepSets architecture (for legacy compatibility)

## Experimental Setup

- **Algorithm**: MAPPO (Multi-Agent PPO)
- **Seeds**: 3 seeds (0, 1, 2) for most experiments, 5 seeds for baseline
- **Iterations**: Default 500 (configurable)
- **Frames per batch**: 6000
- **Total frames**: ~3M per run (500 × 6000)
- **Logging**: WandB + CSV
- **Device**: Configurable (default: cuda:1)

## Device Consistency

All experiments now set all three device parameters to ensure consistency:
- `experiment.sampling_device=$DEVICE`
- `experiment.train_device=$DEVICE`
- `experiment.buffer_device=$DEVICE`

## Naming Convention

Experiments use standardized naming:
```
yyyy-mm-dd-hostname-devicename-task-algo
```
Example: `2025-10-31-c225-cuda1-navigation-mappo`

## Running All Grouped Experiments

To run all grouped experiments sequentially:

```bash
# Make all scripts executable
chmod +x experiments/scripts/grouped/*.sh

# Run all experiments
for script in experiments/scripts/grouped/*_experiments.sh; do
    echo "Running $script..."
    bash "$script" cuda:1 500
    echo "Completed $script"
    echo "---"
done
```

## Monitoring Progress

All experiments log to WandB project: `benchmarl-2025-10-31`

Check progress at: `https://wandb.ai/your-username/benchmarl-2025-10-31`

## Analysis

After experiments complete:
1. Compare model performance across task groups
2. Identify which architectures excel at which task types
3. Analyze scalability trends
4. Measure training time vs. performance trade-offs
5. Validate hypotheses from VMAS_Experiment_Design.md
