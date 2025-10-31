# Model Combination Experiments

This directory contains experimental model architectures that combine existing BenchMARL modules in novel ways.

## 🎯 Goal

Test promising combinations of:
- **Feedforward layers** (MLP)
- **Graph networks** (GNN)
- **Recurrent networks** (LSTM, GRU)
- **Set-based networks** (DeepSets)

## 🧠 Designed Architectures

### 1. **GNN-LSTM Hybrid** (`gnn_lstm_combo.yaml`)
- **Architecture**: MLP[64] → GNN → LSTM → MLP[64]
- **Rationale**: GNN captures spatial relationships, LSTM captures temporal dynamics
- **Best for**: Dynamic multi-agent tasks (flocking, navigation with memory)
- **Est. Params**: ~25K

### 2. **DeepSets-GNN** (`deepsets_gnn_combo.yaml`)
- **Architecture**: DeepSets → GNN → MLP[32]
- **Rationale**: DeepSets for permutation-invariant features, GNN for message passing
- **Best for**: Variable number of agents, coordination tasks
- **Est. Params**: ~30K

### 3. **LSTM-GNN-LSTM** (`lstm_gnn_lstm_combo.yaml`)
- **Architecture**: LSTM[32] → GNN → LSTM[32]
- **Rationale**: Temporal processing before and after spatial aggregation
- **Best for**: Highly dynamic environments with temporal dependencies
- **Est. Params**: ~20K

### 4. **MLP-GNN-GRU** (`mlp_gnn_gru_combo.yaml`)
- **Architecture**: MLP[64] → GNN GATv2 → GRU → MLP[32]
- **Rationale**: Feature extraction → Attention-based graph → Temporal smoothing
- **Best for**: Communication tasks with temporal context
- **Est. Params**: ~22K

### 5. **Multi-GNN Stack** (`multi_gnn_stack.yaml`)
- **Architecture**: MLP[32] → GNN GraphConv → GNN GATv2 → MLP[32]
- **Rationale**: Different GNN aggregations capture different patterns
- **Best for**: Complex graph structures, heterogeneous agents
- **Est. Params**: ~15K

### 6. **Parallel Paths** (`parallel_gnn_deepsets.yaml`)
- **Architecture**: Input → [GNN path || DeepSets path] → Concat → MLP
- **Rationale**: Different aggregation strategies combined
- **Note**: Requires custom implementation (not pure sequence)
- **Best for**: Tasks benefiting from multiple representations

## 📊 Test Protocol

All combinations tested on:
- **Navigation tasks**: navigation, transport
- **Coordination tasks**: flocking, wheel
- **Communication tasks**: simple_spread, simple_reference
- **Adversarial tasks**: simple_adversary, simple_tag

Baseline comparisons:
- Pure MLP
- Pure GNN
- Pure LSTM
- Pure DeepSets

## 🚀 Quick Start

```bash
# Test all combinations on navigation
bash dev/combination/test_all_combinations.sh

# Test specific combination
bash dev/combination/test_single_combination.sh gnn_lstm_combo

# Compare against baselines
bash dev/combination/compare_with_baselines.sh
```

## 📈 Expected Results

Hypotheses:
1. **GNN-LSTM** should excel in dynamic navigation/flocking
2. **DeepSets-GNN** should handle variable agent numbers well
3. **LSTM-GNN-LSTM** should be best for highly temporal tasks
4. **Multi-GNN** should capture complex spatial patterns

## 📁 Files

- `*.yaml` - Model configuration files
- `test_*.sh` - Testing scripts
- `results/` - Experiment outputs
- `analysis.ipynb` - Results analysis notebook
