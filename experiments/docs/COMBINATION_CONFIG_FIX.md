# Fix for Combination Model Configs

## Problem
The combination model scripts (`dev/combination/test_single_combination.sh`) were failing with:
```
Could not find 'model/dev/combination/gnn_lstm_combo'
```

## Root Causes

### 1. Config Location
Hydra can only find configs in registered config paths. The configs were in `dev/combination/` but Hydra searches in `benchmarl/conf/model/`.

### 2. Incorrect RNN Parameters
LSTM and GRU configs were using `num_cells` parameter (which is for MLP layers), but they should use:
- `hidden_size`: Size of hidden state
- `n_layers`: Number of layers
- `bias`, `dropout`, `compile`: Additional parameters

### 3. Logger Configuration Syntax
Scripts were using incorrect logger syntax:
```bash
# Wrong:
experiment.loggers.wandb.project_name=...

# Correct:
experiment.loggers="[wandb,csv]"
experiment.project_name=...
```

### 4. Checkpoint Interval
`checkpoint_interval` must be a multiple of `collected_frames_per_batch` or set to 0.

## Solutions Applied

### 1. Copied Configs to Proper Location
```bash
cp dev/combination/*.yaml benchmarl/conf/model/
```

### 2. Fixed RNN Config Parameters

**LSTM configs** (`gnn_lstm_combo.yaml`, `lstm_gnn_lstm_combo.yaml`):
```yaml
# Before:
l3:
  num_cells: [64]
  activation_class: torch.nn.Tanh

# After:
l3:
  hidden_size: 64
  n_layers: 1
  bias: true
  dropout: 0
  compile: false
```

**GRU configs** (`mlp_gnn_gru_combo.yaml`, `gru_deepsets_combo.yaml`):
```yaml
# Before:
l1:
  num_cells: [48]
  activation_class: torch.nn.Tanh

# After:
l1:
  hidden_size: 48
  n_layers: 1
  bias: true
  dropout: 0
  compile: false
```

### 3. Updated Script References

**Changed from**:
```bash
model=dev/combination/$CONFIG
```

**To**:
```bash
model=$CONFIG
```

### 4. Fixed Logger Configuration

**Changed from**:
```bash
experiment.loggers.wandb.project_name=benchmarl-2025-10-31
experiment.loggers.wandb.mode=online
experiment.loggers.csv.file_name=training_results
```

**To**:
```bash
experiment.loggers="[wandb,csv]"
experiment.project_name=benchmarl-2025-10-31
```

### 5. Fixed Checkpoint Interval

**Changed from**:
```bash
experiment.checkpoint_interval=50000
```

**To**:
```bash
experiment.checkpoint_interval=0
```

## Files Modified

### Config Files (both locations)
1. `benchmarl/conf/model/gnn_lstm_combo.yaml`
2. `benchmarl/conf/model/lstm_gnn_lstm_combo.yaml`
3. `benchmarl/conf/model/mlp_gnn_gru_combo.yaml`
4. `benchmarl/conf/model/gru_deepsets_combo.yaml`
5. `dev/combination/gnn_lstm_combo.yaml`
6. `dev/combination/lstm_gnn_lstm_combo.yaml`
7. `dev/combination/mlp_gnn_gru_combo.yaml`
8. `dev/combination/gru_deepsets_combo.yaml`

### Script Files
1. `dev/combination/test_single_combination.sh`
2. `dev/combination/test_all_combinations.sh`
3. `experiments/scripts/grouped/combination_experiments.sh`

## Updated Combination Names

The `experiments/scripts/grouped/combination_experiments.sh` now uses the actual combination configs:

**Before**:
```bash
COMBOS="attention_gnn,hierarchical_deepsets_gnn,recurrent_gnn,multi_scale_gnn"
```

**After**:
```bash
COMBOS="gnn_lstm_combo,deepsets_gnn_combo,gru_deepsets_combo,mlp_gnn_gru_combo"
```

## Testing

To test a combination model:

```bash
# Single combination test (use benchmarl-dev environment)
conda activate benchmarl-dev
cd /date/nfsShare/sudingli/gnn-rl/BenchMARL-gnn
./dev/combination/test_single_combination.sh gnn_lstm_combo vmas/navigation cuda:1 10

# Test all combinations
./dev/combination/test_all_combinations.sh

# Run grouped combination experiments
bash experiments/scripts/grouped/combination_experiments.sh cuda:1 500
```

## Available Combination Architectures

1. **gnn_lstm_combo**: MLP[64] → GNN GraphConv → LSTM[64] → MLP[64]
2. **deepsets_gnn_combo**: DeepSets → GNN hybrid
3. **gru_deepsets_combo**: GRU[48] → DeepSets → MLP[32]
4. **mlp_gnn_gru_combo**: MLP[64] → GNN GATv2 → GRU[48] → MLP[32]
5. **lstm_gnn_lstm_combo**: LSTM[32] → GNN → LSTM[32]
6. **multi_gnn_stack**: Multiple GNN layers stacked

## Reference: Correct LSTM/GRU Config Format

Based on `benchmarl/conf/model/layers/lstm.yaml` and `gru.yaml`:

```yaml
name: lstm  # or gru

# Required RNN parameters
hidden_size: 128
n_layers: 1
bias: True
dropout: 0
compile: False

# MLP parameters (if using with MLP wrapper)
mlp_num_cells: [256, 256]
mlp_layer_class: torch.nn.Linear
mlp_activation_class: torch.nn.Tanh
mlp_activation_kwargs: null
mlp_norm_class: null
mlp_norm_kwargs: null
```

## Verification

After fixes, the script should:
1. ✅ Find the model config
2. ✅ Load LSTM/GRU layers correctly
3. ✅ Configure loggers properly
4. ✅ Start training without checkpoint errors
5. ✅ Use correct device configuration (sampling, train, buffer all synchronized)
