# WandB Filtering Guide for Model Combinations

## Summary of Changes

The codebase has been updated to make it easier to distinguish different model combinations in WandB. Here's what changed:

### 1. **Updated Experiment Naming**
Experiment names now follow this format:
```
<model_architecture>-<task>-<algorithm>-<date>-<hostname>-<device>
```

**Examples:**
- `gnn_lstm_combo-navigation-mappo-2025-11-02-c225-cuda1`
- `mlp_balanced-flocking-mappo-2025-11-02-c225-cuda1`
- `deepsets_gnn_combo-transport-mappo-2025-11-02-c225-cuda1`
- `gru_deepsets_combo-wheel-mappo-2025-11-02-c225-cuda1`

The **model architecture name comes first**, making it easy to identify at a glance!

### 2. **Added `model_architecture` to WandB Config**
A new field `model_architecture` is now logged to WandB's config/hparams, containing the model configuration name (e.g., `gnn_lstm_combo`, `mlp_balanced`, etc.).

### 3. **Added WandB Tags**
Each run is now automatically tagged with:
- `model_architecture`: e.g., "gnn_lstm_combo"
- `algorithm_name`: e.g., "mappo"
- `task_name`: e.g., "navigation"

## How to Filter in WandB

### Option 1: Filter by Config Field
In the WandB UI, use the config filter:
```
config.model_architecture = "gnn_lstm_combo"
config.model_architecture = "mlp_balanced"
config.model_architecture = "deepsets_gnn_combo"
```

### Option 2: Filter by Tags
Click on tags in the WandB UI or use tag filters:
- Tag: `gnn_lstm_combo`
- Tag: `mlp_balanced`
- Tag: `deepsets_gnn_combo`

### Option 3: Search by Name
Since experiment names now start with the model architecture, you can:
- Search for runs starting with "gnn_lstm_combo-"
- Search for runs starting with "mlp_balanced-"
- etc.

### Option 4: Group Runs
You can group runs by:
- `config.model_architecture`
- Existing grouping by `task_name` is still available

## Model Architecture Names

Based on your experiment scripts, here are the model architectures you're testing:

### Single Models
- `mlp_balanced` - Standard MLP
- `gnn_balanced_graphconv` - GNN with GraphConv
- `gnn_balanced_gatv2` - GNN with GATv2
- `layers/deepsets` - DeepSets architecture

### Combination Models
- `gnn_lstm_combo` - MLP → GNN → LSTM → MLP
- `deepsets_gnn_combo` - DeepSets → GNN hybrid
- `gru_deepsets_combo` - GRU → DeepSets hybrid
- `mlp_gnn_gru_combo` - MLP → GNN → GRU hybrid

## Files Modified

1. **`benchmarl/experiment/experiment.py`**
   - Added `_get_model_architecture_name()` method
   - Modified `_setup_name()` to include model architecture in experiment name
   - Added `model_architecture` parameter to `__init__()`
   - Added `model_architecture` to logged hparams

2. **`benchmarl/experiment/logger.py`**
   - Added `model_architecture` parameter to Logger class
   - Added model_architecture to WandB tags

3. **`benchmarl/hydra_config.py`**
   - Modified `load_experiment_from_hydra()` to extract model choice from Hydra
   - Passes `model_architecture` to Experiment constructor

## Example WandB Queries

### Compare all models on a specific task:
```
task_name = "navigation"
```
Then group by `config.model_architecture`

### Compare a specific model across all tasks:
```
config.model_architecture = "gnn_lstm_combo"
```
Then group by `task_name`

### Compare all combination models:
```
config.model_architecture IN ["gnn_lstm_combo", "deepsets_gnn_combo", "gru_deepsets_combo", "mlp_gnn_gru_combo"]
```

### Find all runs for a specific experiment group:
```
tags = "coordination_experiments"  # if you add this tag in your scripts
```

## Next Steps

1. **Run your experiments** - The new naming will be applied automatically
2. **Check WandB** - Verify that runs have the new naming format
3. **Use filters** - Try the different filtering options above
4. **Add custom tags** (optional) - You can add more tags in your shell scripts if needed

## Need More Tags?

If you want to add more specific tags (like experiment group names), you can modify your shell scripts to add them via the `wandb_extra_kwargs`. For example:

```bash
experiment.wandb_extra_kwargs.tags="[\"scalability\", \"coordination\"]"
```

This would add "scalability" and "coordination" as additional tags beyond the automatic ones.
