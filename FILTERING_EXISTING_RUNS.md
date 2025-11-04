# How to Filter Existing WandB Runs (Before Model Architecture Field)

## Problem
Your existing WandB runs don't have the `model_architecture` field because they were logged before the recent code changes.

## Solution: Use `config.model_config` Field

Your existing runs **DO** have the full model configuration logged in `config.model_config`. Here's how to filter them:

---

## For **Single Models** (MLP, GNN, DeepSets)

### Filter by MLP:
```
config.model_config._target_ = "benchmarl.models.mlp.MlpConfig"
```
OR search for runs with:
```
config.model_name = "mlp"
```

### Filter by GNN (GraphConv):
```
config.model_config.gnn_class = "torch_geometric.nn.conv.GraphConv"
```
OR
```
config.model_config.gnn_class contains "GraphConv"
```

### Filter by GNN (GATv2):
```
config.model_config.gnn_class = "torch_geometric.nn.conv.GATv2Conv"
```
OR
```
config.model_config.gnn_class contains "GATv2"
```

### Filter by DeepSets:
```
config.model_name = "deepsets"
```

---

## For **Combination/Sequence Models**

All combination models have:
```
config.model_name = "sequencemodel"
```

To distinguish between them, you need to look at the **layers**:

### Filter by GNN-LSTM Combo:
Look for runs where:
```
config.model_config.model_configs_1._target_ = "benchmarl.models.gnn.GnnConfig"
AND
config.model_config.model_configs_2._target_ = "benchmarl.models.lstm.LstmConfig"
```

### Filter by DeepSets-GNN Combo:
```
config.model_config.model_configs_0._target_ contains "DeepSets"
AND
config.model_config.model_configs_1._target_ contains "Gnn"
```

### Filter by GRU-DeepSets Combo:
```
config.model_config.model_configs contains "Gru"
AND
config.model_config.model_configs contains "DeepSets"
```

---

## Easier Method: Filter by Run Name/ID

Your run names should follow a pattern. Look at the WandB run names/IDs and they should contain information about the task and other details.

### Use Table View Filters:

1. **Go to WandB project**
2. **Switch to Table view**
3. **Add columns**:
   - `config.model_name`
   - `config.model_config.gnn_class` (for GNN runs)
   - `config.model_config.model_configs` (for sequence models)
4. **Filter and sort** by these columns

---

## Quick Reference: Model Identification

| Model Type | How to Identify in Existing Runs |
|------------|----------------------------------|
| **MLP** | `config.model_name = "mlp"` |
| **GNN GraphConv** | `config.model_name = "gnn"` AND `config.model_config.gnn_class` contains `GraphConv` |
| **GNN GATv2** | `config.model_name = "gnn"` AND `config.model_config.gnn_class` contains `GATv2` |
| **DeepSets** | `config.model_name = "deepsets"` |
| **Sequence Models** | `config.model_name = "sequencemodel"` |

For sequence models, drill down into:
- `config.model_config.model_configs_0._target_`
- `config.model_config.model_configs_1._target_`
- etc.

---

## WandB UI Tips

### Method 1: Use Filters Panel
1. Click **"Filters"** in WandB
2. Click **"+ Add filter"**
3. Type `config.model_name` or `config.model_config`
4. Select the value you want

### Method 2: Use Search Bar
In the search bar, type:
```
model_name:mlp
model_name:gnn
model_name:sequencemodel
```

### Method 3: Group Runs
1. Click **"Group"**
2. Select `config.model_name`
3. This will group all runs by model type

---

## Creating Custom Tags Retroactively

If you have access to WandB API, you can add tags to existing runs:

```python
import wandb

api = wandb.Api()
runs = api.runs("your-entity/benchmarl-2025-10-31")

for run in runs:
    # Get model config
    model_name = run.config.get('model_name', '')
    
    # Add appropriate tag
    if model_name == 'mlp':
        run.tags.append('mlp_balanced')
    elif model_name == 'gnn':
        gnn_class = run.config.get('model_config', {}).get('gnn_class', '')
        if 'GraphConv' in gnn_class:
            run.tags.append('gnn_graphconv')
        elif 'GATv2' in gnn_class:
            run.tags.append('gnn_gatv2')
    elif model_name == 'sequencemodel':
        # Logic to identify combination type
        run.tags.append('combo_model')
    
    run.update()
```

---

## For Future Runs

Once you run new experiments with the updated code, they will automatically have:
- ✅ `config.model_architecture` field
- ✅ WandB tags with model architecture
- ✅ Experiment names starting with model architecture

Making filtering much easier!

---

## Example WandB Queries for Existing Runs

### All MLP runs:
```
config.model_name = "mlp"
```

### All GNN runs:
```
config.model_name = "gnn"
```

### All combination/sequence models:
```
config.model_name = "sequencemodel"
```

### All runs on a specific task:
```
config.task_name = "navigation"
```

### Combine filters:
```
config.model_name = "mlp" AND config.task_name = "navigation"
```
