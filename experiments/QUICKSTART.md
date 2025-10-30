# VMAS Model Survey - Quick Start

This directory contains scripts to run systematic model comparisons across VMAS tasks.

## 🚀 Quick Start

### 1. Test Single Run First
```bash
bash experiments/test_single_hydra.sh
```
This runs a quick 50-iteration test on NAVIGATION task with MLP to verify everything works.

### 2. Run Full Survey
```bash
bash experiments/run_vmas_survey_hydra_multirun.sh
```
This runs all model configurations across selected VMAS tasks (edit the script to add more tasks).

## 📊 Model Configurations

All models have **balanced parameter counts** (~13-20K params):

| Model | Architecture | Est. Params |
|-------|-------------|-------------|
| `mlp_balanced` | MLP [128, 128] | ~19K |
| `gnn_balanced_graphconv` | MLP[64] → GNN GraphConv → MLP[64] | ~14K |
| `gnn_balanced_gatv2` | MLP[64] → GNN GATv2 → MLP[64] | ~14K |
| `layers/deepsets` | DeepSets: Local[128] → Global[128] | ~36K |

## 🔧 Configuration Files

Model configs are in `benchmarl/conf/model/`:
- `mlp_balanced.yaml` - Baseline MLP
- `gnn_balanced_graphconv.yaml` - GNN with GraphConv
- `gnn_balanced_gatv2.yaml` - GNN with attention
- `layers/deepsets.yaml` - Permutation-invariant model

## 📝 Customization

Edit the scripts to customize:
- **Tasks**: Change `TASKS` variable (comma-separated list)
- **Models**: Change `MODELS` variable
- **Device**: Change `DEVICE` (options: `cuda:0`, `cuda:1`, `cpu`)
- **Iterations**: Change `MAX_ITERS` (default: 500, calculated as total_frames/frames_per_batch)
- **Seeds**: Change `SEED` (can use multiple: `1,2,3`)
- **WandB project**: Change to `benchmarl-2025-10-31`

## 📈 Viewing Results

- **WandB**: Check project `benchmarl-2025-10-31`
- **Local CSV**: `outputs/YYYY-MM-DD/HH-MM-SS/*/logs/`
- **JSON configs**: `outputs/YYYY-MM-DD/HH-MM-SS/*/config.json`

## 🎯 Example Commands

### Test different models on one task (CUDA:1, seed 1):
```bash
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation \
    model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2 \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.project_name=benchmarl-2025-10-31
```

### Test one model across multiple seeds (CUDA:0, seeds 2,3):
```bash
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation \
    model=gnn_balanced_graphconv \
    seed=2,3 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:0 \
    experiment.train_device=cuda:0 \
    experiment.project_name=benchmarl-2025-10-31
```

### CPU testing (seed 2, short run):
```bash
python benchmarl/run.py \
    algorithm=mappo \
    task=vmas/navigation \
    model=mlp_balanced \
    seed=2 \
    experiment.max_n_iters=50 \
    experiment.sampling_device=cpu \
    experiment.train_device=cpu \
    experiment.project_name=benchmarl-2025-10-31
```

## ⚙️ Parameter Count Check

Check estimated parameter counts:
```bash
python experiments/check_model_params_simple.py
```

## 🐛 Troubleshooting

- **CUDA errors**: Check device with `nvidia-smi`, verify `DEVICE` setting
- **Import errors**: Ensure `torch_geometric` is installed
- **Memory errors**: Reduce `experiment.on_policy_n_envs_per_worker`
- **Hydra errors**: Check YAML syntax in config files
