# ✅ VMAS Model Survey - Ready to Run

## Setup Complete!

All configurations are working and ready for multi-task/multi-model experiments using Hydra.

---

## 🎯 Quick Commands

### Test First (Recommended)
```bash
# Quick 50-iteration test on one task
bash experiments/test_single_hydra.sh
```

### Run Full Survey
```bash
# Multi-run across all configured models and tasks
bash experiments/run_vmas_survey_hydra_multirun.sh
```

### Custom Hydra Multi-Run
```bash
# Example: Test 3 models on 3 tasks (CUDA:1, seed 1)
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation,vmas/transport,vmas/sampling \
    model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2 \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.project_name=benchmarl-2025-10-31

# Run on CUDA:0 with different seeds
nohup python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation,vmas/transport,vmas/sampling \
    model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2 \
    seed=3 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:0 \
    experiment.train_device=cuda:0 \
    experiment.project_name=benchmarl-2025-10-31 > run_log.txt 

# Run on CPU (slower, for testing)
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation \
    model=mlp_balanced \
    seed=2 \
    experiment.max_n_iters=50 \
    experiment.sampling_device=cpu \
    experiment.train_device=cpu \
    experiment.project_name=benchmarl-2025-10-31
```

---

## 📊 Model Configurations (Balanced Parameters)

| Config File | Architecture | Est. Params | Description |
|------------|--------------|-------------|-------------|
| `mlp_balanced` | MLP[128, 128] | ~19K | Baseline feedforward |
| `gnn_balanced_graphconv` | MLP[64] → GNN GraphConv → MLP[64] | ~14K | GNN with message passing |
| `gnn_balanced_gatv2` | MLP[64] → GNN GATv2 → MLP[64] | ~14K | GNN with attention |
| `layers/deepsets` | Local[128] → Agg → Global[128] | ~36K | Permutation-invariant |

**All GNN models are wrapped with MLPs** to balance parameter counts for fair comparison.

---

## 🌍 Available VMAS Environments

### Navigation & Transport
- `vmas/navigation` - Navigate to goals
- `vmas/transport` - Cooperative transport
- `vmas/reverse_transport` - Transport in reverse
- `vmas/passage` - Navigate through passage
- `vmas/joint_passage` - Coordinated passage
- `vmas/joint_passage_size` - Variable size passage
- `vmas/multi_give_way` - Multi-agent give way

### Coordination & Formation
- `vmas/flocking` - Flocking behavior
- `vmas/wind_flocking` - Flocking with wind
- `vmas/wheel` - Wheel formation
- `vmas/balance` - Balance coordination
- `vmas/dispersion` - Spread out

### Communication & Cooperation
- `vmas/simple_spread` - Spread to landmarks
- `vmas/simple_reference` - Reference communication
- `vmas/simple_speaker_listener` - Speaker-listener
- `vmas/simple_world_comm` - World communication
- `vmas/sampling` - Sampling task

### Adversarial & Competition
- `vmas/simple_adversary` - Adversarial agents
- `vmas/simple_tag` - Tag game
- `vmas/simple_crypto` - Cryptography
- `vmas/simple_push` - Pushing task
- `vmas/football` - Football game

### Physical Manipulation
- `vmas/ball_passage` - Ball through passage
- `vmas/ball_trajectory` - Ball trajectory
- `vmas/buzz_wire` - Buzz wire game
- `vmas/dropout` - Agent dropout
- `vmas/give_way` - Give way task

---

## 🧠 Available Models

### Feedforward Models
- `mlp_balanced` - MLP [128, 128] (~19K params)
- `layers/mlp` - Standard MLP [256, 256]
- `layers/cnn` - Convolutional layers (for image inputs)

### Graph Neural Networks
- `gnn_balanced_graphconv` - MLP→GNN GraphConv→MLP (~14K params)
- `gnn_balanced_gatv2` - MLP→GNN GATv2→MLP (~14K params)
- `layers/gnn` - Standard GNN (single layer)

### Recurrent Models (for temporal sequences)
- `layers/lstm` - LSTM network (captures long-term dependencies)
- `layers/gru` - GRU network (lighter than LSTM)

### Set-based Models
- `layers/deepsets` - DeepSets (~36K params, permutation-invariant)

### Sequence Models (combine multiple layers)
- Create custom sequences in `benchmarl/conf/model/` following `gnn_balanced_*.yaml` pattern

---

## 🔧 Configuration Files Created

### Model Configs (`benchmarl/conf/model/`)
- ✅ `mlp_balanced.yaml`
- ✅ `gnn_balanced_graphconv.yaml`
- ✅ `gnn_balanced_gatv2.yaml`

### Scripts (`experiments/`)
- ✅ `test_single_hydra.sh` - Quick test
- ✅ `run_vmas_survey_hydra_multirun.sh` - Full survey
- ✅ `check_model_params_simple.py` - Parameter estimator

---

## ⚙️ Key Settings

- **Device**: `cuda:1` (second GPU, can use `cuda:0` or `cpu`)
- **Algorithm**: MAPPO
- **Iterations**: 500 per task (= total_frames / frames_per_batch)
  - 500 iters × 6000 frames/batch = 3M total frames
- **Seed**: 1 (can use multiple: `seed=1,2,3`)
- **WandB Project**: `benchmarl-2025-10-31`
- **Logging**: WandB + CSV

---

## 📝 Examples

### Full 3×3 Grid (9 experiments)
```bash
# 3 models × 3 tasks on CUDA:1, seed 1
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation,vmas/transport,vmas/sampling \
    model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2 \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.loggers="[wandb,csv]" \
    experiment.project_name=benchmarl-2025-10-31
```

### Multiple Seeds on CUDA:0
```bash
# Same model/task with seeds 2,3 on CUDA:0
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

### Different VMAS Environments
```bash
# Test coordination tasks (flocking, formation)
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/flocking,vmas/wind_flocking,vmas/wheel \
    model=mlp_balanced,gnn_balanced_graphconv \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.project_name=benchmarl-2025-10-31

# Test multi-agent communication tasks
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/simple_spread,vmas/simple_reference,vmas/simple_speaker_listener \
    model=gnn_balanced_graphconv,layers/deepsets \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:0 \
    experiment.train_device=cuda:0 \
    experiment.project_name=benchmarl-2025-10-31

# Test adversarial tasks
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/simple_adversary,vmas/simple_tag,vmas/simple_crypto \
    model=mlp_balanced,gnn_balanced_gatv2 \
    seed=1,2 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.project_name=benchmarl-2025-10-31
```

### Recurrent Models (LSTM/GRU)
```bash
# Test LSTM on navigation tasks (captures temporal dependencies)
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/navigation,vmas/transport,vmas/reverse_transport \
    model=layers/lstm \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.project_name=benchmarl-2025-10-31

# Test GRU on dynamic environments
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/football,vmas/dropout,vmas/give_way \
    model=layers/gru,layers/lstm \
    seed=1,2,3 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:0 \
    experiment.train_device=cuda:0 \
    experiment.project_name=benchmarl-2025-10-31

# Compare MLP vs LSTM vs GNN
python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/balance,vmas/joint_passage \
    model=mlp_balanced,layers/lstm,layers/gru,gnn_balanced_graphconv \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.project_name=benchmarl-2025-10-31
```

### All VMAS Environments (Full Benchmark)
```bash
# Run complete VMAS benchmark with all tasks
nohup python benchmarl/run.py -m \
    algorithm=mappo \
    task=vmas/ball_passage,vmas/ball_trajectory,vmas/buzz_wire,vmas/balance,vmas/dispersion,vmas/dropout,vmas/flocking,vmas/football,vmas/give_way,vmas/joint_passage,vmas/joint_passage_size,vmas/multi_give_way,vmas/navigation,vmas/passage,vmas/reverse_transport,vmas/sampling,vmas/simple_adversary,vmas/simple_crypto,vmas/simple_push,vmas/simple_reference,vmas/simple_speaker_listener,vmas/simple_spread,vmas/simple_tag,vmas/simple_world_comm,vmas/transport,vmas/wheel,vmas/wind_flocking \
    model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets \
    seed=1 \
    experiment.max_n_iters=500 \
    experiment.sampling_device=cuda:1 \
    experiment.train_device=cuda:1 \
    experiment.loggers="[wandb,csv]" \
    experiment.project_name=benchmarl-2025-10-31 \
    > full_benchmark_log.txt 2>&1 &

# Total runs: 26 tasks × 4 models = 104 experiments
```

### CPU Testing (Slower)
```bash
# Quick test on CPU with seed 2
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

Hydra will automatically create separate runs for each combination!

---

## 🎨 Customization

### Add More Tasks
Edit `run_vmas_survey_hydra_multirun.sh` and add to the `TASKS` variable:
```bash
TASKS="vmas/navigation,vmas/transport,vmas/wheel,vmas/balance"
```

### Change Iterations
```bash
experiment.max_n_iters=1000
```

### Different Seeds
```bash
seed=1,2,3  # Runs each config with 3 seeds (multiplies total runs by 3)
```

### Different Devices
```bash
# CUDA device 0
experiment.sampling_device=cuda:0
experiment.train_device=cuda:0

# CUDA device 1
experiment.sampling_device=cuda:1
experiment.train_device=cuda:1

# CPU (slower, for debugging)
experiment.sampling_device=cpu
experiment.train_device=cpu
```

### WandB Settings
```bash
experiment.project_name=benchmarl-2025-10-31
experiment.wandb_extra_kwargs.tags="[my_experiment,baseline]"
```

---

## 📈 Parameter Count Verification

Check estimated parameters:
```bash
python experiments/check_model_params_simple.py
```

Output:
```
Model                              Est. Params
------------------------------ ---------------
mlp                                     19,202
gnn_graphconv                           13,826
gnn_gatv2                               13,954
deepsets                                35,714
```

---

## 🐛 Verified Working

✅ Configs load successfully  
✅ CUDA:1 device selection works  
✅ MLP baseline runs  
✅ GNN sequence models configured  
✅ Hydra multi-run syntax correct  

---

## 📦 Files Overview

```
benchmarl/conf/model/
├── mlp_balanced.yaml              # MLP [128, 128]
├── gnn_balanced_graphconv.yaml    # MLP→GNN→MLP (GraphConv)
├── gnn_balanced_gatv2.yaml        # MLP→GNN→MLP (GATv2)
└── layers/
    ├── mlp.yaml
    ├── gnn.yaml
    └── deepsets.yaml

experiments/
├── test_single_hydra.sh           # Quick test script
├── run_vmas_survey_hydra_multirun.sh  # Full survey script
├── check_model_params_simple.py   # Parameter counter
├── QUICKSTART.md                  # Quick reference
└── README_VMAS_SURVEY.md          # Detailed docs
```

---

## 🚀 Next Steps

1. **Test**: Run `bash experiments/test_single_hydra.sh`
2. **Verify**: Check output in `./outputs/`
3. **Run Survey**: Execute `bash experiments/run_vmas_survey_hydra_multirun.sh`
4. **Monitor**: Check WandB project `benchmarl-10-30`

---

**Ready to go!** 🎉
