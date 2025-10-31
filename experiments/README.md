# BenchMARL Experiments# BenchMARL Experiments



This directory contains all experiment scripts, utilities, and documentation for running multi-agent reinforcement learning experiments using BenchMARL with VMAS environments.This directory contains experiment scripts for systematically evaluating models and algorithms in BenchMARL.



## 📁 Directory Structure## VMAS Model Survey



```### Overview

experiments/

├── README.md                    # This file - overview of experiments structure`vmas_model_survey.py` runs a comprehensive survey of model architectures across all VMAS environments.

├── docs/                        # All documentation files

│   ├── QUICKREF.md             # Quick reference guide### Models Tested

│   ├── QUICKSTART.md           # Quick start tutorial

│   ├── README.md               # Main documentation (legacy)1. **MLP** - Multi-Layer Perceptron baseline (256-256 cells)

│   ├── README_VMAS_SURVEY.md   # VMAS survey documentation2. **GNN (GraphConv)** - Graph Neural Network with GraphConv layers

│   ├── SETUP_COMPLETE.md       # Complete setup reference3. **GNN (GATv2)** - Graph Attention Network v2 with 4 attention heads

│   ├── VMAS_Experiment_Design.md    # Experiment design and hypotheses4. **DeepSets** - Permutation-invariant set aggregation model

│   ├── TODO_1-3_IMPLEMENTATION.md   # Implementation summary of TODO 1-35. **MLP+GNN Sequence** - Sequential composition of MLP and GNN layers

│   └── COMBINATION_CONFIG_FIX.md    # Fix documentation for combination configs

├── scripts/                     # All executable scripts### VMAS Environments

│   ├── single_tests/           # Single experiment test scripts

│   │   ├── test_single_hydra.sh     # Quick single-task test (recommended)Tests all 28 VMAS environments:

│   │   └── test_single_run.py       # Python-based single test (deprecated)- Balance, Ball Passage, Ball Trajectory, Buzz Wire

│   ├── surveys/                # Full survey scripts- Discovery, Dispersion, Dropout

│   │   ├── run_vmas_survey_hydra_multirun.sh  # Hydra multi-run survey (recommended)- Flocking, Football

│   │   ├── run_vmas_survey_hydra.sh           # Hydra survey (older version)- Give Way, Joint Passage, Joint Passage Size, Multi Give Way

│   │   ├── run_vmas_survey.sh                 # Shell-based survey (deprecated)- Navigation, Passage

│   │   ├── experiment_commands.sh             # Reference commands for all tasks- Reverse Transport, Sampling

│   │   ├── vmas_model_survey.py               # Python survey (deprecated)- Simple Adversary, Simple Crypto, Simple Push, Simple Reference

│   │   └── vmas_model_survey.sh               # Survey wrapper (deprecated)- Simple Speaker Listener, Simple Spread, Simple Tag, Simple World Comm

│   └── grouped/                # Hypothesis-driven grouped experiments- Transport, Wheel, Wind Flocking

│       ├── README.md                          # Grouped experiments documentation

│       ├── run_all_grouped_experiments.sh     # Master script for all groups### Configuration

│       ├── scalability_experiments.sh         # Scalability hypothesis tests

│       ├── coordination_experiments.sh        # Coordination complexity tests- **Algorithm**: MAPPO (Multi-Agent PPO)

│       ├── task_type_experiments.sh           # Task type categorization tests- **Seed**: 1

│       ├── simple_baseline_experiments.sh     # Simple baseline comparison- **Iterations**: 500 per experiment

│       └── combination_experiments.sh         # Combination architecture tests- **Device**: CUDA (if available) or CPU

└── utils/                       # Utility scripts- **Frames per batch**: 6000

    ├── check_model_params.py           # Environment-based param checker- **Environments**: 10 parallel environments

    └── check_model_params_simple.py    # Analytical parameter estimator- **Evaluation**: Every 20 iterations (5 episodes)

```- **Logging**: WandB + CSV



## 🚀 Quick Start### Usage



### Single Test Run#### Python

Test a single configuration quickly:

```bash```bash

bash scripts/single_tests/test_single_hydra.sh# From BenchMARL root directory

```python experiments/vmas_model_survey.py

```

### Run Grouped Experiments

Run hypothesis-driven experiment groups:#### Shell Script

```bash

# All groups```bash

bash scripts/grouped/run_all_grouped_experiments.sh cuda:1 500# Make executable

chmod +x experiments/vmas_model_survey.sh

# Individual groups

bash scripts/grouped/scalability_experiments.sh cuda:1 500# Run

bash scripts/grouped/coordination_experiments.sh cuda:1 500./experiments/vmas_model_survey.sh

bash scripts/grouped/task_type_experiments.sh cuda:1 500```

```

### Results

### Full Survey

Run comprehensive model survey across all VMAS tasks:Results are logged to:

```bash- **WandB Project**: `benchmarl-10-30`

bash scripts/surveys/run_vmas_survey_hydra_multirun.sh- **Local CSV**: `outputs/` directory (created by Hydra)

```- **WandB Tags**: 

  - `model:<model_name>`

## 📊 Experiment Categories  - `vmas_survey`

- **WandB Groups**: `vmas_survey_<model_name>`

### 1. Single Tests (`scripts/single_tests/`)

Quick validation runs for testing configurations.### Total Experiments



**Use when:**- **28 environments** × **5 models** = **140 experiments**

- Testing new model configurations- Estimated time: ~2-4 hours on GPU (depends on hardware)

- Verifying environment setup

- Quick debugging### Customization



**Key Script:** `test_single_hydra.sh`Edit the script to customize:

```bash

bash scripts/single_tests/test_single_hydra.sh```python

# Default: 50 iterations, navigation task, mlp_balanced model, cuda:1# Change models tested

```model_configs = get_model_configs()  # Edit this function



### 2. Full Surveys (`scripts/surveys/`)# Change experiment parameters

Comprehensive experiments across all tasks and models.run_survey_experiment(

    tasks=VMAS_TASKS,

**Use when:**    model_configs=model_configs,

- Running exhaustive comparisons    seed=1,                    # Change seed

- Collecting data for all VMAS tasks    max_iterations=500,        # Change training duration

- Benchmarking new algorithms    wandb_project="benchmarl-10-30",  # Change project name

)

**Key Script:** `run_vmas_survey_hydra_multirun.sh````

```bash

bash scripts/surveys/run_vmas_survey_hydra_multirun.sh### Notes

# Default: 500 iterations, 26 VMAS tasks, 4 models, cuda:1

```- Experiments run sequentially to avoid resource conflicts

- Failed experiments are logged but don't stop the survey

### 3. Grouped Experiments (`scripts/grouped/`)- Results include training curves, evaluation metrics, and final performance

Hypothesis-driven experiments organized by task characteristics.- All experiments use the same hyperparameters for fair comparison



**Use when:**## Adding New Experiments

- Testing specific hypotheses

- Comparing models on task categoriesTo add a new experiment script:

- Focused analysis on coordination/scalability

1. Create a new Python file in this directory

**Key Scripts:**2. Use existing scripts as templates

- `scalability_experiments.sh` - Variable agent count performance3. Follow BenchMARL patterns (use Hydra configs where possible)

- `coordination_experiments.sh` - Simple/Medium/Complex coordination4. Add documentation to this README

- `task_type_experiments.sh` - Navigation/Transport/Flocking/Competitive5. Create a corresponding shell script for easy execution

- `simple_baseline_experiments.sh` - Baseline on simple tasks
- `combination_experiments.sh` - Hybrid architecture evaluation

See `scripts/grouped/README.md` for detailed hypotheses and task groupings.

## 🛠️ Utilities (`utils/`)

### Parameter Estimation
Check model parameter counts before running experiments:

```bash
# Analytical estimation (recommended)
python utils/check_model_params_simple.py

# Environment-based estimation
python utils/check_model_params.py
```

## 📖 Documentation (`docs/`)

### Quick References
- **QUICKREF.md** - One-page reference with all common commands
- **QUICKSTART.md** - Step-by-step tutorial for beginners

### Complete Guides
- **SETUP_COMPLETE.md** - Full reference with all commands and options
- **README_VMAS_SURVEY.md** - Original survey documentation
- **VMAS_Experiment_Design.md** - Experiment design rationale and hypotheses

### Implementation Details
- **TODO_1-3_IMPLEMENTATION.md** - Summary of device consistency, naming, and grouping fixes
- **COMBINATION_CONFIG_FIX.md** - Fix documentation for combination model configs

## 🔧 Configuration

All experiments use standardized configuration:
- **Device**: cuda:0, cuda:1, or cpu (default: cuda:1)
- **Algorithm**: MAPPO (Multi-Agent PPO)
- **Iterations**: Configurable (default: 500 for full runs, 50 for tests)
- **Logging**: WandB + CSV (project: benchmarl-2025-10-31)
- **Naming**: `yyyy-mm-dd-hostname-devicename-task-algo`

### Device Configuration
All scripts now synchronize device settings:
```bash
experiment.sampling_device=$DEVICE
experiment.train_device=$DEVICE
experiment.buffer_device=$DEVICE
```

## 📈 Available Models

### Baseline Models
- `mlp_balanced` - MLP [128, 128] (~19K params)
- `layers/deepsets` - DeepSets architecture

### GNN Models (Balanced)
- `gnn_balanced_graphconv` - MLP[64]→GraphConv GNN→MLP[64] (~14K params)
- `gnn_balanced_gatv2` - MLP[64]→GATv2 GNN→MLP[64] (~14K params)

### Combination Models
- `gnn_lstm_combo` - MLP→GNN→LSTM→MLP
- `deepsets_gnn_combo` - DeepSets→GNN hybrid
- `gru_deepsets_combo` - GRU→DeepSets→MLP
- `mlp_gnn_gru_combo` - MLP→GNN→GRU→MLP

### RNN Models
- `layers/lstm` - LSTM with MLP
- `layers/gru` - GRU with MLP

## 🌍 VMAS Environments (26 Tasks)

Organized by category:

**Scalability:**
- Low-Coordination: `dispersion`, `navigation`
- High-Coordination: `flocking`, `discovery`, `wind_flocking`

**Coordination Complexity:**
- Simple: `navigation`, `give_way`
- Medium: `balance`, `wheel`, `passage`
- Complex: `transport`, `reverse_transport`, `football`, `joint_passage`

**Task Types:**
- Navigation: `navigation`, `dispersion`, `discovery`
- Transport: `transport`, `reverse_transport`, `balance`, `wheel`, `ball_passage`
- Flocking: `flocking`, `wind_flocking`
- Competitive: `football`

See `docs/VMAS_Experiment_Design.md` for complete categorization.

## 🎯 Common Usage Patterns

### Quick Test Before Full Run
```bash
# Test on cuda:1 with 50 iterations
bash scripts/single_tests/test_single_hydra.sh
```

### Run Hypothesis-Driven Experiments
```bash
# Test scalability hypothesis
bash scripts/grouped/scalability_experiments.sh cuda:1 500

# Test coordination hypothesis
bash scripts/grouped/coordination_experiments.sh cuda:1 500
```

### Background Execution
```bash
# Run in background with logging
nohup bash scripts/grouped/scalability_experiments.sh cuda:1 500 \
  > logs/scalability_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor progress
tail -f logs/scalability_*.log
```

### Multi-Device Parallel Execution
```bash
# Terminal 1 - cuda:0
bash scripts/grouped/scalability_experiments.sh cuda:0 500 &

# Terminal 2 - cuda:1
bash scripts/grouped/coordination_experiments.sh cuda:1 500 &
```

## 📊 Monitoring Results

### WandB Dashboard
All experiments log to WandB project: `benchmarl-2025-10-31`
```
https://wandb.ai/your-username/benchmarl-2025-10-31
```

### Local Logs
CSV logs and checkpoints stored in:
```
outputs/yyyy-mm-dd-hostname-devicename-task-algo/
```

### Check Running Experiments
```bash
# See running experiments
ps aux | grep python

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check log files
ls -lth outputs/
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce `on_policy_n_envs_per_worker` (default: 10)
- Use smaller batch size
- Switch to CPU: `bash script.sh cpu 500`

**2. Import errors**
- Activate correct environment: `conda activate benchmarl-dev`
- Install dependencies: `pip install -e .`

**3. WandB login required**
- `wandb login`
- Or disable: change `loggers="[csv]"`

**4. Combination configs not found**
- Configs should be in `benchmarl/conf/model/`
- See `docs/COMBINATION_CONFIG_FIX.md`

### Getting Help

1. Check `docs/QUICKREF.md` for quick reference
2. See `docs/SETUP_COMPLETE.md` for complete documentation
3. Review hypothesis and design in `docs/VMAS_Experiment_Design.md`

## 📝 Next Steps

After running experiments:

1. **Analyze Results** - Compare performance across task groups in WandB
2. **Identify Patterns** - Which models excel at which task types?
3. **Refine Hypotheses** - Use insights to design new experiments
4. **Implement Methods** - Build new GNN-RL algorithms based on findings

See CHANGELOG.md for TODO items 4-5:
- Reimplement existing GNN-RL methods (InforMARL, GCBF+, X-MAGE)
- Design new GNN algorithm

---

**Last Updated:** October 31, 2025  
**Project:** BenchMARL-gnn  
**WandB Project:** benchmarl-2025-10-31
