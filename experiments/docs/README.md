# BenchMARL Experiments

This directory contains experiment scripts for systematically evaluating models and algorithms in BenchMARL.

## VMAS Model Survey

### Overview

`vmas_model_survey.py` runs a comprehensive survey of model architectures across all VMAS environments.

### Models Tested

1. **MLP** - Multi-Layer Perceptron baseline (256-256 cells)
2. **GNN (GraphConv)** - Graph Neural Network with GraphConv layers
3. **GNN (GATv2)** - Graph Attention Network v2 with 4 attention heads
4. **DeepSets** - Permutation-invariant set aggregation model
5. **MLP+GNN Sequence** - Sequential composition of MLP and GNN layers

### VMAS Environments

Tests all 28 VMAS environments:
- Balance, Ball Passage, Ball Trajectory, Buzz Wire
- Discovery, Dispersion, Dropout
- Flocking, Football
- Give Way, Joint Passage, Joint Passage Size, Multi Give Way
- Navigation, Passage
- Reverse Transport, Sampling
- Simple Adversary, Simple Crypto, Simple Push, Simple Reference
- Simple Speaker Listener, Simple Spread, Simple Tag, Simple World Comm
- Transport, Wheel, Wind Flocking

### Configuration

- **Algorithm**: MAPPO (Multi-Agent PPO)
- **Seed**: 1
- **Iterations**: 500 per experiment
- **Device**: CUDA (if available) or CPU
- **Frames per batch**: 6000
- **Environments**: 10 parallel environments
- **Evaluation**: Every 20 iterations (5 episodes)
- **Logging**: WandB + CSV

### Usage

#### Python

```bash
# From BenchMARL root directory
python experiments/vmas_model_survey.py
```

#### Shell Script

```bash
# Make executable
chmod +x experiments/vmas_model_survey.sh

# Run
./experiments/vmas_model_survey.sh
```

### Results

Results are logged to:
- **WandB Project**: `benchmarl-10-30`
- **Local CSV**: `outputs/` directory (created by Hydra)
- **WandB Tags**: 
  - `model:<model_name>`
  - `vmas_survey`
- **WandB Groups**: `vmas_survey_<model_name>`

### Total Experiments

- **28 environments** × **5 models** = **140 experiments**
- Estimated time: ~2-4 hours on GPU (depends on hardware)

### Customization

Edit the script to customize:

```python
# Change models tested
model_configs = get_model_configs()  # Edit this function

# Change experiment parameters
run_survey_experiment(
    tasks=VMAS_TASKS,
    model_configs=model_configs,
    seed=1,                    # Change seed
    max_iterations=500,        # Change training duration
    wandb_project="benchmarl-10-30",  # Change project name
)
```

### Notes

- Experiments run sequentially to avoid resource conflicts
- Failed experiments are logged but don't stop the survey
- Results include training curves, evaluation metrics, and final performance
- All experiments use the same hyperparameters for fair comparison

## Adding New Experiments

To add a new experiment script:

1. Create a new Python file in this directory
2. Use existing scripts as templates
3. Follow BenchMARL patterns (use Hydra configs where possible)
4. Add documentation to this README
5. Create a corresponding shell script for easy execution
