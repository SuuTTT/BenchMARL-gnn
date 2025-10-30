# VMAS Model Survey Experiment

This experiment systematically evaluates different neural network architectures across all VMAS environments in BenchMARL.

## Overview

- **Purpose**: Compare model architectures (MLP, GNN variants, DeepSets, Sequences) across VMAS tasks
- **Algorithm**: MAPPO (Multi-Agent PPO)
- **Environments**: All 26 VMAS tasks
- **Device**: CUDA:1 (second GPU)
- **Iterations**: 500 per task
- **Seeds**: 1 (single seed for initial survey)
- **Logging**: WandB project `benchmarl-10-30` + CSV

## Model Configurations

The following model configurations are tested:

### 1. **MLP** (Multi-Layer Perceptron)
- Architecture: [128, 128] hidden layers with Tanh activation
- Est. Parameters: ~19K (per agent, non-shared)
- Baseline feedforward architecture

### 2. **GNN GraphConv**
- Topology: Full graph (all agents connected)
- Self-loops: False
- Aggregation: Add
- Est. Parameters: ~40 (very parameter-efficient)
- Graph neural network using message passing

### 3. **GNN GATv2** (Graph Attention Network v2)
- Topology: Full graph
- Attention heads: 4 (concat=False)
- Est. Parameters: ~42
- GNN with attention mechanism for selective message passing

### 4. **DeepSets**
- Architecture: Local NN[128] → Aggregation(sum) → Global NN[128]
- Est. Parameters: ~36K
- Permutation-invariant architecture for set-based inputs

### 5. **MLP+GNN Sequence**
- Architecture: MLP[64] → GNN GraphConv
- Intermediate size: 64
- Est. Parameters: ~5.5K
- Hybrid architecture combining feedforward and graph processing

## Parameter Count Notes

**Important**: The parameter counts listed above are rough estimates for a typical VMAS task (e.g., NAVIGATION with 18 obs dims, 2 action dims). 

- GNN models are intentionally very parameter-efficient - this is a feature, not a bug!
- The comparison includes architectural efficiency as part of the evaluation
- Actual counts vary by task (observation/action dimensions)
- Parameters are NOT shared across agents (default BenchMARL setting)
- Both actor and critic use the same model architecture

## Running the Experiment

### Quick Start

```bash
# Run the full survey (all tasks × all models)
python experiments/vmas_model_survey.py
```

### Check Model Parameters (Optional)

```bash
# Get parameter count estimates
python experiments/check_model_params_simple.py
```

## VMAS Tasks Tested

All 26 VMAS tasks are included:

1. BALL_PASSAGE
2. BALL_TRAJECTORY  
3. BUZZ_WIRE
4. DISPERSION
5. DROPOUT
6. FLOCKING
7. FOOTBALL
8. GIVE_WAY
9. JOINT_PASSAGE
10. JOINT_PASSAGE_SIZE
11. MULTI_GIVE_WAY
12. NAVIGATION
13. PASSAGE
14. REVERSE_TRANSPORT
15. SAMPLING
16. SIMPLE_ADVERSARY
17. SIMPLE_CRYPTO
18. SIMPLE_PUSH
19. SIMPLE_REFERENCE
20. SIMPLE_SPEAKER_LISTENER
21. SIMPLE_SPREAD
22. SIMPLE_TAG
23. SIMPLE_WORLD_COMM
24. TRANSPORT
25. WHEEL
26. WIND_FLOCKING

## Experiment Configuration

- **Max Iterations**: 500
- **Frames per Batch**: 6000
- **Envs per Worker**: 10
- **Evaluation**: Every 120K frames (~20 iterations)
- **Evaluation Episodes**: 5
- **Checkpointing**: Disabled (for speed)
- **JSON Export**: Enabled

## Expected Runtime

- Total experiments: 26 tasks × 5 models = 130 runs
- Est. time per run: 5-15 minutes (depends on task complexity)
- **Total estimated time**: 10-30 hours

The experiments run sequentially to avoid GPU memory issues.

## Output Structure

Results are saved to:
```
outputs/
  └── YYYY-MM-DD/
      └── HH-MM-SS/
          ├── task_name/
          │   ├── algorithm_config/
          │   │   └── model_config/
          │   │       └── seed_1/
          │   │           ├── logs/
          │   │           ├── config.json
          │   │           └── ...
```

WandB runs are organized with:
- **Tags**: `model:<model_name>`, `vmas_survey`
- **Group**: `vmas_survey_<model_name>`
- **Project**: `benchmarl-10-30`

## Analyzing Results

After completion, you can:

1. **View on WandB**: Check the `benchmarl-10-30` project
2. **Compare models**: Group by model tag and compare across tasks
3. **CSV logs**: Parse the CSV files in `outputs/` for custom analysis

## Customization

To modify the experiment:

1. **Change device**: Edit `device` variable in `vmas_model_survey.py`
2. **Add/remove tasks**: Modify `VMAS_TASKS` list
3. **Adjust iterations**: Change `max_iterations` parameter
4. **Add model configs**: Extend `get_model_configs()` function
5. **Change algorithm**: Replace `MappoConfig` with another algorithm

## Notes

- The script uses Hydra for configuration management
- All model/task/algorithm configs loaded from YAML defaults
- Errors in individual tasks are caught and logged (experiment continues)
- GPU memory is managed per-task (environments closed after each run)
