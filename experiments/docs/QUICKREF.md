# Quick Reference: Grouped Experiments

## Overview
All TODO items 1-3 from CHANGELOG have been completed. You now have a structured experiment framework.

## What Changed

### ✅ Device Consistency (TODO 1)
All scripts now set:
- `experiment.sampling_device=$DEVICE`
- `experiment.train_device=$DEVICE`
- `experiment.buffer_device=$DEVICE`

### ✅ Standardized Naming (TODO 2)
Experiments now named: `yyyy-mm-dd-hostname-devicename-task-algo`

Example: `2025-10-31-c225-cuda1-navigation-mappo`

### ✅ Grouped Experiments (TODO 3)
5 experiment groups in `experiments/scripts/grouped/`:
1. Scalability
2. Coordination Complexity
3. Task Types
4. Simple Baseline
5. Combination Architectures

## Quick Start

### Run a Single Experiment Group
```bash
# Scalability tests
bash experiments/scripts/grouped/scalability_experiments.sh cuda:1 500

# Coordination complexity tests
bash experiments/scripts/grouped/coordination_experiments.sh cuda:1 500

# Task type tests
bash experiments/scripts/grouped/task_type_experiments.sh cuda:1 500

# Simple baseline tests
bash experiments/scripts/grouped/simple_baseline_experiments.sh cuda:1 500

# Combination architecture tests
bash experiments/scripts/grouped/combination_experiments.sh cuda:1 500
```

### Run All Grouped Experiments
```bash
# Run all 5 experiment groups sequentially
bash experiments/scripts/grouped/run_all_grouped_experiments.sh cuda:1 500

# Or with different device/iterations
bash experiments/run_all_grouped_experiments.sh cuda:0 1000
```

## Experiment Details

| Group | Tasks | Models | Seeds | Total Runs |
|-------|-------|--------|-------|------------|
| Scalability | 5 | 4 | 3 | 60 |
| Coordination | 9 | 4 | 3 | 108 |
| Task Types | 10 | 4 | 3 | 120 |
| Simple Baseline | 2 | 4 | 5 | 40 |
| Combinations | 8 | 4 | 3 | 96 |
| **TOTAL** | | | | **424** |

## Device Options

```bash
```

## Device Options

```bash
# Use CUDA device 0
bash experiments/scripts/grouped/scalability_experiments.sh cuda:0 500

# Use CUDA device 1 (default)
bash experiments/scripts/grouped/scalability_experiments.sh cuda:1 500

# Use CPU
bash experiments/scripts/grouped/scalability_experiments.sh cpu 500
```

## Monitoring

All experiments log to WandB project: `benchmarl-2025-10-31`

Check progress:
```bash
# View WandB dashboard
https://wandb.ai/your-username/benchmarl-2025-10-31

# Or check local CSV logs
ls -la outputs/
```

## Task Groupings

### Scalability
- **Low-Coord**: dispersion, navigation
- **High-Coord**: flocking, discovery, wind_flocking

### Coordination Complexity
- **Simple**: navigation, give_way
- **Medium**: balance, wheel, passage
- **Complex**: transport, reverse_transport, football, joint_passage

### Task Types
- **Navigation**: navigation, dispersion, discovery
- **Transport**: transport, reverse_transport, balance, wheel, ball_passage
- **Flocking**: flocking, wind_flocking
- **Competitive**: football

## Next Steps After Running Experiments

1. **Analyze Results**
   - Compare model performance across task groups
   - Identify patterns in WandB
   - Generate performance plots

2. **Draw Conclusions**
   - Which models excel at which tasks?
   - How do models scale with agent count?
   - Are GNNs better for coordination tasks?

3. **Use Insights**
   - Inform TODO 4: Reimplement GNN-RL methods
   - Inform TODO 5: Design new GNN algorithm

## Files Modified

### Core Changes
- `benchmarl/experiment/experiment.py` - Standardized naming

### Scripts Updated (added buffer_device)
- `experiments/scripts/single_tests/test_single_hydra.sh`
- `experiments/scripts/surveys/run_vmas_survey_hydra_multirun.sh`
- `dev/combination/test_single_combination.sh`
- `dev/combination/test_all_combinations.sh`
- `dev/combination/compare_with_baselines.sh`
- `dev/combination/run_extended_benchmark.sh`

### New Files Created
- `experiments/scripts/grouped/scalability_experiments.sh`
- `experiments/scripts/grouped/coordination_experiments.sh`
- `experiments/scripts/grouped/task_type_experiments.sh`
- `experiments/scripts/grouped/simple_baseline_experiments.sh`
- `experiments/scripts/grouped/combination_experiments.sh`
- `experiments/scripts/grouped/README.md`
- `experiments/scripts/grouped/run_all_grouped_experiments.sh`
- `experiments/docs/TODO_1-3_IMPLEMENTATION.md`

## Documentation

- `CHANGELOG.md` - Updated with completed TODOs
- `experiments/scripts/grouped/README.md` - Detailed experiment group docs
- `experiments/docs/TODO_1-3_IMPLEMENTATION.md` - Implementation summary
- `experiments/docs/QUICKREF.md` - This file

## Troubleshooting

### If experiments fail
Check:
1. CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Correct device specified: `cuda:0` vs `cuda:1`
3. WandB login: `wandb login`
4. Sufficient disk space: `df -h`

### If naming doesn't work
The naming change is in `benchmarl/experiment/experiment.py`. If you get old-style names:
1. Make sure you're using the updated code
2. Check that imports work: `import socket; from datetime import datetime`
3. Verify device string format (should be like "cuda:1")

### If device consistency issues persist
All scripts should have three device parameters. Check with:
```bash
grep -A2 "sampling_device" experiments/scripts/grouped/*.sh
```

Should see `sampling_device`, `train_device`, and `buffer_device` all set to `$DEVICE`.
