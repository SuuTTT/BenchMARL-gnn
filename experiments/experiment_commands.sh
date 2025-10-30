#!/bin/bash
# Common experiment commands for VMAS benchmarking
# Uncomment the command you want to run

# =============================================================================
# QUICK TESTS
# =============================================================================

# Test single task with MLP (50 iters, ~2-3 minutes)
# python benchmarl/run.py algorithm=mappo task=vmas/navigation model=mlp_balanced seed=1 experiment.max_n_iters=50 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# COMPARE MODELS ON SINGLE TASK
# =============================================================================

# Compare 4 models on navigation (CUDA:1, seed 1)
# python benchmarl/run.py -m algorithm=mappo task=vmas/navigation model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# Test recurrent models (LSTM, GRU) vs feedforward (CUDA:0, seed 1)
# python benchmarl/run.py -m algorithm=mappo task=vmas/transport model=mlp_balanced,layers/lstm,layers/gru seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:0 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# NAVIGATION & TRANSPORT TASKS
# =============================================================================

# Navigation tasks with GNN models
# python benchmarl/run.py -m algorithm=mappo task=vmas/navigation,vmas/transport,vmas/reverse_transport model=gnn_balanced_graphconv,gnn_balanced_gatv2 seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# Passage tasks (coordination required)
# python benchmarl/run.py -m algorithm=mappo task=vmas/passage,vmas/joint_passage,vmas/joint_passage_size model=mlp_balanced,layers/deepsets seed=1,2 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:0 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# COORDINATION TASKS (Flocking, Formation)
# =============================================================================

# Flocking tasks - GNNs should excel here
# python benchmarl/run.py -m algorithm=mappo task=vmas/flocking,vmas/wind_flocking,vmas/wheel model=gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# Balance and dispersion tasks
# python benchmarl/run.py -m algorithm=mappo task=vmas/balance,vmas/dispersion model=mlp_balanced,gnn_balanced_graphconv,layers/lstm seed=1,2,3 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:0 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# COMMUNICATION TASKS
# =============================================================================

# Simple communication benchmark
# python benchmarl/run.py -m algorithm=mappo task=vmas/simple_spread,vmas/simple_reference,vmas/simple_speaker_listener model=gnn_balanced_graphconv,layers/deepsets seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# All simple_* tasks
# python benchmarl/run.py -m algorithm=mappo task=vmas/simple_spread,vmas/simple_reference,vmas/simple_speaker_listener,vmas/simple_world_comm,vmas/simple_adversary,vmas/simple_tag,vmas/simple_crypto,vmas/simple_push model=mlp_balanced,gnn_balanced_graphconv seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:0 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# ADVERSARIAL TASKS
# =============================================================================

# Adversarial benchmark (competitive tasks)
# python benchmarl/run.py -m algorithm=mappo task=vmas/simple_adversary,vmas/simple_tag,vmas/simple_crypto model=mlp_balanced,gnn_balanced_gatv2 seed=1,2 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# Competition tasks with recurrent models
# python benchmarl/run.py -m algorithm=mappo task=vmas/football,vmas/simple_tag model=layers/lstm,layers/gru seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:0 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# PHYSICAL MANIPULATION
# =============================================================================

# Ball manipulation tasks
# python benchmarl/run.py -m algorithm=mappo task=vmas/ball_passage,vmas/ball_trajectory,vmas/buzz_wire model=mlp_balanced,gnn_balanced_graphconv seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.project_name=benchmarl-2025-10-31

# Dropout and give way (dynamic agents)
# python benchmarl/run.py -m algorithm=mappo task=vmas/dropout,vmas/give_way,vmas/multi_give_way model=layers/gru,gnn_balanced_graphconv seed=1,2 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:0 experiment.project_name=benchmarl-2025-10-31

# =============================================================================
# FULL BENCHMARKS
# =============================================================================

# Full VMAS benchmark - All 26 tasks with 4 models (104 runs)
# WARNING: This will take 10-30 hours!
# nohup python benchmarl/run.py -m algorithm=mappo task=vmas/ball_passage,vmas/ball_trajectory,vmas/buzz_wire,vmas/balance,vmas/dispersion,vmas/dropout,vmas/flocking,vmas/football,vmas/give_way,vmas/joint_passage,vmas/joint_passage_size,vmas/multi_give_way,vmas/navigation,vmas/passage,vmas/reverse_transport,vmas/sampling,vmas/simple_adversary,vmas/simple_crypto,vmas/simple_push,vmas/simple_reference,vmas/simple_speaker_listener,vmas/simple_spread,vmas/simple_tag,vmas/simple_world_comm,vmas/transport,vmas/wheel,vmas/wind_flocking model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.loggers="[wandb,csv]" experiment.project_name=benchmarl-2025-10-31 > full_benchmark.log 2>&1 &

# Subset benchmark - 10 representative tasks with 6 models (60 runs, ~10 hours)
nohup python benchmarl/run.py -m algorithm=mappo task=vmas/navigation,vmas/transport,vmas/flocking,vmas/balance,vmas/simple_spread,vmas/simple_adversary,vmas/football,vmas/sampling,vmas/wheel,vmas/joint_passage model=mlp_balanced,gnn_balanced_graphconv,gnn_balanced_gatv2,layers/deepsets,layers/lstm,layers/gru seed=1 experiment.max_n_iters=500 experiment.sampling_device=cuda:0 experiment.train_device=cuda:1 experiment.loggers="[wandb,csv]" experiment.project_name=benchmarl-2025-10-31 > subset_benchmark.log 2>&1 &

# =============================================================================
# MULTI-SEED RUNS (for statistical significance)
# =============================================================================

# Key tasks with 3 seeds each
# nohup python benchmarl/run.py -m algorithm=mappo task=vmas/navigation,vmas/transport,vmas/flocking,vmas/simple_spread model=mlp_balanced,gnn_balanced_graphconv seed=1,2,3 experiment.max_n_iters=500 experiment.sampling_device=cuda:1 experiment.train_device=cuda:1 experiment.loggers="[wandb,csv]" experiment.project_name=benchmarl-2025-10-31 > multiseed.log 2>&1 &

# =============================================================================
# USAGE NOTES
# =============================================================================
# - Uncomment a command to run it
# - Use 'nohup ... > logfile.log 2>&1 &' for background runs
# - Monitor with: tail -f logfile.log
# - Check WandB: wandb project benchmarl-2025-10-31
# - Results in: ./outputs/YYYY-MM-DD/HH-MM-SS/
# - Total experiments = tasks × models × seeds
# =============================================================================

echo "Please uncomment a command above to run an experiment"
echo "See SETUP_COMPLETE.md for more details"
