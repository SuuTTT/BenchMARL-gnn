#!/usr/bin/env python3
"""
Test script to run a single VMAS task with one model configuration.
Use this to verify the setup before running the full survey.

Usage:
    python experiments/test_single_run.py
"""

import torch
import torch_geometric
from benchmarl.algorithms import MappoConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models import GnnConfig, MlpConfig, DeepsetsConfig, SequenceModelConfig

# Use second CUDA device (cuda:1)
device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Available CUDA devices: {torch.cuda.device_count()}")


def get_test_model_configs():
    """Model configurations with balanced parameters."""
    return {
        "mlp": MlpConfig(
            num_cells=[128, 128],
            activation_class=torch.nn.Tanh,
            layer_class=torch.nn.Linear,
        ),
        "gnn_graphconv": SequenceModelConfig(
            model_configs=[
                MlpConfig(
                    num_cells=[64],
                    activation_class=torch.nn.Tanh,
                    layer_class=torch.nn.Linear,
                ),
                GnnConfig(
                    topology="full",
                    self_loops=False,
                    gnn_class=torch_geometric.nn.conv.GraphConv,
                    gnn_kwargs={"aggr": "add"},
                ),
                MlpConfig(
                    num_cells=[64],
                    activation_class=torch.nn.Tanh,
                    layer_class=torch.nn.Linear,
                ),
            ],
            intermediate_sizes=[64, 64],
        ),
        "gnn_gatv2": SequenceModelConfig(
            model_configs=[
                MlpConfig(
                    num_cells=[64],
                    activation_class=torch.nn.Tanh,
                    layer_class=torch.nn.Linear,
                ),
                GnnConfig(
                    topology="full",
                    self_loops=False,
                    gnn_class=torch_geometric.nn.conv.GATv2Conv,
                    gnn_kwargs={"heads": 4, "concat": False},
                ),
                MlpConfig(
                    num_cells=[64],
                    activation_class=torch.nn.Tanh,
                    layer_class=torch.nn.Linear,
                ),
            ],
            intermediate_sizes=[64, 64],
        ),
        "deepsets": DeepsetsConfig(
            aggr="sum",
            out_features_local_nn=128,
            local_nn_num_cells=[128],
            local_nn_activation_class=torch.nn.Tanh,
            global_nn_num_cells=[128],
            global_nn_activation_class=torch.nn.Tanh,
        ),
    }


def main():
    """Run a single test experiment."""
    
    # Test configuration
    test_task = VmasTask.NAVIGATION  # Simple task for testing
    test_model = "mlp"  # Start with MLP
    test_iterations = 50  # Short run for testing
    
    print(f"\n{'='*80}")
    print(f"Single Run Test")
    print(f"{'='*80}")
    print(f"Task: {test_task.name}")
    print(f"Model: {test_model}")
    print(f"Iterations: {test_iterations}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load experiment config
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = test_iterations
    experiment_config.max_n_frames = None
    experiment_config.sampling_device = device
    experiment_config.train_device = device
    experiment_config.on_policy_collected_frames_per_batch = 6000
    experiment_config.on_policy_n_envs_per_worker = 10
    experiment_config.loggers = ["csv"]  # Disable wandb for testing
    experiment_config.evaluation = False  # Disable eval for speed
    experiment_config.checkpoint_interval = 0
    experiment_config.create_json = True
    
    # Load algorithm config
    algorithm_config = MappoConfig.get_from_yaml()
    
    # Get model config
    model_configs = get_test_model_configs()
    model_config = model_configs[test_model]
    
    # Load task
    task = test_task.get_from_yaml()
    
    print("Creating benchmark...")
    benchmark = Benchmark(
        algorithm_configs=[algorithm_config],
        tasks=[task],
        seeds={1},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=model_config,
    )
    
    print("Running experiment...")
    try:
        benchmark.run_sequential()
        print(f"\n{'='*80}")
        print(f"✓ Test run completed successfully!")
        print(f"{'='*80}\n")
        print("\nYou can now run the full survey with:")
        print("  python experiments/vmas_model_survey.py")
        print("\nOr test other models by editing test_model in this script:")
        print(f"  Available models: {list(model_configs.keys())}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ Test run failed!")
        print(f"{'='*80}\n")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
