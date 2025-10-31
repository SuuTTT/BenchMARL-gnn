#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
Experiment script to survey all VMAS environments with different model configurations.
This runs a systematic evaluation of model combinations across all VMAS tasks.

Usage:
    python experiments/vmas_model_survey.py
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

# All VMAS tasks to test
VMAS_TASKS = [
    
    VmasTask.BALL_PASSAGE,
    VmasTask.BALL_TRAJECTORY,
    VmasTask.BUZZ_WIRE,
    
    VmasTask.DISPERSION,
    VmasTask.DROPOUT,
    VmasTask.FLOCKING,
    VmasTask.FOOTBALL,
    VmasTask.GIVE_WAY,
    VmasTask.JOINT_PASSAGE,
    VmasTask.JOINT_PASSAGE_SIZE,
    VmasTask.MULTI_GIVE_WAY,
    VmasTask.NAVIGATION,
    VmasTask.PASSAGE,
    VmasTask.REVERSE_TRANSPORT,
    VmasTask.SAMPLING,
    VmasTask.SIMPLE_ADVERSARY,
    VmasTask.SIMPLE_CRYPTO,
    VmasTask.SIMPLE_PUSH,
    VmasTask.SIMPLE_REFERENCE,
    VmasTask.SIMPLE_SPEAKER_LISTENER,
    VmasTask.SIMPLE_SPREAD,
    VmasTask.SIMPLE_TAG,
    VmasTask.SIMPLE_WORLD_COMM,
    VmasTask.TRANSPORT,
    VmasTask.WHEEL,
    VmasTask.WIND_FLOCKING,
]


def get_model_configs():
    """
    Define model configurations to test with balanced parameter counts.
    
    All models are wrapped with MLP layers to ensure comparable parameter counts.
    Target: ~20-30K parameters per model for actor networks.
    
    Architecture patterns:
    - MLP: [128, 128] → ~19K params (baseline)
    - MLP+GNN+MLP: [64] → GNN → [64] → ~20K params (GNN with encoding/decoding MLPs)
    - DeepSets: Local[128] → Global[128] → ~36K params
    
    Note: Parameters are balanced across models to ensure fair comparison.
    """
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


def run_survey_experiment(
    tasks,
    model_configs,
    seed=1,
    max_iterations=500,
    wandb_project="benchmarl-10-30",
):
    """
    Run survey experiment across tasks and models.
    
    Args:
        tasks: List of VmasTask instances
        model_configs: Dict of model_name -> ModelConfig
        seed: Random seed
        max_iterations: Maximum training iterations
        wandb_project: WandB project name
    """
    
    # Load base experiment config from YAML
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Override specific parameters for our survey
    experiment_config.max_n_iters = max_iterations
    experiment_config.max_n_frames = None  # Use iterations instead
    experiment_config.sampling_device = device
    experiment_config.train_device = device
    experiment_config.on_policy_collected_frames_per_batch = 6000
    experiment_config.on_policy_n_envs_per_worker = 10
    experiment_config.loggers = ["wandb", "csv"]
    experiment_config.project_name = wandb_project
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 120_000  # Every ~20 iterations
    experiment_config.evaluation_episodes = 5
    experiment_config.checkpoint_interval = 0  # Disable for speed
    experiment_config.create_json = True
    
    # Load algorithm config from YAML (MAPPO)
    algorithm_config = MappoConfig.get_from_yaml()
    
    # Load all tasks from YAML
    task_instances = [task.get_from_yaml() for task in tasks]
    
    print(f"\n{'='*80}")
    print(f"VMAS Model Survey - Running Experiments")
    print(f"{'='*80}")
    print(f"Total tasks: {len(task_instances)}")
    print(f"Total models: {len(model_configs)}")
    print(f"Total experiments: {len(task_instances) * len(model_configs)}")
    print(f"Seed: {seed}")
    print(f"Max iterations: {max_iterations}")
    print(f"Device: {device}")
    print(f"WandB project: {wandb_project}")
    print(f"{'='*80}\n")
    
    # Run experiments for each model configuration
    for model_name, model_config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"Running with model: {model_name}")
        print(f"{'='*80}\n")
        
        # Update wandb kwargs for this model
        experiment_config.wandb_extra_kwargs = {
            "tags": [f"model:{model_name}", "vmas_survey"],
            "group": f"vmas_survey_{model_name}",
        }
        
        # Create benchmark
        benchmark = Benchmark(
            algorithm_configs=[algorithm_config],
            tasks=task_instances,
            seeds={seed},
            experiment_config=experiment_config,
            model_config=model_config,
            critic_model_config=model_config,  # Use same model for critic
        )
        
        # Run sequentially
        try:
            benchmark.run_sequential()
            print(f"\n✓ Completed all tasks for model: {model_name}")
        except Exception as e:
            print(f"\n✗ Error running model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Survey Complete!")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Run the survey
    run_survey_experiment(
        tasks=VMAS_TASKS,
        model_configs=model_configs,
        seed=1,
        max_iterations=500,
        wandb_project="benchmarl-10-30",
    )


if __name__ == "__main__":
    main()
