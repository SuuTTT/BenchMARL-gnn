#!/usr/bin/env python3
"""
Utility script to check and compare parameter counts across different model configurations.
This helps ensure models have comparable capacity before running experiments.

Usage:
    python experiments/check_model_params.py
"""

import torch
import torch_geometric
from benchmarl.environments import VmasTask
from benchmarl.models import GnnConfig, MlpConfig, DeepsetsConfig, SequenceModelConfig


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_configs():
    """Check parameter counts for each model configuration."""
    
    # Use a sample task to get input/output specs
    task = VmasTask.NAVIGATION.get_from_yaml()
    
    # Create environment using get_env_fun
    env_fun = task.get_env_fun(
        num_envs=1,
        continuous_actions=True,
        seed=0,
        device="cpu"
    )
    env = env_fun()
    
    # Get number of agents from group_map
    group_map = task.group_map(env)
    agent_group = list(group_map.keys())[0]  # Get first agent group name (usually "agents")
    n_agents = len(list(group_map.values())[0])  # Get first group's agent count
    
    # Get observation and action specs for the agent group (they are Composite specs)
    observation_spec = env.observation_spec[agent_group]
    action_spec = env.action_spec[agent_group]
    
    print(f"\n{'='*80}")
    print(f"Model Parameter Count Comparison")
    print(f"{'='*80}")
    print(f"Task: NAVIGATION (sample task)")
    print(f"Number of agents: {n_agents}")
    print(f"Agent group: {agent_group}")
    print(f"Observation spec keys: {list(observation_spec.keys())}")
    print(f"Action spec keys: {list(action_spec.keys())}")
    print(f"{'='*80}\n")
    
    # Model configurations (same as in vmas_model_survey.py)
    model_configs = {
        "mlp": MlpConfig(
            num_cells=[128, 128],
            activation_class=torch.nn.Tanh,
            layer_class=torch.nn.Linear,
        ),
        "gnn_graphconv": GnnConfig(
            topology="full",
            self_loops=False,
            gnn_class=torch_geometric.nn.conv.GraphConv,
            gnn_kwargs={"aggr": "add"},
        ),
        "gnn_gatv2": GnnConfig(
            topology="full",
            self_loops=False,
            gnn_class=torch_geometric.nn.conv.GATv2Conv,
            gnn_kwargs={"heads": 4, "concat": False},
        ),
        "deepsets": DeepsetsConfig(
            aggr="sum",
            out_features_local_nn=128,
            local_nn_num_cells=[128],
            local_nn_activation_class=torch.nn.Tanh,
            global_nn_num_cells=[128],
            global_nn_activation_class=torch.nn.Tanh,
        ),
        "mlp_gnn_sequence": SequenceModelConfig(
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
            ],
            intermediate_sizes=[64],
        ),
    }
    
    param_counts = {}
    
    for model_name, model_config in model_configs.items():
        try:
            # Build model using BenchMARL's get_model method
            model = model_config.get_model(
                input_spec=observation_spec,
                output_spec=action_spec,
                agent_group="agents",
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False,
                share_params=False,
                device="cpu",
                action_spec=action_spec,
            )
            
            params = count_parameters(model)
            param_counts[model_name] = params
            
            print(f"{model_name:25s}: {params:>10,} parameters")
            
        except Exception as e:
            print(f"{model_name:25s}: Error - {str(e)}")
            param_counts[model_name] = None
    
    # Summary statistics
    valid_counts = [v for v in param_counts.values() if v is not None]
    if valid_counts:
        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"  Mean: {sum(valid_counts) / len(valid_counts):>10,.0f} parameters")
        print(f"  Min:  {min(valid_counts):>10,} parameters")
        print(f"  Max:  {max(valid_counts):>10,} parameters")
        print(f"  Range: {max(valid_counts) - min(valid_counts):>10,} parameters")
        print(f"  Std Dev: {torch.tensor(valid_counts, dtype=torch.float32).std().item():>10,.0f}")
        print(f"{'='*80}\n")
    
    env.close()
    return param_counts


if __name__ == "__main__":
    check_model_configs()
