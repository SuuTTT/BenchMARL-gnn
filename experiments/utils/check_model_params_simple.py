#!/usr/bin/env python3
"""
Simple utility to estimate and compare parameter counts across model configurations.
This provides rough estimates based on typical VMAS observation/action dimensions.

Usage:
    python experiments/check_model_params_simple.py
"""

import torch
import torch_geometric.nn as gnn_nn


def estimate_mlp_params(input_dim, hidden_dims, output_dim):
    """Estimate parameters for an MLP."""
    params = 0
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        params += prev_dim * hidden_dim + hidden_dim  # weights + bias
        prev_dim = hidden_dim
    params += prev_dim * output_dim + output_dim  # final layer
    return params


def estimate_gnn_params(input_dim, output_dim, gnn_type="GraphConv", heads=1, concat=False):
    """
    Estimate parameters for a GNN layer.
    Note: Actual params depend on the specific GNN implementation.
    """
    if gnn_type == "GraphConv":
        # GraphConv has: in_channels * out_channels for message transformation
        # Plus optional bias
        params = input_dim * output_dim + output_dim
    elif gnn_type == "GATv2Conv":
        # GAT has more parameters due to attention mechanism
        # Simplified estimate: (in + out) * out * heads + attention params
        if concat:
            params = (input_dim * output_dim * heads + 
                     output_dim * heads + 
                     2 * output_dim * heads)  # attention coefficients
        else:
            params = (input_dim * output_dim + 
                     output_dim + 
                     2 * output_dim)  # attention coefficients
    else:
        params = input_dim * output_dim + output_dim
    
    return params


def estimate_deepsets_params(input_dim, local_hidden, global_hidden, output_dim, local_layers=1, global_layers=1):
    """Estimate parameters for DeepSets architecture."""
    # Local NN (per-agent encoder)
    params = estimate_mlp_params(input_dim, [local_hidden] * local_layers, local_hidden)
    
    # Global NN (after aggregation)
    params += estimate_mlp_params(local_hidden, [global_hidden] * global_layers, output_dim)
    
    return params


def main():
    """Compare parameter counts for different model configurations."""
    
    # Typical VMAS dimensions (e.g., NAVIGATION task)
    n_agents = 3
    obs_dim_per_agent = 18  # observation features per agent
    action_dim_per_agent = 2  # continuous action dimension
    
    print(f"\n{'='*80}")
    print(f"Model Parameter Count Estimates (Balanced Configurations)")
    print(f"{'='*80}")
    print(f"Assumptions:")
    print(f"  - Agents: {n_agents}")
    print(f"  - Observation dim per agent: {obs_dim_per_agent}")
    print(f"  - Action dim per agent: {action_dim_per_agent}")
    print(f"  - Models process per-agent features → per-agent actions")
    print(f"  - GNNs are wrapped with MLPs for balanced parameter counts")
    print(f"{'='*80}\n")
    
    configs = {}
    
    # MLP: [128, 128] → ~19K params
    configs["mlp"] = estimate_mlp_params(
        input_dim=obs_dim_per_agent,
        hidden_dims=[128, 128],
        output_dim=action_dim_per_agent
    )
    
    # GNN GraphConv wrapped with MLPs: MLP[64] → GNN → MLP[64]
    mlp1_params = estimate_mlp_params(
        input_dim=obs_dim_per_agent,
        hidden_dims=[64],
        output_dim=64  # intermediate
    )
    gnn_params = estimate_gnn_params(
        input_dim=64,
        output_dim=64,
        gnn_type="GraphConv"
    )
    mlp2_params = estimate_mlp_params(
        input_dim=64,
        hidden_dims=[64],
        output_dim=action_dim_per_agent
    )
    configs["gnn_graphconv"] = mlp1_params + gnn_params + mlp2_params
    
    # GNN GATv2 wrapped with MLPs: MLP[64] → GNN → MLP[64]
    mlp1_params = estimate_mlp_params(
        input_dim=obs_dim_per_agent,
        hidden_dims=[64],
        output_dim=64
    )
    gnn_params = estimate_gnn_params(
        input_dim=64,
        output_dim=64,
        gnn_type="GATv2Conv",
        heads=4,
        concat=False
    )
    mlp2_params = estimate_mlp_params(
        input_dim=64,
        hidden_dims=[64],
        output_dim=action_dim_per_agent
    )
    configs["gnn_gatv2"] = mlp1_params + gnn_params + mlp2_params
    
    # DeepSets: local_nn[128], global_nn[128]
    configs["deepsets"] = estimate_deepsets_params(
        input_dim=obs_dim_per_agent,
        local_hidden=128,
        global_hidden=128,
        output_dim=action_dim_per_agent,
        local_layers=1,
        global_layers=1
    )
    
    # Print results
    print(f"{'Model':<30s} {'Est. Params':>15s}")
    print(f"{'-'*30} {'-'*15}")
    for model_name, params in configs.items():
        print(f"{model_name:<30s} {params:>15,}")
    
    # Summary
    valid_counts = list(configs.values())
    print(f"\n{'='*80}")
    print(f"Summary Statistics:")
    print(f"  Mean:     {sum(valid_counts) / len(valid_counts):>10,.0f} parameters")
    print(f"  Min:      {min(valid_counts):>10,} parameters")
    print(f"  Max:      {max(valid_counts):>10,} parameters")
    print(f"  Range:    {max(valid_counts) - min(valid_counts):>10,} parameters")
    print(f"  Std Dev:  {torch.tensor(valid_counts, dtype=torch.float32).std().item():>10,.0f}")
    print(f"{'='*80}\n")
    
    print("\nNOTE: These are rough estimates. Actual parameter counts will vary based on:")
    print("  - Specific task observation/action dimensions")
    print("  - Whether parameters are shared across agents")
    print("  - Actor vs. Critic network (critic may have different output dims)")
    print("  - BenchMARL's internal model wrapping\n")


if __name__ == "__main__":
    main()
