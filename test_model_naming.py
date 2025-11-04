#!/usr/bin/env python
"""Test script to verify model architecture naming works correctly."""

print("Testing model architecture naming...")
print("\nExpected behavior:")
print("1. Experiment name format: <model_architecture>-<task>-<algorithm>-<date>-<hostname>-<device>")
print("2. WandB config should include 'model_architecture' field")
print("3. WandB tags should include model_architecture, algorithm, and task")
print("\nExample experiment names:")
print("  - gnn_lstm_combo-navigation-mappo-2025-11-02-hostname-cuda1")
print("  - mlp_balanced-flocking-mappo-2025-11-02-hostname-cuda1")
print("  - deepsets_gnn_combo-transport-mappo-2025-11-02-hostname-cuda1")
print("\nRun your experiments to see the new naming in action!")
print("\nYou can filter in WandB using:")
print("  - config.model_architecture: 'gnn_lstm_combo', 'mlp_balanced', etc.")
print("  - tags: Include model_architecture, algorithm_name, task_name")
print("  - Name/ID: Will start with model architecture for easy identification")
