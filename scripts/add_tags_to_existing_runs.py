#!/usr/bin/env python3
"""
Script to add model_architecture tags to existing WandB runs.
This helps you retroactively tag old runs for easier filtering.

Usage:
    python add_tags_to_existing_runs.py --project benchmarl-2025-10-31 --entity your-entity

Requirements:
    pip install wandb
"""

import argparse
import wandb
from typing import Dict, Any


def get_model_architecture(config: Dict[str, Any]) -> str:
    """Extract model architecture name from run config."""
    model_name = config.get('model_name', '')
    
    # Single models
    if model_name == 'mlp':
        return 'mlp_balanced'
    
    elif model_name == 'deepsets':
        return 'deepsets'
    
    elif model_name == 'gnn':
        # Try to identify GNN type
        model_config = config.get('model_config', {})
        gnn_class = model_config.get('gnn_class', '')
        
        if 'GraphConv' in gnn_class:
            return 'gnn_balanced_graphconv'
        elif 'GATv2' in gnn_class or 'gatv2' in gnn_class.lower():
            return 'gnn_balanced_gatv2'
        else:
            return 'gnn_balanced'
    
    elif model_name == 'sequencemodel':
        # Try to identify combination type from model_configs
        model_config = config.get('model_config', {})
        
        # Check the layer types
        has_gnn = False
        has_lstm = False
        has_gru = False
        has_mlp = False
        has_deepsets = False
        
        # Look for model_configs keys (they're stored as model_configs_0, model_configs_1, etc.)
        for key, value in model_config.items():
            if key.startswith('model_configs'):
                if isinstance(value, dict):
                    target = value.get('_target_', '')
                    if 'gnn' in target.lower():
                        has_gnn = True
                    elif 'lstm' in target.lower():
                        has_lstm = True
                    elif 'gru' in target.lower():
                        has_gru = True
                    elif 'mlp' in target.lower():
                        has_mlp = True
                    elif 'deepsets' in target.lower():
                        has_deepsets = True
        
        # Identify combination type
        if has_gnn and has_lstm:
            return 'gnn_lstm_combo'
        elif has_deepsets and has_gnn:
            return 'deepsets_gnn_combo'
        elif has_gru and has_deepsets:
            return 'gru_deepsets_combo'
        elif has_mlp and has_gnn and has_gru:
            return 'mlp_gnn_gru_combo'
        else:
            return 'sequence_model_combo'
    
    return 'unknown'


def add_tags_to_runs(project: str, entity: str = None, dry_run: bool = True):
    """Add model_architecture tags to existing WandB runs."""
    api = wandb.Api()
    
    # Get all runs
    if entity:
        runs = api.runs(f"{entity}/{project}")
    else:
        runs = api.runs(project)
    
    print(f"Found {len(runs)} runs in project: {project}")
    print()
    
    updated_count = 0
    skipped_count = 0
    
    for run in runs:
        # Check if already has model_architecture tag
        existing_tags = run.tags or []
        
        # Get model architecture
        model_arch = get_model_architecture(run.config)
        
        # Check if tag already exists
        if model_arch in existing_tags:
            print(f"⏭️  Run {run.name}: Already has tag '{model_arch}'")
            skipped_count += 1
            continue
        
        if model_arch == 'unknown':
            print(f"⚠️  Run {run.name}: Could not determine model architecture")
            print(f"   model_name: {run.config.get('model_name', 'N/A')}")
            skipped_count += 1
            continue
        
        # Add tag
        new_tags = list(existing_tags) + [model_arch]
        
        print(f"🏷️  Run {run.name}:")
        print(f"   Adding tag: {model_arch}")
        print(f"   Old tags: {existing_tags}")
        print(f"   New tags: {new_tags}")
        
        if not dry_run:
            run.tags = new_tags
            run.update()
            print(f"   ✅ Updated!")
        else:
            print(f"   🔍 DRY RUN - no changes made")
        
        print()
        updated_count += 1
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total runs: {len(runs)}")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    
    if dry_run:
        print()
        print("This was a DRY RUN. No changes were made.")
        print("Run with --no-dry-run to actually update the tags.")


def main():
    parser = argparse.ArgumentParser(
        description='Add model_architecture tags to existing WandB runs'
    )
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='WandB project name (e.g., benchmarl-2025-10-31)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='WandB entity/username (optional)'
    )
    parser.add_argument(
        '--no-dry-run',
        action='store_true',
        help='Actually update the runs (default is dry-run mode)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WandB Run Tagger")
    print("=" * 60)
    print()
    
    if not args.no_dry_run:
        print("🔍 Running in DRY RUN mode (no changes will be made)")
        print("   Use --no-dry-run to actually update the tags")
        print()
    
    add_tags_to_runs(
        project=args.project,
        entity=args.entity,
        dry_run=not args.no_dry_run
    )


if __name__ == '__main__':
    main()
