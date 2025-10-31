#!/bin/bash
# Master script to run all grouped experiments
# This will execute all experiment groups sequentially

echo "=========================================="
echo "Running All Grouped VMAS Experiments"
echo "=========================================="
echo ""

# Configuration
DEVICE="${1:-cuda:1}"
ITERATIONS="${2:-500}"

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Iterations per experiment: $ITERATIONS"
echo ""

# Calculate total experiments
echo "Experiment Groups:"
echo "  1. Scalability (5 tasks × 4 models × 3 seeds = 60 runs)"
echo "  2. Coordination Complexity (9 tasks × 4 models × 3 seeds = 108 runs)"
echo "  3. Task Types (10 tasks × 4 models × 3 seeds = 120 runs)"
echo "  4. Simple Baseline (2 tasks × 4 models × 5 seeds = 40 runs)"
echo "  5. Combination Architectures (8 tasks × 4 combos × 3 seeds = 96 runs)"
echo ""
echo "Total: ~424 experiment runs"
echo ""
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

# Track start time
START_TIME=$(date +%s)
echo ""
echo "=========================================="
echo "Starting experiments at: $(date)"
echo "=========================================="

# Run each experiment group
EXPERIMENTS=(
    "scalability_experiments.sh"
    "coordination_experiments.sh"
    "task_type_experiments.sh"
    "simple_baseline_experiments.sh"
    "combination_experiments.sh"
)

for i in "${!EXPERIMENTS[@]}"; do
    SCRIPT="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "[$NUM/5] Running: $SCRIPT"
    echo "=========================================="
    
    bash "experiments/scripts/grouped/$SCRIPT" "$DEVICE" "$ITERATIONS"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $SCRIPT"
    else
        echo "✗ Failed: $SCRIPT"
        echo "Continuing to next experiment..."
    fi
done

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "All Grouped Experiments Complete!"
echo "=========================================="
echo "Finished at: $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results available at:"
echo "  WandB: benchmarl-2025-10-31"
echo "  Local: outputs/"
echo ""
echo "Next steps:"
echo "  1. Analyze results in WandB"
echo "  2. Compare model performance across task groups"
echo "  3. Identify best architectures for each task type"
echo "  4. Use insights to design new GNN algorithms"
echo "=========================================="
