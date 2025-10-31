#!/bin/bash
# Shell script to run the VMAS model survey experiment
# This will run all VMAS tasks with all model configurations

echo "=========================================="
echo "VMAS Model Survey Experiment"
echo "=========================================="
echo "Starting experiment at: $(date)"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
echo ""

# Run the experiment
python experiments/vmas_model_survey.py

echo ""
echo "=========================================="
echo "Experiment completed at: $(date)"
echo "=========================================="
echo ""
echo "Check WandB project 'benchmarl-10-30' for results"
echo "Or view local logs in ./outputs/"
