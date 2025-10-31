# Experiments Directory Reorganization

**Date**: 2025-10-31  
**Purpose**: Organize experiments directory into logical categories for better maintainability

## Overview

The `/experiments` directory has been reorganized from a flat structure to a categorized hierarchy for improved organization and clarity.

## New Structure

```
experiments/
├── README.md                    # Main documentation
├── scripts/                     # All executable scripts
│   ├── single_tests/           # Quick validation scripts
│   │   ├── test_single_hydra.sh
│   │   └── test_single_combination.sh
│   ├── surveys/                # Full survey scripts
│   │   ├── run_vmas_survey_hydra_multirun.sh
│   │   ├── test_all_combinations.sh
│   │   ├── compare_with_baselines.sh
│   │   └── run_extended_benchmark.sh
│   └── grouped/                # Hypothesis-driven experiments
│       ├── README.md
│       ├── scalability_experiments.sh
│       ├── coordination_experiments.sh
│       ├── task_type_experiments.sh
│       ├── simple_baseline_experiments.sh
│       ├── combination_experiments.sh
│       └── run_all_grouped_experiments.sh
├── docs/                       # All documentation
│   ├── QUICKREF.md
│   ├── VMAS_Experiment_Design.md
│   ├── TODO_1-3_IMPLEMENTATION.md
│   ├── COMBINATION_CONFIG_FIX.md
│   ├── README_VMAS_SURVEY.md
│   ├── SETUP_COMPLETE.md
│   └── DIRECTORY_REORGANIZATION.md (this file)
└── utils/                      # Utility scripts
    └── (future utility scripts)
```

## Migration Path

### Old → New Locations

#### Scripts
- `experiments/test_single_hydra.sh` → `experiments/scripts/single_tests/test_single_hydra.sh`
- `experiments/run_vmas_survey_hydra_multirun.sh` → `experiments/scripts/surveys/run_vmas_survey_hydra_multirun.sh`
- `experiments/grouped/*.sh` → `experiments/scripts/grouped/*.sh`
- `dev/combination/test_single_combination.sh` → `experiments/scripts/single_tests/test_single_combination.sh`
- `dev/combination/test_all_combinations.sh` → `experiments/scripts/surveys/test_all_combinations.sh`
- `dev/combination/compare_with_baselines.sh` → `experiments/scripts/surveys/compare_with_baselines.sh`
- `dev/combination/run_extended_benchmark.sh` → `experiments/scripts/surveys/run_extended_benchmark.sh`

#### Documentation
- `experiments/QUICKREF.md` → `experiments/docs/QUICKREF.md`
- `experiments/VMAS_Experiment_Design.md` → `experiments/docs/VMAS_Experiment_Design.md`
- `experiments/TODO_1-3_IMPLEMENTATION.md` → `experiments/docs/TODO_1-3_IMPLEMENTATION.md`
- `experiments/COMBINATION_CONFIG_FIX.md` → `experiments/docs/COMBINATION_CONFIG_FIX.md`
- `experiments/README_VMAS_SURVEY.md` → `experiments/docs/README_VMAS_SURVEY.md`
- `experiments/SETUP_COMPLETE.md` → `experiments/docs/SETUP_COMPLETE.md`
- `experiments/grouped/README.md` → `experiments/scripts/grouped/README.md`

## Changes Made

### 1. Directory Structure
- Created `scripts/` subdirectory with three categories:
  - `single_tests/` - Quick validation scripts
  - `surveys/` - Full survey/benchmark scripts
  - `grouped/` - Hypothesis-driven experiment groups
- Created `docs/` subdirectory for all documentation
- Created `utils/` subdirectory for utility scripts

### 2. File Updates
All path references were updated in the following files:

**Documentation Files**:
- `experiments/README.md` (newly created)
- `experiments/docs/QUICKREF.md` (4 path updates)
- `experiments/scripts/grouped/README.md` (8 path updates)
- `experiments/docs/TODO_1-3_IMPLEMENTATION.md` (4 path updates)
- `experiments/docs/COMBINATION_CONFIG_FIX.md` (2 path updates)

**Script Files**:
- `experiments/scripts/grouped/run_all_grouped_experiments.sh` (1 path update)

### 3. Path Changes
All references to old paths were updated:
- `experiments/grouped/` → `experiments/scripts/grouped/`
- `experiments/*.md` → `experiments/docs/*.md`
- `experiments/*.sh` → `experiments/scripts/{category}/*.sh`

## Benefits

1. **Better Organization**: Scripts and docs clearly separated
2. **Easier Navigation**: Category-based structure makes it easy to find relevant files
3. **Scalability**: New scripts can be added to appropriate categories
4. **Clarity**: Purpose of each directory is immediately clear
5. **Maintainability**: Related files grouped together

## Usage Examples

### Running Scripts

**Single Test**:
```bash
bash experiments/scripts/single_tests/test_single_hydra.sh
```

**Survey**:
```bash
bash experiments/scripts/surveys/run_vmas_survey_hydra_multirun.sh cuda:0 1000
```

**Grouped Experiments**:
```bash
bash experiments/scripts/grouped/scalability_experiments.sh cuda:1 500
```

### Finding Documentation

**Quick Reference**:
```bash
cat experiments/docs/QUICKREF.md
```

**Grouped Experiments Details**:
```bash
cat experiments/scripts/grouped/README.md
```

**Main Overview**:
```bash
cat experiments/README.md
```

## Verification

To verify all paths are correctly updated:

```bash
# Check for old path references (should return nothing)
grep -r "experiments/grouped/" experiments/

# List new structure
find experiments/ -type f -o -type d | sort
```

## Backward Compatibility

⚠️ **Breaking Change**: Old paths will no longer work. All scripts and documentation now use the new structure.

If you have any scripts or aliases that reference old paths, update them to the new structure:
- `experiments/grouped/` → `experiments/scripts/grouped/`
- `experiments/*.md` → `experiments/docs/*.md`

## Next Steps

Future improvements:
1. Add more utility scripts to `utils/` directory
2. Consider adding `results/` directory for output organization
3. Add `configs/` directory for shared configuration files
4. Document any new organizational patterns in this file

## Summary

The reorganization successfully categorizes all experiment scripts and documentation into logical groups, making the project more maintainable and easier to navigate. All path references have been updated to reflect the new structure.
