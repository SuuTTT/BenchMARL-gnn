# CHANGELOG

## 2025-10-31 - VMAS Survey & Model Combination Development

### Built On
- **BenchMARL Framework**: Multi-agent RL benchmarking library with Hydra configuration
- **VMAS**: Vectorized Multi-Agent Simulator with 26 task environments
- **PyTorch Geometric**: For GNN implementations (GraphConv, GATv2Conv)
- **TorchRL**: Base environment and data structures
- **Hydra**: Configuration management with multi-run support
- **WandB**: Experiment tracking (project: benchmarl-2025-10-31)
- **MAPPO Algorithm**: Multi-Agent Proximal Policy Optimization as primary test algorithm

### What Has Been Done

#### 1. Model Configurations
Created balanced model configurations with ~13-19K parameters:
- `benchmarl/conf/model/mlp_balanced.yaml` - Baseline MLP with [128, 128] layers (~19K params)
- `benchmarl/conf/model/gnn_balanced_graphconv.yaml` - MLP[64]→GraphConv GNN→MLP[64] (~14K params)
- `benchmarl/conf/model/gnn_balanced_gatv2.yaml` - MLP[64]→GATv2 GNN→MLP[64] (~14K params)

**Parameter Balancing Strategy**: Wrapped GNNs with MLP layers to achieve comparable parameter counts across different architectures.

#### 2. Experiment Scripts
Created comprehensive experiment infrastructure:

**Testing Scripts**:
- `experiments/test_single_hydra.sh` - Single task test (verified working)
- `experiments/check_model_params_simple.py` - Analytical parameter estimator

**Survey Scripts**:
- `experiments/run_vmas_survey_hydra_multirun.sh` - Full multi-run across 26 VMAS tasks × 4 models = 104 runs
- `experiments/experiment_commands.sh` - Reference commands for all 26 VMAS environments + 7 model types

**Configuration**:
- Device: CUDA:1 as primary (with CUDA:0 and CPU options documented)
- Iterations: 500 for full survey, 50 for testing
- WandB Project: benchmarl-2025-10-31
- Logging: WandB + CSV
- Iteration Calculation: `total_frames / frames_per_batch`

#### 3. Documentation
Created comprehensive documentation:
- `experiments/README_VMAS_SURVEY.md` - Original survey documentation
- `experiments/QUICKSTART.md` - Quick start guide
- `experiments/SETUP_COMPLETE.md` - Complete reference with all commands

**VMAS Environments Covered** (26 total):
- Navigation: balance, navigation, reverse_transport, transport, wheel
- Adversarial: adversarial_comms, discovery, dropout, flocking, give_way, passage, simple_adversary, simple_crypto, simple_reference, simple_speaker_listener, simple_spread, simple_tag, simple_world_comm
- Physics: buzz_wire, joint_passage, joint_passage_size
- Communication: football, wind_flocking
- Multi-modal: dispersion, sampling

#### 4. Model Combination Architectures
Designed 4 promising model combinations in `dev/combination/`:

**Attention-GNN** (`configs/attention_gnn.yaml`):
- Architecture: Attention[heads=4]→GNN GraphConv→Attention[heads=2]
- Motivation: Combine self-attention with graph structure learning

**Hierarchical DeepSets-GNN** (`configs/hierarchical_deepsets_gnn.yaml`):
- Architecture: DeepSets→GNN GraphConv→MLP[128]
- Motivation: Set aggregation for global features, GNN for relational structure

**Recurrent-GNN** (`configs/recurrent_gnn.yaml`):
- Architecture: GRU[hidden=64]→GNN GraphConv→GRU[hidden=32]
- Motivation: Temporal reasoning combined with spatial graph structure

**Multi-Scale GNN** (`configs/multi_scale_gnn.yaml`):
- Architecture: Parallel GNN branches with different aggregations (add/mean/max)
- Motivation: Capture different scales of inter-agent relationships

**Testing Scripts**:
- `dev/combination/test_combinations.sh` - Quick test (100 iterations)
- `dev/combination/run_all_combinations.sh` - Full runs (500 iterations)
- `dev/combination/README.md` - Architecture documentation

#### 5. Verification Status
- ✅ Single test run verified working (`test_single_hydra.sh`)
- ✅ Parameter estimates validated (models balanced 13-19K params)
- ✅ Hydra multi-run syntax tested and documented
- ⏳ Full survey not yet executed (scripts ready)
- ⏳ Combination architectures designed but not tested

### TODO

#### High Priority
1. **✅ Fix Device Consistency Bug** - COMPLETED
   - ~~Current issue: Devices split across operations causing slow iterations~~
   - **Solution Implemented**: Added `experiment.buffer_device=$DEVICE` to all experiment scripts
   - **Files Updated**:
     - `experiments/test_single_hydra.sh`
     - `experiments/run_vmas_survey_hydra_multirun.sh`
     - `dev/combination/test_single_combination.sh`
     - `dev/combination/test_all_combinations.sh`
     - `dev/combination/compare_with_baselines.sh`
     - `dev/combination/run_extended_benchmark.sh`
   - All three device parameters now synchronized: `sampling_device`, `train_device`, `buffer_device`

2. **✅ Standardize Experiment Naming Convention** - COMPLETED
   - ~~Current: Auto-generated Hydra timestamps~~
   - **Implemented**: `yyyy-mm-dd-hostname-devicename-task-algo`
   - **Example**: `2025-10-31-c225-cuda1-navigation-mappo`
   - **Modified**: `benchmarl/experiment/experiment.py` - Updated `_setup_name()` method to generate standardized names
   - Automatically includes: date, hostname, device (cuda0/cuda1/cpu), task name, algorithm name

3. **✅ Group Experiments by Detailed Task Types** - COMPLETED
   - ~~Current: Flat list of 26 VMAS tasks~~
   - **Solution**: Created structured experiment scripts in `experiments/grouped/`
   - **Created Files**:
     - `experiments/grouped/scalability_experiments.sh` - Tests model scalability with varying agent counts
     - `experiments/grouped/coordination_experiments.sh` - Simple/Medium/Complex coordination tasks
     - `experiments/grouped/task_type_experiments.sh` - Navigation, Transport, Flocking, Competitive
     - `experiments/grouped/simple_baseline_experiments.sh` - Baseline on simple tasks
     - `experiments/grouped/combination_experiments.sh` - Tests hybrid architectures
     - `experiments/grouped/README.md` - Complete documentation
   - **Task Groupings** (based on VMAS_Experiment_Design.md):
     - **Scalability**: Low-coord (dispersion, navigation) + High-coord (flocking, discovery, wind_flocking)
     - **Coordination**: Simple (navigation, give_way) + Medium (balance, wheel, passage) + Complex (transport, reverse_transport, football, joint_passage)
     - **Task Types**: Navigation/Dispersion, Transport/Manipulation, Flocking/Formation, Competitive/Mixed
   - All scripts use unified device configuration and standardized naming

#### Research Goals
4. **Reimplement Existing GNN-RL Methods**
   - Algorithms to implement:
     - **InforMARL**: Information-aware MARL with GNNs
     - **GCBF+**: Graph-based control barrier functions
     - **X-MAGE**: Cross-agent message aggregation with GNNs (it's task allocation phase maybe universal)
   - Action: Research papers, implement in BenchMARL framework, compare against baselines

5. **Design New GNN Algorithm**
   - Goal: After understanding existing methods through experiments, propose novel GNN architecture for MARL
   - Approach: Build on insights from survey experiments and reimplementations
   - Action: Design, implement, and validate new method

6. try using the group and deepset function in benchmarl.

### Notes
- **Iteration Calculation**: For 500 iteration runs: 500 = `on_policy_collected_frames` / `frames_per_batch`
- **Parameter Balancing**: GNNs wrapped with MLPs to achieve ~14K params (vs pure GNN ~38 params)
- **Hydra Multi-Run**: Use `-m` flag for parallel execution across tasks/models
- **Device Options**: Scripts support cuda:0, cuda:1, and cpu via command-line overrides
