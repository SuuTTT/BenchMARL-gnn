# BenchMARL Design Notes

## High-Level Goals
- Provide a configurable multi-agent RL (MARL) benchmark with unified experiment, algorithm, and environment abstractions.
- Support graph-based policies/critics so GNNs can encode agent relations while reusing TorchRL collectors, buffers, and logging.
- Enable on- and off-policy training with shared infrastructure for evaluation, checkpointing, and experiment reproducibility.

## Experiment Lifecycle
1. **Hydra Entry Point** (`benchmarl/run.py`)
   - Loads Hydra config (`algorithm`, `task`, `model`, `critic_model`, `experiment`).
   - Constructs an `Experiment` from dataclass-backed configs (see `benchmarl/hydra_config.py`).
2. **Experiment Setup** (`benchmarl/experiment/experiment.py`)
   - Validates that model sequences respect GNN placement constraints (e.g., graph layers first when using positional edges).
   - Chooses action modality (continuous vs discrete) per task and algorithm capabilities.
   - Builds evaluation and training env factories with task-defined transforms; injects RNN transforms when models are recurrent.
   - Instantiates algorithm-specific modules (losses, buffers, optimizers) per agent group.
3. **Data Collection**
   - Obtains an explorative policy from the algorithm; wraps deterministic heads with stochastic exploration modules.
   - Uses either `SyncDataCollector` (typical) or manual rollouts when gradients must flow through collection.
   - Buffers initial random frames for off-policy warm-up when supported.
4. **Training Iteration**
   - Logs collection stats (returns, timing) and splits batches by agent group.
   - Applies algorithm preprocessing (e.g., advantage computation, centralized value expansion) and reshapes for RNN/GNN stacks.
   - Streams each group batch into a replay buffer sized per on/off-policy requirements; supports prioritized replay on demand.
   - Runs optimizer steps: samples mini-batches, evaluates `LossModule`s, applies gradient clipping (norm or value), updates targets via Polyak or hard sync.
   - Anneals exploration layers when available and updates collector weights.
5. **Evaluation & Checkpointing**
   - Periodically runs deterministic rollouts on evaluation envs, logging to configured sinks and optional JSON artifacts.
   - Manages checkpoint cadence, retention, and optional buffer exclusion; persists full config (task, model, algo, seed) alongside checkpoints.
   - Graceful shutdown tears down collectors/envs and cleans on-disk buffer storage.

## Algorithm & Buffer Abstractions
- `AlgorithmConfig` dataclasses wrap algorithm parameters; `get_algorithm()` instantiates a concrete `Algorithm` with experiment context.
- `Algorithm` base class enforces conventions around specs (`observation_spec`, `action_spec`, `state_spec`), multi-group policies, and replay buffers.
- Provides hooks to:
  - Construct per-group deterministic and explorative policies (`_get_policy_for_loss`, `_get_policy_for_collection`).
  - Build replay buffers with size adjustments for sequence models (RNN/GNN) and prioritized sampling when requested.
  - Define losses and target network updaters, handle batch reshaping, and aggregate loss terms prior to optimization.

## GNN Integration Details
- `benchmarl/models/gnn.py` implements a graph policy/critic module backed by PyG message passing classes.
- Key configuration fields:
  - `topology` (`full`, `empty`, `from_pos`) and `self_loops` determine edge structure.
  - Optional `position_key` and `velocity_key` compute relative features for edge attributes; `from_pos` dynamically builds adjacency using `edge_radius`.
  - `share_params` controls whether all agents share a GNN or each agent owns a distinct copy; centralized critics pool outputs (mean) when needed.
- Safeguards ensure:
  - Graph layers only appear in positions that preserve agent dimensions and required keys (especially within `SequenceModelConfig`).
  - Positional/velocity specs match configured feature counts and that centralized critics either place GNN first or avoid incompatible inputs.
- Result: policies can encode interaction structure while staying compatible with TorchRL collectors, replay buffers, and optimization loops.

## Implementation Touchpoints
- **Configs**: YAML under `benchmarl/conf/` parametrizes experiments, algorithms, models, and tasks; dataclasses provide validation and defaults.
- **Environment Registry**: Tasks supply env constructors, transforms, reward accumulation hooks, and group mappings consumed during setup.
- **Logging**: `benchmarl/experiment/logger.py` standardizes metric sinks (TensorBoard, Weights & Biases) and metadata capture (hyperparameters, seeds, configs).
- **Callbacks**: `benchmarl/experiment/callback.py` enables lifecycle hooks (on batch collected, on train step/end) so downstream projects can extend behavior without forking core loops.

## GNN + RL Challenges Addressed
- **Relational Credit Assignment**: GNN policies/critics aggregate neighbor features, enabling coordination signal propagation beyond local observations.
- **Topology Flexibility**: Supports dense, sparse, or sensory-derived graphs via runtime topology selection, allowing experimentation with graph structure impact.
- **Integration with TorchRL**: Maintains compatibility with existing TorchRL data structures (TensorDicts, collectors, replay buffers), reducing custom glue code.
- **Sequence Handling**: Adjusts buffer sizing and batching to respect temporal slices when GNNs appear inside RNN sequences, preventing shape mismatches.
- **Task Diversity**: Works across vectorized envs (VMAS, SMACv2, MAGent, MeltingPot) by delegating spec normalization to task registries and allowing shared policy params across heterogeneous agent groups.
