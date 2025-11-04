# InformARL-onpolicy (Graph MAPPO) Design Notes

## High-Level Intent
- Extend MAPPO with graph neural policies/critics so agents exploit relational structure in MPE-style environments (`informarl-onpolicy`).
- Preserve the familiar MAPPO training loop—warmup, rollout, GAE/returns, PPO updates—while swapping dense MLPs for configurable Graph Neural Networks.
- Provide toggles for standard MAPPO vs. graph-enhanced variants so baselines and GNN ablations share identical infrastructure.

## Execution Flow
1. **Script Entry** (`scripts/train_mpe.py`)
   - Parses CLI + graph-specific overrides (`onpolicy.config`, `graph_config`).
   - Sets seeds/devices, init logging (wandb or TensorBoard), builds train/eval vector envs (`GraphDummyVecEnv`/`GraphSubprocVecEnv`).
   - Chooses shared vs. separated runner; Graph pipelines use `runner.shared.graph_mpe_runner.GMPERunner`.
2. **Runner Setup** (`runner/shared/base_runner.py`)
   - Instantiates policy/trainers using algorithm selection: `graph_mappo.GR_MAPPO` + `graph_MAPPOPolicy.GR_MAPPOPolicy` when `env_name=GraphMPE`.
   - Builds buffers; GNN path uses `utils.graph_buffer.GraphReplayBuffer` capturing `node_obs`, `adj`, `agent_id`, `share_agent_id` along with classical MAPPO tensors.
3. **Warmup** (`GMPERunner.warmup`)
   - Resets env, writes initial observations/graph tensors into buffer slot 0.
   - If centralized critic enabled, reshapes shared observations/agent IDs to match expected per-agent layout.
4. **Rollout Loop** (`GMPERunner.run`)
   - For each episode step: `collect` queries policy for value, action, log-prob, updated RNN states using the batched graph inputs; env executed with one-hot actions; buffer `insert` stores rollouts with masks and graph context.
   - After episode horizon: `compute` runs value bootstrap + return/GAE and `train` yields PPO updates (possibly multiple minibatches/epochs) via trainer.
   - Handles LR decay, checkpointing, logging, eval, gif rendering.
5. **Training Update** (`graph_mappo.GR_MAPPO`)
   - Standard PPO objective with policy/value losses, entropy bonus, optional gradient clipping.
   - Supports AMP (`torch.cuda.amp.GradScaler`), PopArt/value normalization, recurrent or feed-forward policies, active masks.
   - Data generators in buffer provide feed-forward or recurrent minibatches, delivering graph tensors alongside advantages.

## Core Components
- **Policy wrapper** (`graph_MAPPOPolicy.GR_MAPPOPolicy`)
  - Holds `GR_Actor` and `GR_Critic`, each parameterized by shared graph settings.
  - `get_actions`/`get_values`/`evaluate_actions` funnel numpy buffers through actor/critic, returning values, actions, log-probs, updated states.
  - Linear LR schedules keep actor/critic optimizers synchronized with training horizon.
- **Graph Actor** (`graph_actor_critic.GR_Actor`)
  - `GNNBase` encodes per-agent graph context (node features + learned entity-type embeddings + optional edge attrs) using PyG `TransformerConv` stacks; aggregator selectable via `args.actor_graph_aggr`.
  - Concatenates GNN output with local observation, passes through `MLPBase`, optional `RNNLayer`, and `ACTLayer` for stochastic policies (discrete MultiDiscrete handling).
  - Supports `split_batch` to chunk oversized batches before GNN forward for memory efficiency.
- **Graph Critic** (`graph_actor_critic.GR_Critic`)
  - Uses matching `GNNBase` but aggregator (`args.critic_graph_aggr`) can be `global` or per-agent to suit centralized vs decentralized value functions.
  - Optionally concatenates centralized observation (if `use_cent_obs`) before MLP + optional RNN + output head (PopArt op when configured).
- **Graph Buffer** (`utils.graph_buffer.GraphReplayBuffer`)
  - Mirrors MAPPO buffer semantics with extra slots for graph data and agent identifiers.
  - Computes returns with/without GAE, PopArt, value norm, and handles proper-time-limit masking.
  - Generators output appropriately flattened tensors for feed-forward, naive recurrent, or chunked recurrent updates.
- **Environment Wrappers** (`envs.env_wrappers`, `envs.mpe`) 
  - `GraphDummyVecEnv`/`GraphSubprocVecEnv` extend baseline vector env wrappers to return tuples `(obs, agent_id, node_obs, adj, reward, done, info)`.
  - Graph observations encode per-entity positions/velocities plus entity type index; adjacency stores inter-entity distances used to build PyG edge sets.

## GNN + RL Specifics
- **Entity-Type Encoding**: `GNNBase` uses `EmbedConv` layers to embed discrete entity types and merge with continuous node features, enabling heterogeneous agents/goals/obstacles within the same attention-based graph.
- **Dynamic Graph Construction**: `TransformerConvNet.process_adj` thresholds pairwise distances (`max_edge_dist`) to prune edges; supports batched conversions to PyG `Data` objects without Python loops.
- **Aggregation Choices**:
  - Actor-level typically keeps per-agent node embeddings (`graph_aggr='node'`) and gathers each agent’s latent using its index.
  - Critic can operate on pooled graph descriptors (`graph_aggr='global'`) or stacked per-agent embeddings when centralized V requires agent-specific context.
- **Batch Splitting**: `split_batch` / `max_batch_size` guard against GPU OOM by chunking GNN forwards while keeping gradients intact.
- **Recurrent Support**: Optional RNN layers wrap the post-GNN MLP features so temporal dependencies coexist with spatial graph structure; masks reset hidden states on episode boundaries.
- **Training Stability**: PPO update uses AMP scaling, gradient clipping, PopArt or ValueNorm, and per-agent active masks to handle variable team sizes or dead agents.

## Key Differences vs. Vanilla MAPPO
- Additional observation channels (`node_obs`, `adj`, `agent_id`, `share_agent_id`) propagate through runner, buffer, trainer, and policy interfaces.
- GNN encoders replace pure MLPs but preserve MAPPO’s interface so toggling `env_name`/`algorithm_name` switches between graph-aware and baseline versions without altering outer loop code.
- Graph components introduce hyperparameters for embedding widths, attention heads, aggregation strategy, edge radius, and batch handling; these live alongside classic PPO knobs in `config.py`.
- Replay buffer normalization and advantage computation remain unchanged, ensuring performance gains stem from representation learning rather than pipeline tweaks.

## Practical Considerations
- GraphMPE envs expect `all_args.max_edge_dist`, `num_embeddings`, etc., to match environment entity counts; misconfiguration yields invalid edge construction or dimension mismatches.
- `agent_id` tensors are used as indices for gathering node embeddings—must remain integer and consistent with node ordering; buffer resets carefully mirror env outputs to avoid stale IDs.
- When `use_centralized_V` is enabled, shared observations and agent IDs are reshaped to `(threads, agents, agents * feat)` to match critic expectations.
- AMP scaler adds `unscale_` calls before gradient clipping; leaving `amp` disabled reverts to standard `loss.backward()`, so docs should note mixed-precision dependency on CUDA.
- Rendering / evaluation reuse the same policy interfaces (`policy.act` with deterministic flag) and thus automatically benefit from graph features without extra glue code.
