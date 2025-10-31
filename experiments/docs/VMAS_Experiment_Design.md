# VMAS Experiment Design and Environment Grouping

## 1. Introduction

This document proposes a structured approach to evaluating various neural network architectures on the Vectorized Multi-Agent Simulator (VMAS) environments. The goal is to understand the strengths and weaknesses of different models based on task characteristics. We group the environments and design a set of experiments to test specific hypotheses about model performance.

## 2. VMAS Environment Grouping

We can group the VMAS environments based on several axes. This allows us to select a representative subset of tasks for our experiments.

### Group 1: Scalability (Variable Number of Agents)

These tasks are designed to test how well a model scales with an increasing number of agents.

*   **Low-Coordination:** `dispersion`, `navigation`
*   **High-Coordination:** `flocking`, `discovery`, `wind_flocking`, `road_traffic`

### Group 2: Coordination Complexity

These tasks require different levels of coordination among agents.

*   **Simple Coordination (Collision Avoidance):** `navigation`, `give_way`
*   **Medium Coordination (Cooperative Manipulation/Movement):** `balance`, `wheel`, `passage`
*   **Complex Coordination (Strategic/Spatially-aware):** `transport`, `reverse_transport`, `football`, `joint_passage`, `ball_trajectory`

### Group 3: Task Type

Categorizing by the high-level objective.

*   **Navigation & Dispersion:** `navigation`, `dispersion`, `discovery`
*   **Transport & Manipulation:** `transport`, `reverse_transport`, `balance`, `wheel`, `ball_passage`
*   **Flocking & Formation:** `flocking`, `wind_flocking`
*   **Competitive/Mixed:** `football`

## 3. Experiment Design

### 3.1. Objective

To systematically evaluate the performance, scalability, and sample efficiency of different neural network architectures (MLP, GNN, DeepSets, AttentionGNN) across a diverse set of VMAS environments.

### 3.2. Research Questions

1.  How do GNN-based models perform on tasks with high coordination complexity compared to simpler architectures like MLP?
2.  How do different models scale in terms of performance and training time as the number of agents increases?
3.  Are certain architectures better suited for specific task types (e.g., GNN for transport, DeepSets for flocking)?

### 3.3. Models to Evaluate

*   **MLP (Multi-Layer Perceptron):** As a baseline. Each agent's policy is an MLP that takes its local observation as input. No explicit communication.
*   **GNN (Graph Neural Network):** Agents form nodes in a graph, and observations are passed over edges. This allows for explicit communication and relational reasoning.
*   **DeepSets:** An architecture for permutation-invariant inputs, suitable for tasks where agents are homogeneous and their ordering does not matter.
*   **AttentionGNN:** A GNN with an attention mechanism, allowing agents to weigh the importance of information from different neighbors.

### 3.4. Metrics

*   **Average Episode Reward:** Primary measure of performance.
*   **Success Rate:** Task-specific metric (e.g., percentage of successful transports).
*   **Scalability:** Performance curve as the number of agents increases.
*   **Training Time:** Wall-clock time to reach a certain performance threshold.

### 3.5. Proposed Experiments

#### Experiment 1: Scalability

*   **Hypothesis:** GNNs, DeepSets, and AttentionGNNs will show better performance and scalability than MLPs in environments with a large and variable number of agents.
*   **Environments:** `dispersion`, `flocking`, `navigation`.
*   **Method:**
    1.  Train MLP, GNN, and DeepSets on these environments.
    2.  Vary the number of agents (e.g., 4, 8, 16, 32).
    3.  Plot Average Episode Reward vs. Number of Agents for each model.
    4.  Plot Training Time vs. Number of Agents.

#### Experiment 2: Coordination Complexity

*   **Hypothesis:** GNN and AttentionGNN will outperform MLP and DeepSets in tasks requiring complex, spatially-aware coordination.
*   **Environments:** `transport`, `football`, `joint_passage`.
*   **Method:**
    1.  Train all four models (MLP, GNN, DeepSets, AttentionGNN) on these environments with a fixed number of agents.
    2.  Compare the final converged performance (Average Episode Reward).
    3.  Analyze agent behaviors to understand coordination strategies (e.g., in `transport`, do GNN agents position themselves optimally around the package?).

#### Experiment 3: Simple Tasks Baseline

*   **Hypothesis:** For simple tasks with a fixed, small number of agents, the performance difference between models will be minimal, and MLPs may offer the best trade-off between performance and training speed.
*   **Environments:** `balance`, `give_way`.
*   **Method:**
    1.  Train all four models on these environments.
    2.  Compare final performance and training time.

### 3.6. Experimental Setup

*   **Algorithm:** Use a standard MARL algorithm like MAPPO for all experiments to ensure consistency.
*   **Hyperparameters:** Keep hyperparameters (learning rate, batch size, entropy coefficient, etc.) consistent across all models and experiments where possible.
*   **Reproducibility:** Use multiple random seeds (e.g., 5) for each experimental run and report the mean and standard deviation of the results.
*   **Hardware:** All experiments should be run on the same hardware to ensure fair comparison of training times.
