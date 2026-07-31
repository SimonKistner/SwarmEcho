# SwarmEcho: System Overview

## Introduction
SwarmEcho is a GPU-accelerated Multi-Agent Reinforcement Learning (MARL) environment built in JAX. The objective is to train a drone swarm to explore a 2D environment and form a delay-tolerant communication relay chain between a base station and a discovered target.

## Codebase Architecture
The project heavily utilizes functional separation. The core logic lives in `src/`, designed to keep neural network architecture separate from physical simulators and hyperparameter configurations.

### Directory Mapping
* **`src/swarmecho/core/`**: Core utilities and base configuration loading (`config.py`).
* **`src/swarmecho/curriculum_config/`**: Contains the curriculum progression `levels/` and map geometry in `maps/`. Defaults are defined by structured dataclasses in `src/swarmecho/core/config.py`. See **[01_configuration_guide](../02_guide/01_configuration_guide.md)**.
* **`src/swarmecho/env/`**: Houses the JAX physics simulator, raycasting, map rasterization, and agent-centric observations. See **[02_environment_and_physics](02_environment_and_physics.md)**.
* **`src/swarmecho/models/`** and **`src/swarmecho/training/`**: MAPPO models, centralized critics, rollout buffers, and training/evaluation orchestration. See **[03_marl_and_training](03_marl_and_training.md)**.
* **`src/swarmecho/analysis/`** and **`src/swarmecho/visualize/`**: Streamlit dashboards and OpenCV rendering used to inspect training and evaluation runs. See **[02_analysis_and_tools](../02_guide/02_analysis_and_tools.md)**.
* **Architecture Defense**: Rationale for the current MAPPO + Agent-Centric Critic design over privileged world-state critics. See **[04_mappo_acc_architecture_defense](04_mappo_acc_architecture_defense.md)**.

## Core Concepts
### Centralized Training, Decentralized Execution (CTDE)
- **Actor (decentralized):** Each drone only sees its local observation. This is what runs on the real drone.
- **Critic (centralized):** During training only, the critic receives the concatenated observations of all N agents (using self-attention to maintain permutation invariance). This produces low-variance value estimates.

### JAX and vmap
The environment transition, observations, and rewards are pure JAX functions. Training vectorizes them with `vmap` over the configured batch, whose maintained default is **4,000 parallel environments**. The same functions also support smaller validation and evaluation batches.

### Environment-state ownership

`EnvState` is a nested ownership container. Kinematics live in `PhysicsState`;
communication facts and persistent target knowledge live in
`CommunicationState`; the temporary 2D coverage aid lives in
`ExplorationState`; and discrete finder-path relay bookkeeping lives in
`RelayTaskState` and `env/relay_task.py`. The renderer creates its flat CPU
frame projection only at the rendering boundary.
