# SwarmEcho: System Overview

## Introduction
SwarmEcho is a GPU-accelerated Multi-Agent Reinforcement Learning (MARL) environment built in JAX. The objective is to train a drone swarm to explore a 2D environment and form a delay-tolerant communication relay chain between a base station and a discovered target.

## Codebase Architecture
The project heavily utilizes functional separation. The core logic lives in `src/`, designed to keep neural network architecture separate from physical simulators and hyperparameter configurations.

### Directory Mapping
* **`src/core/`**: Core utilities, base configuration loading (`config.py`).
* **`src/curriculum_config/`**: Contains the defining parameters of the world `base_params.yaml`, curriculum progression `levels/`, and geometry of the maps `maps/`. See **[02_configuration_guide](02_configuration_guide.md)**.
* **`src/env/`**: Houses the JAX physics simulator: action-space rules, raycasting, map rasterizing, and the ego-centric observation calculations. See **[03_environment_and_physics](03_environment_and_physics.md)**.
* **`src/models/`** & **`src/training/`**: The neural brain. Multi-Agent PPO logic, centralized critics, custom rollout buffers, and Actor-Critic networks. See **[04_marl_and_training](04_marl_and_training.md)**.
* **`src/analysis/`** & **`src/visualize/`**: Dashboard tools and OpenCV/Matplotlib logic used to render the rollout videos and evaluate training runs. See **[05_analysis_and_tools](05_analysis_and_tools.md)**.

## Core Concepts
### Centralized Training, Decentralized Execution (CTDE)
- **Actor (decentralized):** Each drone only sees its local observation. This is what runs on the real drone.
- **Critic (centralized):** During training only, the critic receives the concatenated observations of all N agents (using self-attention to maintain permutation invariance). This produces low-variance value estimates.

### JAX and vmap
The entire environment logic is written using pure JAX. Because the physics and observations are written as tensor operations (without python-level loops or external states), we use JAX's `vmap` (vectorizing map) to simulate exactly **1024 parallel universes** instantly on the GPU.
