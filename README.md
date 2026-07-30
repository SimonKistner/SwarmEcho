# SwarmEcho

<p align="center">
  <img src="docs/Preview_vid/demo_2x.gif" width="600" alt="SwarmEcho Behaviour Preview" />
</p>

A **GPU-accelerated Multi-Agent Reinforcement Learning** environment built in JAX, training a drone swarm to explore a 2D environment and form a **delay-tolerant communication relay chain** between a base station and a discovered target.

Agents are trained using **MAPPO** (Multi-Agent PPO with a Centralised Critic).

---

## Architecture

The project uses a `src/swarmecho/` package with separate environment, model, training, analysis, visualization, and map-configuration modules.

```
SwarmEcho/
├── docs/                     ← Technical reports & documentation
├── src/
│   └── swarmecho/
│       ├── core/             ← Structured configuration & utilities
│       ├── curriculum_config/← Retained levels, maps, and maze builder
│       ├── env/               ← JAX physics, state, rewards, observations
│       ├── models/             ← MAPPO actor, critic, and recurrent modules
│       ├── training/           ← Rollouts, PPO, evaluation, and analysis jobs
│       └── visualize/          ← OpenCV rendering and preview tooling
└── README.md
```

---

## Installation

**Prerequisites:** WSL2 with CUDA, `uv` package manager.

```bash
# Clone the repo
git clone https://github.com/yourname/SwarmEcho
cd SwarmEcho

# Install all dependencies (including JAX CUDA wheels)
uv sync

# The training banner reports the devices selected by JAX.
```

---

## Quick Start

### 1. Run the Core Workflow Validation
Run the minimal train-update-evaluate-checkpoint-render workflow.
```bash
uv run swarmecho-validate
```

### 2. Start Curriculum Training
Train the maintained default small-maze stage.
```bash
uv run swarmecho-curriculum
```

### 3. Training a Single Level
```bash
# Start small-maze training with its level settings
uv run swarmecho-train level=M01_small_maze

# Overriding parameters via CLI
uv run swarmecho-train level=M01_small_maze training.num_envs=512 logging.wandb_mode=online
```

### 4. Multi-seed Training

Run sequential, reproducible training runs. The command appends `_seed_N` to
the supplied run name:

```bash
uv run swarmecho-multi-train level=M01_small_maze \
  logging.run_name=small_maze_v1 seeds=3 base_seed=42
```

### 5. 2D Grid Maze Builder
Launch the browser-based grid editor to draw maze walls and target no-spawn cells.
```bash
uv run swarmecho-maze-builder
```


### 6. Consolidated Map Preview & Renderer
Render static blueprint images or simulated video/GIF rollouts of any map blueprint.
```bash
uv run swarmecho-render M03_big_maze
```

### 7. Discovery & Analysis Dashboards
Inspect training parameters, curriculum evolution, and evaluation videos across all runs.
```bash
uv run swarmecho-dashboard
# Checkpoint-oriented evaluation artifacts:
uv run swarmecho-eval-dashboard
```

---

## Documentation Directory

To maintain a clean separation of concerns, all deep-dive technical details have been modularized and moved into the `docs/` folder, directly mirroring the `src/` codebase structure:

1. **[01_system_overview.md](docs/01_system_overview.md)**: High-level CTDE architectural layout and JAX `vmap` logic.
2. **[02_configuration_guide.md](docs/02_configuration_guide.md)**: The single source of truth for global parameters, curriculum scale up, and map geometry.
3. **[03_environment_and_physics.md](docs/03_environment_and_physics.md)**: Current 37-dimensional default observation space, optional transitional observation aids, Euler physics, and continuous action clipping.
4. **[04_marl_and_training.md](docs/04_marl_and_training.md)**: Details the MAPPO execution loop, the Centralized Critic Self-Attention, and the exact team reward formulation.
5. **[05_analysis_and_tools.md](docs/05_analysis_and_tools.md)**: Guide to using the local dashboard, exporting OpenCV render videos, and a reference for W&B logging dictionaries.
6. **[06_connectivity_investigation.md](docs/06_connectivity_investigation.md)**: Raw connectivity ownership findings and the deterministic before/after verification workflow.
7. **[07_mappo_acc_architecture_defense.md](docs/07_mappo_acc_architecture_defense.md)**: Argument for the current MAPPO + Agent-Centric Critic architecture over privileged world-state alternatives.

---

## Output Structure

Each training run creates its own directory:
```
outputs/
└── {run_name}/
    ├── checkpoints/
    │   └── ckpt_001000/    ← Orbax checkpoint (Flax NNX state dict)
    └── artifacts/
        ├── train/          ← scheduled training-evaluation artifacts
        └── eval/           ← checkpoint-scoped evaluation artifacts
```

Timestamp suffixes are used only when `logging.use_timestamp_postfix` is
enabled. Curriculum runs use `outputs/curriculum/<curriculum_name>/`.

---

## Core Workflow Validation

```bash
uv run swarmecho-validate
```
