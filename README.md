# SwarmEcho

<p align="center">
  <img src="docs/05_assets/Preview_vid/demo_2x.gif" width="600" alt="SwarmEcho Behaviour Preview" />
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

### 1b. Run the 3D Baseline

Train the strict 3D baseline with the recurrent MAPPO/TarMAC stack:

```bash
uv run swarmecho-train-3d --level M00_no_maze_open_cuboid_3D
```

Inspect the latest replay independently while training continues:

```bash
uv run swarmecho-inspect-3d outputs/M00_no_maze_open_cuboid_3D/replays/latest.json
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

The deep-dive technical documentation is organized by topic in [`docs/`](docs/README.md):

- [Reference](docs/01_reference/): current system architecture, environment, training, and assumptions.
- [Guides](docs/02_guide/): configuration, commands, and analysis tooling.
- [Roadmap](docs/03_roadmap/): cleanup work and future features.
- [Archive](docs/04_archive/): historical and experimental context.

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
