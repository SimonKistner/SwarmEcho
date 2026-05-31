# SwarmEcho

<p align="center">
  <img src="docs/Preview_vid/demo_2x.gif" width="600" alt="SwarmEcho Behaviour Preview" />
</p>

A **GPU-accelerated Multi-Agent Reinforcement Learning** environment built in JAX, training a drone swarm to explore a 2D environment and form a **delay-tolerant communication relay chain** between a base station and a discovered target.

Agents are trained using **MAPPO** (Multi-Agent PPO with a Centralised Critic).

---

## Architecture

The project uses a clean `src/` layout with logical modules and a centralized `curriculum_config` package.

```
SwarmEcho/
├── docs/                     ← Technical reports & documentation
├── src/
│   ├── core/                 ← Configuration & global utilities
│   ├── curriculum_config/    ← Semantic curriculum settings
│   │   ├── base_params.yaml  ← Global base hyperparameters
│   │   ├── levels/           ← level_00..02 yaml overrides
│   │   └── maps/             ← Map YAML geometry definitions
│   │       └── scripts/      ← Map tools (Builder, Baker, Prepper)
│   ├── env/                  ← JAX physics, rewards, and observations
│   ├── models/               ← MARL network architectures (MAPPO, IPPO)
│   ├── tests/                ← Environment smoke tests
│   ├── training/             ← PPO trainers, rollout buffers, and runners
│   └── visualize/            ← Video renderers (OpenCV, Matplotlib)
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

# Verify GPU is detected
uv run python -c "import jax; print(jax.devices())"
# → [CudaDevice(id=0)]
```

---

## Quick Start

### 1. Run the Environment Smoke Test
Verify the JAX physics and reward functions are working correctly.
```bash
uv run python src/tests/smoke_test_env.py
```

### 2. Start Curriculum Training
Train through stages: L0 (Open Field) → L1 (Warehouse) → L2 (Complex Maze).
```bash
uv run python src/training/curriculum.py
```

### 3. Training a Single Level
```bash
# Start Level 0 training with default settings
uv run python src/training/train.py level=00

# Overriding parameters via CLI
uv run python src/training/train.py level=01 training.num_envs=512 logging.wandb_mode=online
```

### 4. Architectural Map Builder
Launch the Streamlit-based architectural designer to create new drone environments.
```bash
uv run streamlit run src/curriculum_config/maps/scripts/map_builder.py
```


### 5. Validate Map Geometry
Run a short trajectory test on all blueprints to verify JAX compatibility and render validation videos.
```bash
uv run python src/tests/validate_maps.py
```

### 6. Discovery & Analysis Dashboard
Inspect training parameters, curriculum evolution, and evaluation videos across all runs.
```bash
uv run streamlit run src/analysis/dashboard.py
```

---

## Documentation Directory

To maintain a clean separation of concerns, all deep-dive technical details have been modularized and moved into the `docs/` folder, directly mirroring the `src/` codebase structure:

1. **[01_system_overview.md](docs/01_system_overview.md)**: High-level CTDE architectural layout and JAX `vmap` logic.
2. **[02_configuration_guide.md](docs/02_configuration_guide.md)**: The single source of truth for global parameters (`base_params.yaml`), map curriculum scale up, and the visual Map Builder.
3. **[03_environment_and_physics.md](docs/03_environment_and_physics.md)**: Deep dive into the 57-dimensional observation space, Euler physics, and continuous action clipping.
4. **[04_marl_and_training.md](docs/04_marl_and_training.md)**: Details the MAPPO execution loop, the Centralized Critic Self-Attention, and the exact team reward formulation.
5. **[05_analysis_and_tools.md](docs/05_analysis_and_tools.md)**: Guide to using the local dashboard, exporting OpenCV render videos, and a reference for W&B logging dictionaries.
6. **[07_forward_history_diversity.md](docs/07_forward_history_diversity.md)**: Optional CDS-inspired role diversity extension for MAPPO.

---

## Output Structure

Each training run creates its own directory:
```
outputs/
└── {run_name}_{timestamp}/
    ├── checkpoints/
    │   └── ckpt_001000/    ← Orbax checkpoint (Flax NNX state dict)
    └── videos/
        └── eval_update_001000.mp4
```

---

## Core Self-Tests

```bash
# Environment Smoke Test (RECOMMENDED)
uv run python src/tests/smoke_test_env.py

# Physics Engine Standalone
uv run python src/env/physics.py

# Observation Space Standalone
uv run python src/env/observations.py

# Neural Network Architecture
uv run python src/models/mappo.py
```
