# SwarmEcho

> GPU-accelerated Multi-Agent Reinforcement Learning — JAX drone swarm forming a delay-tolerant communication relay chain.

A swarm of N drones explores a bounded environment to locate a target, then autonomously positions to bridge a communication chain between the target and a fixed base station. Trained end-to-end with MAPPO using JAX + Flax NNX entirely on GPU.

---

## Prerequisites

### 1. WSL2 with CUDA (Windows users)

JAX GPU support on Windows requires WSL2. Set it up once:

```powershell
# In Windows PowerShell (Admin):
wsl --install         # installs Ubuntu by default
wsl --update
```

After reboot, open your Ubuntu terminal. NVIDIA drivers stay on Windows — WSL2 bridges automatically. Verify GPU is visible:

```bash
nvidia-smi   # should show your RTX 4090
```

> **Important — venv location**: Create the uv virtual environment **inside the WSL2 filesystem** (e.g. `~/venvs/swarmecho/`), NOT on the mounted Windows drive (`/mnt/q/...`). Cross-filesystem venvs cause slow I/O and version resolution issues. The source code on Q: is accessed in-place via the mount.

### 2. Install `uv` inside WSL2

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc    # or restart terminal
```

---

## Setup

```bash
# Navigate to the project (inside WSL2, Q: is at /mnt/q/)
cd /mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho

# Create the virtual environment inside WSL2's own filesystem to avoid
# cross-filesystem penalties. uv respects UV_PROJECT_ENVIRONMENT.
export UV_PROJECT_ENVIRONMENT=~/venvs/swarmecho
uv sync

# Verify JAX sees your GPU
uv run python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

> If `jax.devices()` shows only `[CpuDevice]`, check that `jax[cuda12]` installed correctly:
> ```bash
> uv run python -c "import jaxlib; print(jaxlib.__version__)"
> # Should contain "cuda"
> ```

---

## Project Structure

```
SwarmEcho/
├── pyproject.toml          # uv project manifest
├── configs/
│   └── default.yaml        # master config (OmegaConf)
├── swarmecho/
│   ├── config.py           # config loader
│   ├── env/
│   │   ├── state.py        # EnvState JAX PyTree
│   │   ├── physics.py      # physics_step() — pure JAX
│   │   ├── observations.py # get_observations(), adjacency
│   │   └── rewards.py      # two-phase reward function
│   ├── models/
│   │   └── actor_critic.py # Flax NNX Actor-Critic
│   ├── training/
│   │   ├── ppo.py          # PPO rollout + loss
│   │   └── train_state.py  # TrainState dataclass
│   └── visualize/
│       └── renderer.py     # render_video() → .mp4
├── scripts/
│   ├── test_physics.py     # physics sanity check → mp4
│   └── train.py            # main training entry point
└── outputs/                # .gitignored — videos, checkpoints
```

---

## Running

### Physics sanity check (Step 3)
```bash
uv run python scripts/test_physics.py
# Opens outputs/videos/physics_test.mp4
```

### VRAM dry-run check
```bash
uv run python scripts/test_physics.py --dry-run
# Prints estimated VRAM usage without running the full rollout
```

### Training
```bash
uv run python scripts/train.py
```

With config overrides (OmegaConf CLI syntax):
```bash
uv run python scripts/train.py env.num_agents=16 training.lr=1e-4 training.num_envs=2048
```

### WandB login (first time)
```bash
uv run wandb login
```

---

## Config

All hyperparameters live in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `env.num_agents` | 8 | Swarm size |
| `env.comm_radius` | 25.0 m | Communication link range |
| `env.visual_radius` | 15.0 m | Target detection range |
| `training.num_envs` | 1024 | Parallel environments (RTX 4090) |
| `training.total_timesteps` | 50M | Training budget |

---

## Algorithm

- **Algorithm**: IPPO (Independent PPO) with parameter sharing + one-hot agent IDs
- **Reward**: Global cooperative — all agents share the same scalar reward
- **Observation**: Ego-centric with 4-bit target knowledge encoding (self-seen / comm-seen + their complements)
- **Action**: Continuous 2D force vector, clipped to `max_force`
- **Vectorisation**: `jax.vmap` across `num_envs`, `jax.lax.scan` for rollouts — entire training loop on GPU
