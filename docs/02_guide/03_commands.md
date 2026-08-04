# SwarmEcho Command Reference

This document serves as a cheat sheet for all common execution commands in the SwarmEcho project, organized by the logical workflow of designing, testing, training, and analyzing.

Unless noted otherwise, run all commands from the project root directory.

---

## Environment Setup & Smoke Tests

### 1. Activate Virtual Environment
If you prefer to work inside the virtual environment rather than prefixing commands with `uv run`:
```bash
source .venv/bin/activate
```

---

## Map Design & Analysis Dashboard

### 2D Grid Maze Builder
Launch the browser-based editor to draw grid mazes and target no-spawn cells:
```bash
uv run swarmecho-maze-builder
```

### Discovery & Analysis Dashboard
Launch the Streamlit visualization tool to check run stats, compare parameters, and view evaluation/rendering videos:
```bash
uv run swarmecho-dashboard
```

### Consolidated Map Preview & Renderer
Render static images or short video/GIF rollouts (always simulated) of any map blueprint. All outputs are automatically saved to `outputs/map_previews/`:
```bash
# Render default static PNG blueprint image (showing spawn zones and real spawn positions)
uv run swarmecho-render M03_big_maze

# Render a simulated video rollout (MP4, 10 frames, 3 drones, 1 target, 1 base)
uv run swarmecho-render M03_big_maze --mode video

# Render a clean architectural SVG blueprint without spawns or zones
uv run swarmecho-render M03_big_maze --format svg --no-spawns --no-zones

# Render a snappy simulated GIF preview using a specific level config
uv run swarmecho-render M01_small_maze --mode gif --level M01_small_maze
```

---

## Drone Swarm Training

### Single Run Training
Train a drone swarm on a specific level:
```bash
uv run swarmecho-train level=M01_small_maze logging.run_name=maze01_v1 logging.wandb_mode=disabled
```

### Multi-Seed Sequential Training
Run multiple training iterations with the same configuration, utilizing different random seeds. The run name will automatically receive a `_seed_N` suffix:
```bash
uv run swarmecho-multi-train level=M01_small_maze logging.run_name=maze01_v2 seeds=5 base_seed=99
```

### General Grid Search
Sweep arbitrary configuration values for one level. Repeat `--grid` for each
parameter axis and use `--set` for overrides shared by every run:
```bash
uv run swarmecho-grid-search M01_small_maze \
  --grid training.lr=0.0001,0.0003 \
  --grid training.ent_coef=0.0,0.01 \
  --set evaluation.eval_video=false
```

### Curriculum Training
Train through sequential levels (inheriting checkpoint weights from the previous level) either with default stages or custom levels:
```bash
# Run the maintained default small-maze stage
uv run swarmecho-curriculum

# Run a custom sequence of levels
uv run swarmecho-curriculum levels=M00_no_maze_open_square,M03_big_maze,M02_mid_maze,M01_small_maze
```

---

## Spatial Failure Analysis Pipeline

### Run Evaluation Sweep & Generate Heatmaps
Simulate the configured parallel evaluation batch, save comprehensive target/outcome/distance data, and generate heatmaps (failed chain targets, and found-and-delivered or split target-not-found heatmaps):
```bash
uv run swarmecho-evaluate-pipeline checkpoint=outputs/my_run/checkpoints/ckpt_001000
```

---

## 3D Migration Performance Gate

Run config-driven recurrent 3D training. The command writes a final Orbax
checkpoint, scalar metrics, and a renderer-independent replay below the output
directory:

```bash
uv run swarmecho-train-3d level=M00_no_maze_open_cuboid_3D
```

The 3D entry point deliberately uses the same OmegaConf-style `key=value`
contract as maintained 2D training. For example, the quickest normal-pipeline
inspector smoke run is:

```bash
uv run swarmecho-train-3d level=M00_no_maze_open_cuboid_3D training.total_timesteps=400000 logging.run_name=inspector_smoke logging.wandb_mode=disabled
```

The artifact layout is unchanged from maintained 2D runs. Checkpoints remain in
`outputs/<run>/checkpoints/`; scheduled training inspection artifacts live in
`outputs/<run>/artifacts/train/replays/`; and manual checkpoint evaluations live
in `outputs/<run>/artifacts/eval/ckpt_<update-and-steps>/replays/`. A 3D replay
occupies the role of a 2D MP4 and uses the same canonical
`u<update>_s<environment-steps>` suffix. There is intentionally no special
`replays/latest.json` path that bypasses this artifact contract.

Branch a new run from existing weights with
`training.checkpoint_path=outputs/M00_no_maze_open_cuboid_3D/checkpoints/ckpt_000500`.

Evaluate a saved checkpoint deterministically and write a standalone replay:

```bash
uv run swarmecho-evaluate-3d checkpoint=outputs/M00_no_maze_open_cuboid_3D/checkpoints/ckpt_001000
```

The evaluator defaults to `mode=parallel`, which writes the checkpoint-scoped
parallel evaluation CSV. Use `mode=selective_auto_pick result=success offset=0`
to replay a target selected from the closest CSV, or
`mode=selective_manual_pick target_position=x,y,z` for an explicit target.
The old `parallel_eval=true|false` argument remains accepted for compatibility.

Inspect any completed replay from a separate terminal. The inspector is a
standalone browser process with orbit/zoom/pan, playback and scrubbing, coverage
and communication toggles, reward/status readouts, and transparent shell. It
discovers completed replays below `outputs/` and presents them in a selector, so
no artifact path is required:

```bash
uv run swarmecho-inspect-3d
```

Validate the complete 3D environment → recurrent TarMAC actor/critic → rollout
buffer → GAE → MAPPO gradient-update contract on a deliberately small batch:

```bash
uv run swarmecho-validate-3d
```

Run the minimum 3D cuboid environment through JIT and VMAP. Comma-separated
values produce the CPU/CUDA comparison matrix:

```bash
uv run swarmecho-benchmark-3d num_envs=256,1024,4000 radar_bins=8,16,32 grid=4x4x4,12x12x8 steps=200 output=benchmark_3d_cuda.json
```

The harness uses random actions and one compiled `lax.scan` that calculates
observations on every step, matching rollout structure more closely than a
Python loop around a step-only kernel. The JSON report records the selected JAX
backend and devices, compilation and run times, environment steps per second,
state/observation shapes, ideal chain margin, and device memory statistics when
the backend exposes them. The `grid` matrix is important: `4x4x4` is only a
correctness case, while larger entries expose volumetric-coverage scaling.
