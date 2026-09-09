# SwarmEcho Command Reference

This document serves as a cheat sheet for all common execution commands in the SwarmEcho project, organized by the logical workflow of designing, testing, training, and analyzing.

Unless noted otherwise, run all commands from the project root directory.

## HOW TO EVAL 3D

### Design and inspect the static M02 obstacle layout

Edit `EVAL_FIXED_OBSTACLE_BOUNDS` at the top of
`tools/inspect_obstacle_eval_layout.py`. Each row is exactly
`(min_x, min_y, min_z, max_x, max_y, max_z)`, the format accepted by
`evaluation.eval_fixed_obstacle_bounds`. Generate and inspect the cuboids,
roadmap, and up to five routes:

```bash
uv run python tools/inspect_obstacle_eval_layout.py
uv run swarmecho-inspect-3d root=outputs
```

Copy the edited tuple rows into the M02 level YAML.

### Robust evaluation plus diverse successful replays

Obstacle levels automatically use their configured static layout for all 4,000
targets and store that layout once beside the target CSV. Replay candidates are
100%-agreement successes across the robustness runs. `replays=N` greedily
balances final chain length with spatial separation.
Selection is performed entirely from the completed CSV's target positions,
unanimous outcome stage, and final chain lengths. Only those selected target
and shared-layout configurations are subsequently simulated and recorded as
replays; the 4,000 evaluation episodes are not recorded.

```bash
CHECKPOINT=outputs/<run>/checkpoints/ckpt_<update>
uv run swarmecho-evaluate-3d \
  checkpoint="$CHECKPOINT" \
  mode=parallel \
  replay_after=true \
  result=success \
  eval_name=agents7 \
  replays=3
uv run swarmecho-inspect-3d root=outputs
```

Use `replays=1` for only the longest representative.

If the robustness CSV already exists, skip the 4,000-environment evaluation and
render only the selected replays:

```bash
uv run swarmecho-evaluate-3d \
  checkpoint="$CHECKPOINT" \
  mode=selective_auto_pick \
  result=success \
  replays=3
```

---

## Environment Setup & Smoke Tests

### 1. Activate Virtual Environment
If you prefer to work inside the virtual environment rather than prefixing commands with `uv run`:
```bash
source .venv/bin/activate
```

---

## Map Design & Analysis Dashboard

### Layered 3D Map Builder
Launch the browser editor to paint floor tiles, solid walls, storeys, the base,
and target no-spawn cells. Existing `swarmecho-map/v1` building maps can be
opened directly:
```bash
uv run swarmecho-maze-builder
```

After saving, run the standalone map contract and optional CPU runtime smoke
check. This constructs and resets the same environment functions used by
training but does not train a model or require CUDA:

```bash
uv run swarmecho-validate-building path/to/map.yaml --runtime-smoke
uv run pytest -q tests/test_3d_map_builder.py
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

## Required pre-training checks

Run these commands from the repository root **before the first training
command**. Do not start `swarmecho-train-3d` until every automated command exits
with status 0 and the visual inspection looks correct.

### 1. Focused obstacle, configuration, and inspector tests

```bash
uv run pytest -q \
  tests/test_obstacles3d.py \
  tests/test_config3d.py \
  tests/test_inspector3d.py
```

Expected output: pytest reaches `[100%]`, reports `15 passed in ...s`, and
prints no `FAILED` or `ERROR` section.

### 2. Complete regression suite

```bash
uv run pytest -q
```

Expected output: pytest reaches `[100%]` and ends with all tests passed. Tests
explicitly marked as manual CUDA diagnostics may be reported as skipped; there
must be no failures or errors.

### 3. Generate the inspectable roadmap fixture

```bash
uv run python tests/generate_obstacle_roadmap_testresult.py
```

Expected output:

```text
outputs/testresults/obstacles.roadmap.json
```

### 4. Visually inspect the exact generated layouts

```bash
uv run swarmecho-inspect-3d root=outputs
```

Expected result: the browser opens without a server error and the artifact
dropdown contains `TESTRESULT_[Obstacle roadmap]`. Loading it must show five
selectable layouts, three cuboids per layout, the base and top-layer dummy
target, roadmap nodes and edges, with shortest path 1 enabled by default and up
to four additional path toggles. Verify that no displayed path crosses a
cuboid before continuing.

### 5. Small maintained 3D update validation

```bash
uv run swarmecho-validate-3d
```

Expected output: the validation completes one rollout/GAE/MAPPO update, prints
finite training statistics, and exits successfully without a traceback or
non-finite-value error.

For M02, each evaluation writes a per-environment randomized-layout CSV and a
fixed-layout companion CSV suitable for a spatial heatmap. Both CSVs contain
the obstacle bounds and final physical chain length for every lane. Automatic
successful replay offsets are ordered by descending final chain length, so
`offset=0` selects the longest successful relay route rather than the target
with the greatest straight-line base distance.

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

The 3D trainer has the same sequential multi-seed workflow:

```bash
uv run swarmecho-multi-train-3d \
  level=M00_no_maze_open_cuboid_tall_3D \
  logging.run_name=tall_v1 logging.wandb_group=tall_v1 \
  seeds=3 base_seed=9
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

For 3D, provide the ordered level list directly. The final checkpoint from
each stage is used to initialize the next stage:

```bash
uv run swarmecho-curriculum-3d \
  levels=M00_no_maze_open_cuboid_3D,M00_no_maze_open_cuboid_tall_3D \
  logging.run_name=tall_curriculum logging.wandb_group=tall_curriculum
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

After completing the required pre-training checks, train the randomized
three-cuboid level with its new M02 name:

```bash
uv run swarmecho-train-3d level=M02_random_cuboid_obstacles_3D
```

The 3D entry point deliberately uses the same OmegaConf-style `key=value`
contract as maintained 2D training. For example, the quickest normal-pipeline
inspector smoke run is:

```bash
uv run swarmecho-train-3d level=M00_no_maze_open_cuboid_3D training.total_timesteps=400000 logging.run_name=inspector_smoke logging.wandb_mode=disabled
```

To train with the same bounded pre-`tanh` action perturbation used by robust
checkpoint evaluation, add `training.training_noise=true`. It is disabled by
default; `training.noise_level` defaults to `0.011` (the robust-evaluation
noise level) and can be overridden independently:

```bash
uv run swarmecho-train-3d level=M00_no_maze_open_cuboid_3D training.training_noise=true training.noise_level=0.02
```

This noise is used only to step the training environments. Periodic
during-training evaluations remain unperturbed.

The artifact layout is unchanged from maintained 2D runs. Checkpoints remain in
`outputs/<run>/checkpoints/`; scheduled training inspection artifacts live in
`outputs/<run>/artifacts/train/replays/`; and manual checkpoint evaluations live
in `outputs/<run>/artifacts/eval/ckpt_<update-and-steps>/eval_<timestamp>/replays/`.
Every manual evaluator invocation creates its own `eval_<timestamp>` folder;
use `eval_name=<label>` to name it `eval_<label>_<timestamp>`. A 3D replay
occupies the role of a 2D MP4 and uses the same canonical
`u<update>_s<environment-steps>` suffix. There is intentionally no special
`replays/latest.json` path that bypasses this artifact contract.

Branch a curriculum run from existing weights with
`training.checkpoint_path=outputs/M00_no_maze_open_cuboid_3D/checkpoints/ckpt_000500`.
For a clean new run initialized from existing weights, also set
`training.ckpt_loading_mode=init`; its update counter, logged steps, and
checkpoint history all begin at zero.

Evaluate a saved checkpoint deterministically and write a standalone replay:

```bash
uv run swarmecho-evaluate-3d checkpoint=outputs/M00_no_maze_open_cuboid_3D/checkpoints/ckpt_001000
```

The evaluator defaults to `mode=parallel`, which writes the checkpoint-scoped
parallel evaluation CSV. Use `mode=selective_auto_pick result=success offset=0`
to replay a target selected from the closest CSV, or
`mode=selective_manual_pick target_position=x,y,z` for an explicit target.
For the combined workflow, add `replay_after=true` to a
`mode=parallel` command; it runs the parallel evaluation and then replays the
selected target, using `result=success` and `offset=0` unless overridden.
When launching from WSL, a Windows checkpoint path is accepted directly; quote
the `checkpoint=` argument so Bash preserves the backslashes:

```bash
uv run swarmecho-evaluate-3d mode=parallel replay_after=true result=success offset=0 checkpoint='Q:\_0_Projects\000_SwarmEcho\SwarmEcho\outputs\curr_added_noise_v2_M01\checkpoints\ckpt_001201'
```
It is converted internally to `/mnt/q/_0_Projects/...` before the checkpoint
and its evaluation/replay artifacts are accessed.

If you want a command with no quoting, use forward slashes in the Windows path:

```bash
uv run swarmecho-evaluate-3d mode=parallel replay_after=true result=success offset=0 checkpoint=Q:/_0_Projects/000_SwarmEcho/SwarmEcho/outputs/curr_added_noise_v2_M01/checkpoints/ckpt_001201
```

Alternatively, run the command from PowerShell through `wsl.exe`; PowerShell
does not consume the backslashes:

```powershell
wsl.exe uv run swarmecho-evaluate-3d mode=parallel replay_after=true result=success offset=0 checkpoint=Q:\_0_Projects\000_SwarmEcho\SwarmEcho\outputs\curr_added_noise_v2_M01\checkpoints\ckpt_001201
```

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
## Inspect randomized 3D obstacle roadmaps

Generate five deterministic training-style layouts, their visibility roadmaps,
and up to five alternative shortest routes per layout, then open the ordinary
3D inspector. The generated artifact appears as `TESTRESULT_[Obstacle roadmap]`.

```bash
uv run python tests/generate_obstacle_roadmap_testresult.py
uv run swarmecho-inspect-3d root=outputs
```
