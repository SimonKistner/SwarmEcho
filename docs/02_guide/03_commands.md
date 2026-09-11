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
The map builder uses port **8766**; the 3D inspector uses **8765**, so both
can run together. Override with `swarmecho-maze-builder --port <port>` or
`swarmecho-inspect-3d port=<port>` if needed.

Launch the browser editor to paint floor tiles, solid walls, storeys, the base,
and target no-spawn cells. Existing `swarmecho-map/v1` building maps can be
opened directly. Drag across tiles or no-target cells to paint rectangular
selections. Click and hold to preview a straight row of walls (cyan to add,
red to remove); drag back to shorten the row and release to apply it.
Use each edge's
`+1`/`-1` controls to copy-expand or trim the footprint. Storeys inherit the
interior volume, walls, and no-target pattern below, removing the old roof and
leaving the new floor and roof empty. Cells count cubic interior volumes,
independently of tiles; painting or removing a tile does not delete the volume
or its no-target setting. **Show no-target filling** hides/shows the overlay
without changing spawn exclusions. **Add roof** creates roof tiles without
switching views, excluding facade cutouts open to the outside and removing old
top-roof overhangs. Enclosed upper rooms do not need floor tiles to receive a roof.
Select **Roof** in Current storey or Structure to inspect the
whole building with its opaque roof. **Copy current YAML** copies the exact payload used by validation
and saving. Drag outside the floor (or select Orbit) to rotate the structure;
the mouse wheel zooms:
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

The editor's volume/tile and roof controls also have standalone JavaScript
regression checks: `node --test tests/test_3d_map_builder_ui.cjs`.

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
uv run python tests/generate_obstacle_roadmap_testresult.py --obstacles
```

Expected output:

```text
outputs/testresults/M01_no_maze_open_cuboid_tall_[<x>-<y>-<z>].roadmap.json
```

### 4. Visually inspect the exact generated layouts

```bash
uv run swarmecho-inspect-3d root=outputs
```

Expected result: the browser opens without a server error and the artifact
dropdown contains the map-named Roadmap entry. Loading it must show five
selectable layouts, three cuboids per layout, the base and top-layer dummy
target, roadmap nodes and edges, with all returned paths enabled and individually
selectable, ranked by length. Verify that no displayed path crosses a
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

The left visibility panel controls interior walls per storey (all on except
the top storey), outer walls (off), and roof (off). Outer walls follow the
storey checkboxes; roof visibility is independent. Concrete walls start at 50%
transparency. The optional experimental height fade starts at an adjustable
height and increases upward to the transparency slider's value. Floors and
roofs remain opaque. Dark tile floors and the wall/floor cell grid are on
by default, with separate visibility toggles. The artifact picker groups each
run into one entry, with its replays and evaluations in a submenu.
Authored geometry is read from the replay's
named map as it currently exists. Visibility persists during playback and
rotation; the default camera elevation is 25°. Elevation, auto-rotation, and
rotation-speed controls are available for both replays and heatmaps.

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
## Inspect 3D map roadmaps and ranked paths

To scan each interior raster point's farthest reachable roadmap partner:

```bash
uv run python tests/scan_roadmap_distances.py --level B01_office --spacing 5
uv run swarmecho-inspect-3d root=outputs
```

This writes a CSV under `outputs/testresults` with point coordinates, farthest
partner, adjusted distance in equivalent metres and drone communication ranges.
It generates only `shortest_farthest` and `longest_farthest` inspectable roadmaps,
one best route for each extreme pair. The longest is the largest shortest-path
distance across sampled pairs, not a deliberately long detour. The restricted
minimum for allowed targets and valid base/drone spawn positions is still printed
in the stats. Unreachable pairs are excluded and reported separately. The scan
uses static authored geometry, the level's roadmap setting and communication
range; generated obstacles are not included. `--map` can override the map.
Spacing defaults to the map cell size; smaller spacing increases cost sharply
because all raster pairs are checked. Results approximate the continuous volume
and do not prove that no qualifying position exists between raster samples.

`env.roadmap_corner_bonus_m` defaults to `8.0`. The scan, standalone roadmap
generator and training spawn-separation check include this allowance **before**
shortest-path selection and ranking. Set it to zero for physical distance alone,
or override scripts with `--corner-bonus-m 8`. The chain reward is unchanged.
The efficient roadmap approximation charges one allowance for the first
intermediate waypoint group and another for each internal link longer than
`wall_thickness_m + 2 * (drone_radius + obstacle_planning_clearance_m)`.
Consecutive shorter links stay in one group, accounting for wall-depth detours.
A direct visible route has no allowance. This is waypoint grouping, not exact
angle-aware corner counting; dense chains of short waypoints can form one group.
The inspector displays the saved adjusted score rather than recomputing physical
polyline length. Older artifacts keep their original scores until regenerated.

Choose a level (uses its map and clearance settings), a target XYZ point in
metres, and the maximum number of paths. For example, for B00:

```bash
uv run python tests/generate_obstacle_roadmap_testresult.py --level B00_test --target 2.5 2.5 12.5 --paths 5
uv run swarmecho-inspect-3d root=outputs
```

Use your desired target coordinates. `--map easy_room` selects a map directly;
it can also override a level's map. `--start X Y Z` overrides the base as the
starting point. The default filename includes the target coordinates, for example
`outputs/testresults/easy_room_[25-14-8].roadmap.json`, so different target points
do not overwrite each other. The same map/target is overwritten on rerun;
`--output` still overrides the filename.

Generated obstacles are **off by default**; authored map walls/tiles remain.
`env.roadmap_merge_walls` defaults to `true`: authored rectangular wall/tile
unions are merged for roadmap generation, with clearance-offset edge samples
spaced evenly at most one map cell apart. Collision geometry is unchanged.
The generator inherits this setting from `--level`, or the general environment
default when using `--map` alone. Use `--no-merge-walls` to compare with the old
generator, or `env.roadmap_merge_walls=false` for training.
To generate randomized layouts with a level's obstacle configuration, use
`--level M02_random_cuboid_obstacles_3D --obstacles --layouts 5 --seed 9000`.

The inspector ranks distinct loopless visibility-graph paths by total length
in metres and displays the difference from the shortest in metres and percent.
All returned paths have individual toggles; fewer than requested may exist.
These are geometric diagnostic costs, not episode reward scores. The diagnostic
includes authored solids and constrains vertices to the building interior; it
does not change the level's configured chain reward or training roadmap.

Training also supports `reward.chain_reward_system=obstacle_geodesic` with
`env.num_obstacles=0`. Authored walls and tiles participate in its roadmap.
For a static map, the roadmap is precomputed on the CPU and shared across
environments; it is not copied into each environment's rollout state. The
in-process cache keys include solid geometry, world bounds and clearance.
Generated obstacles use a per-layout roadmap containing both authored and
generated solids, reused across episode resets. Euclidean training skips
roadmap construction. Future assembled maps will need to supply their own
layout geometry; map assembly itself is not implemented here.

To retry the office level with geodesic rewards in your activated WSL environment:

```bash
uv run swarmecho-train-3d level=B01_office reward.chain_reward_system=obstacle_geodesic
```

B01 now uses 5 m coverage voxels (540 instead of 4,320). Timestamped `[INIT]`
messages identify setup stages, shared-roadmap construction, the first rollout
dispatch/host transfer and the first optimizer update. First-update messages
are emitted once per training invocation, including a resumed run; they are
not recurring training-step logs. They locate the blocking stage but do not
identify individual XLA compiler passes.
