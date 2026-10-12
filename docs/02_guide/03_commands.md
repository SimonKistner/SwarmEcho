# Supported commands

Run Python/ML commands in the existing WSL environment from the repository root.

## Train

```bash
uv run swarmecho-train level=M00_no_maze_open_cuboid
uv run swarmecho-train level=B01a_office_find_only logging.run_name=office logging.wandb_mode=disabled
uv run swarmecho-multi-train level=M00_no_maze_open_cuboid seeds=3 base_seed=42 logging.run_name=baseline
uv run swarmecho-curriculum levels=M00_no_maze_open_cuboid,M01_no_maze_open_cuboid_tall logging.run_name=curriculum
```

Resume compatible weights and run accounting:

```bash
uv run swarmecho-train level=M00_no_maze_open_cuboid_training.checkpoint_path=outputs/RUN/checkpoints/ckpt_000050 training.ckpt_loading_mode=resume
```

Replace RUN and checkpoint placeholders with real paths. Use `branch` for
cumulative handoff or `init` for weights with fresh counters.

## Evaluate and generate replays

Parallel evaluation:

```bash
uv run swarmecho-evaluate checkpoint=outputs/RUN/checkpoints/ckpt_000050 mode=parallel
```

Parallel evaluation followed by selected successful replays:

```bash
uv run swarmecho-evaluate checkpoint=outputs/RUN/checkpoints/ckpt_000050 mode=parallel replay_after=true result=success replays=3 offset=0 eval_name=review
```

Select from an existing evaluation, or request an XYZ position:

```bash
uv run swarmecho-evaluate checkpoint=outputs/RUN/checkpoints/ckpt_000050 mode=selective_auto_pick result=fail replays=1
uv run swarmecho-evaluate checkpoint=outputs/RUN/checkpoints/ckpt_000050 mode=selective_manual_pick target_position=7.5,12.5,7.5
```

Manual positions must be valid for the checkpoint's map. The evaluator infers
the level where checkpoint metadata permits; an explicit `level=...` takes
precedence. Other configuration overrides remain available.
`replay_execution=parallel_capture` selects the action-capturing replay path.

## Inspect and analyze

```bash
uv run swarmecho-inspect root=outputs
uv run swarmecho-inspect root=outputs/RUN port=8765 open=false
uv run swarmecho-dashboard
```

`swarmecho-eval-dashboard` and `swarmecho-artefacts` are aliases for
`swarmecho-inspect` with identical options.

Export a selected replay or evaluation heatmap as a static public page:

```bash
uv run swarmecho-inspect export=outputs/RUN/artifacts/eval/replays/REPLAY.json name=example-replay "title=Selected replay"
uv run swarmecho-inspect export=outputs/RUN/artifacts/eval/HEATMAP_info.csv name=example-heatmap
```

The inspector also provides an **Export public page** button. See
[the public inspector guide](04_public_inspector.md) for export options,
previewing, and GitHub Pages setup.

## Build maps

```bash
uv run swarmecho-maze-builder --port 8766
uv run swarmecho-validate-building src/swarmecho/curriculum_config/maps/custom_building.yaml
```

The builder command serves the building editor.

Generate the C curriculum maps and preview the requested checkpoint on them:

```bash
python src/swarmecho/curriculum_config/maps/scripts/C_curriculum_map_generator.py
uv run swarmecho-preview-c-curriculum --checkpoint outputs/curr_cellarBuilding_B02cs1/checkpoints/ckpt_000540 --wall-sensing-layout fibonacci --wall-sensing-rays 16 --wall-sensing-ray-max-distance 4
```

See [C curriculum](05_C_curriculum.md) for sequential training, level settings,
target-band certificates, evaluation budgets and replay sampling options.

## Validate and benchmark

```bash
uv run swarmecho-validate
uv run swarmecho-benchmark num_envs=64 steps=64 wall_sensing_rays=8 drone_radar_bins=8
uv run swarmecho-benchmark num_envs=64 steps=64 wall_sensing_rays=32 drone_radar_bins=16 radar_mode=band_sep_counting distance_bands=4 radar_count_cap=3
uv run swarmecho-benchmark num_envs=64 steps=64 wall_sensing_layout=xyz wall_sensing_rays=14,18,26 wall_sensing_ray_max_distance=8 drone_radar_bins=16 radar_mode=band_sep_counting distance_bands=3
uv run pytest tests
uv run pytest tests/test_environment.py -k counting_radar -v
JAX_PLATFORMS=cpu uv run pytest tests/test_wall_sensing.py -v
node --test tests/test_map_builder_ui.cjs tests/test_inspector_visibility.cjs
```

Validation and benchmark implementations are packaged under
`src/swarmecho/training/`, matching their console entry points.
The validator checks one small update; the full tests and ordinary
training/resume/evaluation workflows provide broader coverage.

Benchmark radar options, including wall layout and range, accept comma-separated
values and run their Cartesian product. Every layout/count combination must
be valid; XYZ supports only 14, 18, or 26 rays. Use
`radar_mode=band_sep_counting distance_bands=1` to benchmark counts
without distance subdivision; `distance_bands=0` is equivalent. Legacy mode
ignores band and count-cap settings.

The benchmark measures compilation/runtime performance and observation shapes;
it does not assert correct spatial assignment or visibility. The focused
`counting_radar` tests run JIT observations for controlled scenes and compare
exact cells: capped category counts, each radial boundary crossing, 0/1 single
bands, self/inactive/out-of-range filtering, and authored/generated obstruction
checks that preserve visible peers while excluding blocked ones.

`test_wall_sensing.py` verifies XYZ unit directions, the six exact axes,
14/18/26 ray groups and cubic symmetry, plus unchanged legacy direction order.
It checks every XYZ ray against analytic first-hit distances in a closed cell,
then checks actual compiled 2 x 2 m windows from both sides of X/Y walls: frame
hits, passage through the opening, the first wall beyond it and range clipping.
Eager and JIT observations are compared. Separate checks verify that wall range
controls both reach and normalization independently of target visibility, that
target visibility still follows `visual_radius`, and that the C levels select
XYZ/26/8 m. The suite includes a window that XYZ detects at 6 m while the nearest
legacy Fibonacci ray hits its frame. These are geometric correctness tests;
they do not measure policy learning or success rates.
