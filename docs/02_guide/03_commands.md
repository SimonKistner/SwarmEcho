# Supported commands

Run Python/ML commands in the existing WSL environment from the repository root.

## Train

```bash
uv run swarmecho-train level=M00_no_maze_open_cuboid_3D
uv run swarmecho-train level=B01a_office_find_only logging.run_name=office logging.wandb_mode=disabled
uv run swarmecho-multi-train level=M00_no_maze_open_cuboid_seeds=3 base_seed=42 logging.run_name=baseline
uv run swarmecho-curriculum levels=M00_no_maze_open_cuboid_3D,M01_no_maze_open_cuboid_tall_logging.run_name=curriculum
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

## Build maps

```bash
uv run swarmecho-maze-builder --port 8766
uv run swarmecho-validate-building src/swarmecho/curriculum_config/maps/custom_3d_building.yaml
```

The builder command serves the building editor.

## Validate and benchmark

```bash
uv run swarmecho-validate
uv run swarmecho-benchmark num_envs=64 steps=64 radar_bins=8
uv run pytest tests
node --test tests/test_3d_map_builder_ui.cjs tests/test_inspector3d_visibility.cjs
```

Validation and benchmark implementations are packaged under
`src/swarmecho/training/`, matching their console entry points.
The validator checks one small update; the full tests and ordinary
training/resume/evaluation workflows provide broader coverage.
