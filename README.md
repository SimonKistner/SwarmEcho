# SwarmEcho

Runtime modules, APIs, and commands use unsuffixed names (for example,
`training/train.py`, `load_level`, and `swarmecho-train`). Map/level names and
new artifact/checkpoint markers also use unsuffixed names. Readers accept the
previous identifiers for saved-run compatibility. After updating this checkout, refresh installed CLI
entry points with `uv sync` in the existing WSL environment.

SwarmEcho trains drone swarms to explore **buildings and cuboid worlds** and
form a communication relay between a base and a target. The runtime uses JAX,
recurrent MAPPO, and optional TarMAC communication.

Training runs in the project's existing WSL2/CUDA environment. From the repository
root in that environment:

```bash
uv sync
uv run swarmecho-train level=M00_no_maze_open_cuboid
```

Inspect replays and evaluation heatmaps, compare runs, or edit buildings:

```bash
uv run swarmecho-inspect
uv run swarmecho-dashboard
uv run swarmecho-maze-builder
```

The general Streamlit run-analysis dashboard remains available for runs.
The inspector opens on port 8765; the building editor uses port 8766.
`swarmecho-eval-dashboard` and `swarmecho-artefacts` are aliases for the spatial
inspector and accept its `key=value` options.

## Training and evaluation

```bash
uv run swarmecho-multi-train level=M00_no_maze_open_cuboid seeds=3 base_seed=42 logging.run_name=baseline
uv run swarmecho-curriculum levels=M00_no_maze_open_cuboid,M01_no_maze_open_cuboid_tall logging.run_name=curriculum
uv run swarmecho-evaluate checkpoint=outputs/RUN/checkpoints/ckpt_000050 mode=parallel replay_after=true replays=3
```

Replace checkpoint placeholders with an existing checkpoint. Level files and
`key=value` overrides control training, rewards, observations, evaluation, and
logging. Existing command names, checkpoint contracts, maps, and levels are
preserved.

## Package layout

- `src/swarmecho/env/`: state, physics, building geometry, roadmaps, and observations.
- `src/swarmecho/models/`: actors, critics, and recurrent communication.
- `src/swarmecho/training/`: training, PPO, checkpoints, evaluation, and validation.
- `src/swarmecho/visualize/`: replay serialization and interactive inspector.
- `src/swarmecho/analysis/`: the general run-analysis dashboard.
- `src/swarmecho/curriculum_config/`: maps, levels, and the building editor.

## Validation

Run these in the existing WSL environment:

```bash
uv run swarmecho-validate
uv run pytest tests
node --test tests/test_map_builder_ui.cjs tests/test_inspector_visibility.cjs
```

The workflow validator exercises a small rollout and PPO update; it is not a
substitute for checkpoint, training/resume, evaluation, and UI regression checks.
The benchmark is available as `swarmecho-benchmark`.

See the [documentation index](docs/README.md), [commands](docs/02_guide/03_commands.md),
[PPO controls](docs/01_reference/05_ppo_controls.md), and
[removal audit](docs/03_roadmap/05_2d_removal_audit.md).
Historical 2D material is retained only as explicitly labeled documentation.
