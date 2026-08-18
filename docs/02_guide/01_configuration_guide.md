# Configuration & Curriculum Guide

SwarmEcho uses a modular, layered configuration ecosystem. Default values are defined directly in Python dataclasses, and custom settings or difficulty levels are layered on top via YAML files.

> **3D migration note:** the executable 3D slice uses strict configs from
> `curriculum_config/levels/` rather than merging them through the legacy 2D
> level loader. `M00_no_maze_open_cuboid_3D.yaml` selects its map and declares its
> physics, radar, episode, connectivity, target-distance, and reward parameters.
> Unknown fields are rejected, and a level is rejected when its ideal straight
> relay cannot reach the map's farthest top corner. This becomes the main
> configuration path as production training moves to 3D.

For 3D levels, `env.coverage_voxel_size` controls only exploration-coverage
resolution, in metres. It defaults to the map's `cell_size_m`, preserving the
legacy one-voxel-per-building-cell behaviour. For example, a 20 m cuboid map
with `cell_size_m: 5.0` and `coverage_voxel_size: 2.5` keeps its 4 × 4 × 4
building grid but uses an 8 × 8 × 8 coverage grid. The voxel size must evenly
divide all three world dimensions.

All environments default to `env.no_movement_termination_steps: 50`: an
episode ends when no agent displaces by more than `env.movement_epsilon`
(default `0.001` m) for 50 consecutive transitions. This catches stalled
rollouts while allowing slow residual motion to settle. It is an ordinary
failed termination with no additional terminal reward penalty.

## 1. Parameters & Where to Find Them
Rather than a global YAML file, baseline configurations are declared in the structured dataclasses in `src/swarmecho/core/config.py` (specifically `SwarmEchoConfig`). The configuration structure is separated into semantic domains:

| Domain | Description / Usage |
|---|---|
| `env` | World physics, vision and communication limits, agent count, and map name. |
| `reward` | Scaling coefficients for exploration, chain gaps, and success bonuses. |
| `training` | PPO optimization, rollout shape, total budget, seed, and checkpoint loading/resume behavior. |
| `evaluation` | Parallel evaluation, evaluation-success early exit, videos, evaluation data/heatmaps, and checkpoint saving. |
| `network` | Hidden layer dimensions, number of layers, activation types. |
| `logging` | Run naming, output roots, Weights & Biases settings, and training-log frequency. |
| `visualize` | Appearance and diagnostic-overlay controls used by the maintained renderer. |

The recurrent MAPPO extension is controlled from `network`:

| Key | Default | Effect |
|---|---:|---|
| `actor_memory` | `true` | Uses the maintained per-agent GRU actor. |
| `critic_memory` | `true` | Uses the maintained per-agent GRU critic before cross-agent attention. |
| `critic_type` | `observation` | `observation` preserves the original joint-observation critic exactly; `privileged` uses compact simulator-state tokens during training only. |
| `memory_comm_enabled` | `true` | Enables TarMAC communication for the recurrent actor. Requires `actor_memory: true`. |
| `memory_comm_every_k_steps` | `5` | Defines the static agent-agent communication slots; base replay also uses this cadence. |
| `tarmac_sig_dim` | `16` | Query/signature dimension used for TarMAC sender addressing. |
| `tarmac_val_dim` | `32` | Value/message dimension used for TarMAC communicated payloads. |
| `tarmac_include_self` | `false` | If enabled by a level override, adds the receiver's own previous signature/value as an attention candidate when it is already receiving an external message. |

The privileged critic remains training-only and does not change actor inputs or
evaluation actions. It consumes exact normalized kinematics, base/target
geometry, task and connectivity flags, the communication adjacency graph,
previous-step collision/coverage diagnostics, compact global coverage
summaries. The full map and coverage grid are not copied into the rollout
buffer. Its attention is masked by the exact
communication graph, so the critic GRU can accumulate multi-hop information
over time without receiving actor hidden states.

When recurrent actor communication is enabled, SwarmEcho uses a single-round TarMAC-style mechanism rather than the older configurable hidden-state attention path. Agents carry GRU hidden state plus previous `(signature, value)` communication state; the base relay stores the first target-knowing reporter's emitted TarMAC signature/value and replays that token to eligible non-knowing agents in base range.

### 1.1 Evaluation, Early Exit, and Saved Artifacts

Evaluation metrics always come from the parallel evaluator. The following
settings live under `evaluation`:

| Key | Default | Description / Effect |
|---|---:|---|
| `eval_freq` | `20` | Evaluation-metric frequency in training updates. |
| `eval_offset` | `1` | Update offset applied to the metric schedule. |
| `eval_min_train_success` | `0.0` | Minimum recent training success required before a scheduled evaluation runs. |
| `eval_parallel_envs` | `4000` | Number of environments in each parallel evaluation batch. |
| `eval_broadcast_on_curriculum_early_stop` | `false` | Fills remaining scheduled W&B evaluation points with the terminal evaluation metrics when a level exits early. |
| `early_exit` | `false` | Enables training exit when parallel evaluation success reaches the threshold. |
| `early_exit_threshold` | `0.99` | Parallel evaluation success rate required for early exit. |
| `eval_video` | `true` | Enables scheduled training replays and a checkpoint-scoped evaluation plus replay after the final checkpoint is saved. |
| `eval_video_freq` | `20` | Evaluation-video frequency in training updates. |
| `eval_video_offset` | `1` | Update offset applied to the video schedule. |
| `training_heatmap_creation` | `false` | Persists during-training evaluation CSV/manifest data and heatmaps. Standalone evaluation always creates its checkpoint-scoped heatmap data. |
| `eval_not_deliv_not_visual_splitt_in_two` | `false` | If true, splits the not-delivered/not-visual heatmap into two separate files instead of one found-and-delivered heatmap. |
| `save_model` | `true` | Enables scheduled and final checkpoint saves. |
| `checkpoint_freq` | `50` | Scheduled checkpoint frequency in training updates. |
| `checkpoint_offset` | `0` | Update offset applied to the checkpoint schedule. |
| `checkpoint_dir` | `null` | Optional checkpoint output directory; otherwise the run checkpoint directory is used. |

When `training_heatmap_creation` is enabled, the during-training evaluation CSV is written under `artifacts/train/data`; manual checkpoint evaluation always writes it under that checkpoint's
`artifacts/eval/.../data` folder. A filename such as
`eval_info_u000700_s00070M.csv` contains `x`, `y`, `stage`, and
`distance_to_base`. `stage` is one of `not_found`, `visually_found`,
`found_and_delivered`, or `chain_success`. Collection is performed once after
the parallel evaluation batch; it does not add work to the simulation loop.
Heatmaps are filtered from this canonical CSV. Evaluation heatmap and video
filenames use synchronized update/step suffixes such as
`u000700_s00070M`.

When `early_exit` is reached, training always saves a handoff checkpoint and
returns. This advances a curriculum runner to the next level and ends a solo
run. `save_model` controls scheduled and final saves; it does not suppress this
required handoff checkpoint.

Checkpoint loading remains under `training`: `checkpoint_path`,
`checkpoint_step_offset`, and `ckpt_loading_mode` control input semantics.
`resume` continues the checkpoint's update and cumulative-step timeline;
`branch` starts updates at zero while preserving that timeline for curriculum
history; `init` restores weights only and starts a completely new run at zero
steps with no inherited history. Checkpoint saving belongs to `evaluation` because it follows the
same update schedule as evaluation and video artifacts.

When `training_heatmap_creation` is enabled, the training pipeline
combines "Not Visually Found" (sky blue) and "Not Delivered" (dark blue)
target groups into one found-and-delivered heatmap by default. Setting
`eval_not_deliv_not_visual_splitt_in_two: true` writes separate heatmaps.

## 2. Curriculum Overrides & Level Structure
The training system scales difficulty sequentially via the `levels/` directory.
- The structured defaults in `src/swarmecho/core/config.py` act as the overarching default configuration.
- Level override files are located in `src/swarmecho/curriculum_config/levels/` and act as patches. The selected level YAML overrides the base definitions.
- Maintained levels are the M-series maze/open-square configurations.

## 3. Map Geometry (`src/swarmecho/curriculum_config/maps/`)
Maps are fundamentally defined via YAML layouts (rooms, hallways, walls) which are then rasterized into boolean occupancy grids for simulation.
- **The Grid Maze Builder:** A browser-based cell editor for drawing maze walls and target no-spawn cells. It creates matching map and level YAML files and remains a useful reference for future 3D map tooling. (`src/swarmecho/curriculum_config/maps/scripts/maze_builder/`)
- **Validation Tools:** Scripts inside `src/swarmecho/curriculum_config/maps/scripts/` check generated and hand-authored maps for compatibility with the maintained environment rules.

Each map defines `spawn_zones.base`, `spawn_zones.drone`, and
`spawn_zones.target` as `[x_min, y_min, x_max, y_max]`. Base, drone, and target
positions are always sampled from these map-defined zones. A zero-area zone
such as `[25, 25, 25, 25]` defines one exact spawn point.

Valid target positions must be free cells inside the target zone. Maps can
further restrict them with:

- `target_wall_clearance`: minimum distance from walls, in metres.
- `target_exclude_zones`: rectangular no-spawn areas written as
  `[x_min, y_min, x_max, y_max]`.
- `target_exclude_circles`: circular no-spawn areas written as
  `[centre_x, centre_y, radius]`.

Map loading raises a configuration error if these constraints leave no valid
target positions. A single-point target zone remains valid when its point is
free and outside all exclusions.

### Map Gallery Preview
The retained level and map sequence is:

1. `M00_no_maze_open_square`
2. `M03_big_maze` (9×9 cells)
3. `M02_mid_maze` (7×7 cells)
4. `M01_small_maze` (5×5 cells)

`M04_tiny_grid_maze_CORE_ONLY_TEST` remains a dedicated core validation
profile and uses the retained `M01_small_maze` map.
### 3D obstacle evaluation

When `env.num_obstacles > 0`, evaluation automatically uses one static layout
for every independently sampled target. Define handcrafted cuboids with
`evaluation.eval_fixed_obstacle_bounds` entries in
`[min_x,min_y,min_z,max_x,max_y,max_z]` order. The target CSV stores only
per-target results; one adjacent layout JSON stores the shared cuboids.
