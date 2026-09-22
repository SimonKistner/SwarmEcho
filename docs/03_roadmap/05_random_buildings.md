# Random buildings

## Start training

```sh
uv run swarmecho-train level=B02_random_buildings
```

This level generates one persistent building for each of `training.num_envs`
(default 4,000). The default envelope is 30 × 30 m, with three 5 m stories.
The generator runs at initialization, never during a simulation step. Episode
resets keep the environment's map ID and geometry. Mission sampling continues
to follow the existing training settings.

Existing levels default to `random_buildings.enabled: false`. They retain
their authored map and obstacle behavior. Enabling random buildings requires
`env.map_names: []`, null/omitted map names, or `[random]`. Specifying an authored
map is an initialization error. Random buildings and `env.num_obstacles > 0`
are mutually exclusive.

```yaml
random_buildings:
  enabled: true
  length_m: 30.0                 # X extent, multiple of cell_size_m
  width_m: 30.0                  # Y extent, multiple of cell_size_m
  stories: 3
  cell_size_m: 5.0               # Also story height
  wall_thickness_m: 0.25
  tile_thickness_m: 0.25
  rooms_per_story_min: 4
  rooms_per_story_max: 8
  extra_door_probability: 0.15
  window_probability: 0.2
  staircase_max: 1
  shaft_max: 1
  save_training_maps: false
```

## Generation and geometry

1. Create a closed rectangular exterior, bottom slab, roof, and intermediate floors.
2. Recursively partition each story into randomly sized rectangular rooms.
3. Find room adjacencies. Randomized spanning-tree connections receive doors;
   additional doors and windows create alternate routes.
4. Reserve stair/shaft columns, remove the appropriate intermediate tiles, and
   clear internal walls immediately around connectors on both participating stories.
5. Choose a ground-floor base outside the connector cells. Exclude the base cell
   and connector cells from target sampling; compile and validate the map.

The room-count settings apply before clearing connector access. Clearing walls
can merge some of the partitioned spaces. No minimum final room count or
architectural realism score is claimed.

For each adjacent story pair:

| Stair maximum | Shaft maximum | Sampled connectors |
| --- | --- | --- |
| positive | positive | 1..stair maximum, 0..shaft maximum |
| positive | zero | 1..stair maximum, no shafts |
| zero | positive | no stairs, 1..shaft maximum |
| zero | zero | one staircase override |

Counts are sampled inclusively. With both maxima equal to one, a staircase is
guaranteed and a shaft is optional. Columns are selected once per building;
stairs and shafts use separate columns. A one-story building needs no vertical
connector.

Walls belong to shared faces, not to individual cells:

- `x_walls: [x_boundary, y_cell, z_cell]`
- `y_walls: [x_cell, y_boundary, z_cell]`
- `tiles: [x_cell, y_cell, z_boundary]`

`x_doors`, `y_doors`, `x_windows`, and `y_windows` annotate existing wall faces.
One face cannot contain both opening types. Editing it from either side affects
the same wall, so adjacent cells cannot produce duplicate walls. An opening must
have interior volume on both sides. Exterior doors/windows are rejected.

Doors have a centred opening 40% of cell width and 70% of cell height. Windows
are centred, 40% wide, and span 30–70% of cell height. Windows are open holes,
without glass. The compiler creates solid frame pieces; physics, line of sight,
communication, roadmaps, and rendering use those pieces.

`stairs: [x, y, lower_story, direction]` describes eight solid treads in a two-story
cell column. Directions are 0=+X, 1=+Y, 2=−X, 3=−Y in world coordinates. Treads
occupy the middle 45% of the transverse width, leaving side passages for flying
agents. The tile above the staircase must be absent. These are drone-navigation
structures, not a pedestrian locomotion or building-code simulation.

## Guarantees and limits

- Exterior sealing and cell connectivity from the base are checked by the shared compiler.
- Every adjacent story pair has at least one connector.
- Generator aperture and stair-passage dimensions must admit the configured
  drone radius plus planning clearance.
- Generated training roadmaps are checked for connectivity before training when
  geodesic planning is enabled. A failed graph is reported at initialization.
- Seeds, generator version, and settings are deterministic. Exact regeneration
  also requires the saved environment/planning configuration and generator version.

Connectivity does **not** guarantee a successful relay chain for every target
with the configured number of agents, communication radii, or episode horizon.
The existing ideal-chain bound is only a coarse necessary check. Minimum
geodesic spawn separation can also be infeasible for some targets, just as on
authored maps; its existing bounded sampling/error behavior remains in place.

Generated maps use a compact visibility roadmap with cell centres, doorway/window
portals, and clear lanes around vertical connectors. These candidates are saved
in `roadmap_nodes_m`. Exact shortest routes are computed on that graph; they are
approximations to continuous-space shortest paths. Generation and inspection use
the same host roadmap implementation as training. Corner allowances retain the
existing distinction: the inspection scanner includes `roadmap_corner_bonus_m`,
and training applies that allowance to minimum-separation sampling when enabled.

## Inspect five maps

```sh
uv run python tests/generate_random_buildings_testresult.py
# Equivalent:
uv run python -m swarmecho.analysis.generate_random_maps
```

The script loads `B02_random_buildings`, calls the training generator, and writes:

```text
outputs/testresults/random_buildings_<timestamp>_seed<seed>/
  generation.json
  generation_level.yaml
  maps/map_0000.yaml ... map_0004.yaml
  map_0000_...shortest_farthest.roadmap.json
  map_0000_...longest_farthest.roadmap.json
  map_0000_...csv
  ...
```

Options include `--level`, `--count`, `--seed`, `--spacing`, `--output-dir`,
`--maps-only`, and repeatable `--set KEY=VALUE` level overrides. The seed selects
the same first map IDs as training; it is not a different preview generator.

The existing scanner is reused, preserving both old commands:

```sh
uv run python tests/scan_roadmap_distances.py --map office_mirrored
uv run python tests/generate_obstacle_roadmap_testresult.py --map office_mirrored
```

The scanner rasterizes valid points, finds each point's farthest reachable
partner by shortest roadmap distance, and exports the smallest and largest of
those farthest distances. This is **not** a longest looping path. It is a raster
approximation, not enumeration of infinitely many continuous spawn pairs.
The existing all-point extreme exports and target/base eligibility diagnostics
are retained; extreme endpoints are not restricted to target-eligible cells.

Open `uv run swarmecho-inspect` and refresh discovery. Nested inspection folders
are discovered; generated roadmap groups show their timestamp and map ID.
Each map has two selectable roadmap artifacts.

## Manual building editor

Use `uv run swarmecho-maze-builder`. The door/window tools select the nearest
shared face. The staircase tool selects a lower cell and removes its overhead
tile; a direction selector controls ascent. Clicking an existing staircase
removes it. Use the tile tool to close the remaining opening if desired.

The template selector opens small editable door, window, and staircase examples
using the same primitives. It replaces the current document, like New/Open.
Generated maps can be imported using the YAML file picker or opened directly:

```sh
uv run swarmecho-maze-builder --map-dir outputs/testresults/<export>/maps
```

With `--map-dir`, Open and Save use that folder. Without it, the existing map
directory remains the default. Story inspection, wall painting, resizing, and
layer editing remain available. Shared-face annotations move with those edits.
Import/save preserves the physical geometry and base position. Saving through
the editor discards generated roadmap hints, so subsequent training builds a
fresh general-purpose roadmap from the edited geometry.

## Fixed random-map evaluation

```yaml
evaluation:
  random_eval: true              # Default false; enabled in B02
  random_eval_envs: 5            # Number of persistent MAPS
  random_eval_maps: null         # Or exactly five map names/YAML paths
  training_robustness: false     # Existing robust-eval toggle; required off
  eval_parallel_envs: 4000       # Episodes/parallel lanes on EACH map
  training_heatmap_creation: true
  eval_video: true
```

Maps use a separate seed stream from training, or the exact files referenced by
`random_eval_maps`. Small frozen copies are always written to `random_eval_maps/`
in the run directory. Reopening the run uses those copies, even if the original
reference files were edited or removed. The number/settings should remain fixed
when resuming the same run. Heatmaps and replays share these frozen buildings.

Maps are evaluated sequentially without action noise. Each map gets the existing
parallel evaluation batch. Aggregate metrics are arithmetic means across maps;
the early-stop hold advances once per complete evaluation, not once per map.
Per-map metrics are saved alongside the aggregate. This is not a robustness
agreement ensemble; each target has one noise-free result.

The existing evaluation/replay frequencies, offsets, success gate, heatmap
toggle, and replay toggle are respected. A replay event creates one replay per
fixed map; replay-only events do not advance metric/early-stop counters. The
final training update uses this suite instead of the legacy final noisy ensemble.

The inspector groups a map suite into one entry per artifact kind/evaluation
event, with a map-ID selector. Heatmap/replay switching preserves the map ID.
Heatmap target commands include `random_eval_map_id`, so selected-target replays
use the correct saved building. Standalone `swarmecho-evaluate mode=parallel`
also uses the persistent map suite when `random_eval` is enabled.

## Memory, initialization, and future regeneration

Training arrays have fixed padded capacities, shared dimensions, and one
persistent map ID per environment. Inactive solid slots sit outside the world;
inactive roadmap entries have an unreachable cost, and padded spawn boxes have
zero sampling weight. Geometry and roadmaps are built once on the CPU and
transferred once. The immutable roadmap bank has no JAX rollout-tree leaves, so
its matrices are not copied into timestep histories. Per-lane collision bounds
remain persistent state arrays. Existing bounded solid-intersection loops are
reused; there is no new spatial acceleration structure in this version.

`logging.terminal_log_init: true` reports generation/compilation progress, padded
array sizes, and bank memory. Startup planning cost and memory grow with map
count, solids, and especially roadmap vertex count squared. The compact portal
graph controls this cost, but 4,000-map GPU throughput and memory still require a
measurement on the training machine. Increasing dimensions can be expensive.

`save_training_maps: false` avoids thousands of YAML writes. A small
`random_buildings.json` seed/settings manifest is always saved. Enabling map
saving writes the definitions into `generated_maps/`; it does not reduce GPU
memory or provide an OOM recovery mechanism.

Automatic regeneration on a success threshold is not implemented. The explicit
bank and map IDs provide a place to add versioned replacement banks later, at a
rollout/episode boundary. Evaluation maps should remain fixed when that is added.

## Checks performed for this change

Only lightweight CPU artifact generation, raster reachability checks for the
five examples, editor geometry/base round-trips, template compilation, and
JavaScript syntax checks were run. No GPU training run or heavy test suite was
run. A small training-machine smoke run should precede a full run:

```sh
uv run swarmecho-train level=B02_random_buildings training.num_envs=20 training.num_minibatches=1 training.num_steps=8 training.total_timesteps=160 evaluation.random_eval_envs=2 evaluation.eval_parallel_envs=4 evaluation.eval_video=false evaluation.save_model=false logging.wandb_mode=disabled
```
