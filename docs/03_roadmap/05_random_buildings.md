# Random buildings

## Start training

```sh
uv run swarmecho-curriculum levels=B02a_random_buildings_find_only,B02b_random_buildings_deliver,B02c_random_buildings logging.run_name=random_buildings
```

Each level generates one persistent building for each of `training.num_envs`
(default 4,000). B02 uses a 20 × 20 m envelope with three 5 m stories.
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
  save_training_maps: false
  load_maps_from_bank: false      # true: reuse a compatible offline map pool
```

Roadmap resolution is configured separately from the raster spacing used by
inspection scans:

```yaml
env:
  roadmap_approach: full                 # full (default) or minimal
  roadmap_node_density: 1.0             # full: at most 1 m between feature nodes on 5 m cells
  roadmap_merge_wall_end_nodes: false   # full only; merge close nodes at perpendicular wall joins
```

For building maps, `full` derives vertices from the completed wall grid:
clearance-safe door and window rims on both wall faces, exposed wall tips,
and stair entrances, exits, and flight spans. It does not include saved cell
centres or sample the horizontal edges of solid wall boxes. Candidate lines
are divided evenly with a maximum interval of
`cell_size_m / (5 * roadmap_node_density)`. A density of 1.0 means 1 m on
B02's 5 m cells; a short line needs no intermediate vertex. Opening corners
and wall-tip endpoints remain at any density. The door bottom has one line
through the clear opening; a window sill and both opening tops have mirrored
lines on the two wall faces. Collision and visibility checks still use the
original solids expanded by drone radius plus planning clearance. This also
applies to authored building maps; explicit saved `roadmap_nodes_m` are used
only by `minimal`. `minimal` uses the compact
cell-centre, portal, and stair hints saved with generated maps. The density
and wall-end settings do not affect `minimal`. Maps without saved compact hints
derive interior cell centres and available door, window, and stair anchors.
The random-map inspection command accepts overrides such as
`--set env.roadmap_approach=minimal` or
`--set env.roadmap_node_density=4.0`.

Optional wall-end merging considers only the close pair left by a perpendicular
L-shaped join. A wall face with a door or window can form such a join because
its ends are solid jambs; it is included. Nodes on the actual opening rims are
protected and remain separate on the two wall faces. A corner pair is replaced
at its midpoint only when the point is clear and can see every other roadmap
vertex visible from either original. Other wall ends remain unchanged. The
completed wall grid already joins collinear wall faces before exposed tips are
selected. Stair enclosure walls use this same rule: their L joins are eligible,
while stair treads, flight nodes, and T or cross joins are not merge candidates.

## Generation and geometry

1. Create a closed rectangular exterior, bottom slab, roof, and intermediate floors.
2. Choose stairs for each adjacent story pair. A stair cannot sit directly
   above another stair. Its lower approach and upper exit must both face an
   ordinary interior cell; its side faces may lie along the exterior wall.
   Reject placements that would split the remaining cells on a story.
3. Recursively partition each story into randomly sized rectangular rooms.
   Randomized spanning-tree connections receive doors; additional doors can
   create alternate routes.
4. Remove the floor tile above each stair. Keep the lower face opposite ascent
   and the upper face along ascent open, using a door if a wall was already
   there. Make the other three faces solid walls on both stories. Add doors
   between ordinary cells where the stair walls cut an existing room route.
   Room reachability at this stage uses only doors and missing walls.
5. Randomly add windows to remaining eligible room walls, excluding stair
   walls. Windows never replace a required door.
6. Choose one free ground-floor corner cell for the saved base, placing it at the
   cell centre on top of the floor tile. Exclude the base cell from target
   sampling; compile and validate the map.

Targets can spawn in the free space of both staircase cells: above/beside the
lower treads and in the upper landing cell. Stair treads use the same
`env.target_wall_buffer_fraction * cell_size_m` clearance from their solid
surfaces as ordinary walls. Spawn boxes are split around buffered treads and
sampled by their remaining volume, shared by training and evaluation. Generated
cuboid obstacles still use `env.obstacle_target_buffer_m`.

Generator version `partition_connect_v6` saves only the base-cell exclusion.
When compiling older generated maps (including B02c static maps and saved
evaluation suites), an exclusion list matching the old base-plus-stairs default
is upgraded in memory to the base cell alone. Custom exclusion lists are
preserved. Source YAML files and existing replay/heatmap data are unchanged.
Older bank records are incompatible with the updated generator/source digest;
bank setup builds a compatible pool instead of reusing obsolete spawn boxes.

The room-count settings apply before stair walls and door repairs. The repairs
may change the final room boundaries. No minimum final room count or
architectural realism score is claimed.

Each adjacent story pair gets between one and `staircase_max` staircases,
subject to the available non-stacked cells and a connected layout. Stairs on
neighbouring cells are allowed, and a cell may be reused after a story pair
with no stair in that column. `staircase_max` must leave at least one other
cell available. A one-story building needs no vertical connector. Shafts are
not generated.

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

The generator may add windows after door connectivity is established, but a
window never satisfies a room's required access. A one-cell enclosed room
cannot be left with windows as its only openings.

`stairs: [x, y, lower_story, direction]` describes eight solid treads in a two-story
cell column. Directions are 0=+X, 1=+Y, 2=−X, 3=−Y in world coordinates. New
random buildings set `geometry.stair_full_width: true`, so treads span the
entire cell width. Older maps without this flag retain 45%-width treads. The
tile above the staircase is absent. For a +X ascent, the lower -X face is the
entrance and the upper +X face is the exit. The other three faces of each cell
are solid; the same rule rotates with the stair direction. Entry and exit can
be doors or missing walls but never solid walls or exterior faces. The walls
restrict access to one face per level; movement through the stair remains
bidirectional. These are drone-navigation structures, not a pedestrian
locomotion or building-code simulation.

## Guarantees and limits

- Exterior sealing and cell connectivity from the base are checked by the shared compiler.
- Every ordinary cell has a route through doors or missing walls; windows are
  excluded when establishing this connectivity.
- Every adjacent story pair has at least one connector.
- Generator aperture dimensions must admit the configured drone radius plus
  planning clearance. New full-width stairs have an open lower entrance and
  upper exit, with roadmap nodes near both openings.
- Generated training roadmaps are checked for connectivity before training when
  geodesic planning is enabled. A failed graph is reported at initialization.
- Seeds, generator version, and settings are deterministic. Exact regeneration
  also requires the saved environment/planning configuration and generator version.

Connectivity does **not** guarantee a successful relay chain for every target
with the configured number of agents, communication radii, or episode horizon.
The existing ideal-chain bound is only a coarse necessary check. Minimum
geodesic spawn separation can also be infeasible for some targets, just as on
authored maps; its existing bounded sampling/error behavior remains in place.

Generated maps save compact cell-centre, doorway/window portal, and stair
entrance/exit candidates in `roadmap_nodes_m` for the `minimal` approach.
`full` derives opening rims and exposed wall tips directly from
geometry, and adds stair entrance/exit spans and a clearance-aware route above
the treads when its points pass collision and visibility checks. Floor slabs,
wall tops, and individual treads remain collision geometry without
contributing all their box-edge candidates. Exact shortest routes on
either visibility graph approximate continuous-space shortest paths. Generation
and inspection use the same host roadmap implementation as training. Corner
allowances retain the existing distinction: the inspection scanner includes
`roadmap_corner_bonus_m`, and training applies that allowance to
minimum-separation sampling when enabled.

## Inspect five maps

```sh
uv run python tests/generate_random_buildings_testresult.py
# Equivalent:
uv run python -m swarmecho.analysis.generate_random_maps
```

The script loads `B02a_random_buildings_find_only` by default, calls the training generator, and writes:

```text
outputs/testresults/random_buildings_<timestamp>_seed<seed>/
  generation.json
  generation_level.yaml
  maps/map_0000.yaml ... map_0004.yaml
  map_0000_...shortest_farthest.roadmap.json
  map_0000_...top5_longest_farthest.roadmap.json
  map_0000_...csv
  ...
```

Options include `--level`, `--count`, `--seed`, `--spacing`, `--top-pairs`, `--output-dir`,
`--maps-only`, and repeatable `--set KEY=VALUE` level overrides. The seed selects
the same first map IDs as training; it is not a different preview generator.

The existing scanner is reused, preserving both old commands:

```sh
uv run python tests/scan_roadmap_distances.py --map office_mirrored
uv run python tests/generate_obstacle_roadmap_testresult.py --map office_mirrored
```

The scanner rasterizes valid points, finds each point's farthest reachable
partner by shortest roadmap distance, and exports the smallest farthest pair
and, for generated random maps, the five largest distinct farthest pairs in
one inspectable artifact. Mirrored pairs count once. Each pair has its own
shortest route; the five routes can be toggled independently in the inspector.
`--top-pairs N` changes the count. This is **not** a longest looping path. It is a raster
approximation, not enumeration of infinitely many continuous spawn pairs.
The existing all-point diagnostics and target/base eligibility diagnostics
are retained; ranked endpoints are not restricted to target-eligible cells.
The longest-farthest pair is chosen by path distance, so its endpoints need not
be on the bottom and top stories. Older generated maps stacked stairs in one
column and left side lanes, allowing a route to jump vertically through the
middle story without a waypoint there. Newly generated maps place successive
stairs in different cells, requiring a horizontal connection on that story.

Open `uv run swarmecho-inspect` and refresh discovery. Nested inspection folders
are discovered. A generated `random_buildings_<timestamp>` suite appears as one
run entry; its submenu lists `map_0000_shortest_farthest`,
`map_0000_top5_longest_farthest`, and the corresponding entries for the other maps.
The ranked view shows each route with its own coloured B (base) and T (target)
markers. Unchecking a route hides its markers as well.
The Roadmap paths panel shows the selected layout's roadmap node count above
the visibility controls; switching layouts updates the count.
Seed and raster settings remain in artifact files but are omitted from the
selector labels.

Roadmap artifacts use the same concrete-wall and dark-tile building renderer as
replays. The left panel can hide an entire storey (including its floor, doors,
windows, stairs, walls and roof pieces), show outer walls and roofs, and adjust
transparency for all visible building pieces. Dark tile floors are always shown
for visible storeys, and transparency starts at 0% across roadmap, replay, and
heatmap views. Roadmap visibility edges start hidden. Extra random obstacles
remain orange. The **Orange obstacle rendering** checkbox at the bottom of the
panel restores the older obstacle-box view. Older artifacts without a usable
building snapshot or matching map also fall back to orange boxes.

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
  replay_targets_from_storey: 3  # 1-based; null samples all valid storeys
  training_robustness: false     # Existing robust-eval toggle; required off
  eval_parallel_envs: 4000       # Episodes/parallel lanes on EACH map
  training_heatmap_creation: true
  eval_video: true
```

Without an offline bank, maps use a separate seed stream from training, or the
exact files referenced by `random_eval_maps`. With `load_maps_from_bank: true`,
training and evaluation select distinct maps from the same compatible pool;
explicit `random_eval_maps` still take priority. Small frozen copies are always written to `random_eval_maps/`
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
`replay_targets_from_storey` limits only automatically generated replay targets
to the requested storey of each frozen building. It is `null` by default, and
all B02 levels set it to `3`. Evaluation episodes, heatmaps, and metrics continue
to sample targets from every valid storey. The replay target still follows the
usual spawn exclusions and geodesic-separation rule.

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
count, solids, and especially roadmap vertex count squared. `full` adds nodes
and can use substantially more memory than `minimal`; the largest graph sets
the padded size for every map in the bank. A 4,000-map run should be measured
on the training machine before increasing density. Increasing building
dimensions can be expensive.

`save_training_maps: false` avoids thousands of YAML writes. A small
`random_buildings.json` seed/settings manifest is always saved. Enabling map
saving writes the definitions into `generated_maps/`; it does not reduce GPU
memory or provide an OOM recovery mechanism.

## Offline random-building pool

Set `random_buildings.load_maps_from_bank: true` to select persistent buildings
and precomputed roadmaps from
`src/swarmecho/curriculum_config/maps/random_building_bank/pools/`. The default
is `false`, which keeps the existing per-run generation and roadmap compilation.
To build a pool ahead of training, run:

```sh
uv run swarmecho-build-random-bank --level B02a_random_buildings_find_only
```

Then enable loading for the curriculum with the shared override:

```sh
uv run swarmecho-curriculum levels=B02a_random_buildings_find_only,B02b_random_buildings_deliver,B02c_random_buildings random_buildings.load_maps_from_bank=true logging.run_name=random_buildings
```

Each manual invocation adds a `DEFAULT_POOL_SIZE` batch to a compatible
pool (currently 5,000 in `generate_bank.py`), or creates a pool when the settings
do not match any existing one. Change that constant for a different manual batch
size. `--workers` selects CPU processes (default: up to four). Without `--seed`,
each batch gets a fresh operating-system random seed; an explicit `--seed` makes
the candidate sequence reproducible. Repeating the same explicit seed skips
maps already in the pool, so it may add fewer than the batch size. Each pool has a
`metadata.json`, one YAML map per building, and one unpadded `.npz` record per
roadmap. The records include solids, nodes, shortest-path matrices, spawn boxes,
and base positions. When the effective corner bonus is zero, only one distance
matrix is stored. A new pool becomes selectable only after its first batch is finished.
Pool data are excluded from Git.
B02a training loads only the solids, base spawn data, and a per-map coverage
eligibility mask from these records. Target spawn boxes and roadmap matrices are
loaded for B02b and B02c. Coverage ignores voxel centres inside authored solids,
so 100% means every observable interior voxel centre has been seen. To certify
that the selected pool's eligible voxels can be reached or viewed, run:

```sh
uv run python -m swarmecho.analysis.check_bank_coverage
```

The checker reports any voxel it cannot certify; `--limit 100` checks a quick
prefix. Its geometric certificate is sufficient for reachability, while an
uncertified voxel may still have a curved route the checker does not model.
The standalone builder prefixes every progress line, warning, and final pool
path with the same `[HH:MM:SS]` timestamp used by training. Automatic pool
generation retains the training setup timing alongside that timestamp.

The metadata records the generator version, a hash of the geometry and roadmap
source files, geometry settings, roadmap approach and density, drone and
planning clearance, corner bonus, and target spawn buffer. Runtime flags and
the training seed do not determine compatibility. Every map also has a SHA-256
content hash that excludes its name and seed metadata. New batches skip repeated
seeds before compiling a roadmap and skip any duplicate geometry before saving.
Older pools acquire content hashes when first extended; existing map IDs remain
unchanged. A pool lock prevents concurrent generation jobs from appending the
same batch at once, and metadata publishes new IDs only after their files exist.
Training logs every setting that differs from an existing pool. If a compatible
pool has too few maps, training extends that pool to at least the number it
needs (or `DEFAULT_POOL_SIZE`, whichever is larger). If no compatible pool
exists, training creates one and continues automatically.
Selection is reproducible from `training.seed`; evaluation gets distinct maps
from the same pool and freezes copies in the run directory. The run manifest
records the pool path and selected map IDs. The 4,000 selected records are
loaded and padded for JAX; the remaining pool stays on disk.

### Check spawn-pair feasibility without training

Run the training startup reset and its base/target acceptance check without
creating a model or starting rollouts:

```sh
uv run swarmecho-check-spawn-pairs --level B02b_random_buildings_deliver --manifest outputs/my_run/random_buildings.json
```

`--manifest` reuses the exact selected map order from a bank-backed training
run, even if the pool has since grown. Without it, the checker selects maps
from the largest compatible pool using the level's training seed. Repeat every
training override with `--set KEY=VALUE`; for example,
`--set env.roadmap_corner_bonus_m=2.0`. The script never builds or extends a
pool. It uses the same initial JAX keys, map IDs, reset function, and attempt
limit as training. Failed lanes report their map file, fixed target, final
rejected base, and base spawn volume by storey. Exit status 1 means at least
one reset would raise the training error. `--rounds 10` checks nine additional
sets of targets; those extra rounds are stress samples, not a replay of later
training episodes.

With a randomized base, one target is sampled first and held fixed during all
base retries. Base points are weighted by available spawn volume across
storeys; no cross-episode or cross-environment uniqueness rule is applied.
The geodesic threshold is a *minimum shortest-route* distance, and the corner
bonus applies only to routes that use roadmap waypoints. A direct visible
route gets no bonus. Because every training lane must pass, even a small
per-lane failure probability can fail a large initial batch.

Automatic regeneration on a success threshold is not implemented. The explicit
bank and map IDs provide a place to add versioned replacement banks later, at a
rollout/episode boundary. Evaluation maps should remain fixed when that is added.

## Verify a generator change

Generate a fresh five-map inspection suite and check that each adjacent story
pair has a stair, successive stairs use different cells, and all raster points
remain mutually reachable:

```sh
uv run python tests/generate_random_buildings_testresult.py
```

Then run a small training-machine smoke run before a full run:

```sh
uv run swarmecho-train level=B02a_random_buildings_find_only training.num_envs=20 training.num_minibatches=1 training.num_steps=8 training.total_timesteps=160 evaluation.random_eval_envs=2 evaluation.eval_parallel_envs=4 evaluation.eval_video=false evaluation.save_model=false logging.wandb_mode=disabled
```
