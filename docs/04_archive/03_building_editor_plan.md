# volumetric map builder: architecture and implementation plan

> Archived editor design proposal, retained for historical context.
> See the current [tools guide](../02_guide/02_analysis_and_tools.md) for the
> implemented building editor. This document is not an operational guide.

Status: **implementation proposal**. This document follows the completed volumetric
baseline rather than the pre-transition assumptions in the discovery brief. It
defines the smallest path from the current sealed-cuboid maps to an optional,
layer-oriented building editor without creating a second environment runtime.

## Product target

The builder should make a building by editing a stack of square-cell plans:

- grow or shrink a rectangular canvas in any horizontal direction;
- paint real floor/ceiling tiles and toggle real wall segments;
- edit one storey at a time, with separate **Tiles** and **Walls** modes;
- add, remove, duplicate, and select storeys;
- add every outward-facing wall around the selected floor footprint in one
  operation (a 2 x 2 footprint creates eight wall segments);
- select one interior cell for the base and paint target-exclusion cells;
- save, reload, and manually edit a compact, human-readable map;
- produce both low, wide office/corridor plans and tall buildings through the
  same operations; and
- refuse to save a map whose usable volume leaks to the exterior.

Tiles and walls are solid rectangular prisms. Their thickness is part of the
map, is shown by the editor, and is used by collision and line-of-sight code.
There is no separate “builder environment”: a builder map is an ordinary volumetric
map selected by a level.

### Deliberate first-release limits

The grid stays axis-aligned, all cells are cubic, and one map has one cell
size, wall thickness, and tile thickness. Sloped roofs, stairs, doors that
animate, arbitrary meshes, materials, windows to the exterior, and multiple
bases are out of scope. Openings between storeys are supported, so stairs or
lifts can be represented later without changing the topology.

## Current-state audit

The transition already established useful seams which should be retained:

1. `BuildingArrays` is the level-facing map product. It already stores
   `(X,Y,Z+1)` tiles, `(X+1,Y,Z)` X walls, `(X,Y+1,Z)` Y walls, thicknesses,
   target exclusions, base position, and world size.
2. `load_level` selects exactly one map and compiles it before constructing
   training functions. Map dimensions are consequently static for a compiled
   level, which is the right behavior for JAX batching.
3. Target sampling already clips each cell's continuous volume against the six
   adjacent tile/wall prisms.
4. The volumetric environment already has reusable AABB segment tests for procedural
   cuboids, and its collision, communication LOS, target visibility, coverage,
   radar, evaluation artifacts, and inspector all understand those cuboids.
5. The old maze-builder server is a small local standard-library HTTP service.
   Its endpoint and file-writing pattern are reusable, but its 2D DOM canvas,
   legacy map conversion, and German-only controls are not an architectural
   base for the new editor.

There is one important gap: authored interior `tiles`, `x_walls`, and
`y_walls` currently affect target-spawn clearance only. Motion is bounded by
one analytic outer cuboid, while collision and every LOS path only see the
random AABBs in `EnvState`. The inspector likewise renders the world
box and procedural obstacles, not authored building prisms. Therefore an
editor-only implementation would look correct but train on different
geometry. Runtime geometry integration must precede the polished editor.

The current shell validator is also narrower than the intended product. It
requires a complete rectangular bottom, roof, and four side walls, but it does
not prove that sparse intermediate geometry is sealed, that the base is in
free space, or that usable cells are mutually reachable. It cannot express an
irregular floor footprint unambiguously: a missing tile may mean an atrium,
not “this cell is outside the building.”

## Architecture decisions

### 1. Keep one compiled building contract

Extend `BuildingArrays`; do not add an editor-specific runtime map class.
Authoring YAML is parsed and validated on the CPU once, then compiled into the
immutable NumPy arrays captured by the JAX functions.

Add these fields:

- `interior_cells: bool[X,Y,Z]`: cells belonging to usable enclosed volume;
- `solid_min_m`, `solid_max_m: float32[S,3]`: AABBs for every authored tile
  and wall after merging adjacent coplanar segments;
- `solid_kind: uint8[S]`: tile/X-wall/Y-wall, for rendering and diagnostics;
- `map_hash`: hash of normalized authoring topology and physical dimensions.

`S` is fixed for one loaded map. Different map shapes already require separate
level compilations, so a global maximum-size tensor is unnecessary. If a
curriculum ever batches several maps in one compiled update, that future
loader can pad to the maximum `S` in that explicitly selected map set and add
an AABB validity mask.

The lattice arrays remain authoritative for editing and validation; merged
AABBs are a derived acceleration representation and are never saved.

### 2. Use a compact `swarmecho-map/v2` authoring format

The two existing v1 cuboids prove the compiler contract, but coordinate lists
make even an empty building verbose. V2 should encode each lattice plane as a
row string. It remains readable in a text editor and has deterministic output.
Coordinates are always `[x,y,z]`, while each displayed string is a Y row with
X increasing left to right.

```yaml
format: swarmecho-map/v2
name: office_demo
grid: {cols: 6, rows: 3, layers: 1}
cell_size_m: 5.0
tile_thickness_m: 0.25
wall_thickness_m: 0.25

# Each interior plane has `rows` strings of `cols` characters.
interior_layers:
  0: ["111111", "111111", "111111"]

# Tile keys are Z boundaries (0..layers).
tile_boundaries:
  0: ["111111", "111111", "111111"]
  1: ["111111", "111111", "111111"]

# X-wall rows contain cols+1 characters; Y-wall planes contain rows+1 rows.
wall_layers:
  0:
    x: ["1000001", "1010101", "1000001"]
    y: ["111111", "000000", "000000", "111111"]

mission:
  base_cell: [0, 1, 0]
  target_exclusion_layers:
    0: ["110000", "110000", "000000"]
```

`0` and `1` are the only legal plane characters. Missing planes mean all
zeroes, except that editor creation commands explicitly materialize their
defaults before validation. YAML output orders numeric layers ascending and
always emits the grid and physical parameters, making diffs stable.

The base remains a static point as decided for the volumetric environment; selecting a
cell is the authoring convenience. Compilation places it at the horizontal
cell center and immediately above that cell's lower tile. This removes the
current possibility of manually placing the base inside a wall while retaining
the runtime's point representation.

Do not maintain two permanent parser branches. Implement a one-shot
`migrate_v1_to_v2` converter, convert both bundled maps and their tests in the
same change, and make the production loader accept v2 only after that change.
The migration command may continue to read v1 for users with local files, but
the environment loader must not. This is the minimal legacy bridge and avoids
silently reinterpreting old maps.

### 3. Define enclosure over explicit interior cells

`interior_cells` distinguishes an atrium/opening from nonexistent exterior
volume. Validation treats every interior cell as a free-space node with six
faces. A face is sealed when it has the corresponding tile or wall prism. A
face between two interior cells may be open or sealed. A face from an interior
cell to a non-interior cell or outside the grid must be sealed.

This gives exact, actionable checks:

1. every interior-to-exterior face is sealed;
2. no interior cell is completely consumed by wall/tile thickness;
3. the base cell is interior and has usable volume;
4. target-exclusion cells are interior;
5. at least one non-excluded target volume satisfies wall clearance and the
   level's base-distance rule;
6. all interior cells intended for target spawning are reachable from the base
   through open faces; and
7. floating tiles are allowed (ceilings and bridges), but a configurable
   warning reports unsupported authored floor islands rather than rejecting a
   physically valid flying-drone map.

The primary enclosure test should report the first leaking cell and face, for
example `interior cell [4,2,1] leaks through +X; add x_wall [5,2,1] or remove
the cell from the interior footprint`. A flood fill from a one-cell padded
exterior can be retained as a cross-check in tests.

“Add outer walls” is then deterministic: for the current storey, inspect each
selected interior cell's four horizontal neighbors and add a wall wherever
the neighbor is absent or outside the grid. It does not add walls between two
selected cells and does not alter ceilings. For a solid 2 x 2 selection this
adds exactly eight segments. A separate **Seal building** action may add all
missing horizontal outward walls plus lower/upper tiles; it must preview the
count and remain undoable.

### 4. Reuse AABB kernels, but replace the obstacle-only assumptions

Compile authored solids to AABBs with their real thickness. Merge consecutive
coplanar tile or wall cells into larger boxes to reduce `S`; never merge across
a gap or a material kind. Pass these static arrays beside any procedural
obstacles to a shared geometry query layer.

Update, in this order:

- **Target spawn:** filter out non-interior cells, then retain the existing
  continuous-volume sampling and distance constraints.
- **Collision:** swept-sphere versus authored and procedural AABBs. Inflate
  boxes by `drone_radius`, find the earliest hit fraction, move to contact, and
  project the remaining velocity along the hit surface. The current all-or-
  nothing rollback is acceptable as a temporary integration slice but not for
  narrow corridors because it causes sticking at corners.
- **Communication and visibility:** call the existing vectorized segment/AABB
  slab test over the combined solids. Use an epsilon policy that permits a
  segment starting on a floor but rejects traversal through it.
- **Radar:** return nearest intersection distance, rather than only “any box
  blocks the full ray,” across authored and procedural solids.
- **Coverage:** mark only interior voxel centers and apply the same LOS query;
  exterior and solid voxels do not count in the coverage denominator.
- **Inspector/artifacts:** persist `map_name` and `map_hash`; let the inspector
  load immutable authored geometry from the map and verify the hash. Continue
  storing procedural obstacle bounds in a replay because those vary by reset.

Do not feed every building prism into the existing eight-corners-per-obstacle
Floyd-Warshall roadmap. Its vertices and cubic all-pairs work grow too quickly
for an office plan. Compute a six-neighbor cell graph from authoring topology
on the CPU and store base-to-cell/topological distances for diagnostics and
the discrete finder reward. Use AABB LOS for actual radio/vision edges. Keep
the current cuboid roadmap only for the small optional procedural-obstacle
mode until that mode is either retired or given a measured replacement.

### 5. Build a Three.js editor as a replaceable client

Keep the local HTTP server and map-directory security boundary, but serve a
new editor application. Use Three.js `InstancedMesh` groups for tiles, X walls,
and Y walls, `Raycaster` for picking, and `OrbitControls` for rotation. The
server owns parsing, validation, normalized serialization, file writes, and
template discovery; the browser owns transient editing and rendering.

The UI has four regions:

- **Top toolbar:** new/open/save-as, undo/redo, template, dimensions, physical
  sizes, validate, and seal building;
- **Left tool rail:** Select, Tiles, Walls, Base, Target exclusion, erase, and
  Add outer walls;
- **Viewport:** perspective top/side view, orbit/pan/zoom, fixed 90-degree
  rotate buttons, grid axes, hover preview, and real-depth instanced solids;
- **Storey panel:** elevator-style storey list, up/down buttons, add, duplicate,
  and delete, plus visibility controls for layers above/below.

Only one edit mode is active. In Tiles mode, clicking toggles the lower tile of
the current storey; a distinct ceiling control edits its upper boundary. In
Walls mode, picking the nearest cell edge toggles the corresponding X/Y wall.
Shared boundaries have one identity, preventing overlapping duplicate walls.
Drag painting, box selection, keyboard shortcuts, undo/redo, and batch outer
walls all use commands over the same normalized document model.

The current storey is opaque. Lower storeys default to 25% opacity and upper
storeys to hidden or 8% “ghost” opacity. A section-height slider and isolate
storey toggle solve closed-building inspection without changing geometry.
Camera rotation changes only presentation, never lattice coordinates.

The first web slice may import Three.js as vendored ES modules. Do not require
a Node build pipeline merely for this optional tool. Pin the version and serve
it locally so map authoring does not depend on a CDN.

### 6. Make server APIs document-oriented

Use a versioned API rather than extending the 2D maze payload:

- `GET /api/maps` and `GET /api/templates` return names and summaries;
- `GET /api/maps/{name}` returns normalized v2 authoring data;
- `POST /api/validate` returns errors, warnings, derived dimensions, solid
  count, reachable cell count, and maximum base distance without writing;
- `POST /api/maps/{name}` validates and atomically writes a new file;
- `PUT /api/maps/{name}` requires an explicit overwrite flag and the last-read
  map hash to prevent accidental lost updates.

Reject traversal and invalid names, cap dimensions/body size, and write via a
temporary file plus `Path.replace`. Creating a level is a separate opt-in
action: it writes a small level copied from a selected volumetric template and changes
only `env.map_names`. The builder must never silently pick agent/radius/training
values based on building size; it should display the ideal chain margin using
the chosen level settings and warn if negative.

## Delivery sequence and gates

### Phase 0 — lock topology and migrate maps

1. Add v2 dataclasses/parser/serializer and precise plane-shape errors.
2. Add `interior_cells`, base-cell compilation, enclosure/reachability checks,
   map hashing, and adjacent-prism merging.
3. Convert the two bundled cuboids to v2 and update the building tests.
4. Add a standalone v1-to-v2 migration command, not a runtime fallback.

**Gate:** both bundled levels load unchanged in world dimensions, base point,
target exclusions, and shell geometry; serialization round-trips byte-stably;
the validator identifies deliberate leaks by cell and face; a 2 x 2 perimeter
unit test produces eight walls.

### Phase 1 — make authored geometry real in the environment

1. Introduce combined static authored AABBs plus reset-specific procedural
   AABBs and validity masks where needed.
2. Integrate target sampling, collision, LOS, radar distance, and coverage.
3. Replace building use of the cuboid Floyd-Warshall roadmap with the compiled
   cell graph.
4. Add a hand-authored one-storey corridor with two rooms and a doorway as the
   reference integration map.

**Gate:** a scripted drone cannot cross any floor/wall at high speed, can pass
through the doorway, loses radio and vision through the wall, radar reports
the nearer wall, targets never appear outside interior cells, and coverage
excludes solid/exterior volume. Run the existing empty-cuboid workflow to prove
no regression and benchmark the corridor at representative batch sizes.

### Phase 2 — editor core

1. Add the document APIs and a minimal Three.js scene.
2. Implement storey navigation, Tiles/Walls modes, picking, opacity/isolation,
   undo/redo, dimension growth in four directions, and outer-wall generation.
3. Implement base and target-exclusion painting, live validation, import,
   atomic save, and reopen.

When prepending a row/column, translate all affected coordinates—including
base and exclusions—so the visible building stays stationary relative to its
content. Removing a boundary is blocked if it would discard content unless
the user confirms a previewed destructive crop.

**Gate:** create and reopen both a 30 x 4 x 1 corridor office and a 4 x 4 x 8
tower; the saved YAML diff is deterministic; all solids have visible depth;
the 2 x 2 outer-wall action is visually and programmatically eight segments;
and server validation agrees with direct `load_building` validation.

### Phase 3 — workflow finish and templates

Add drag/box painting, duplicate storey, keyboard help, save-conflict handling,
level creation, and three bundled templates: empty cuboid, single-storey
corridor office, and small multi-storey house/atrium. Add the same map geometry
renderer to replay and heatmap views, with section/opacity controls shared in
appearance with the editor.

**Gate:** each template validates, runs through reset/step/replay, and can be
edited and saved as a new map without hand-editing YAML. Accessibility checks
cover keyboard focus, non-color-only tool state, and readable validation errors.

### Phase 4 — performance decision, not speculative complexity

Measure solid count before/after merging, JIT compile time, step throughput,
device memory, editor frame time, and replay payload/load time for at least:

- 4 x 4 x 4 empty cuboid;
- 30 x 4 x 1 corridor office;
- 12 x 12 x 8 room-heavy building; and
- the largest editor-supported dimensions.

Only if the shared AABB scan misses the agreed throughput budget should it be
replaced by volumetric DDA over lattice faces or a spatial index. Preserve the geometry
query interface so this is a kernel substitution, not another map format.

## Test matrix

### Pure Python

- v2 schema, dimensions, characters, ordering, duplicates, and stable YAML;
- v1 migration equivalence;
- each shell face and irregular footprint leak;
- base/exclusion legality and target volume availability;
- six-neighbor reachability including doors and vertical openings;
- prism dimensions, centers, kinds, merging, and non-merging across gaps;
- expand/prepend/crop coordinate transforms; and
- outer-wall generation for rectangle, L shape, hole, and adjacent cells.

### JAX/runtime

- swept-sphere collision from both directions and corner contact;
- LOS endpoint/epsilon behavior and nearest radar hit;
- authored plus procedural obstacle composition;
- target/coverage masks for non-interior cells;
- JIT and VMAP shape stability; and
- existing empty-map training/evaluation artifact compatibility.

### Browser/server

- API traversal, size limits, overwrite/hash conflicts, and atomic writes;
- browser document reducer tests for every undoable command;
- deterministic pick-to-lattice coordinate tests at four camera rotations;
- save/reload end-to-end checks; and
- screenshots of the corridor and tower at current-storey and section views.

## Compatibility and rollout

- Existing **level** YAML remains unchanged because it already references a map
  by name. No dimension flag or separate legacy training entry point is added.
- The two repository maps are mechanically migrated to v2 before the loader
  switches, preserving their names so checkpoints, commands, and levels keep
  resolving them.
- Replays gain a map hash. Older replays without one continue to show their
  world box and stored procedural obstacles; this compatibility belongs in the
  out-of-process inspector, not in training.
- The 2D builder can remain available during Phase 0/1, but after the volumetric editor
  can create the corridor acceptance map its CLI should be redirected to the
  new UI and the old converter removed in one cleanup change.
- Checkpoints are compatible only when their observation/action/network shapes
  are compatible; map geometry alone does not change those shapes. A map may
  nevertheless be a harder task, so compatibility must not be presented as a
  promise of learned-policy transfer.

## Risks and explicit non-solutions

- **Large plans:** coordinate lists and one mesh per segment are rejected;
  plane strings plus instancing/merged AABBs address this first.
- **JAX recompilation:** editing a map changes static geometry and intentionally
  causes a new level compilation. Hot-reloading geometry into a running trainer
  is not a goal.
- **Thin-wall tunneling:** endpoint occupancy is insufficient; swept tests are
  mandatory before calling authored walls supported.
- **Opacity ambiguity:** ghost layers supplement, not replace, isolate/section
  controls and a clear current-storey label.
- **Automatic sealing surprises:** batch actions preview changes, are undoable,
  and never run implicitly on save.
- **Procedural cuboids:** they are not silently converted into authoring tiles.
  They remain an optional level feature and are composed at the geometry-query
  boundary.
- **Premature DDA:** lattice DDA may ultimately outperform AABB scans, but the
  current tested slab code is the pragmatic first integration. Measurements in
  Phase 4 decide whether to replace it.

## Definition of done

The feature is complete when a user can create either the corridor office or
the tower entirely in the browser, validate and save it, select it from an
otherwise ordinary volumetric level, and observe identical solid geometry in physics,
radio/vision, radar, coverage, replay, and the editor. Invalid exterior leaks
must never reach training, and the existing named cuboid levels must continue
to load after their one-time format migration.
