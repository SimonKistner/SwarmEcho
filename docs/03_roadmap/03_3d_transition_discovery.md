# 3D transition: discovery brief and first decision gate

Status: **discussion draft**, not an implementation specification. This brief
records what is clear, identifies choices that materially affect architecture,
and deliberately details only the first executable slice.

## Outcome and non-goals

SwarmEcho will become 3D-only. Positions, forces, collision, communication,
visibility, coverage, spawn volumes, relay paths, observations, maps,
evaluation, and replay data must all have a real third axis. The transition
should preserve the project's strongest property: thousands of fixed-shape
environments stepped and trained together using JAX.

The first baseline does not need aerodynamics, attitude, rotors, rigid-body
contact, lighting, or photorealism. Those add simulator and policy complexity
without testing relay learning.

## Clear product decisions

- **3D only:** no dimension flag or retained 2D runtime.
- **Grid-authored buildings:** cubic cells have solid floor/roof tiles and solid
  vertical walls with configurable thickness; editing happens one storey at a time.
- **Closed-map invariant:** every reachable free-space component is sealed from
  the exterior, checked on save and load rather than treated as an editor hint.
- **Simple mission placement:** the base is a static point at one cell centre,
  drones spawn at that point, and only target exclusion is painted as cells.
- **Headless training, separate inspection:** training emits replay data;
  interactive rendering consumes it out of process and is never in `env_step`.
- **Incremental delivery:** each sprint ends in an executable artifact with a
  numeric or visual acceptance check.

## Current 2D coupling audit

This is not a mechanical `2 -> 3` edit.

| Area | Current assumption | Required 3D equivalent |
|---|---|---|
| Map | width/height, XY zones and line walls | XYZ lattice, solid tiles/walls, exclusion cells |
| Compilation | 2D occupancy raster | validated tile/wall arrays and optional voxels |
| State/action | `(N,2)` position, velocity, force | `(N,3)` and world extent `(3,)` |
| Collision | 2D Euler sweep | swept sphere against solid tiles and walls |
| Visibility | 2D raster DDA | analytic baseline test; 3D voxel DDA for buildings |
| Communication | 2D distance and LOS | 3D distance/LOS; graph logic stays reusable |
| Radar | circle angle bins | approximately equal-solid-angle directions |
| Coverage | 2D boolean raster/circular probes | voxel field/spherical probes |
| Finder path | XY cells, four neighbours | XYZ cells, six neighbours |
| Critic | XY geometry, four quadrants | XYZ geometry, octants or pooled voxels |
| Models | actor output fixed at 2 | action width 3; recalculate observation contract |
| Rewards | reusable semantics, 2D spatial inputs | same semantics over 3D primitives |
| Evaluation | heatmaps and frame video | voxel metrics and interactive replay |
| UI | 2D HTML editor and OpenCV/SVG | WebGL editor/replay with layer controls |

PPO/GAE, recurrent sequence handling, checkpoint plumbing, graph reachability,
TarMAC, and most logging are dimension-agnostic. Change them only where tensor
or artifact contracts change.

## Physics engine decision

### Recommendation: retain native JAX for the baseline

Keep the pure-function/JIT/VMAP architecture and replace its spatial kernel
with a small 3D kernel. Model drones as spheres (or points with inflated
obstacles), integrate XYZ velocity, and collide with the cuboid's tiles and walls.
JAX has no 2D restriction; today's restriction is in SwarmEcho's shapes and
algorithms. The first empty cuboid has analytic boundary collision and internal
line of sight, so a general physics engine is unnecessary.

### Why not MuJoCo/MJX now

MJX becomes attractive if the mission requires rigid bodies, orientation,
joints, complex contact, or actuator dynamics. It supports JAX transformations
and batched device simulation, but supplies much more machinery than spherical
drones need. Editable voxel buildings must also be converted to collision
geometry, potentially increasing model size, compilation cost, and difficulty
of batching different maps. Brax has the same mismatch: it primarily solves a
broader articulated-physics problem.

The gate is empirical: benchmark one native JAX cuboid kernel. Reconsider MJX
if realistic rigid-body dynamics becomes required or the kernel misses its
correctness/performance budget.

Primary references for the later architecture decision record:

- [JAX `vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
- [JAX `jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/)
- [Brax repository](https://github.com/google/brax)

## Proposed map contract

Save authoring topology in YAML, not meshes or a dense metre-resolution physics
volume. Compile immutable fixed-shape arrays at load time. A compact sketch:

```yaml
format: swarmecho-map/v1
name: M00_no_maze_open_cuboid
width: 40.0
height: 40.0
depth: 30.0
building_cell_grid: {cols: 8, rows: 8, layers: 6}
cell_size_m: 5.0
tile_thickness_m: 0.25
wall_thickness_m: 0.25
geometry:
  tiles: []         # [x_cell, y_cell, z_boundary]
  x_walls: []       # [x_boundary, y_cell, z_cell]
  y_walls: []       # [x_cell, y_boundary, z_cell]
base_cell: [3, 3, 0]
target_exclusion_cells: [[3, 3, 0]]
```

The syntax is open, but semantics should be fixed early:

1. integer cells are canonical; metres derive from `cell_size_m`;
2. tiles and walls are solid rectangular prisms with explicit thickness;
3. a wall is centred on a cell boundary and extends half its thickness into
   each neighbouring cell;
4. the base is a static point at its cell centre and all drones spawn there;
5. only target exclusion is painted as cells; runtime target candidates must
   additionally satisfy `comm_radius_base + 0.5 * comm_radius + buffer < distance`;
6. validation checks bounds, shell completeness, support, non-empty spawns,
   exterior sealing, and reachability;
7. the format is explicitly versioned and never silently reinterpreted.

Only the bottom floor and top roof are required to be complete. Intermediate
tile layers may be complete, sparse, or empty, enabling shafts, atria, and
staircase openings without adding a more complex room abstraction.

## Spherical radar

The screenshot's eight pieces naturally represent spherical octants. They are
useful for debugging but likely too coarse for control; latitude/longitude bins
also over-represent poles.

Make the direction count configurable. `radar_bins: 8` must use the eight
spherical octants shown in the reference image so the low-cost representation
can be tested. Larger values use fixed approximately uniform unit directions
(for example, a Fibonacci sphere), casting one wall ray per direction. Assign
drone signals by maximum dot product and retain the four existing channels.
Treat 8/16/32/64 as measured ablations. This permits environment complexity to
increase without forcing observation and policy compute to increase at once.

The first CUDA rollout matrix established **8 bins as the baseline default**.
At 4,000 environments and 200 steps, retained device memory was approximately
0.66 GB for 8 bins, 1.17 GB for 16 bins, and 2.19 GB for 32 bins; reported peak
memory was approximately 1.89/3.42/6.50 GB. All variants were fast enough, so
the lower-dimensional policy and substantially lower rollout memory decide the
default. The parameter remains available for controlled 16/32-bin experiments.

## Visualization direction

Use a browser WebGL client, preferably Three.js, for both editor and replay. It
provides orbit/free cameras, transparency, picking, instanced voxels, and normal
web controls without coupling Unity or another compiled application to Python.
Unity is justified only if presentation-quality or standalone distribution
becomes a requirement.

Replay should be a versioned data product, not an MP4: map hash, config,
timestep, positions, velocities, active masks, target/base state,
connectivity/knowledge, reward terms, coverage deltas or keyframes, and optional
radar debug data. Prototype with chunked NumPy plus JSON metadata; assess Zarr
after measuring real sizes. Atomic completed files allow a separately launched
local server to inspect replays while training continues. The training process
must never launch or manage the inspection server. If shared hardware proves
restrictive, an optional cooperative pause flag may be polled only at safe
evaluation boundaries. MP4/GIF becomes optional export from the same record.

References: [Three.js `InstancedMesh`](https://threejs.org/docs/#api/en/objects/InstancedMesh),
[OrbitControls](https://threejs.org/docs/#examples/en/controls/OrbitControls),
and [Raycaster](https://threejs.org/docs/#api/en/core/Raycaster).

## First executable slice

### Sprint 0 — contracts and benchmark harness

Deliver a minimal `map/v1` parser/validator, one hand-written cuboid, and
shape contracts for state/action/observation/replay. Add a benchmark reporting
reset/step throughput for representative environment counts on CPU and the
available accelerator.

**Gate:** the cuboid validates; deliberate leaks and invalid spawns fail with
actionable messages; compiled arrays have documented static shapes; the 2D
runtime is not yet left half-converted.

### Sprint 1 — minimum 3D environment

Replace the spatial core with XYZ state/action, analytic cuboid collision, 3D
distance communication, unobstructed internal LOS, volumetric target spawning
outside the base exclusion, spherical radar, and voxel coverage. Add a
deterministic scripted rollout and random-policy batched rollout. Do not build
the full editor or interior-wall DDA yet.

Every building reports the unobstructed Euclidean distance from the base to its
farthest top corner in a generated comment near the top of its YAML. Level
configuration—not the map—chooses enough agents and suitable radii. For five
drones, the sensing/connectivity chain is `base --(base comm radius)--> D1 -->
D2 --> D3 --> D4 --> D5 --(visual radius)--> target`; each of the four
drone-to-drone arrows uses `comm_radius`. Its unobstructed maximum length is
therefore `comm_radius_base + 4 * comm_radius + visual_radius`.

**Gate:** a scripted five-drone vertical chain connects and fails when one hop
exceeds radius; targets never spawn in exclusion; high-speed sphere collision
stays in bounds; coverage grows across Z; JIT/VMAP and throughput checks pass;
one replay opens in a minimal orbitable browser scene.

Interior tiles/walls and 3D DDA, the complete editor, MAPPO baseline,
analytics, templates, and cleanup should only be detailed after Sprint 1 measurements.

## Confirmed owner decisions

1. Drones use translation-only spherical physics; the inspector may render a
   nicer visual model without changing collision geometry.
2. The bottom floor, top roof, and all four outer walls are complete. Every
   intermediate tile layer may be complete, sparse, or absent, supporting
   shafts, atria, and staircase openings.
3. Maps report their maximum unobstructed base-to-top-corner distance. Agent
   count and connectivity/visibility radii remain level concerns.
4. The base is one static point at a cell centre; drones spawn exactly there.
   Only target exclusion uses painted cells.

## Evidence needed next

- JAX cuboid and voxel-DDA throughput/memory at training batch sizes;
- coverage memory (`envs × X × Y × Z`) and packed/sparse alternatives;
- replay size with coverage deltas/keyframes;
- radar resolution needed for stable avoidance;
- whether observation/action changes mean training from scratch (expected);
- transparent stacked-layer editor usability via a clickable prototype.

These measurements, rather than framework preference, drive the next review.
