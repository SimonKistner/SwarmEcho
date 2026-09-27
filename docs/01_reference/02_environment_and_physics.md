# environment and physics

`EnvState` stores XYZ positions and velocities, the base and target,
agent activation, coverage voxels, visibility and connectivity, persistent
target knowledge, chain progress, episode state, and obstacle geometry.

## Motion and geometry

Actions contain three components. The environment clips normalized actions,
scales them by maximum force, applies drag and the timestep, caps speed, and
advances position. World bounds constrain motion. Segment checks against solid
bounds expanded by the drone radius prevent crossing authored or generated
obstacles. Collided velocity components are zeroed.

Authored maps use `swarmecho-map/v1`, including world width, height, depth,
building cells, tiles, walls, spawn settings, and target exclusions. All bundled
maps use volumetric geometry. `obstacles.py` provides visibility checks and physical/geodesic
route calculations.

## Observations

The actor receives local information, with width given by
`observation_dim(cfg)`:

```text
6 + 4 * radar_bins
  + 3 * observe_base_vector
  + 3 * observe_target_vector
  + radar_bins * observe_coverage_probe
  + observe_chain_contributor
  + observe_current_timestep
```

The six mandatory self values are normalized XYZ velocity, connection to the
base chain, connection to the target chain, and target knowledge. Spherical
radar bins encode walls, nearby drones, target-connected drones, and
base-connected drones. Target vectors are masked until the target is known.
Inactive agents receive zero observations.

Coverage is a voxel field. `coverage_voxel_size` may differ from the authoring
cell size and must evenly divide all world dimensions.

## Relay task and rewards

Communication depends on range and visibility. Target knowledge and delivery
to the base persist within an episode. Reward settings govern exploration,
collisions, discovery, delivery, chain success, gap shaping, idle termination,
and optional redundancy/chain-efficiency bonuses.

`chain_reward_system` supports `euclidean` and `obstacle_geodesic`.
For a partial chain, `euclidean` chooses the base-connected drone closest in
straight-line distance to the target and the target-connected drone closest to
the base. `obstacle_geodesic` jointly selects both leaders by the smallest
free-space roadmap gap. The team pays `time_penalty_per_step` on every step,
divided equally among active drones. For a target mission, after the required
target knowledge reaches the base, each contributing drone receives
`gap_reduction_meter_bonus * max(L - gap, 0)` per step,
where `L` is the selected system's base-to-target distance. In obstacle mode,
both `L` and the gap use the free-space roadmap. The bonus scales with the
actual route length and reaches its maximum at zero gap, including during the
chain hold. Only drones on the shortest communication paths from the respective
endpoint to its selected leader receive the bonus. When redundancy reward is
enabled and the chain is complete, all drones on a valid complete path can
receive it. Coverage missions have no gap bonus or chain-gap computation. The optional
chain-contributor observation uses the same mode-specific selection once a
complete chain exists.
Authored buildings and generated obstacles share the planning machinery.
Episode completion can depend on holding a chain, finding/delivering the target,
idleness, or the episode length, according to the selected level.

Authoritative behavior remains in `env/environment.py`, `env/obstacles.py`,
and the chosen level YAML.
