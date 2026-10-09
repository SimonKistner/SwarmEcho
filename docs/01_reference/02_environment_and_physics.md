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
With [adaptive swarm size](06_adaptive_swarm_size.md) enabled, the actor adds a
fourth discrete vote action. The environment uses only the first three
components for motion.

Authored maps use `swarmecho-map/v1`, including world width, height, depth,
building cells, tiles, walls, spawn settings, and target exclusions. All bundled
maps use volumetric geometry. `obstacles.py` provides visibility checks and physical/geodesic
route calculations.

## Observations

The actor receives local information, with width given by
`observation_dim(cfg)`:

```text
6 + wall_radar_bins
  + 3 * drone_radar_bins                        [legacy only]
  + 3 * drone_radar_bins * max(1, distance_bands) [band_sep_counting only]
  + 3 * observe_base_vector
  + 3 * observe_target_vector
  + wall_radar_bins * observe_coverage_probe
  + observe_chain_contributor
  + observe_current_timestep
```

The six mandatory self values are normalized XYZ velocity, connection to the
base chain, connection to the target chain, and target knowledge. Target
vectors are masked until the target is known. Inactive agents receive zero
observations.

### Wall and drone radar

`wall_radar_bins` and `drone_radar_bins` independently control spherical
direction counts (integers of at least 4, both defaulting to 8). Eight directions
use exact octants; other counts use a deterministic spherical distribution.
Walls retain one continuous first-hit proximity signal per direction within
`visual_radius`. Coverage probes follow the wall directions.

In legacy mode, each drone direction has three nearest-neighbor proximity signals: all
active neighbors, target-connected neighbors, and base-connected neighbors.
Signals are `max(0, 1 - distance / comm_radius)`, with self and obstructed
neighbors excluded. Each category can select a different nearest drone.

`radar_mode: legacy` is the default and preserves the original behavior.
`distance_bands` and `radar_count_cap` have no effect in this mode. With equal
wall/drone direction counts, the original per-direction order
`[wall, all-drone, target-connected-drone, base-connected-drone]` is preserved
for checkpoint compatibility. With unequal counts, radar consists of a wall
block followed by a drone block, where drone categories remain interleaved per
direction.

`radar_mode: band_sep_counting` replaces drone proximity signals with counts
for the same three categories in each direction × distance-band cell.
`distance_bands` defaults to 1; 0 is an alias for 1. Bands have equal width
`comm_radius / max(1, distance_bands)`. Internal boundaries belong to the outer
band, and the last band includes the communication-radius boundary. For four
bands at a 15 m radius, the borders are 3.75, 7.5, 11.25, and 15 m. A neighbor
occupies exactly one cell per applicable category. Counts include only visible,
active, non-self neighbors within communication range; a drone exactly at the
radius is counted in the last band. No continuous drone distance features are
included in this mode; wall proximity remains continuous.

`radar_count_cap` is a positive integer, defaulting to 3. Each count is encoded
as `min(count, radar_count_cap) / radar_count_cap`: the default represents
0, 1, 2, and 3+ as 0, 1/3, 2/3, and 1. Counting is enabled by the mode, not by
changing the cap. To add counts without distance subdivision, select counting
mode with `distance_bands: 1` (or 0).

Counting mode uses radar blocks in this order: wall proximity, drone counts.
Counts flatten `[direction, band, category]`, from inner to outer
band, with categories ordered all / target-connected / base-connected. The
self and optional observation prefix retains its existing order. Observation
width stays independent of team size. With seven self values, 16 wall and
drone directions, and four bands, counting mode has 215 values per drone;
with three bands it has 167 values.

B02c variants enable counting with a cap of 3 and retain their previous angular
resolutions. `B02c_random_buildings` uses three bands; the static and mini variants
use four. Other levels remain in legacy mode. Frozen YAML configs
using `radar_bins` are accepted by assigning that value to both new parameters;
explicit new fields take precedence. Checkpoint contracts record the separate
resolutions, mode, layout, band count, and cap. Old contracts map to legacy
interleaved radar; changing feature meanings requires a compatible checkpoint
or fresh training. Legacy band/cap changes and the 0/1 band alias are treated as
semantically equivalent during checkpoint validation. Counts-only radar uses
the `band_counts_v2` layout identifier; checkpoints from the earlier counting
layout that also included drone proximity cannot be loaded into this layout.

Coverage is a voxel field. `coverage_voxel_size` may differ from the authoring
cell size and must evenly divide all world dimensions.

## Relay task and rewards

Communication depends on range and visibility. Target knowledge and delivery
to the base persist within an episode. Reward settings govern exploration,
collisions, discovery, delivery, chain success, gap shaping, idle termination,
and optional redundancy/chain-efficiency bonuses.

### First visual sighting is not rewarded

In a delivery-gated target mission, the first visual sighting updates the
drone's target knowledge but pays no `finder_bonus` or `target_found_bonus`.
Exploration already earns coverage reward. The team can learn where and how to
spread out, but with randomized target placement it cannot reliably control
which drone happens to see the target first. Rewarding that identity would add
noisy credit to the exploration policy. Despite its name, `finder_bonus` is
paid to a target-aware drone directly linked to the base when target knowledge
first reaches the base; `target_found_bonus` is shared among active drones at
that delivery event. Peer informing is rewarded separately when a previously
uninformed drone first learns the target through a direct peer link.

The intended team behavior is to spread out and explore, then have the first
drone that sees the target inform at least one peer. Informed drones propagate
that knowledge toward the base and to other drones. The first drone may stay
near the target as an endpoint or return there after a handoff. Once the base
and other drones are informed, shared and recurrent memory should help some
drones form the target-side relay while others build from the base. The two
sides then meet and hold the connection. These are desired roles, not fixed
drone assignments or a guaranteed result of the current rewards.

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

Four optional reward switches default to `false`. With
`gap_reward_uses_change`, each drone instead receives
`gap_change_meter_bonus` times the change in its own credited closed metres.
Joining a selected relay path can produce a positive reward; leaving or breaking
it can produce a negative reward. Without delivery vesting, the first transition
that opens the delivery or discovery gate pays no accumulated gap reward.
`delivery_gap_vesting_enabled` requires the change reward and target delivery.
At delivery, it records each drone's credited metres and releases that existing
credit over `delivery_gap_vesting_steps` subsequent steps while the contribution
persists. Newly made progress pays immediately. Losing progress cancels unpaid
credit first; only credit already paid can be reclaimed. Episode termination
releases any remaining credit to current contributors. With
`success_bonus_as_hold_record`, each new episode record for consecutive full-chain
hold steps releases `success_bonus / hold_chain_for` to current complete-chain
contributors, divided among them. The terminal success payout is then omitted;
repeating an already achieved hold length pays nothing. This mode requires
`success_condition: chain_held`. With `peer_informing_reward_enabled`, a drone
earns a share of `peer_informing_bonus` when a previously uninformed peer first
learns the target through a direct communication link. Multiple senders split
one recipient's bonus; direct sighting and base replay take precedence.

Training logs report peer-transfer counts, the best hold, and positive and
negative gap reward totals separately. The current B02c level enables peer
informing, pays success at completion, uses the per-step absolute gap reward,
and leaves delivery gap vesting disabled.

Authored buildings and generated obstacles share the planning machinery.
Episode completion can depend on holding a chain, finding/delivering the target,
idleness, or the episode length, according to the selected level.

Authoritative behavior remains in `env/environment.py`, `env/obstacles.py`,
and the chosen level YAML.
