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

**Design-status warning:** `observe_base_vector`, `observe_target_vector`, and
`observe_coverage_probe` are debugging-only legacy options. They are retained
solely to inspect/debug behavior; their presence, settings, or historical use
must under no circumstances be treated as design decisions, endorsed design
choices, or candidates to consider in response to any design question.

The actor receives local information, with width given by
`observation_dim(cfg)`:

```text
6 + wall_sensing_rays
  + 3 * drone_radar_bins                        [legacy only]
  + 3 * drone_radar_bins * max(1, distance_bands) [band_sep_counting only]
  + 3 * observe_base_vector                 [debugging-only legacy option]
  + 3 * observe_target_vector               [debugging-only legacy option]
  + wall_sensing_rays * observe_coverage_probe [debugging-only legacy option]
  + observe_chain_contributor
  + observe_current_timestep
  + observe_seeing_target
  + 3 * observe_visually_gated_target_displacement
```

The six mandatory self values are normalized XYZ velocity, connection to the
base chain, connection to the target chain, and target knowledge. Target
vectors are masked until the target is known. Inactive agents receive zero
observations.

`env.observe_seeing_target` defaults to `false` and is enabled in `G01_static`.
When enabled, it adds one float in the optional block before radar: `1.0`
when the agent currently sees the target within `visual_radius` with unobstructed
line of sight, otherwise `0.0`. It returns to zero after losing sight even if
target knowledge persists. Inactive agents and coverage missions report zero.

`env.observe_visually_gated_target_displacement` defaults to `false` and is enabled
in `G01_static`. It adds three floats after the optional direct-sighting flag and
before radar: `(target_position - drone_position) / visual_radius`, in the same
fixed world axes as velocity. The vector carries direction and range; with the
5 m G01 visual radius, a target offset `[3,0,1]` m produces `[0.6,0,0.2]`.
It is nonzero only during current direct sighting within visual range and with
clear line of sight. Loss of sight, obstacles, inactive agents and coverage
missions produce zeros even when remembered or received target knowledge remains
true. A coincident visible target also has zero displacement; the existing
`observe_seeing_target` flag can disambiguate that case. The displacement toggle
does not require the binary flag and does not provide positions to relays.

This is an idealized local stereo/RGB-D detection measurement. Real object
detection combined with depth can produce camera-relative XYZ coordinates, as
demonstrated by [Luxonis spatial detection](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/spatial_detection_network/).
Calibration/attitude would transform those coordinates into the velocity frame.
The simulator uses exact, noise-free displacement and its existing omnidirectional
range/occlusion visibility; it does not simulate a directional camera field of
view, missed detections or depth noise. A single monocular image requires scale
assumptions to estimate metric distance. The legacy debugging target vector above
is not this sensor: it is gated by persistent knowledge instead of direct sight.

G01 observation width is now **139**, versus **136** for the Phase 0 checkpoint.
Checkpoint contracts record the new toggle and its enabled normalization radius;
older contracts interpret a missing toggle as false. Sensor-enabled loading requires
matching metadata and input width; disabled old configurations retain their layout.

### Wall and drone radar

`wall_sensing_layout` selects `fibonacci` (default, legacy directions) or `xyz`.
`wall_sensing_rays` defaults to 16. Fibonacci accepts any integer of at least 4;
eight preserves the original octant directions, and other counts use the
original deterministic Fibonacci sphere and feature order. XYZ supports only
14, 18, or 26 rays: 14 uses six axes plus eight corner diagonals; 18 uses six
axes plus twelve plane diagonals; 26 uses all three groups. Directions are
normalized, fixed in world coordinates, and ordered axes, plane diagonals
when present, then corner diagonals when present.

Walls retain one continuous first-hit proximity signal per thin ray:
`max(0, 1 - hit_distance / wall_sensing_ray_max_distance)`. The range defaults
to **8 m**, independently of `visual_radius`, which still controls target
visibility. This spans a 5 m cell and part of the next cell. It is a plausible
compact-LiDAR range; for comparison, the manufacturer specifies 7 m on dark
targets and 12 m on reflective targets for the [TFmini-S](https://ai.benewake.com/en/products/tfmini-s/).
The simulator uses ideal thin rays without reflectivity or sensor noise.
Coverage probes follow the wall directions; their probe spacing still uses
`coverage_voxel_size`. This is a debugging-only legacy observation option and
must not be treated as a design choice or considered for design decisions.

`drone_radar_bins` remains independent and defaults to 8. It always uses the
original octant/Fibonacci directions and nearest-direction assignment; every
visible neighbor belongs to one bin. It has no wall-layout setting.

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
use four. C levels use 26 XYZ wall rays, 16 drone bins and three counting bands.
Frozen YAML configs using `wall_radar_bins` map that field to `wall_sensing_rays`;
the older `radar_bins` maps to both wall rays and drone bins. Explicit current
fields take precedence. These saved configs retain their original wall range
from `visual_radius` unless the separate range is explicitly supplied.
Checkpoint contracts record wall ray count, direction layout and range alongside
drone resolution, mode, feature layout, band count, and cap. Old contracts
default to Fibonacci directions and recover their wall range from the run's
saved `config.yaml` (or the old 4 m default if that file is unavailable).
Changing ray count, direction layout or range requires compatible weights or
fresh training. Legacy band/cap changes and the 0/1 band alias are treated as
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
first reaches the base; each active drone receives the full `target_found_bonus`
once at that delivery event, independent of the active population. Inactive
drones receive zero. With `target_found_requires_delivery=false`, the same
per-active-drone payout occurs at the first discovery instead. A configured
bonus of 500 therefore pays 500 to each active drone, and the total team payout
is 500 times the active population. Existing configured bonus values now have
this per-drone meaning. Peer informing is rewarded separately when a previously
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

Additional reward switches default to `false`. With
`progress_based_cluster_tiebreaker_soft`, gap credit can additionally be shared
among equal-hop alternatives, weighted by their hypothetical frontier-leader
progress. Shared mandatory links keep full credit and separate clusters split
independently. This defaults off and is enabled in G01; terminal contributor
selection and actor observations remain unchanged. See
[soft cluster credit](09_soft_cluster_credit.md) for its formula.

With
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

### Reward scaling with population

`target_found_bonus` is a fixed amount per active drone. The following terms
still divide a team amount by a recipient count; their behavior is unchanged.
Here `N` is the current active population and `N_previous` is the population
that acted before deployment or decommissioning on this transition.

| Setting | Per-recipient amount before event gates |
| --- | --- |
| `success_bonus` (ordinary success, including coverage) | `success_bonus / N` |
| `success_bonus_contributors_only` | `success_bonus / contributor_count`, paid only to selected contributors |
| `success_bonus_as_hold_record` | `success_bonus / hold_chain_for / holder_count`, paid only on new hold records |
| `chain_efficiency_bonus` | `chain_efficiency_bonus * individual_efficiency_credit / N`; even active noncontributors increase the divisor |
| `time_penalty_per_step` | `-time_penalty_per_step / N`; adaptive missions instead charge the acting population using `N_previous` |
| `no_movement_termination_penalty` | Signed configured amount divided by `N` at idle termination |
| `all_decommissioned_penalty` | `-all_decommissioned_penalty / N_previous` to the last acting drones |
| `success_unused_agent_penalty` | `-success_unused_agent_penalty * unused_active_count / N` to every active drone at success |

Success and efficiency rewards therefore still become smaller per recipient
as their divisor grows. The shared penalties likewise become smaller in
magnitude per recipient for the same team cost; the unused-agent team cost
also grows with the number of unused drones.

Other rewards split local contribution credit rather than dividing by the
whole active population: exploration splits each newly covered voxel among
its simultaneous viewers, peer informing splits each newly informed peer's
bonus among its senders, and optional soft cluster credit splits gap credit
among local equal-hop alternatives. Additional active drones elsewhere do not
directly dilute these credits. Gap shaping otherwise pays per contributing
drone and credited metre. Finder, reinforcement, decommission, rejected-call,
and individual noncontributor amounts are also paid in full per eligible
recipient. Reference rewards pay each caller its own decaying share of a
recruit's normal reward and inherit any population scaling in that reward.

Training logs report peer-transfer counts, the best hold, and positive and
negative gap reward totals separately. The current B02c level enables peer
informing, pays success at completion, uses the per-step absolute gap reward,
and leaves delivery gap vesting disabled.

Authored buildings and generated obstacles share the planning machinery.
Episode completion can depend on holding a chain, finding/delivering the target,
idleness, or the episode length, according to the selected level.

Authoritative behavior remains in `env/environment.py`, `env/obstacles.py`,
and the chosen level YAML.
