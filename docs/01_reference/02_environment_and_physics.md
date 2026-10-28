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

With both base-feedback channels disabled, G01's target-displacement sensor
increases observation width to **139**, versus **136** for the Phase 0 checkpoint.
Checkpoint contracts record the new toggle and its enabled normalization radius;
older contracts interpret a missing toggle as false. Sensor-enabled loading requires
matching metadata and input width; disabled old configurations retain their layout.

### Base-network feedback

Adaptive-size episodes also provide optional base-network feedback after the
drone's own vote and decommission-zone inputs. Both flags default to false
globally; G02 enables `observe_base_vote_ratio` and `observe_base_population`. The actor then sees
qualified base voters / active population and active population / the global
`DEPLOYED_AGENTS_OBS_REF = 10.0`. Both are zero unless the drone currently has a
physical radio path to the base. These observations are not latched; no path
means zero on that observation. See [adaptive swarm size](06_adaptive_swarm_size.md)
for receipt semantics, layout and legacy-checkpoint handling.

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

Communication depends on range and unobstructed visibility. Target knowledge
and delivery to the base persist within an episode. All reward amounts are
signed individual coefficients: negative values subtract reward directly.
No payout is divided by the active population. See
[signed individual rewards](11_individual_rewards.md) for every default,
recipient, calibration choice and supported level.

### First visual sighting is not rewarded

In a delivery-gated mission, seeing the target first updates the drone's
knowledge but pays neither `finder_reward` nor `target_found_reward`.
Exploration already rewards coverage; the identity of the first visual finder
is incidental under randomized target placement. At first delivery,
`finder_reward` pays each target-aware direct base neighbor, while every active
drone receives the full `target_found_reward` once. Inactive slots receive zero.
Without delivery gating, both events instead use first visual discovery.
Re-entry into sight or reconnecting after delivery cannot pay the events again.

The intended behavior is to explore, inform peers, propagate knowledge toward
the base, and form a relay with drones working from both endpoints. These are
learned roles rather than fixed assignments. Peer informing rewards eligible
direct senders when a previously uninformed drone first learns the target;
direct sighting and base replay take precedence. Simultaneous senders split
that peer's credit, while simultaneous viewers split a new voxel's exploration
credit. These are local contribution shares, not population-split budgets.

### Chain geometry and progress credit

`chain_reward_system` supports `euclidean` and `obstacle_geodesic`. Euclidean
partial chains select the base-connected drone closest to the target and the
target-connected drone closest to the base. Obstacle-geodesic selection jointly
minimizes the free-space roadmap gap. After the information gate opens, a
contributing drone receives `gap_progress_reward_per_meter * max(L - gap, 0)`
per step, where both `L` and the gap use the selected distance system.

Only drones on the selected shortest communication paths from each endpoint to
its frontier leader receive credit. A complete chain selects a shortest-hop
base-target path. Optional redundancy credit allows complete-path alternatives.
Coverage missions calculate no target-chain gap reward. The contributor
observation follows the same geometry once a complete chain exists.

With `progress_based_cluster_tiebreaker_soft`, equal-hop alternatives share
local credit according to hypothetical frontier progress. Mandatory links keep
full credit, and independent clusters split independently. Terminal usefulness
selection and actor observations remain unchanged; see
[soft cluster credit](09_soft_cluster_credit.md).

`gap_reward_uses_change`, its coefficient `gap_change_reward_per_meter`, and
delivery-gap vesting are legacy diagnostic tools only. They are retained for
historical debugging, not training design, tuning or plateau remedies. Normal
training uses recurring gap shaping.

In this diagnostic mode, each drone receives
`gap_change_reward_per_meter` times its change in credited metres instead of
recurring progress reward. Without vesting, gate activation pays no accumulated
credit. `delivery_gap_vesting_enabled` requires change mode and delivery gating:
it records pre-delivery credit and releases it over
`delivery_gap_vesting_steps`, while new progress pays immediately. Loss cancels
unpaid credit first; only paid credit can be reclaimed. Termination releases
remaining current credited progress.

### Success and time pressure

The legacy idle shortcut is separate from successful chain holding.
`env.no_movement_termination_steps` must currently be a positive integer: zero
does not disable it and is rejected. To prevent idle termination within an
episode, set it strictly above `env.max_steps` (for example, 701 for a 700-step
episode, or 10000000). This leaves the ordinary episode time limit and
`chain_held` success check active. If the idle and chain-hold thresholds are
reached on the same transition, successful completion still takes precedence
and no idle-failure reward is paid.

Every active drone receives the full `success_reward` at first success. At
terminal chain success, each active drone outside the selected success chain
also receives `useless_agent_at_success_reward`. In addition, every active
drone receives `useless_agent_at_success_all_agents_reward` per useless active
drone, without a population divisor, capped in absolute amount per recipient by
`useless_agent_at_success_all_agents_reward_cap`. Defaults are success +3000,
the individual adjustment -3000, and the all-agents adjustment `-min(750 * U, 2000)`
for `U` useless drones. Contributors receive `3000 - min(750 * U, 2000)` and
useless drones receive `-min(750 * U, 2000)` from these terminal components.
The cap leaves useful drones at least +1000 from success and this adjustment,
preserving an incentive to finish even with many unused drones. Inactive drones receive zero.
Both adjustments trigger only once at terminal chain success; zero disables
either independently. There is no contributors-only success switch.

`success_reward_as_hold_record` instead pays each qualifying holder
`success_reward / hold_chain_for` on each new episode hold record. Repeated
records do not pay, and no extra terminal success payout is added. Holders do
not divide a shared budget. This requires `success_condition: chain_held`.

Each acting drone receives `time_presure_reward` every transition, including
its removal transition. The default is -0.75 regardless of population. The
critic learns per-agent returns and PPO averages active-agent samples; the
summed team return in telemetry is not the training target.

G02 enables peer informing and soft cluster credit, uses recurring progress
reward and terminal success, and disables delivery vesting and hold records.
Other authored levels retain unsupported legacy reward blocks with commented
replacement defaults beneath them. Code in `env/environment.py`, the current
`RewardConfig`, and G02 define the current behavior.
