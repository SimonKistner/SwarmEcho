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
