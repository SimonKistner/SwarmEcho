# Adaptive swarm size

Adaptive population control is enabled in G02 and disabled globally by default. Other authored levels retain unsupported legacy reward blocks until migrated. The environment retains a fixed JAX array of `env.num_agents` slots (seven in G02). `adapt_size.vote_protocol` selects accumulating `credits` (the global default) or timestamped `latest_decision` (enabled in G02). Credit packets use fixed-shape packed receipts sized from `env.max_steps`; latest decisions use one timestamp and YES/NO value per receiver/source pair. A slot can be decommissioned and reused indefinitely in one episode. Its replacement gets a new lifetime ID and starts with zero target knowledge, vote state, actor/critic recurrent memory and TarMAC message state. PPO keeps the shared learned weights and previously collected rollout samples. Decommissioning cuts the per-lifetime return, including when a replacement occupies the same slot immediately.

## Deployment and voting

`adapt_size.initial_agents` reserves the initial cohort. Drone 1 starts at reset; the others activate at their normal `slot_index × env.spawn_delay` episode step. A successful vote deploys one reinforcement immediately in a free, unreserved slot. Thus a call can arrive before a delayed initial drone without cancelling that drone's scheduled arrival. The episode terminates if every active drone is decommissioned, even while an initial drone is still scheduled.

The actor has the existing three Gaussian movement outputs plus one discrete vote output from its shared policy trunk. Only movement is passed through `tanh`. `hold` samples a Bernoulli YES/NO action which sets the bit each step. `keep_yes_no` samples a categorical KEEP, YES or NO action; KEEP retains the bit. The bit starts at zero. Each actor observes its own vote and whether it is inside the decommission sphere. Both modes use the existing PPO entropy coefficient.

Two additional adaptive observations default to false globally and are enabled
only in G02: `env.observe_base_vote_ratio` and `env.observe_base_population`. A drone currently
connected to the base, directly or through physical radio relays, receives:

1. Qualified active voters at the base / currently active deployed drones.
2. Currently active deployed drones / `DEPLOYED_AGENTS_OBS_REF`.

The global constant lives in `core/config.py` and is 10.0 for every map and
curriculum stage. It is a normalization reference, not a deployment cap; the
population observation is not clipped. Population means the live fleet, not
reserved initial slots, lifetime deployment IDs or cumulative replacement calls.
In `credits`, a voter qualifies after the base receives `vote_holding` unspent
YES credits in its current round. In `latest_decision`, it qualifies when the
base's newest received decision for that source and round is YES. A received NO
withdraws that approval; missing information is UNKNOWN and does not approve.
Neither mode reads a disconnected source's current action directly.
Successful deployment clears base approval and starts the next round with
a zero vote-ratio observation. Latest-decision capacity rejections do so too.
These inputs are instantaneous: both become zero when the drone disconnects,
and reconnecting exposes the current values rather than remembered snapshots.
No observation receipt is cached. The recurrent policy can still learn to
remember previous inputs. Inactive drones receive zero throughout.
There is no raw vote tally, round number, episode clock or configured capacity
in these two observations. Changing the fleet size does not change their width.

In both modes, communication runs every `network.memory_comm_every_k_steps` steps through unobstructed, in-range direct links. One pre-receive snapshot enforces one physical hop per communication tick. The base receives vote records but does not relay them back out. `quorum_fraction` accepts values in [0, 1]. Below 1, approval requires **strictly more** than that fraction of currently active drones: 0.49 admits 4/8 and 5/9. A value of **1.0 means unanimity**: every currently active drone must have qualified approval at the base. G02 uses 1.0. An empty active population never reaches quorum. Decommissioning immediately invalidates the removed source's records everywhere and updates the denominator. A removed relay loses its copies, but copies at other relays survive. `quorum_hold` optionally adds consecutive environment steps with quorum before an attempt.

With `vote_protocol: latest_decision`, every connected source sends its current YES **or NO**, timestamped with the episode transition step. A receiver keeps only the newest timestamp for each source in its known round, independent of arrival route. Relays preserve source timestamps; a delayed old YES cannot overwrite a newer NO. Duplicate copies and repeated YES messages do not accumulate credits. Unchanged source choices refresh their timestamps on communication ticks. A source emits its current choice directly; it does not queue its own past choices, and its carried-vote diagonal stays empty. Relay copies persist without a timeout, even after the source disconnects or changes its mind. Thus approval is the base's latest known agreement, not a guarantee of simultaneous current agreement. `vote_holding` is ignored in this mode: one latest YES approves, and a newer received NO revokes it. The existing `hold` and `keep_yes_no` policy action modes both work.

With `vote_protocol: credits`, a connected YES source emits one numbered credit. Each receiver accepts at most one new credit per source per tick, with packet receipts deduplicating routes. Relays retain all received packets for that round; NO stops new emissions without retracting stored or delivered credits. An isolated source cannot bank credits for later reconnection, and a source does not retain its own packets. Qualification needs `vote_holding` delivered, unspent credits, regardless of the source's current YES/NO bit.

Successful approval deploys one drone immediately, advances the call round and clears base approval. In latest-decision mode a capacity rejection also advances the round and clears the base's records, so one cached agreement can fund only one attempt. Further attempts require fresh decisions in the new round; there is no extra cooldown. The base repeatedly announces its current round through the same one-hop network. Disconnected drones catch up when they reconnect, clearing their carried records and local vote; acknowledgement overrides a simultaneous policy YES. Old-round records cannot approve in the new round. A newborn knows the current round at the station but has no mission knowledge.

At capacity an attempted call deploys nobody and penalizes **each qualified YES voter**. In credits mode the round and raw receipts remain, but base spends all credits already received from qualified voters. Each needs a fresh `vote_holding` batch before another rejection. Voting time also incurs the existing episode time cost. G02 pays `reinforcement_reward=10` to each qualified source on successful deployment; carrying somebody else's approval does not make a relay a qualified source.

## Optional reinforcement vote shaping

`reward.priveleged_vote_reward.enabled` defaults to `false`; G02 enables
it. It requires adaptive size with `vote_protocol: latest_decision`. This switch
controls both new reward terms, suppression of ordinary reinforcement reward on
unnecessary grants, and the remembered chain population. With it off, the old
reward terms, reward arithmetic, vote transport, deployment rules and policy
observations are retained. No shaping arrays or replay fields are allocated.

```yaml
reward:
  reinforcement_reward: 10
  priveleged_vote_reward: # Estimator/chain-aware vote penalties; suppress bonuses on unnecessary calls.
    enabled: true
    withheld_reinforcement_vote_reward: -1.0
    unnecessary_reinforcement_call_reward: -200.0
```

The effective population budget is the smaller of the accepted target's
estimator count and the smallest **active population** that has formed a valid
physical chain during this episode, including at reset. This records population,
not the length of a selected path. Estimator counts are certified sufficient
constructions, not proofs of a continuous minimum. Without an estimator count,
there is no budget-based supervision until a chain has been observed. A currently
connected chain always makes another deployment unnecessary once the base knows
the target. A later break retains the observed budget; decommissioning can make
replacement recruitment up to that budget necessary again. Reserved initial
arrivals count toward pending population so they do not justify extra calls.

`withheld_reinforcement_vote_reward` pays each informed, base-network-connected
source choosing NO on an eligible communication tick while the population is
below budget and no chain is connected. Every withholding source is charged,
not only a single pivotal voter. It does not pay between communication ticks,
when a call completes, while at capacity, for inactive/removed/newborn sources,
or while learning/catching up to a new round. The source must already know the
target and the base must already know it before the action. Connectivity excludes
agents removed on this transition. Current YES choices are never charged because
an older NO is still in transit. A previously received YES is no exemption for
a source now choosing NO, since NO can withdraw approval. There is no credit
batch or dependence on `vote_holding` in this protocol. G02 communicates every
step, so sustained eligible NO costs -1 per step; a cadence of five charges
only once per five steps, without multiplying the amount by five.

`unnecessary_reinforcement_call_reward` pays each responsible qualified YES
**source** on a granted call exceeding the effective budget or adding a drone
while a valid chain is connected. Decision-time context travels with the same
timestamped record as YES/NO. An uninformed approval or a YES sent while the
recruit was needed is not retroactively penalized when it arrives after a chain
forms. A call that is necessary again after population loss is also exempt.
All classification uses the pre-action state, not knowledge or connectivity
first acquired after movement. A simultaneous decommission invalidates the old
chain-connectivity override, allowing a replacement within the budget without
being charged for a chain that the removal may have broken. The grant's ordinary `reinforcement_reward` is
suppressed on unnecessary calls, including for exempt delayed sources. Calls
before base delivery retain ordinary reinforcement reward. Capacity rejection
continues to pay only `rejected_call_reward`, without stacking an unnecessary
deployment penalty. No vote is blocked or changed by shaping.

Both coefficients are finite signed per-agent amounts; zero disables that
component. Suppression of ordinary reinforcement reward on unnecessary grants
is the only direct reward change independent of these two coefficients: it
remains active with both amounts zero as long as `enabled=true`. Optional
reference rewards inherit changes to recruits' normal rewards. To restore the
complete old behavior, including ordinary call reward,
use `reward.priveleged_vote_reward.enabled=false`. The terms enter normal
reward before optional caller-reference shares. W&B reports both under
`rewards/<coefficient_name>`; replays store both additive components and the
optional `adapt_vote_shaping_*` budget, provenance and event arrays. Inspector
reward plots label both components. No estimator/budget observation is added.
Checkpoint contracts record the enabled behavior as a noncritical reward change;
changing it warns while retaining compatible policy tensors.

## Replay vote inspection

New replays record the full receiver/source matrices at every frame, including the base: `adapt_packet_round`, `adapt_packet_count`, `adapt_packet_bits`, `adapt_decision_step`, `adapt_decision_yes`, and the episode `adapt_vote_step` clock. Timestamp -1 means no received decision. Credit packet IDs are zero-based emission sequence numbers. Lifetime IDs are recorded alongside the matrices, and removed/reused slots have their old traffic invalidated before a new identity can use them.

The inspector's **Carried votes** card lists every active drone's retained records, including an explicit **None** for empty storage. Hovering its marker, name, or YES indicator shows the same list together with coordinates, current vote and lifetime ID. Latest-decision entries show source, YES/NO, source timestamp, age in environment steps, and round; credit entries show source, count and exact packet ID ranges. Base rows show YES, NO or UNKNOWN in latest-decision mode and credit totals in credits mode. A carrier can still show records from an older round until it hears the base announcement. Older replays remain readable and show **Not recorded in this replay** where carrying data is unavailable; their aggregate totals cannot reconstruct those matrices.

## Decommissioning and rewards

A sphere of configurable radius is placed above each base, outside the spawning drone's body, with a collision-free route from the spawn point. A reset without a valid sphere fails visibly in training/evaluation. A drone inside for `decommission_hold_steps` consecutive environment steps is removed permanently. Leaving resets dwell. Its slot can later hold a fresh reinforcement. The sphere and dwell progress appear in the replay inspector.

All reward values are signed individual amounts. `reinforcement_reward` pays
every qualified deployment voter in full, `decommission_reward` pays each removed
drone, and `rejected_call_reward` pays each qualified rejected voter. Initial
spawns do not earn a reinforcement reward. A voter's own source must qualify
under the selected protocol; relaying another voter's packet does not qualify. Removal
rewards and time pressure remain payable to the lifetime that acted.

`all_decommissioned_penalty` is also signed and pays its full amount to each
last acting drone on extinction. No population divisor remains. The same
individual convention applies to `success_reward`, `target_found_reward`,
`time_presure_reward`, and `no_movement_termination_penalty`.

At terminal chain success, `useless_agent_at_success_reward` pays only each
currently active drone outside the selected shortest-hop success chain.
Contributors and inactive drones receive zero from this component. Separately,
`useless_agent_at_success_all_agents_reward=-750` applies to every active drone
per useless active drone, without a population divisor. Its per-recipient
absolute amount is limited by `useless_agent_at_success_all_agents_reward_cap=2000`.
With `U` useless drones, default terminal success components are
`3000 - min(750 * U, 2000)` for contributors and `-min(750 * U, 2000)` for
useless drones. The cap preserves at least +1000 from these components for useful
drones even when many drones are unused, keeping completion worthwhile. Inactive drones
receive neither term. Both trigger only once at terminal chain success and
zero disables either independently. The contributors-only success switch remains removed.
Partial-chain shaping still uses the selected frontier leaders and optional
local soft cluster shares. This individual success adjustment also works
without adaptive size and enters normal signed recruit reward for references.

See [signed individual rewards](11_individual_rewards.md) for the full default
table, four-agent calibration and local attribution rules. Legacy reward names
and cost aliases are no longer converted or accepted.

## Reward by reference

With `adapt_size.reward_by_reference_enabled=true` and adaptive size enabled, every qualified source whose approval helped grant a reinforcement receives an extra fraction of that recruit's normal reward. Qualification follows the selected vote protocol. The caller group is captured at the grant, before base approval is reset. Relaying somebody else's vote does not make a drone a caller. Rejected calls and scheduled initial drones create no references.

For recruit `j` with age `a`, its extra reward to **each** caller is:

```text
reward_by_reference_fraction × max(1 - a / reward_by_reference_steps, 0) × normal_reward[j]
```

Age is zero on the activation transition and advances on every environment step, regardless of communication cadence. Defaults pay 10% at activation, 5% at age 100, 0.05% at age 199, and zero from age 200 onward: 200 rewarded transitions including activation. Each caller receives the full share independently; it is not divided among callers or deducted from the recruit. A caller's bonuses from multiple recruits add together.

The normal reward includes all configured individual signed rewards, including terminal success and useless-agent rewards. Reference bonuses are excluded from the source reward, so they do not cascade up recruitment chains. If D1 calls D2, then D2 calls D3, D1 receives a share only of D2's normal reward. If D2 and D3 later call D4, each receives a full decaying share of D4's normal reward; D1 gets no D4 share unless it also qualified for that call.

Every active recruit receives the fixed success reward at terminal success. If
it is useless to the selected chain, its individual useless-agent adjustment
also enters its normal reward. Each caller receives the signed decaying share
of that combined reward, including the all-agents useless-drone charge. That
charge applies to useful recruits too whenever any active drone is useless.
There is no payout for future steps after the episode ends.
References end when either lifetime is decommissioned, including on the removal transition, and reset with the episode. A replacement in a reused slot inherits no references, and a fresh voted recruit starts with a new caller group and age zero. The signed extra reward is exposed as `reward_by_reference` in reward terms and `rewards/reward_by_reference` in W&B. Replay files also store `adapt_reference_callers` (caller rows, recruit columns) and `adapt_reference_age` for inspecting the attribution.

W&B also reports `rewards/reward_by_reference_positive` and
`rewards/reward_by_reference_negative`. These sum positive and negative
caller-recruit payments separately per episode, retaining the negative sign,
before averaging over the same completed episodes. They split payments before
different recruits can cancel each other. The two diagnostics sum to the net
reference reward and do not change actual drone rewards.

## Exposed settings

All settings have normal YAML and CLI `key=value` overrides. G02 is the migrated
reference level; old level reward blocks need their commented replacement
before loading. The reference controls belong to `adapt_size`.

| Setting | Global default | G02 | Meaning |
| --- | ---: | ---: | --- |
| `adapt_size.enabled` | `false` | `true` | Enable population control |
| `adapt_size.initial_agents` | `1` | `1` | Reserved initial cohort |
| `env.num_agents` | `5` | `7` | Hard maximum slot count |
| `env.spawn_delay` | `5` | `5` | Stagger reserved initial drones only |
| `env.observe_base_vote_ratio` | `false` | `true` | Qualified base voters / live population, zero when disconnected |
| `env.observe_base_population` | `false` | `true` | Live population / global reference 10, zero when disconnected |
| `adapt_size.vote_mode` | `hold` | `hold` | Vote action layout |
| `adapt_size.vote_protocol` | `credits` | `latest_decision` | Accumulating credits or newest timestamped YES/NO |
| `reward.priveleged_vote_reward.enabled` | `false` | `true` | Enable budget/chain-aware individual vote rewards |
| `adapt_size.vote_holding` | `1` | `1` | Delivered YES credits per voter; ignored by latest_decision |
| `adapt_size.quorum_fraction` | `0.49` | `1.0` | Strictly greater below 1; 1 means unanimity |
| `adapt_size.quorum_hold` | `0` | `0` | Required consecutive quorum steps |
| `network.memory_comm_every_k_steps` | `5` | `1` | Vote and message cadence |
| `adapt_size.decommission_radius_m` | `1` | `0.75` | Sphere radius in metres |
| `adapt_size.decommission_hold_steps` | `15` | `15` | Consecutive dwell steps |
| `reward.reinforcement_reward` | `1` | `10` | Full signed amount per qualified deployment voter; suppressed on unnecessary grants when shaping is enabled |
| `reward.priveleged_vote_reward.withheld_reinforcement_vote_reward` | `-1` | `-1` | Per eligible communication tick choosing NO; shaping toggle required |
| `reward.priveleged_vote_reward.unnecessary_reinforcement_call_reward` | `-200` | `-200` | Per responsible YES source on an unnecessary grant; shaping toggle required |
| `reward.decommission_reward` | `0` | `0` | Full signed amount per removed drone |
| `reward.rejected_call_reward` | `-200` | `-200` | Full signed amount per qualified rejected voter |
| `reward.all_decommissioned_penalty` | `-1000` | `-1000` | Full signed amount per last acting drone |
| `reward.useless_agent_at_success_reward` | `-3000` | `-3000` | Only each useless active drone at terminal chain success |
| `reward.useless_agent_at_success_all_agents_reward` | `-750` | `-750` | Every active drone per useless active drone at terminal chain success; no divisor; capped per recipient |
| `reward.useless_agent_at_success_all_agents_reward_cap` | `2000` | `2000` | Nonnegative maximum absolute adjustment per active drone after counting useless drones |
| `adapt_size.reward_by_reference_enabled` | `false` | `false` | Enable caller rewards |
| `adapt_size.reward_by_reference_fraction` | `0.10` | `0.20` | Full initial fraction per caller |
| `adapt_size.reward_by_reference_steps` | `200` | `500` | Linear decay window in transitions |

To enable reference rewards in an adaptive level, add or override:

```yaml
adapt_size:
  reward_by_reference_enabled: true
  reward_by_reference_fraction: 0.10
  reward_by_reference_steps: 200
```

Enabling adaptive population control or changing vote mode changes action and observation shapes. The two base-feedback toggles each add one observation channel; G02's adaptive width changes from 140 to 142 with both enabled. Checkpoint contracts record both flags and the enabled population normalization reference, rejecting changed input meanings even when widths match. Old checkpoints interpret missing flags as false: disable both for legacy evaluation, or train a fresh policy to use the added inputs. Enabling reward by reference changes rewards without changing the policy's action or observation shapes.

The vote protocol is also recorded in checkpoint contracts. Missing protocol metadata means `credits` for an adaptive checkpoint; loading it as `latest_decision` is rejected even though policy tensor shapes match. New G02 training uses latest decisions. To select the old transport for a new run, override `adapt_size.vote_protocol=credits`, disable `reward.priveleged_vote_reward.enabled`, and set the desired `adapt_size.vote_holding` (for example, 3 for the previous G02 threshold).

## Verification

From WSL in the repository root:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/test_reward_by_reference.py
uv run pytest -q tests/test_adapt_size.py
node --test tests/test_inspector_adapt_size.cjs
```

The dedicated reference-reward checks use only 2-4 drone slots, the real vote
protocol, and known reward inputs. They check caller assignment, unqualified
voters and relays, full shares for multiple callers, default/custom decay
boundaries, signed rewards without cascading, terminal success and useless-agent rewards,
the disabled toggle, decommissioning, slot reuse, rejected calls, and fresh
episode state. They sample ages instead of simulating 200 steps, disable whole
function JIT compilation, and require no map loading, model, gradients, or
training. Run just this file for the lightweight feature check.

The broader adaptive-size tests exercise compiled/vmapped population churn, delayed initial spawns versus immediate calls, both vote heads and PPO updates, isolated vote loss, one-credit-per-hop delivery, announcements, repeated rejections, decommission termination with sparse/dense autoreset, lifetime returns, rewards, checkpoint contracts, metrics, evaluation capture/replay and inspector state. They use a small authored building and require no random-building bank or W&B account. A full G02 training run uses its five frozen authored maps. See [adaptive diagnostics](07_adaptive_swarm_diagnostics.md) for interpreting training curves.
