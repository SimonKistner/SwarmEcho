# Adaptive swarm size

Adaptive population control is enabled in `B02c_random_buildings` and disabled by default for other levels. The environment retains a fixed JAX array of `env.num_agents` slots (nine in B02c). Vote packets use fixed-shape packed receipts sized from `env.max_steps`. A slot can be decommissioned and reused indefinitely in one episode. Its replacement gets a new lifetime ID and starts with zero target knowledge, vote state, actor/critic recurrent memory and TarMAC message state. PPO keeps the shared learned weights and previously collected rollout samples. Decommissioning cuts the per-lifetime return, including when a replacement occupies the same slot immediately.

## Deployment and voting

`adapt_size.initial_agents` reserves the initial cohort. Drone 1 starts at reset; the others activate at their normal `slot_index × env.spawn_delay` episode step. A successful vote deploys one reinforcement immediately in a free, unreserved slot. Thus a call can arrive before a delayed initial drone without cancelling that drone's scheduled arrival. The episode terminates if every active drone is decommissioned, even while an initial drone is still scheduled.

The actor has the existing three Gaussian movement outputs plus one discrete vote output from its shared policy trunk. Only movement is passed through `tanh`. `hold` samples a Bernoulli YES/NO action which sets the bit each step. `keep_yes_no` samples a categorical KEEP, YES or NO action; KEEP retains the bit. The bit starts at zero. Each actor observes only its own vote and whether it is inside the decommission sphere. It has no round number, clock, global population, or vote tally. Both modes use the existing PPO entropy coefficient.

On every `network.memory_comm_every_k_steps` communication tick, a YES drone authors one vote credit **only if it has a live drone or base link**. An isolated drone cannot store YES votes for later delivery, and the sender cannot replay old votes after reconnecting. Credits travel one unobstructed, in-range hop per communication tick alongside the normal TarMAC message; each receiver accepts at most one new credit per source per tick. Relays retain individual packets they already received, so one can deliver an earlier vote after its sender disconnects. Per-source, per-round packet receipts prevent duplicated routes from multiplying credits. NO pauses new credits but does not withdraw credits already received. The base counts a drone as approving once it has at least `vote_holding` unspent credits for the round. Approval requires **strictly more** than `quorum_fraction` of currently active drones: 0.49 admits 4/8 and 5/9. Decommissioning immediately removes a drone's credit and updates the denominator. `quorum_hold` optionally adds consecutive environment steps with quorum before an attempt.

Successful approval deploys one drone, advances the deployment round and resets base credits. The base repeatedly announces its latest completed round through the same one-hop network. Disconnected drones catch up when they reconnect and clear their local vote on receipt. A newborn knows the current protocol round at the station but has no mission knowledge. The base round and received credits are visible in the inspector, not in actor observations.

At capacity an attempted call deploys nobody and penalizes **each qualified YES voter**. The round and raw receipts remain, but the base spends all credits received from those voters. Each needs a fresh `vote_holding` batch before another rejection. This can repeat without a hard call limit, cooldown, reinforcement spawn delay, or capacity observation. Voting time already incurs the existing episode time cost. B02c sets reinforcement_reward to -50 for each qualified voter on a successful deployment.

## Decommissioning and rewards

A sphere of configurable radius is placed above each base, outside the spawning drone's body, with a collision-free route from the spawn point. A reset without a valid sphere fails visibly in training/evaluation. A drone inside for `decommission_hold_steps` consecutive environment steps is removed permanently. Leaving resets dwell. Its slot can later hold a fresh reinforcement. The sphere and dwell progress appear in the replay inspector.

`reward.reinforcement_reward` is a signed amount paid in full to each qualified source whose delivered base credits triggered a successful deployment; `reward.decommission_reward` is a signed amount paid in full to each removed drone; `reward.rejected_call_penalty` is paid by each qualified voter on a rejected batch; and `reward.all_decommissioned_penalty` is a team cost paid by the last acting drones. At successful chain termination, `reward.success_unused_agent_penalty` charges the team once per **currently active** drone outside the selected contributing chain. That total is divided across active drones. Previously decommissioned and inactive drones are excluded. A complete chain selects one shortest-hop base-to-target relay path; partial-chain shaping still uses the selected frontier leaders. `reward.success_bonus_contributors_only` optionally pays the unchanged team success bonus only to those contributors. It defaults to false and is enabled only in `B02c_random_buildings_small`. B02c keeps the chain-efficiency reward off. The original time cost continues every step, including the decommissioning step.

`reward.success_noncontributor_penalty` adds an independent **individual** cost
at terminal `chain_held` success. Each currently active drone outside the same
selected contributing chain receives the full negative amount itself; contributors
and inactive/decommissioned drones receive zero from this term. It is charged once,
on top of the existing shared unused-agent penalty and all other reward terms.
The config defaults to `0`; all C levels set it to `4000`, giving each
noncontributor an additional `-4000`. It also works without adaptive swarm size.
Its reward component is `success_noncontributor_penalty`. When reference rewards
are enabled, this cost is included in the recruit's normal signed reward used
for the existing caller shares.

Both signed event rewards default to zero and accept any finite positive or negative value: positive rewards the recipient, negative punishes. Reinforcement credit uses all qualified sources captured before the base clears its receipts, independently of reference rewards. With vote_holding=5, voters at 5/5 qualify and a voter at 3/5 does not. If eight voters qualify when only two are needed, all eight receive the exact configured amount; it is never divided. Scheduled initial spawns and rejected calls do not receive the reinforcement reward. Decommission reward remains paid on the removal transition. Legacy saved reinforcement_cost/decommission_cost fields load with their signs inverted; explicit new names take precedence.

## Reward by reference

With `adapt_size.reward_by_reference_enabled=true` and adaptive size enabled, every qualified voter whose delivered YES credits helped grant a reinforcement receives an extra fraction of that recruit's normal reward. The caller group is captured at the grant, before base credits are reset. Relaying somebody else's vote does not make a drone a caller. Rejected calls and scheduled initial drones create no references.

For recruit `j` with age `a`, its extra reward to **each** caller is:

```text
reward_by_reference_fraction × max(1 - a / reward_by_reference_steps, 0) × normal_reward[j]
```

Age is zero on the activation transition and advances on every environment step, regardless of communication cadence. Defaults pay 10% at activation, 5% at age 100, 0.05% at age 199, and zero from age 200 onward: 200 rewarded transitions including activation. Each caller receives the full share independently; it is not divided among callers or deducted from the recruit. A caller's bonuses from multiple recruits add together.

The normal reward includes all configured individual and shared rewards and penalties, including terminal success and unused-agent penalties. Reference bonuses are excluded from the source reward, so they do not cascade up recruitment chains. If D1 calls D2, then D2 calls D3, D1 receives a share only of D2's normal reward. If D2 and D3 later call D4, each receives a full decaying share of D4's normal reward; D1 gets no D4 share unless it also qualified for that call.

A recruit called shortly before successful termination receives a success bonus only if it qualifies under the configured success reward rules. Its normal share of the global unused-agent penalty is still included in the reference calculation at the remaining fraction. That extra penalty is paid independently by each caller in addition to their own normal rewards and penalties. There is no payout for future steps after the episode ends.

References end when either lifetime is decommissioned, including on the removal transition, and reset with the episode. A replacement in a reused slot inherits no references, and a fresh voted recruit starts with a new caller group and age zero. The signed extra reward is exposed as `reward_by_reference` in reward terms and `rewards/reward_by_reference` in W&B. Replay files also store `adapt_reference_callers` (caller rows, recruit columns) and `adapt_reference_age` for inspecting the attribution.

W&B also reports `rewards/reward_by_reference_positive` and
`rewards/reward_by_reference_negative`. These sum positive and negative
caller-recruit payments separately per episode, retaining the negative sign,
before averaging over the same completed episodes. They split payments before
different recruits can cancel each other. The two diagnostics sum to the net
reference reward and do not change actual drone rewards.

## Exposed settings

All settings have normal YAML and CLI `key=value` overrides. Both `vote_mode` choices are commented in code and the level. Global defaults leave the feature off; B02c values are starting points for experiments.

The three reference-reward settings belong to `adapt_size`. All C curriculum
levels explicitly enable them (`true`, `0.10`, `200`); the global toggle default
remains `false`. Other levels that leave reference rewards off inherit the
defaults without YAML entries.

| Setting | Global default | B02c | Meaning |
| --- | ---: | ---: | --- |
| `adapt_size.enabled` | `false` | `true` | Enable vote action, observations and population control |
| `adapt_size.initial_agents` | `1` | `2` | Reserved initial cohort, in `[1, env.num_agents]` |
| `env.num_agents` | `5` | `9` | Maximum population / fixed JAX slot count |
| `env.spawn_delay` | `5` | `5` | Stagger only the reserved initial cohort |
| `adapt_size.vote_mode` | `hold` | `hold` | `hold` or `keep_yes_no` |
| `adapt_size.vote_holding` | `1` | `10` | Delivered YES credits per voter; isolated YES ticks add none |
| `adapt_size.quorum_fraction` | `0.49` | `0.49` | Approval requires a strictly greater active-voter fraction |
| `adapt_size.quorum_hold` | `0` | `0` | Minimum consecutive environment steps with quorum; 0 acts immediately |
| `network.memory_comm_every_k_steps` | see network config | `3` | Existing communication cadence, used by votes |
| `adapt_size.decommission_radius_m` | `1.0` | `1.0` | Sphere radius in metres |
| `adapt_size.decommission_hold_steps` | `15` | `15` | Consecutive environment steps inside sphere |
| `reward.reinforcement_reward` | `0` | `-50` | Full signed amount per qualified deployment voter; initial spawns are exempt |
| `reward.decommission_reward` | `0` | `0` | Full signed amount per removed drone |
| `reward.rejected_call_penalty` | `0` | `500` | Cost per qualified voter per rejected batch |
| `reward.all_decommissioned_penalty` | `0` | `1000` | Team cost on extinction |
| `reward.success_unused_agent_penalty` | `0` | `500` | Team cost per active noncontributor at success |
| `reward.success_noncontributor_penalty` | `0` | `0` | Individual cost per active noncontributor at terminal chain success; C levels use `4000` |
| `reward.success_bonus_contributors_only` | `false` | `false` (`true` in B02c small) | Split terminal success bonus among contributing drones only; requires `chain_held` |
| `adapt_size.reward_by_reference_enabled` | `false` | `false` | Enable signed caller bonuses; requires adaptive size |
| `adapt_size.reward_by_reference_fraction` | `0.10` | `0.10` | Full initial share per caller, finite and in `[0, 1]` |
| `adapt_size.reward_by_reference_steps` | `200` | `200` | Positive number of rewarded environment transitions; linear decay |

To enable reference rewards in an adaptive level, add or override:

```yaml
adapt_size:
  reward_by_reference_enabled: true
  reward_by_reference_fraction: 0.10
  reward_by_reference_steps: 200
```

Enabling adaptive population control or changing vote mode changes action and observation shapes. Checkpoint restore rejects incompatible contracts; train a fresh policy for each vote architecture. Enabling reward by reference changes rewards without changing the policy's action or observation shapes.

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
boundaries, signed rewards without cascading, terminal success and unused-agent rewards,
the disabled toggle, decommissioning, slot reuse, rejected calls, and fresh
episode state. They sample ages instead of simulating 200 steps, disable whole
function JIT compilation, and require no map loading, model, gradients, or
training. Run just this file for the lightweight feature check.

The broader adaptive-size tests exercise compiled/vmapped population churn, delayed initial spawns versus immediate calls, both vote heads and PPO updates, isolated vote loss, one-credit-per-hop delivery, announcements, repeated rejections, decommission termination with sparse/dense autoreset, lifetime returns, rewards, checkpoint contracts, metrics, evaluation capture/replay and inspector state. They use a small authored building and require no random-building bank or W&B account. A full B02c training run still needs the configured random-building bank. See [adaptive diagnostics](07_adaptive_swarm_diagnostics.md) for interpreting training curves.
