# Signed individual rewards

Every reward coefficient is a signed amount: positive adds reward, negative
subtracts reward, and zero disables the term. The environment never negates a
configured amount implicitly. The retained names `all_decommissioned_penalty`
and `no_movement_termination_penalty` follow this same signed convention.
`time_presure_reward` retains the requested spelling in the public config.

## Defaults and recipients

The global numeric defaults follow G02. Previously population-split success,
delivery and time amounts use a four-agent reference: `12000 / 4 = 3000`,
`500 / 4 = 125`, and `-3 / 4 = -0.75`. Four is a calibration reference, not a
runtime divisor or a measured population average. Optional switches remain
off by default; G02 explicitly enables peer informing and soft cluster credit,
and selects obstacle-geodesic chain geometry.

| Setting | Default | Recipient and trigger |
| --- | ---: | --- |
| `exploration_reward` | `2` | Each credited new coverage voxel; stops for a drone after it knows the target |
| `finder_reward` | `100` | Each target-aware direct base neighbor on first delivery; without delivery gating, each first visual finder |
| `target_found_reward` | `125` | Every currently active drone once on first delivery, or first discovery without delivery gating |
| `success_reward` | `3000` | Every currently active drone once on first success |
| `time_presure_reward` | `-0.75` | Each acting drone every transition, including its removal transition |
| `collision_reward` | `-0.5` | Each colliding active drone on that transition |
| `useless_agent_at_success_reward` | `-3000` | Each active drone outside the selected contributing chain at terminal chain success |
| `useless_agent_at_success_all_agents_reward` | `-750` | Every active drone per useless active drone at terminal chain success; no population divisor; capped per recipient |
| `useless_agent_at_success_all_agents_reward_cap` | `-2000` | Signed bound on the all-agents adjustment per active drone; same sign as the coefficient, or zero to disable |
| `reinforcement_reward` | `1` | Every qualified deployment voter in full; initial spawns and rejected calls are excluded |
| `priveleged_vote_reward.withheld_reinforcement_vote_reward` | `-1` | Informed connected source choosing NO on an eligible communication tick while a recruit is needed; nested enabled toggle required |
| `priveleged_vote_reward.unnecessary_reinforcement_call_reward` | `-200` | Responsible qualified YES source on an unnecessary grant; nested enabled toggle required |
| `decommission_reward` | `0` | Each removed drone, paid even after removal |
| `non_contributor_return_reward_per_meter` | `0` | Surplus drone's positive obstacle-geodesic progress toward the decommission-zone centre during an intact chain hold; G03: `3` |
| `non_contributor_departure_reward_per_meter` | `0` | Surplus drone's metres away from that centre, including inside the zone; G03: `-3` |
| `non_contributor_decommission_reward` | `0` | One terminal payment to a removed surplus lifetime if the chain survives; local G03: `1500`, G03 CLU: `3000` |
| `rejected_call_reward` | `-200` | Each qualified YES voter on a rejected fresh voting batch |
| `all_decommissioned_penalty` | `-1000` | Each last acting drone in full when all drones are decommissioned |
| `no_movement_termination_penalty` | `0` | Every active drone on idle failure; success and time-limit termination are excluded |
| `gap_progress_reward_per_meter` | `0.2` | Per credited closed route metre per step, once the target-information gate is open |
| `gap_change_reward_per_meter` | `3` | Legacy diagnostic only: multiplies each drone's signed change in credited closed metres |
| `chain_efficiency_reward` | `0` | Each selected drone times its path efficiency when enabled |
| `peer_informing_reward` | `50` | One newly informed peer's credit, allocated to its eligible direct senders when enabled |

No coefficient is divided by active population, contributor count or holder
count. Inactive slots receive zero, except that removal-transition event
rewards and time pressure still belong to the lifetime that acted.
All-decommissioned deliberately preserves the former single-final-agent loss
of 1000 rather than calibrating that failure event down to 250.

At default terminal chain success with `U` useless active drones, every active
drone receives `max(-750 * U, -2000)` from the all-agents adjustment. There is no
population divisor: the cap applies separately to each recipient. It limits
the loss so useful drones retain at least +1000 from their +3000 success reward
and this adjustment even when many drones are unused, preserving an incentive
to finish. This bound concerns those components only; other rewards and reference
shares still affect the final total. Necessary team size is not penalized when `U=0`.
The useless drones themselves also receive the capped term alongside their
separate individual adjustment. With two useless drones, a contributor receives
`3000 - 1500 = 1500` and a useless drone receives `3000 - 3000 - 1500 = -1500`
from these terminal components. With three or more, those amounts are +1000 and
-2000. With no useless drones the all-agents term is zero.
Both useless-agent terms fire once at terminal chain success, work with or
without adaptive size, and are zero for inactive drones, failures and other
mission types. Each setting is signed and zero disables it independently.
The cap is a finite signed bound; zero suppresses the all-agents term.
Nonzero coefficient and cap must have the same sign. A negative cap is a lower
bound applied with `max`; a positive cap is an upper bound applied with `min`.
For example, +750 with a +2000 cap pays `min(750 * U, 2000)`.
Existing configs pairing a negative coefficient with a positive magnitude cap
must change that cap to its negative value. `min(-750 * U, -2000)` would impose
at least a 2000 loss even with no useless drones and would not limit larger losses.
Both enter recruits' normal rewards before reference shares are computed.
The contributors-only success switch remains removed.

## Local attribution and shaping

### Independent surplus-agent rewards

Both `G03_random_buildings` and `G03_CLU_random_buildings` explicitly enable
return progress at +3/metre and departure at -3/metre. Qualified decommission
pays +1500 in local G03 and +3000 in G03 CLU. All three global defaults are zero. Each coefficient independently
disables its own term; when all are zero the new calculation, component keys
and additional path planning are absent, preserving existing behavior.
The local +1500 exit amount is half its +3000 success reward; G03 CLU's +3000
exit amount matches its success reward. Each is a standalone setting: changing
or disabling success reward does not change the exit amount. G03 CLU also
enables geometric chain-efficiency reward at 50, with the submission config
comparing coefficients 25/50/100 under forward and reverse curricula.

These rewards apply only to adaptive `chain_held` or `chain_held_and_cleand_up` missions, starting during
the holding period, with a complete chain before and after the action. They
use the same selected contributor membership as `observe_chain_contributor`,
which both G03 levels already enable. An acting non-contributor need not be
connected to either endpoint to receive guidance. A surviving drone must also
remain a non-contributor; becoming a contributor or losing the chain produces
zero movement shaping. Eligibility changes never create progress payments.

Movement uses the signed change of shortest obstacle-aware distance to the
decommission-zone centre
on the existing roadmap, not the legacy gap-change switch. Positive and
negative progress are multiplied by their separate signed coefficients.
There is no path-length normalization: a longer return earns more total
undiscounted reward. Stationary agents receive zero. This permits equivalent
shortest routes rather than requiring the inspector's exact reference polyline.

There is no radius cutoff or zero-reward region. Decreasing distance to the
zone centre earns positive movement reward; increasing it incurs the departure
penalty, including inside the decommission sphere. Arriving at the exact centre
earns the final progress payment; remaining there earns zero, and moving away
is penalized immediately. Unchanged distance earns zero at any position.
The actual decommission sphere and its dwell requirement are unchanged.

The terminal exit payment requires an actual new decommission event, prior
non-contributor membership, and an intact chain before and after removal.
It belongs to the removed lifetime even when the slot is immediately reused,
and does not pay again on later steps. Contributors and chain-breaking exits
receive none. It is additive to the existing unconditional `decommission_reward`;
enabling either does not enable the other. Existing rewards and penalties are
unchanged. Enabled components enter training metrics, replay plots and normal
reference-reward accounting.

`reward.priveleged_vote_reward.enabled=false` preserves the previous
reinforcement rewards and protocol exactly. G02 enables it with timestamped
latest decisions. When enabled, an eligible NO costs -1 per communication tick;
an unnecessary grant costs each responsible source -200 and suppresses the
ordinary grant reward. Budget uses the estimator's sufficient count, tightened
by the smallest population that actually formed a chain. Decision-time context
protects delayed previously necessary approvals. Capacity rejection does not
stack the new penalty. Both new terms enter normal reward/reference credit and
the existing training/replay component pipeline. See the
[full timing and eligibility rules](06_adaptive_swarm_size.md#optional-reinforcement-vote-shaping).

Local credit normalization remains intentional: simultaneous viewers split a
new voxel, simultaneous eligible senders split one newly informed peer, and
optional soft cluster credit allocates progress among equal-hop relay
alternatives. An unrelated active drone does not dilute these credits. See
[soft cluster credit](09_soft_cluster_credit.md).

`gap_reward_uses_change`, `gap_change_reward_per_meter` and the companion
`delivery_gap_vesting_enabled` / `delivery_gap_vesting_steps` are **legacy
diagnostic tools only**, retained to inspect historical experiments. Do not
recommend them as training settings, reward-design alternatives or plateau
remedies. Recurring gap shaping is the normal training setting.

For diagnostic interpretation, `gap_reward_uses_change` replaces the recurring
gap term with signed progress changes. `delivery_gap_vesting_enabled` requires
change mode and delivery gating; it vests pre-delivery credited metres over `delivery_gap_vesting_steps`
(default 10), paying any remaining current credit at termination. Negative
progress coefficients are supported as signed coefficients, without silently
disabling their computation.

With `success_reward_as_hold_record=true`, every qualifying holder receives
`success_reward / env.hold_chain_for` on each new episode hold record. Repeated
records do not pay, and there is no additional terminal success payout.
Population does not enter that calculation. This mode requires a chain-hold
success condition. With `chain_held_and_cleand_up`, completing hold-record
payments does not bypass cleanup or pay additional records while cleanup continues.

Reference rewards remain additive decaying shares of each recruit's complete
normal signed reward, independently paid to every qualified caller, with no
deduction from the recruit and no cascading of reference rewards. G02 keeps
20% initial credit over 500 transitions; global adapt-size defaults remain
disabled, 10%, and 200 transitions.

## Training, telemetry and supported levels

The critic predicts per-agent returns. PPO uses per-agent advantages and
averages over active samples; it does not train against the summed episode
return shown in W&B. Fixed time pressure makes each acting drone face the
same urgency at every population size. More agents can still alter the mix of
training samples and caller reference rewards.

Reward component keys use the current config names, except that the selected
recurring/change gap component is `gap_reward` and caller credit remains
`reward_by_reference`. `success` in transition info is the terminal boolean;
the additive component is `success_reward`. Positive/negative reference
diagnostics and gap gain/loss diagnostics must not be added a second time.
See [adaptive diagnostics](07_adaptive_swarm_diagnostics.md).

G02 is the migrated level. Every other authored level retains its original
reward block, marked unsupported, with a commented copy of the current global
defaults immediately underneath. Replace that original block and review its
level-specific switches before using it again. Old reward names and removed
switches are rejected, including in old saved configs and CLI overrides;
there are no reward aliases or automatic conversions. Checkpoint contracts
record `signed_individual_v1` so loading older weights reports the behavioral
change. Existing historical run metrics and archived results are not rewritten.

Runtime verification must be performed in WSL; this change was inspected
without running the project's WSL virtual environment or Python compilation.
