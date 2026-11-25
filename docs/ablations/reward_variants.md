# Removed mission and reward variants

The current individual reward semantics are defined in [environment and physics](../02_environment_and_physics.md#coverage-and-rewards). Historical logs and coefficients must be interpreted using their original reward implementation.

## Success and reward allocation

Earlier tasks could terminate on exploration coverage, target discovery or delivery to the base. They isolated stages of the mission. The current success criteria require holding a physical relay chain, with an additional cleanup condition available.

Historical reward systems included team sharing and population-based scaling. A reward divided by configured population can have a different aggregate magnitude and incentive from the current full per-recipient amount. Old comparisons also used a four-agent calibration reference. Reward names alone cannot establish numerical equivalence across revisions.

## Relay and information shaping

| Removed mechanism | Historical incentive or attribution |
| --- | --- |
| Euclidean relay gap | Rewarded progress using straight-line distances, which could treat positions across walls as close |
| Hard cluster credit | Selected a discrete relay branch with deterministic tie-breaking, concentrating credit on that branch |
| Redundant-chain reward | Credited members of multiple valid simple base–target paths, allowing alternative routes to earn simultaneously |
| Discovery-triggered finder bonus | Paid on direct target sighting before base delivery |
| Peer-informing reward | Paid sending drones for newly informed peers, with variants sharing credit across simultaneous senders |
| Hold-record payout | Paid increments when a chain exceeded its previous consecutive-hold record |
| Gap-change reward | Paid changes in relay progress between transitions |
| Delivery vesting | Accumulated pre-delivery gap-change credit and released it after delivery over subsequent transitions |
| Proximity shaping | Used distances to the base, target or neighbours as movement incentives |
| Reference reward | Paid callers a decaying share of a recruited drone's rewards |

The redundant-path implementation used subset dynamic programming to summarize simple routes and their lengths. Its state cost grew with the number of agent subsets. This also changed which drones counted as contributors in its associated observation and replay views.

An early post-delivery split task rewarded a finder returning toward the target while other drones handled reporting. Another experiment pre-covered the base neighbourhood to reduce repeated local exploration reward. The old notes record design motivations for these variants, without a controlled numerical benefit claim.

## Idle termination

An alternate terminal branch stopped an episode after a configured period without meaningful movement and could apply an idle penalty. The winning G03 setup had a 1,000-step threshold beyond its 700-step horizon and a zero penalty. Movement counters can still appear in diagnostic records without providing this removed termination rule.

The [population-voting record](population_voting.md) describes privileged vote shaping and reward by reference. The current soft geodesic credit, delivery event and recurring closed-gap reward are explained in the maintained environment reference.

## Generic removal and terminal-surplus payments

The generic `decommission_reward` paid every removed drone and was zero in G03. The retired `useless_agent_at_success_reward` penalized each active surplus drone at terminal success. `useless_agent_at_success_all_agents_reward` penalized every active drone for the surplus count, bounded by `useless_agent_at_success_all_agents_reward_cap`.

Cleanup success requires no active surplus, so those terminal penalties contributed zero to G03 cleanup training. They could contribute under chain-only inspection and validation. Their removal changes those reward totals while preserving the success criteria. The separate intact-chain surplus return, departure and decommission rewards remain part of G03.
