# Current problems

## Bug fix

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 9. **Nearby-drone observations do not directly establish valid relay edges.** | • Proximity does not imply wall-valid connectivity.<br>• Summaries lack associated records of usable neighbour links, weakening reasoning around walls/corners. | • Intended: walls block peer visibility and communication; agents do not choose usable links.<br>• Inspection: communication checks line of sight; radar checks only distance and active status, detecting peers through walls.<br>• If radar should show only usable links, this is an implementation bug. | |

## Explicit relay topology information

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 4. **The actor cannot reliably distinguish situations that receive different rewards.** | • Useful/redundant drones can have similar observations and histories.<br>• Rewards depend on hidden topology/path selection, producing conflicting signals.<br>• Issue: conditional indistinguishability, not RL averaging inherently erasing fine distinctions. | | |
| 5. **There is no explicit mechanism for allocating complementary relay roles.** | • Shared policy must decide who holds and who repositions.<br>• Without usable priority, negotiated roles, or other asymmetry, clustered drones may all hold or all move.<br>• Spatial understanding alone does not solve coordination. | • Explicit topology information may help learn complementary roles.<br>• Conditional benefit: interchangeable drones may still act symmetrically; information alone may be insufficient. | |
| 6. **Neighbour geometry is not associated with individual neighbour messages.** | • Spatial radar and aggregated messages are separate.<br>• Cannot associate a message with a particular nearby drone.<br>• Harder to assess alternative links, upstream/downstream roles, and who can safely leave. | | |
| 7. **The drone radar discards neighbour multiplicity.** | • Keeps maximum signal per angular bin.<br>• Several drones in similar directions can collapse into one signal.<br>• Replay-visible clusters may be less apparent in observations. | | |
| 8. **Communication aggregation also loses multiplicity information.** | • Softmax-weighted averaging gives identical results for one sender or several identical senders.<br>• Cannot reliably recover neighbour counts lost by radar. | | |
| 10. **The contributor observation provides no guidance during partial-chain construction.** | • Enabled in 9a, but zero until a complete chain exists.<br>• Does not distinguish useful, redundant, or dead-end incomplete-chain positions. | | |
| 11. **Direct target contact is not clearly distinguished from indirect target connectivity.** | • Observes target-component membership, not separate direct sighting with bearing/range.<br>• Harder to recognize the endpoint anchor that should hold.<br>• Limits what the discovering drone can communicate about the target. | • Add a flag indicating that the target is in the drone's visual range. | |
| 12. **There is no explicit distributed routing representation.** | • RL must teach messages to encode endpoint reachability, alternatives, roles, and temporal validity.<br>• Inputs lack associated hop estimates, message age, and routing relationships.<br>• Larger message vectors add capacity, but do not guarantee those meanings emerge. | | |

## Reward and contribution design

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 2. **Holding a partial chain generates substantial recurring reward.** | • 9a repeatedly rewards maintained progress.<br>• Last training record: mean shaping reward ≈5,831; success reward ≈9,306.<br>• Moving risks reliable income for uncertain future benefit.<br>• Magnitude tuning alone may not fix the incentive. | • Chain-change reward removes recurring income from freezing.<br>• Lost credited progress still yields negative reward.<br>• Path-credit reassignment can punish movement without worsening team position.<br>• Reward-mode changes alone do not solve rerouting. | |
| 3. **Selected-path membership is an unstable proxy for causal contribution.** | • Credit follows one selected relay path.<br>• Interchangeable drones can receive very different credit.<br>• Small movements can transfer credit without comparable changes in team usefulness. | • Consider credit across multiple eligible paths instead of selecting one through tie-breaking.<br>• May introduce new problems; better contribution-credit design needs further thought. | |


## Explicit geographic memory

Proposed intervention:

- Auxiliary loss: reconstruct building dimensions and interior geometry from the GRU hidden state.
- Trains and assesses memory representation; does not establish that the action policy uses the geometry.

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 14. **Obstacle-aware routing depends on an unverified spatial memory.** | • Drones do not know which corridors/staircases connect base and target.<br>• Detours require retaining and using previously observed geometry.<br>• Unverified: memory stores that geometry; action head uses it. | | |
| 17. **Spatial memory receives no explicit auxiliary supervision.** | • RL alone must teach motion integration, geometry retention, communication, and relay control.<br>• Easy successful behaviour may provide insufficient pressure to learn spatial representations.<br>• Targeted auxiliary prediction is promising; reward and coordination problems still need separate attention. | | |
| 18. **The central critic does not receive the privileged geometry/topology underlying the reward.** | • 9a critic aggregates team observations/histories.<br>• Must infer geometry/topology known directly to the reward calculation.<br>• Centralization alone does not resolve unstable contribution credit. | | |

## Training parameters

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 15. **Discounting strongly reduces the value of delayed rerouting benefits.** | • `gamma=0.99`: reward after 200 steps retains about 13% of immediate weight.<br>• At `dt=0.1`, this is a 20-second manoeuvre.<br>• Immediate loss of partial-chain income can outweigh delayed improvement. | | |
| 16. **Direct recurrent training spans only 100 steps.** | • Memory persists across rollout boundaries; gradients span only the rollout sequence.<br>• Longer-history retention/use is harder to learn.<br>• This does not mean the GRU forgets after 100 steps. | | |

