# Current problems

## Bug fix

| problem | description | notes | solved? |
| --- | --- | --- |---------|
| 9. **Nearby-drone observations do not directly establish valid relay edges.** | Peer radar previously checked only distance and active status, allowing signals through walls despite communication being blocked. | All three peer radar channels now apply the same solid-intersection check used by communication, covering authored walls and generated obstacles. Regression test: `tests/test_environment.py::test_peer_radar_hides_drones_behind_walls`; failure confirmed before the fix. Individual neighbour association and multiplicity remain separate issues (6–8). | Fixed✅ |

## Explicit relay topology information

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 4. **The actor cannot reliably distinguish situations that receive different rewards.** | Useful and redundant drones can have very similar observations and histories, while the reward depends on hidden topology and path selection. This creates conflicting learning signals. The problem is conditional indistinguishability, rather than RL averaging inherently destroying fine distinctions. | | |
| 5. **There is no explicit mechanism for allocating complementary relay roles.** | A shared policy must somehow decide which drone holds and which repositions. Without a usable priority, negotiated role, or other asymmetry, clustered drones can all choose to hold—or all choose to move. Spatial understanding alone does not solve this coordination problem. | Explicit relay topology information may help the policy learn complementary hold and reposition roles. This problem is only conditionally addressed: interchangeable drones may still choose symmetric actions, so better information alone may not be sufficient. | |
| 6. **Neighbour geometry is not associated with individual neighbour messages.** | The actor receives a spatial radar summary and an aggregated communication message. It cannot directly identify “this message belongs to that drone over there,” making it harder to reason about alternative links, upstream/downstream roles, and who can safely leave. | | |
| 7. **The drone radar discards neighbour multiplicity.** | It keeps the maximum signal in each angular bin. Multiple drones in approximately the same direction can collapse into one signal. The cluster visible in the replay is therefore not necessarily equally visible in the observation. | | |
| 8. **Communication aggregation also loses multiplicity information.** | Softmax-weighted averaging produces the same result for one sender and several senders with identical contents. The communication channel cannot reliably recover the neighbour count discarded by radar. | | |
| 10. **The contributor observation provides no guidance during partial-chain construction.** | Although enabled in 9a, it remains zero until a complete chain exists. It cannot tell an agent whether its current incomplete-chain position is useful, redundant, or a dead end. | | |
| 11. **Direct target contact is not clearly distinguished from indirect target connectivity.** | The actor observes target-component membership rather than a separate direct-sighting measurement with target bearing/range. Consequently, identifying “I am the endpoint anchor that should hold” is harder than intended. This also limits what the discovering drone can communicate about the target. | Add another flag that clearly indicates when the target is in visual range of the drone. | |
| 12. **There is no explicit distributed routing representation.** | Learned messages must discover how to encode endpoint reachability, alternatives, roles, and temporal validity through the RL objective. The current inputs do not directly supply associated hop estimates, message age, or routing relationships. Larger message vectors provide capacity, but do not guarantee that these meanings emerge. | | |

## Reward and contribution design

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 2. **Holding a partial chain generates substantial recurring reward.** | 9a repeatedly rewards maintained progress. The last training record averaged about 5,831 reward from this shaping, compared with 9,306 from success. Moving can sacrifice reliable income for an uncertain future benefit. Reward magnitudes alone may not fix this structural incentive. | Chain-change reward removes recurring income from freezing, but does not solve rerouting: losing previously credited progress produces a negative reward, and path-credit reassignment can punish movement without worsening the team's position. Changing the reward mode alone is insufficient. | |
| 3. **Selected-path membership is an unstable proxy for causal contribution.** | Reward credit follows one selected relay path. Nearly interchangeable drones can receive very different credit because one wins the path selection. Small movements can transfer that credit without producing a comparable change in actual team usefulness. | Possibly allocate reward across multiple eligible relay paths instead of selecting a single path through tie-breaking. This could introduce new problems, so a better approach to contribution credit still needs to be considered. | |



## Explicit geographic memory

Proposed intervention: add an auxiliary loss for reconstructing building dimensions and interior geometry from the GRU hidden state. This addresses memory representation and supervision, but does not establish that the action policy uses the learned geometry.

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 14. **Obstacle-aware routing depends on an unverified spatial memory.** | The drones do not know which corridors or staircases connect the base and target. The policy must retain previously observed geometry and use it to evaluate detours. We have not established that its memory contains that information, or that its action head uses it. | | |
| 17. **Spatial memory receives no explicit auxiliary supervision.** | The RL objective must simultaneously teach motion integration, geometry retention, useful communication, and relay control. The easy successful behaviour may provide too little pressure to learn the harder spatial representation. This is why targeted auxiliary prediction is promising—but it would not independently fix the reward or coordination problems above. | | |
| 18. **The central critic does not receive the privileged geometry/topology underlying the reward.** | 9a uses an observation-based critic. It aggregates team observations and histories, but must still infer information that the reward calculation knows directly. A central critic therefore does not automatically resolve the unstable contribution credit described above. | | |

## Training parameters

| problem | description | notes | solved? |
| --- | --- | --- | --- |
| 15. **Discounting strongly reduces the value of delayed rerouting benefits.** | With `gamma=0.99`, a reward 200 steps later has about 13% of its immediate weight. At `dt=0.1`, that is a 20-second manoeuvre. Immediate loss of partial-chain income can therefore dominate a delayed improvement. | | |
| 16. **Direct recurrent training spans only 100 steps.** | Memory persists across rollout boundaries, but gradients through its construction span the rollout sequence. This makes learning to preserve and use information over longer navigation histories harder. It does not mean the GRU forgets after 100 steps. | | |

