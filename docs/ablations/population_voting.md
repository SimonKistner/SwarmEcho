# Population voting and reinforcement

The removed recruitment system let drones request additional agents. It coupled policy actions, vote transport, base decisions, population changes and reward attribution. Current deployment is described in the [environment reference](../02_environment_and_physics.md#physics-and-deployment).

## Policy decisions and transport

Earlier actors had an additional recruitment decision alongside movement. A hold-style decision and a categorical KEEP/YES/NO decision were supported. KEEP retained the previous decision, while YES and NO changed the source's published choice. Learned TarMAC messages and explicit vote records were distinct channels.

The credit protocol circulated qualified YES records with source identity and freshness. The latest-decision protocol propagated each source's latest YES or NO timestamp, letting a newer NO invalidate an older YES. Peer transport used a pre-receive snapshot, enforcing one physical hop per communication tick. Relays could carry other drones' records. The base received records without broadcasting them back as peer vote records.

The base tested agreement among active sources. Below a quorum fraction of one, approval required strictly more than that fraction. A fraction of one meant unanimity, and an empty active population could not approve a call. `quorum_hold` required consecutive steps of agreement. Successful recruitment reset the hold and began a spawn-delay cooldown. During cooldown, the base cleared or invalidated approval and kept the hold counter at zero. Initial scheduled launches were reserved separately.

## Population changes and rewards

Reinforcements could activate available slots. Historical variants reused decommissioned slots, requiring new source identities and cleared recurrent/vote state so old records could not authorize a new agent's choices. Decommissioning invalidated that source's records and changed the quorum denominator. Copies carried by other active relays could outlive removal of a relay.

Privileged vote shaping used simulator knowledge to judge whether an agent was still needed. It could penalize connected informed NO decisions when reinforcements were needed and penalize responsible YES sources after an unnecessary grant. The budget estimate could tighten using an observed successful active chain population. Capacity-rejected requests had a separate penalty. Historical G02 used −200 for unnecessary grants, and CLU G03 used −500. These numbers describe the corresponding old configurations.

Reward by reference linked a granted recruit to the qualified caller sources. Each caller received an additional fraction of the recruit's ordinary rewards, decaying linearly with recruit age. The default fraction was 0.10 over 200 transitions, while G02 used 0.20 over 500. Each caller received its full share, multiple recruits could contribute, and the payment was additive. Reference bonuses themselves did not cascade recursively. Age advanced on environment transitions, independently of communication cadence.

## Removed observations and diagnostics

The system exposed source decisions, carried-record matrices, base agreement and cooldown state. A base vote-ratio input could enter observations. Replays and the inspector displayed base-vote cards, carried-vote records and drone decision icons. Recruitment diagnostics tracked grants, rejected calls, quorum timing and reference-reward attribution.

The old winning checkpoint retained a vote-ratio input that was always zero under its configuration. Its explicit [conversion record](../04_validation_evaluation_and_inspection.md#experimental-records) explains the current 140-input checkpoint copy. That historical input provides no evidence that voting was enabled in the winning run.
