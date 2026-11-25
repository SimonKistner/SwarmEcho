# Removed observation and broadcast variants

Earlier observation layouts supported more feature combinations than the current branch. Observation widths recorded in old experiments belong to their exact configuration. The [current observation contract](../02_environment_and_physics.md#observations) defines the maintained sensors and their dimensions.

| Removed input or mode | Historical information supplied |
| --- | --- |
| Raw relative base vector | Direction and displacement to the base in the observation frame |
| Raw relative target vector | Target displacement under the older knowledge-based visibility rules |
| Separate visual Boolean | A direct-sighting flag alongside persistent target knowledge |
| Clock/timestep input | Progress through the episode horizon |
| Coverage probes | Local samples of the shared exploration-coverage state |
| Own-vote/base-vote signals | Recruitment decisions or agreement in the removed voting system |
| XYZ wall-ray layouts | Fixed axis, plane-diagonal and corner-direction arrangements with 14, 18 or 26 rays |

The historical target vector could expose position after knowledge propagated. The maintained visually gated displacement follows direct sight. The early 136-input frozen checkpoint used binary sighting and produced the [recorded ambiguity results](observation_ambiguity.md). The later sensor-only run added three displacement values to that historical layout. Its 139-input count belongs to that revision.

Always-inform base broadcast admitted the base's cached learned message for drones that already knew the target. It was an alternate receiver condition. The current recipient condition and message timing are documented under [communication](../02_environment_and_physics.md#communication-and-knowledge).

The archived directional-ray selector is separate from the two peer-radar representations that remain available. Population feedback, decommission-zone membership, chain contribution and visible displacement also remain live settings and are explained in the current reference.
