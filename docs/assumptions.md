# Project Assumptions

This is a living, intentionally incomplete record of assumptions behind the
maintained SwarmEcho workflow. When future work exposes another implicit
assumption, add it at the most appropriate level.

## Project-level assumptions

### Research task

- The maintained task is decentralized multi-agent exploration followed by
  formation of a communication relay between one base station and one target.
- There is exactly one base station and exactly one target in an episode.
- Finding or reporting the target is not the final objective. Success requires
  a continuous multi-hop base-to-target chain for the configured hold time.
- The base station is a passive endpoint, not a permanently available,
  omniscient communication channel to every drone.
- Agents execute from local observations under a shared policy. Centralized
  information is training-time critic context, not privileged actor input.

### Scope

- The maintained environment is the two-dimensional core workflow.
- Normal training and evaluation use map-defined levels rather than diagnostic
  memory tasks or special-purpose toy objectives.
- Removed experiments are preserved as ideas and rationale, not as dormant
  implementation branches.

## Environment assumptions

### World and entities

- Maps are static during an episode. Walls, spawn zones, and no-spawn zones are
  map data.
- Base, drone, and target spawns are sampled from their map-defined zones.
  A zero-area zone may define a single fixed spawn point.
- Target exclusion around a base is represented by a map no-spawn area rather
  than a runtime spawning policy.
- Movement, visibility, and communication respect the maintained solid-wall
  model; there are no communication-transparent mesh walls.

### Knowledge and communication

- A drone cannot use the target position before it has observed or received
  that information through the modeled communication process.
- Target knowledge is persistent once acquired.
- Information reaches the base only through the modeled, range-limited,
  wall-aware communication links. Any supported base memory replay remains
  subject to its explicit range, timing, and validity rules; it is not a
  permanent base broadcast.
- Communication connectivity is evaluated through the current multi-hop
  network, including the base and target endpoints.

## Reward assumptions

### Credit assignment

- Exploration and collision rewards remain local to the responsible agent.
- Target-found and terminal-success team bonuses are intrinsically shared and
  divided by the number of agents.
- Rewards are not globally averaged after their individual components are
  computed.
- Chain progress counts both the base-connected and target-connected fronts.
- The dynamic chain-gap reward is assigned to one deterministically selected
  shortest communication route. Equally short alternative routes do not all
  receive that dynamic reward.
- Non-contributing agents receive the configured maximum chain-gap penalty,
  divided by swarm size.

### Removed shaping and objectives

- Drone separation is not directly shaped by an agent-agent proximity penalty.
- Motion toward the base or target is not directly shaped by proximity bonuses.
  The policy must learn useful positioning from exploration, collision,
  information-delivery, and chain rewards.
- The finder bonus is awarded at the maintained discovery/delivery event. It is
  not split with a later bonus for returning to the target.
- Returning to the target after delivery is not an episode objective.

## Training and evaluation assumptions

- Training uses parallel randomized environment resets drawn from the selected
  map definition.
- Evaluation metrics always come from one parallel deterministic-policy batch;
  there is no sequential metric evaluator or parallel-mode switch.
- The only supported automatic early-exit metric is parallel evaluation
  success. Recent training success and target-found rates do not terminate a
  run.
- Reaching the configured early-exit threshold always saves a handoff
  checkpoint before `train()` returns. A solo run then ends; a curriculum
  runner advances to the next level.
- `evaluation.save_model` controls scheduled and final saves, not the mandatory
  early-exit handoff checkpoint.
- Evaluation policy actions are deterministic, while an evaluation episode's
  initial state may still vary according to map spawn zones and its random key.
- An evaluation video contains exactly one independently collected episode. It
  is not selected from, and need not reproduce an episode in, the parallel
  metric batch.
- Saved evaluation CSV rows describe realized episode positions and outcomes,
  not a separate target-generation policy.
- Evaluation, video, heatmap, and checkpoint-saving schedules are configured
  under `evaluation`. Checkpoint loading and branch/resume behavior remain under
  `training`.
- Checkpoints are tied to the maintained observation, model, and reward
  semantics; compatibility with removed experimental branches is not assumed.

## Technical assumptions

- The environment state represents one base position and one target position,
  not variable-length entity collections.
- The renderer uses the same single deterministic shortest-route convention as
  the reward implementation.
- The maze builder emits maps and levels for the maintained core behavior; it
  does not expose archived reward-routing or diagnostic-task switches.
