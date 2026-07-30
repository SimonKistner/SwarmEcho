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

### Transitional learning scaffolds

- Base-vector, target-vector, and local coverage-probe observations are
  temporary aids, comparable to training wheels. They are disabled in the
  maintained 2D levels but retained for the beginning of the 3D transition,
  where they may make early learning and diagnosis easier.
- These observation aids are not part of the intended final product. Once 3D
  training works without them, all three switches and their implementation
  should be removed rather than becoming permanent optional features.

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
- Physics is the sole authority for direct wall-aware communication edges,
  direct target visibility, and multi-hop base/target connectivity. Actor
  observations, TarMAC, reward routing, and evaluation rendering consume that
  result rather than rebuilding competing graphs.
- Direct adjacency is retained in the current state because maintained reward
  routing and communication consume it. Direct target visibility and the two
  final connectivity flags are compact per-drone vectors. Full reachability
  is temporary and is not stored across steps or in the rollout buffer.

## Reward assumptions

### Credit assignment

- Exploration and collision rewards remain local to the responsible agent.
- Target-found and terminal-success team bonuses are intrinsically shared and
  divided by the number of agents.
- Rewards are not globally averaged after their individual components are
  computed.
- Chain progress counts both the base-connected and target-connected fronts.
- The dynamic chain-gap reward is assigned to one deterministically selected
  shortest wall-aware communication route from the physics graph. Equally
  short alternative routes do not all receive that dynamic reward.
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
- The maintained 2D map progression is `M00_no_maze_open_square`,
  `M03_big_maze`, `M02_mid_maze`, then `M01_small_maze`.
- `M04_tiny_grid_maze_CORE_ONLY_TEST` is a validation run profile that uses
  the maintained `M01_small_maze` map; its historic filename is not a fifth map.

## Technical assumptions

- The environment state represents one base position and one target position,
  not variable-length entity collections.
- `EnvState` is an ownership container, not a flat collection of every array
  used during an episode. Kinematics and world entities, communication and
  knowledge, exploration coverage, relay-task progress, and per-step reward
  signals have separate nested state objects.
- General physical state does not own maze cells or finder-path history. The
  maintained discrete finder-path transition belongs to the 2D relay-task
  layer, which consumes physical and communication results.
- The 2D coverage grid belongs to the exploration scaffold rather than
  physical state. This keeps the temporary coverage training aid identifiable
  when it is reconsidered after the 3D transition.
- Collision and newly covered-cell results are per-step signals retained
  because rewards consume them; they are not persistent world geometry.
- Rendering receives the nested environment trajectory and creates a flat,
  CPU-only frame projection at the renderer boundary. Renderer presentation
  data is not added to environment state.
- The renderer uses the same single deterministic shortest-route convention as
  the reward implementation.
- The maze builder emits maps and levels for the maintained core behavior; it
  does not expose archived reward-routing or diagnostic-task switches.
