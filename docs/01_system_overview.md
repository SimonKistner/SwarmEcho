# 01 System overview

SwarmEcho trains communicating drones to find a target inside a three-dimensional building, report it to a base, and maintain a radio relay chain between the base and target. The policy must explore with local sensing, carry information through recurrent memory and learned messages, and position enough drones to keep the chain connected. The cleanup task additionally requires surplus drones to return to the base and decommission.

The [root README](../README.md) covers installation and commands. Read the detailed references in order.

1. This overview introduces the task, configuration and implementation boundaries.
2. [Environment and physics](02_environment_and_physics.md) explains geometry, dynamics, deployment, observations, communication, success and rewards.
3. [MARL and training](03_marl_and_training.md) explains the actor and critic, PPO, sampling curricula, auxiliary supervision and checkpoint lifecycle.
4. [Validation, evaluation and inspection](04_validation_evaluation_and_inspection.md) explains measurement protocols, diagnostics and saved artifacts.

## Maps, levels and configuration

A **map** is one building. Its **map file** describes dimensions, walls, floors, openings, stairs, the base position and target-spawn exclusions. A **level** defines an experiment. Its **level file** selects or generates maps and configures the environment, reward coefficients, network, training and measurement protocols. The same building can support several levels, and one level can train across many buildings.

| Preset | Map selection | Purpose |
| --- | --- | --- |
| `G01_static` | One authored building, `g01_static_map.yaml` | A local single-map experiment with same-map validation |
| `G02_static` | Five authored buildings, `g02_static_map_01.yaml` through `05` | A local multi-map experiment with success-weighted sampling |
| `G03_CLU_random_buildings` | 3,600 training buildings and 400 separate validation buildings | A generated-building experiment with forward target curriculum and path supervision |

These names identify presets. They also select different training and measurement settings. The level files contain their complete overrides. G01's map is stored independently of generated pools. Authored buildings live in `src/swarmecho/curriculum_config/maps/custom_maps/`, and compiled random pools live in `maps/random_building_bank/`. The map editor and standalone checkpoint evaluation also accept custom building YAMLs. [Level files](../src/swarmecho/curriculum_config/levels/G) · [Authored maps](../src/swarmecho/curriculum_config/maps/custom_maps)

`level=` accepts a preset name or a level YAML path. The configuration loader combines the schema defaults, level overrides and command-line `section.field=value` overrides. The resolved run `config.yaml` records the resulting experiment. `inspection_evaluation` is the descriptive YAML section name for inspection, and `evaluation` is its supported alias. Validation has its own section and success criterion. [Configuration schema and loader](../src/swarmecho/core/config.py)

The loader validates section names, fields and supported combinations. Saved experiment configurations pass through compatibility handling for historical keys before loading. That handling preserves access to existing artifacts. The current schema and level files define selectable mechanisms for new experiments.

G03 supplies the running example in these references. The system also supports different building dimensions, fleet caps, force and sensing ranges, reward coefficients, network widths and rollout sizes. Some adaptations require a fresh model because they change observation or decoder dimensions. Others have explicit compatibility constraints, such as fixed-base requirements for validation and adaptive target sampling. Each reference explains those constraints alongside the relevant mechanism.

## One environment episode

1. Reset selects a building, a legal target and a base position. The configured number of drone slots is reserved, and `spawn_delay` controls automatic launch timing.
2. Each active drone receives its local observation. Its recurrent actor combines that observation with received TarMAC messages and samples three movement commands.
3. Information propagates along the pre-action communication graph. The simulator integrates movement, checks collisions, updates target sightings and relay connectivity, and calculates individual rewards.
4. A held base–target chain satisfies `chain_held`. The `chain_held_and_cleand_up` criterion additionally requires all active surplus drones to leave through the decommission zone.
5. Success, the episode horizon, or exhaustion of both active and scheduled drones terminates the episode. Training lanes reset and continue collecting experience.

The target and base remain fixed within an episode. There is one base and one target. The simulator models motion and communication in a building using idealized geometric visibility and range checks. All drones use a common coordinate orientation. Sensor noise, camera field of view, vehicle attitude and a detailed flight controller are outside this model. [Environment transition](../src/swarmecho/env/environment.py)

## Learning and information boundaries

The shared actor acts from a drone's observation, recurrent state and received learned messages. The centralized critic uses the team's observations and recurrent states during training and produces a separate value estimate for each drone. PPO learns from individual reward streams and masks inactive slots. Runtime policy actions use the actor. [Actor](../src/swarmecho/models/actor.py) · [Critic](../src/swarmecho/models/critic.py) · [PPO](../src/swarmecho/training/ppo.py)

Geometry is available to the simulator for collision checks, relay estimates, reward attribution and supervision. The actor's inputs are defined by the observation function. The auxiliary decoder predicts a reference geodesic path from actor memory, using privileged geometry to construct training labels. Its prediction is available for diagnostics, and its loss can shape the actor representation. Actions are produced by the actor's movement head. [Path labels](../src/swarmecho/analysis/path_targets.py) · [Auxiliary training](../src/swarmecho/training/geo_auxiliary.py)

## Implementation and experiment records

| Area | Responsibility |
| --- | --- |
| `env/` | Building compilation, collision geometry, observations, knowledge, relay graphs and rewards |
| `models/` | Recurrent actor, TarMAC communication, centralized attention critic and action distributions |
| `training/` | Rollouts, PPO, sampling controllers, validation, checkpoint recovery and evaluation |
| `analysis/` | Geodesic labels, map preparation and diagnostic tools |
| `visualize/` | Replay recording, inspector, validation views and public export |
| `cluster/` | Cluster submission, source snapshots, run management and retrieval |

Cluster execution uses the same training implementation. Its separate [cluster documentation](../cluster/README/README.md) covers the infrastructure workflow. The [geometry experiments](../geo_experiments/README.md) document frozen-policy probes and their own artifacts.

For a scientific claim, retain the source revision, resolved configuration, checkpoint, selected maps, validation cases and metric records together. A preset describes a reproducible recipe, while each run records its actual overrides and inputs. The [evaluation reference](04_validation_evaluation_and_inspection.md#experimental-records) identifies the current baseline records and historical comparison artifacts.

The [ablation index](ablations/README.md) is a fallback for terminology or results involving mechanisms removed from this branch. The live source and these numbered references define the current implementation.
