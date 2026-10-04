# configuration

`src/swarmecho/core/config.py` is the single place to find the user-facing
setting definitions and their defaults, in the order below. `load_level` and
`load_level_cli` load a YAML level and apply OmegaConf `key=value` overrides.
Fields are validated against the dataclasses. Unknown fields or sections fail
explicitly.

| Section | Definition / purpose |
| --- | --- |
| `env` | `EnvConfig`: motion, perception, spawning, coverage, and obstacles |
| `random_buildings` | `RandomBuildingConfig`: generated map dimensions and layout; disabled by default |
| `reward` | `RewardConfig`: relay objectives and reward weights |
| `training` | `TrainingConfig`: PPO, noise, rollout sizes, and checkpoint loading |
| `network` | `NetworkConfig`: actor/critic memory, architecture, and TarMAC |
| `evaluation` | `EvaluationConfig`: schedules, robustness, maps, and artifacts |
| `logging` | `LoggingConfig`: run names, output roots, W&B, and terminal output |

There is no visualization configuration section in a level. Inspector controls
belong to the independent inspector.

Reward switches `success_bonus_as_hold_record`, `gap_reward_uses_change`,
`delivery_gap_vesting_enabled`, and `peer_informing_reward_enabled` default to
`false`. Delivery gap vesting requires gap-change reward and delivery gating;
`delivery_gap_vesting_steps` defaults to 10 and must be positive.
`gap_change_meter_bonus` and `peer_informing_bonus` set the respective enabled
reward amounts. The hold-record switch spreads the existing `success_bonus`
across first achieved consecutive chain-hold lengths rather than adding to it.
The current B02c level enables peer informing and terminal success reward
while keeping the per-step absolute gap reward. Its time cost is 3 per step;
hold records and delivery gap vesting are disabled.

**First visual sighting is not a reward event in delivery-gated missions.**
Coverage rewards the exploration pattern; randomized target placement makes
the identity of the first drone to see the target incidental. `finder_bonus`
rewards a target-aware drone that delivers knowledge directly to the base,
while `target_found_bonus` is shared at delivery. See
[First visual sighting is not rewarded](../01_reference/02_environment_and_physics.md#first-visual-sighting-is-not-rewarded)
for the intended exploration, handoff, and relay sequence.

## Levels and maps

Levels live in `src/swarmecho/curriculum_config/levels/`; maps live in the
adjacent `maps/` directory. With `random_buildings.enabled=false`,
`env.map_names` contains one or more authored map names or YAML paths relative
to `maps/`. A one-map list keeps the classic authored-map training and
evaluation path, including the M levels. With `random_buildings.enabled=true`,
set `env.map_names` to `[]` or `[random]`; omitted generation fields use the
defaults in `RandomBuildingConfig`. Use a level name or an existing YAML path:

```bash
uv run swarmecho-train level=B01a_office_find_only logging.run_name=office
```

`B02c_static` copies the current B02c mission settings and names three 15 m
buildings copied from the compatible random-building pool into `maps/`. Its
`random_buildings.enabled` and `evaluation.random_eval` flags are false. Training
uses each named layout persistently, distributing lanes as evenly as possible
(2,000 lanes across three maps: 667/667/666). The maps must share grid and world
dimensions, and the number of maps cannot exceed `training.num_envs`.
Evaluation uses those same layouts, with `evaluation.eval_parallel_envs` episodes
**per map**; its heatmaps and scheduled replays are saved for every map. This
measures performance on the training layouts. The run records map paths,
content hashes, and lane counts in `static_maps.json` and rejects changes to
that selection in the same run directory. Comment out two `env.map_names`
entries and use a new run name to use the classic single-map path, whose
evaluation also uses that map and retains its usual final robust evaluation.

```bash
uv run swarmecho-train level=B02c_static logging.run_name=B02c_static
```

The supported M-series names are `M00_no_maze_open_cuboid`,
`M01_no_maze_open_cuboid_tall`, and `M02_random_cuboid_obstacles`.
The B-series names are `B00_test`, `B01a_office_find_only`, `B01b_office`,
`B02a_random_buildings_find_only`, `B02b_random_buildings_deliver`, and
`B02c_random_buildings`, and `B02c_static`.

Building cell size controls authoring geometry. `env.coverage_voxel_size`
controls coverage resolution and must divide every world dimension; null uses
the building cell size. Optional actor observation flags change network input
width, so they must agree with a loaded checkpoint.

`env.success_condition` selects `coverage`, `discovery`, `delivery`, or
`chain_held`. Coverage episodes omit the target and end when every eligible
interior voxel has been seen. Voxel centres inside solid geometry are excluded.
`evaluation.success_condition` can override the training goal; B02a uses
`coverage` during training and `discovery` during evaluation. The target remains
present in B02a evaluation for comparable target heatmaps and replays.

Target knowledge persists per drone. A visual sighting informs that drone
immediately; on each `network.memory_comm_every_k_steps` communication tick,
informed drones pass knowledge one direct, unobstructed communication hop.
An informed drone can report to the base, which then informs drones in its
communication range on later ticks. `env.base_keeps_informing` defaults to
`false`: the base replays its saved neural message only to uninformed drones.
Set it to `true` to replay that message to all drones in base range, including
already informed drones.

## Training and checkpoints

Recurrent training requires `training.num_envs` divisible by
`training.num_minibatches`. The timestep budget must cover at least one rollout.
Production actor and critic memory must both be enabled.

The W&B metric `train/visually_found_rate` is the fraction of the last
`training.num_envs` completed episodes in which any drone directly saw the
target. It uses the same sliding window as `train/target_found_rate`.

`training.checkpoint_path` loads a checkpoint. `ckpt_loading_mode=resume`
continues run accounting, `branch` carries cumulative progress into another run,
and `init` loads weights with fresh accounting. Use compatible architecture,
observation settings and normalization.

See [PPO controls](../01_reference/05_ppo_controls.md) for clipping, entropy,
and normalization. Training action perturbations use
`training.training_noise` and `training.noise_level`.

## Evaluation

`evaluation.num_agents` defaults to `null` (`None` in Python), which uses the
training count from `env.num_agents`. Set it to a positive integer to deploy
that many agents with the trained policy during periodic and final evaluation,
standalone checkpoint evaluation, heatmap generation, and all replay modes.
Recurrent actor memory and TarMAC messages use the evaluation count; training
and the checkpoint's weights keep their original count and architecture.
`B02c_random_buildings` trains with 7 agents and evaluates with 9, so newly
generated replays contain 9 agents. Existing saved artifacts retain their
original agent count. Override with `evaluation.num_agents=null` to evaluate
with the training count, or, for example, `evaluation.num_agents=9` to use 9.

Evaluation can use a different map through the existing
`evaluation.eval_differes_from_training_map` and `evaluation.eval_map`
fields. The spelling is retained for existing configuration compatibility.

`evaluation.apply_trainings_min_geo_separation` defaults to `false`. When
enabled, evaluation keeps the map's fixed base and applies the training
geodesic base/target minimum, multiplier, and spawn-attempt limit while
sampling targets. With a fixed base, each attempt samples another target,
so `training.skip_on_no_pair_found` is unnecessary. Evaluation reports an
error if no qualifying target is found within the attempt limit. The B02
levels enable this setting.

`evaluation.minimum_geodesic_separation` optionally overrides that inherited
setting. B02a disables target/base separation during targetless training and
enables it during target-based evaluation. Early exit requires
`evaluation.early_exit_success_rate` for the configured evaluation goal; setting
`evaluation.early_exit_min_success_length_reduction` also requires successful
episodes to finish sooner on average.

`evaluation.training_robustness` enables robust periodic metrics.
`eval_robustness_runs` and `eval_action_noise_max` control robustness.
Standalone parallel checkpoint evaluation uses the robust evaluation path.

Evaluation, replay, and checkpoint schedules each have frequency and offset
settings. The retained `eval_video`/frequency fields schedule replay
artifacts. `training_heatmap_creation` controls evaluation heatmap artifacts.
`evaluation.replay_targets_from_storey` defaults to `null`. Set a 1-based
storey number to restrict automatically generated random-building replay
targets to that storey; evaluation metrics and heatmaps still sample across
all valid target storeys. All B02 levels set it to `3`.
Early-exit settings gate training on evaluation success, optionally requiring
a hold period.

## Reward defaults

The defaults live in `RewardConfig`: delivery required; Euclidean chain
distance; exploration 0.25; collision 0.5; finder 0; target found 100;
success 500; idle termination -1000. Each active drone shares a team time
penalty of 5 per step. A contributing drone on a target mission earns a
per-step gap reduction bonus of 0.125 per metre of the selected base-to-target
distance already closed. Coverage missions, including B02a training, receive
the time penalty but no gap bonus. Optional redundancy and efficiency rewards
are disabled by default; the efficiency bonus is 0.5. Level overrides take
precedence. B02c explicitly sets the 0.2 per-metre bonus.
At a 40 m roadmap distance, closing 1 m pays the default 0.125 to each
contributing drone each step; a held zero-gap chain pays 5 to each contributor
each step. With B02c's 0.2 setting, those amounts are 0.2 and 8.
