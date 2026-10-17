# configuration

`src/swarmecho/core/config.py` is the single place to find the user-facing
setting definitions and their defaults, in the order below. `load_level` and
`load_level_cli` load a YAML level and apply OmegaConf `key=value` overrides.
Fields are validated against the dataclasses. Unknown fields or sections fail
explicitly.

| Section | Definition / purpose |
| --- | --- |
| `env` | `EnvConfig`: motion, perception, spawning, coverage, and obstacles |
| `adapt_size` | `AdaptSizeConfig`: opt-in voting, deployment, decommission, and reference-reward controls |
| `random_buildings` | `RandomBuildingConfig`: generated map dimensions and layout; disabled by default |
| `reward` | `RewardConfig`: relay objectives and reward weights |
| `training` | `TrainingConfig`: PPO, noise, rollout sizes, and checkpoint loading |
| `training.adaptive_map_sampling` | Default-off map-selection probabilities, per-map windows, warmup, floor, smoothing and support/evaluation budgets; enabled in G02; see the [reference](../01_reference/10_adaptive_map_sampling.md) |
| `network` | `NetworkConfig`: actor/critic memory, architecture, and TarMAC |
| `evaluation` | `EvaluationConfig`: schedules, robustness, maps, and artifacts |
| `logging` | `LoggingConfig`: run names, output roots, W&B, and terminal output |

There is no visualization configuration section in a level. Inspector controls
belong to the independent inspector.

### Training-only adaptive target spawning

`training.adaptive_target_spawn` is a nested `AdaptiveTargetSpawnConfig`.
The [feature reference](../01_reference/08_adaptive_target_spawning.md) specifies
the formulation, smoothing, warning rules and focused verification command.
It defaults off; disabled training retains the existing volume sampler, keyed
random draws, reset path, statistics and artifacts. Evaluation always retains
its ordinary sampler. Enable it explicitly, for example:

```bash
uv run swarmecho-train level=G01_static training.adaptive_target_spawn.enabled=true logging.run_name=G01_adaptive_spawn
```

```yaml
training:
  adaptive_target_spawn:
    enabled: false
    min_probability_fraction: 0.20
    success_window_episodes_per_cell: 500
    probability_smoothing_half_life_updates: 3.0
    plateau_warning_enabled: true
    plateau_warning_horizon_updates: 50
    plateau_warning_improvement_threshold_pp: 3.0
```

The first implementation supports one static authored target map with a fixed
base, no generated obstacles and no minimum-geodesic-separation filtering.
Unsupported enabled combinations fail explicitly. Each spawnable XYZ cell is its
own category, ordered by Manhattan distance to the base and then XYZ.

Baseline probability `b_i` is each cell's buffered valid free volume divided by
total valid volume. Stair solids, wall/floor buffers and no-spawn cells are
accounted for by the existing geometry sampler. Within a cell, probability is
split among its free-volume boxes in proportion to volume; positions remain
continuous. G01 has 23 spawnable cells, represented by 51 boxes with the current
default target buffer. Its baseline cell shares range from approximately 1.28%
to 5.87%, rather than equal 4.35% shares.

For success estimates `s_i`, largest failure rate `m=max(1-s_i)` and floor
fraction `f`, desired probability is:

```text
p_i = f*b_i + (1-f)*((1-m)*b_i + m*b_i*(1-s_i)/sum_j(b_j*(1-s_j)))
```

All-success returns to the baseline. Estimates initialize at zero, reproducing
the baseline until outcomes arrive. Each cell measures success over its latest
500 completed training episodes, including failures, using available outcomes
while filling. These independent windows do not alter the general run-statistics
window (currently sized to `training.num_envs`). Estimates persist when no new
episode completes in a cell. After every PPO update, probabilities move toward
the desired distribution with the configured half-life; the default moves 20.6%
of the remaining difference per update, preserving the minimum `f*b_i`.

Weights remain fixed throughout each rollout and affect new episodes, without
resampling episodes already underway. Plateau warnings use the window-smoothed
success estimate and average applied spawn probability over 50 PPO updates.
They require prior improvement above 3 percentage points somewhere in the run,
new completions in the flagged cell and sufficient headroom. No value during
the interval may exceed its starting success by more than the threshold. A
continuing plateau warns once and rearms after a significant gain.

No per-cell metrics are sent to W&B. The independent
[adaptive spawning dashboard](02_analysis_and_tools.md#adaptive-spawning-dashboard)
reads a small local journal appended once per PPO update. Checkpoints carry
`adaptive_spawn_state.json` with windows, applied-next probabilities and warning
state. `ckpt_loading_mode=resume` restores that controller only with matching
configuration and geometry; `branch` and `init` start new controller windows.
Resuming an older checkpoint without controller state prints a warning and starts
from the baseline. A new branch should use a new run name.

Wall sensing has three independent controls: `env.wall_sensing_layout`
(`fibonacci`, the legacy default, or `xyz`), `env.wall_sensing_rays` (default 16),
and `env.wall_sensing_ray_max_distance` (default 8.0 metres). Fibonacci accepts
any integer count >= 4; XYZ accepts only 14, 18, or 26. XYZ uses six axis rays
plus cube corners (14), plane diagonals (18), or both (26). All C levels use
XYZ with 26 rays and 8 m range. `env.visual_radius` controls target visibility
independently; `env.drone_radar_bins` retains the existing direction layout.
Saved configs using `wall_radar_bins` or the older combined `radar_bins` are
still readable and preserve their original wall range from `visual_radius`.
New explicit settings take precedence. Wall count, layout and range must match
a loaded checkpoint; renaming an old count does not change its sensor semantics.

Reward switches `success_bonus_as_hold_record`, `success_bonus_contributors_only`, `gap_reward_uses_change`,
`delivery_gap_vesting_enabled`, and `peer_informing_reward_enabled` default to
`false`. Delivery gap vesting requires gap-change reward and delivery gating;
`delivery_gap_vesting_steps` defaults to 10 and must be positive.
`gap_change_meter_bonus` and `peer_informing_bonus` set the respective enabled
reward amounts. The hold-record switch spreads the existing `success_bonus`
across first achieved consecutive chain-hold lengths rather than adding to it.
`success_bonus_contributors_only` keeps the terminal success bonus total unchanged
but distributes it only among selected chain contributors; it is enabled in
`B02c_random_buildings_small` and requires `chain_held` success.
The current B02c level enables peer informing and terminal success reward
while keeping the per-step absolute gap reward. Its time cost is 3 per step;
hold records and delivery gap vesting are disabled.

`B02c_random_buildings` also enables adaptive population control, starting with
two reserved drones staggered by `env.spawn_delay` and capacity nine. Its
voted reinforcements arrive immediately. Other levels keep their previous
population behavior unless `adapt_size.enabled=true` is set. See the
[adaptive swarm reference](../01_reference/06_adaptive_swarm_size.md) for settings
and [diagnostics guide](../01_reference/07_adaptive_swarm_diagnostics.md) for W&B curves.

`reward.reinforcement_reward` and `reward.decommission_reward` default to zero.
Positive values reward; negative values punish. Each qualified deployment voter
receives the full reinforcement amount, and each removed drone receives the full
decommission amount. Neither is divided across agents. Legacy saved cost fields
are converted by negating their values.

`adapt_size.reward_by_reference_enabled` defaults to `false` and requires adaptive
size. When enabled, each qualified caller receives an additional signed share
of a voted recruit's normal reward, excluding reference bonuses. Set
`adapt_size.reward_by_reference_fraction` (default `0.10`, range `[0, 1]`) and
`adapt_size.reward_by_reference_steps` (default `200`, positive integer) to control
the initial share and linear decay window in environment steps. The recruit's
reward is unchanged, each caller receives the full share, and references stop
on decommissioning or episode reset. The separate W&B reward curve is
`rewards/reward_by_reference`.

**First visual sighting is not a reward event in delivery-gated missions.**
Coverage rewards the exploration pattern; randomized target placement makes
the identity of the first drone to see the target incidental. `finder_bonus`
rewards a target-aware drone that delivers knowledge directly to the base,
while `target_found_bonus` pays its full configured value once to every active
drone at delivery, independent of the number of active drones. With
`target_found_requires_delivery=false`, it pays the same way at first discovery.
Existing configured amounts are now per drone, so total team payout grows with
the active population. See
[First visual sighting is not rewarded](../01_reference/02_environment_and_physics.md#first-visual-sighting-is-not-rewarded)
for the intended exploration, handoff, and relay sequence.

## Levels and maps

Levels live in `src/swarmecho/curriculum_config/levels/M/`, `levels/B/`, and
`levels/C/`; short CLI names, grouped paths, and legacy level paths still
resolve through the shared loader. Maps live in the
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
(2,000 lanes across three maps: 667/667/666). Maps must share cell size and
wall/tile thicknesses; grid/world dimensions may differ. Training pads arrays
internally and uses each map's physical bounds. The number of maps cannot exceed
`training.num_envs`.
Evaluation uses those same layouts, with `evaluation.eval_parallel_envs` episodes
**per map**; its heatmaps are saved for every map. Scheduled replays default to
every map; `evaluation.map_suite_replay_limit` can cap only replay generation. This
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

`env.observe_visually_gated_target_displacement=false` is the default. `G01_static`
enables it, adding three direct-visible target offsets normalized by `visual_radius`
in fixed XYZ axes. It returns zeros after sight is lost, even if the target is
still known through memory/communication. Together with the existing sighting flag,
G01 has 139 inputs. Train a new matching checkpoint before running its geo probes;
the Phase 0 checkpoint has 136 inputs. The enabled radius is checkpoint-critical
because it sets the measurement's normalization. See the
[sensor semantics and realism assumptions](../01_reference/02_environment_and_physics.md#observations).

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

When `reward.enable_chain_efficiency_reward=true`, W&B logs
`rewards/efficiency` alongside the other reward components. It sums the paid
reward across agents and steps per episode, then averages over the last
`training.num_envs` completed episodes. Logging starts once that window is
full, including both successful and failed episodes.

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

`evaluation.final_early_exit_success_rate` defaults to `null`. In an agent-count
curriculum it overrides the success threshold only in the final stage, when
both the geometric target budget and deployable population cap equal the original
`env.num_agents`. Intermediate stages retain `early_exit_success_rate`, including
stages that already reach the population cap through spare agents. Outside an
agent-count curriculum this setting has no effect.

`evaluation.training_robustness` enables robust periodic metrics.
`eval_robustness_runs` and `eval_action_noise_max` control robustness.
`evaluation.final_robustness` defaults to `true` for final training and standalone
parallel checkpoint evaluation. Set it to `false` for one deterministic pass
without added action noise; training reuses its ending metric pass when available.
All C levels disable both periodic and final robustness.
`evaluation.eval_min_train_success` skips scheduled metrics/replays below the
specified rolling training-success rate; ending evaluations bypass that gate.

Evaluation, replay, and checkpoint schedules each have frequency and offset
settings. The retained `eval_video`/frequency fields schedule replay
artifacts. `training_heatmap_creation` controls evaluation heatmap artifacts.
`evaluation.replay_targets_from_storey` defaults to `null`. Set a 1-based
storey number to restrict automatically generated random-building replay
targets to that storey; evaluation metrics and heatmaps still sample across
all valid target storeys. All B02 levels set it to `3`.
Early-exit settings default to evaluation success (`early_exit_source:
evaluation`), optionally requiring a hold period. Authored levels can select
`early_exit_source: training` to require fresh completed-episode windows on
every map; `early_exit_hold_evals` then counts passing training windows.
The existing success and successful-episode speed thresholds apply to each
map. Final evaluation still runs for diagnostics. See the
[C curriculum guide](05_C_curriculum.md) for the stages, target-distance
certificates, budgets and preview command.

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
