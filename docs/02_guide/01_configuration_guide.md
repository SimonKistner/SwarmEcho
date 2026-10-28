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
| `inspection_evaluation` | Visual inspection schedules, maps, robustness and artifacts; `evaluation` remains a legacy alias |
| `validation` | Independent held-out statistical episodes, batching, result tables and best-model selection |
| `logging` | `LoggingConfig`: run names, output roots, W&B, and terminal output |

There is no visualization configuration section in a level. Inspector controls
belong to the independent inspector.

`training.num_maps` separates persistent random training-map count from
`training.num_envs` (parallel lanes). Its `null` default preserves the old
lane-count fallback; authored levels keep their configured map list.
`inspection_evaluation.inspection_maps` replaces `random_eval_envs`; random
buildings automatically use a persistent inspection subset. G03 reserves 400
validation maps beside 3600 training maps, using three held-out maps for visual
inspection. See [inspection and validation](../01_reference/12_validation.md)
for the independent 10000-episode validation pass, fixed targets and checkpoint
selection. G02 keeps inspection-based stopping and disables validation.
Random training maps are sampled uniformly at episode resets when adaptive map
sampling is disabled. The split and validation remain available independently;
enabling adaptation only changes training map probabilities.

### Optional reinforcement vote rewards

G02 enables `reward.priveleged_vote_reward.enabled=true`; the global
default is false. It requires `adapt_size.enabled=true` and
`adapt_size.vote_protocol=latest_decision`. Set it to false to retain the old
reinforcement reward calculation and voting behavior, without remembered budgets
or shaping arrays. The two signed coefficients are
`reward.priveleged_vote_reward.withheld_reinforcement_vote_reward=-1.0` per eligible communication tick
and `reward.priveleged_vote_reward.unnecessary_reinforcement_call_reward=-200.0` per responsible YES
source on an unnecessary grant. No new observations or timing settings are
introduced. See [adaptive size](../01_reference/06_adaptive_swarm_size.md#optional-reinforcement-vote-shaping)
for decision-time attribution, observed-chain budgets and replacement calls.

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

For large banks, detailed diagnostics can be disabled independently of the
adaptive samplers:

```yaml
logging:
  adaptive_cell_diagnostics: false
  individual_map_diagnostics: false
```

Both switches default to `true` globally and are `false` in G03. The cell switch
disables dashboard metadata/history creation, per-update cell diagnostic rows,
individual cell plateau reports and per-map cell-support startup messages.
Cell-level metrics were already absent from W&B. The map switch suppresses
individual-map W&B series, full map-statistic history arrays, top-map history
entries and per-map evaluation statistics. Summary history and W&B retain
map-balanced mean, worst, best, median and lower-decile (p10) success, plus map
warmup/coverage and probability-range metrics. Training map success summaries
use only maps with completed outcomes; monitor `adaptive_maps/observed_fraction`.
The best/median/p10 summaries are added only in summary-only mode; G02 keeps its
existing detailed W&B keys and journal formats with the default switches on.

These are reporting switches: map/cell selection, support discovery, outcome
windows and complete checkpoint state remain active. They are outside sampler
identity checks, so changing reporting does not require discarding adaptive
state on resume. Existing historical files are preserved; disabling cell
diagnostics stops new dashboard data but does not erase earlier charts.
Evaluation heatmaps and replays retain their own `evaluation` switches.

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

Rewards now use signed individual amounts: positive rewards, negative subtracts,
and zero disables a term. No reward divides by active-agent count. The names
`all_decommissioned_penalty` and `no_movement_termination_penalty` are retained,
but their values are signed too. `time_presure_reward` is the public field name.

Global numeric defaults follow G02 with a four-agent reference: success +3000,
target delivery +125, and time pressure -0.75 per acting drone per step.
`useless_agent_at_success_reward=-3000` applies only to each useless active
agent at terminal chain success. In addition,
`useless_agent_at_success_all_agents_reward=-750` applies to every active drone
per useless active drone at that success, without dividing by population.
`useless_agent_at_success_all_agents_reward_cap=2000` limits the absolute amount
per recipient: each gets `-min(750 * useless_count, 2000)`. The min cap leaves a
useful drone at least +1000 from success and this adjustment, so completing still
pays even with many unused drones. For two useless drones each active drone gets
-1500 from this term; useless drones also receive their separate -3000 adjustment.
The cap is a finite nonnegative magnitude. Zero disables either
term independently. Contributors-only success remains removed. Local voxel, peer-sender and soft
cluster attribution remain. See [signed individual rewards](../01_reference/11_individual_rewards.md).

Optional `success_reward_as_hold_record` and `peer_informing_reward_enabled`
default to false. Each new hold record pays every qualifying holder
`success_reward / hold_chain_for`, without dividing by holder count.

For relay-filtered adaptive-size training,
`logging.required_agents_metrics_window_episodes: 500` controls separate
completed-episode windows for each exact target estimator count. W&B's
`req_agent_K/active_agents_at_EP_end_mean` tracks actual active population at
episode end across successes and failures. `req_agent_K/success_rate` uses the
same smaller window. `req_agent_K/total_eps_in_eps_window` measures the category's
fraction in the ordinary `training.num_envs` completed-episode window. The
curriculum's spare slot does not change `K`. See
[adaptive diagnostics](../01_reference/07_adaptive_swarm_diagnostics.md).

Adaptive-size actors default to `env.observe_base_vote_ratio: false` and
`env.observe_base_population: false`; only G02 enables both. While connected to the base network they
observe qualified base voters divided by the currently active population, and
active population divided by the global `DEPLOYED_AGENTS_OBS_REF = 10.0` in
`core/config.py`. A qualified voter has at least `adapt_size.vote_holding`
unspent YES credits delivered in the current base round. Successful deployment
clears the base credits, so the vote ratio starts at zero in the next round. Both inputs become
zero immediately off-network. Disable both observation flags to evaluate an
older checkpoint with its original adaptive observation layout; fresh training
is required to use the two added input channels.

`gap_reward_uses_change`, `gap_change_reward_per_meter` and delivery-gap vesting
are **legacy diagnostic tools only**, retained for historical debugging. Do not
consider or recommend them as training design/tuning settings or plateau
remedies. Normal training uses recurring gap shaping. Diagnostic vesting
requires gap-change reward and delivery gating; its step count defaults to 10.
G02 keeps peer informing and soft cluster credit enabled, uses
obstacle-geodesic geometry, and leaves hold records and vesting disabled.

G02 is the migrated level. Every other level's original reward block remains
untouched and unsupported, with a TODO and commented global-default replacement
underneath. Replace that original block before using the level. Old reward
names, old cost aliases and removed switches are rejected rather than converted.

`reinforcement_reward` defaults to +1 per qualified deployment voter,
`decommission_reward` to 0 per removed drone, and `rejected_call_reward` to -200
per qualified rejected voter. Removal events pay the lifetime that acted.

Reference rewards remain optional under `adapt_size`: defaults are disabled,
0.10 initial fraction and 200 transitions. G02 keeps them enabled at 0.20 over
500 transitions. Each caller receives its full decaying share independently,
without deducting it from the recruit or cascading reference rewards. See the
[adaptive swarm reference](../01_reference/06_adaptive_swarm_size.md).

First visual sighting is not rewarded in delivery-gated missions. At delivery,
`finder_reward` pays each eligible direct base deliverer and
`target_found_reward` pays every active drone once. Without delivery gating,
they use first discovery instead. See
[First visual sighting is not rewarded](../01_reference/02_environment_and_physics.md#first-visual-sighting-is-not-rewarded).

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
`rewards/chain_efficiency_reward` alongside the other reward components. It sums the paid
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

`RewardConfig` contains the commented source of truth. Defaults include
exploration +2 per credited voxel, collision -0.5, finder +100, target delivery
+125, success +3000, time pressure -0.75 per acting drone per step, useless agent
at success -3000, and idle termination 0. Gap progress pays +0.2 per credited
closed metre per step; a 40 m fully credited route pays +8 per contributor per
step. Coverage missions have no gap reward. Efficiency is disabled with a zero
coefficient; change mode is disabled with coefficient 3. Optional switches keep
their normal global defaults; G02 explicitly overrides its geometry and enabled
features. The full list and exact triggers are in
[signed individual rewards](../01_reference/11_individual_rewards.md).
