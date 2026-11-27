# Inspection evaluation and held-out validation

`inspection_evaluation` is the canonical name of the former `evaluation`
section. It retains the existing visual inspection runner, heatmaps, replays,
episode budgets, thresholds and checkpoint scheduling. Old `evaluation.*` config
and CLI keys remain readable, including `random_eval_envs` as an alias for
`inspection_maps`. The old early-exit source `evaluation` remains readable as
`inspection_evaluation`. New run snapshots use the canonical names.

Random-building runs automatically prepare a persistent inspection suite;
`inspection_maps` controls its size. Authored levels use their configured maps.
G02 still evaluates its five authored maps, with validation disabled.

## Inspection target distribution

```yaml
inspection_evaluation:
  target_sampling: training # independent (default) | training
  success_condition: null # Follow env.success_condition, or select an inspection-only goal.
  dynamic_max_geo_distance: null
  num_agents: null
```

`target_sampling: training` makes inspection resets use a snapshot of the live
adaptive target sampler's cell probabilities and current virtual target budget.
It samples new target positions with the same proposal weights and feasibility
rejection rules as training; it does not replay training's already sampled
targets. Only target eligibility changes with the virtual budget. All configured
recruitment slots remain available throughout evaluation, as in training.
`eval/success_rate` therefore measures the current training distribution, and
`eval/target_budget` identifies the budget used by that pass. The existing
`eval/spawn_curriculum_*` metrics still drive progression when inspection is the
configured curriculum source; every inspection episode is eligible in this mode.

This mode currently supports a single authored training map with adaptive target
spawning, including the G01 virtual curriculum. Population, task, map and
feasibility overrides that change that training distribution are rejected.
G03's held-out inspection maps cannot reuse training maps' cell weights and keep
independent sampling. `dynamic_max_geo_distance` only controls feasibility;
setting it to `true` alone does not select the live sampler or virtual budget.

Periodic metrics, final metrics, heatmaps and newly sampled replays use the
selected distribution. Robustness passes share a frozen sampler snapshot.
Standalone checkpoint inspection restores the saved adaptive sampler and virtual
budget; a missing sampler checkpoint is an error, with no uniform fallback.
Explicit replay targets and CSV-selected targets keep their original positions.

Validation never receives inspection's adaptive weights or virtual budget.
G01 uses `validation.mode: uniform_full_task` for independent uniform solvable
targets under the full seven-agent budget. With training-distribution inspection
and `early_exit_source: inspection_evaluation`, inspection continues to control
virtual-curriculum advancement, while **validation controls full-level early
stopping** using the existing success, length and hold thresholds. This requires
enabled `uniform_full_task` validation. With `early_exit_source: validation`,
validation retains control of both progression and stopping.

`target_sampling: independent` restores the previous inspection distribution and
stopping source. Existing level files default to this mode; both G01 level files
explicitly select `training`. When resuming across this option change, the saved
virtual budget is retained and its progression hold resets so measurements from
different distributions are not combined.

## Independent map counts and the split

`training.num_envs` counts parallel simulation lanes. `training.num_maps`
controls how many persistent random training buildings to load or create. Its
`null` default retains the original lane-count fallback. Authored levels use
their map list; an explicit count must agree with that list. Random-map datasets
may have more buildings than live lanes with either uniform or adaptive sampling.
Frozen selections survive OOM lane
reductions, checkpoint resume and agent-stage handoffs.

G03 has 4000 training lanes and a dataset of 3600 training plus 400 validation
maps. One seeded bank selection reserves disjoint map IDs. Its three inspection
maps are a fixed subset of those 400 validation maps. The bank manifest and
checkpointed selection retain the whole split. Generated-once runs also reject
validation geometry already present in training and persist their map manifest.

The dataset assignment is independent of adaptive map weighting. Random-map runs
always select a training map at episode reset: uniformly when
`training.adaptive_map_sampling.enabled: false`, or using the adaptive map CDF
when enabled. Disabling adaptation creates no map warmup or adaptive map outcome
windows and does not disable validation or adaptive target spawning. For backward
compatibility the split still lives at `adaptive_maps/selection.json` and is
copied into every random-map checkpoint as `adaptive_map_selection.json`, even
when no adaptive map sampler exists. Missing/corrupt frozen banks are not replaced.

## Configuration

```yaml
validation:
  enabled: true
  success_condition: null # Follow env.success_condition independently of inspection.
  num_maps: 400
  num_episodes: 10000
  seed: 456789
  fixed_targets: true
  parallel_envs: 500
  val_freq: 100
  val_on_finalization: true
  save_per_map_stats: true
  save_best_checkpoint: true
```

Validation currently supports target tasks on persistent random buildings with
fixed bases and no minimum-geodesic-separation constraint. Adaptive map sampling
may be enabled or disabled. It uses the training stage population and defaults to
the training success condition; inspection-only overrides do not change validation.

Set `validation.success_condition` and `inspection_evaluation.success_condition`
independently to select each runner's goal. G03 CLU uses a 50-step hold with
surplus cleanup for training, validation and inspection. Selecting `chain_held`
for an individual runner lets its episodes end with active surplus agents. The
selected goal determines success rates, episode lengths and any progression,
stopping or checkpoint decisions based on those rates. A `null` override follows
training. Changing the effective validation goal changes its frozen task identity
and requires a new run.

When validation controls curriculum progression, inspection may set
`dynamic_max_geo_distance: false` to sample the permitted spawn volume without
the stage's reachability filter. Inspection may also override `num_agents`.
When inspection itself controls progression, it must retain the stage filter
and use `num_agents: null`. Map exclusions and wall buffers apply in either case.

Each pass has `num_episodes` TOTAL episodes. Map IDs are sampled independently
and uniformly; neither adaptive map nor cell weights are used. Counts can differ
between maps. Unobserved maps have null estimates and are excluded from map-rate
percentiles. Case keys depend on the validation seed, stage and episode identity,
never the execution batch. With `fixed_targets: true`, accepted map-target pairs
are persisted and reused across passes, batch-size changes and resume. Different
agent-budget stages have their own certified cases and best-score history. With
the switch off, each update gets a new deterministic update-specific sample.

The runner uses one padded validation bank and bounded batches. Only the final
batch is padded, and its surplus results are discarded. Policy execution shares
the normal deterministic actor, communication, vote and reward code. No full
trajectories, heatmaps or replays are retained by validation.

## Scheduling, reporting and checkpoints

Validation runs on positive multiples of `val_freq` PPO updates.
`val_on_finalization` measures the normal ending/stopping model unless that
update has already been validated. Inspection retains its separate schedule.
`inspection_evaluation.early_exit_source: validation` makes validation alone
advance the existing success/length/hold criterion. The inspection source keeps
G02's previous behavior.

`validation/success_rate` is the pooled success fraction of the IID uniform
map-target episodes. `map_balanced_success_rate` separately averages observed
map rates equally. Summaries include a Wilson 95% episode-success interval,
observed-map coverage, worst/best/median/p10 map success, lengths, returns and
terminal active population. These estimates do not prove mastery of every map.

With `chain_held_and_cleand_up` validation, `validation/chain_hold_rate` is the
fraction of all episodes that reached `hold_chain_for` consecutive connected
steps before termination. `first_hold_surplus_episodes` counts episodes with
active agents outside the selected relay at the first qualifying hold.
`first_hold_surplus_successes` counts how many of those episodes subsequently
achieved full mission success, including after a broken and re-established relay.
`first_hold_surplus_success_rate` divides those counts and is omitted for an
empty denominator. The same summaries appear under
`validation/required_agents_<n>/` when target requirement estimates are available.

With adaptive swarm size enabled, `validation/useless_agents_at_success` and
`eval/useless_agents_at_success` report the mean number of active agents outside
the success-contributing chain, over successful episodes only, matching training's
`adapt_size/useless_agents_at_success`. The metric is omitted when no episode
succeeds. Inspection map suites and robustness runs pool successful episodes
instead of averaging their per-map or per-run means equally.

A strict success-rate improvement replaces `checkpoints/best_val_checkpoint/`.
The existing checkpoint publisher saves actor, critic, value normalizer and
adaptive state together, plus validation cases, identity, best score and hold
count. There is no actor/critic mixing. Ordinary resume still uses the selected
checkpoint. Best-score tracking survives resume; a newer completed best in the
same run is retained when rewinding to an older ordinary checkpoint.

Files under each run/stage's `validation/`:

- `metadata.json`: task identity, map identities and settings.
- `cases.npz`: fixed map IDs, keys and accepted target positions.
- `history.jsonl`: summaries and resume rollback markers.
- `results_uXXXXXX.json`: one pass summary and, when enabled, a compact per-map table.

Only summaries go to W&B under `validation/*`. Local per-map results are controlled
by `validation.save_per_map_stats`, independently of training diagnostic switches.
The inspector lists validation passes in the run's artifact menu. Its statistics
view shows a success trend, map-rate distribution and searchable/sortable table,
without loading geometry or replay data.

`validation/peer-informed-pct` and `validation/base-informed-pct` show how newly
informed agents learned the target, pooled over this validation pass's episodes
(excluding padding lanes). Direct discoveries are included in the denominator;
the remaining percentage represents them. No training-window smoothing applies.
An agent counts once per lifetime, with simultaneous sources attributed to direct
sight, then base, then peer. A pass without informed agents omits both metrics.
Raw `[all, peer, base]` recipient counts are retained in local result files.

CPU checks cover real config loading, aliases, count independence, disjoint map
selection and host statistics. JAX/GPU execution, throughput and OOM continuation
still require verification in the project's WSL environment.
