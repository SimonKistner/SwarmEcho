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
may be enabled or disabled. It uses the training task and stage population; inspection-only overrides
do not change validation.

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

CPU checks cover real config loading, aliases, count independence, disjoint map
selection and host statistics. JAX/GPU execution, throughput and OOM continuation
still require verification in the project's WSL environment.
