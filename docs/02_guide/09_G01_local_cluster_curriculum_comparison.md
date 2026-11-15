# G01 spawn-filter curriculum: local 4090 and cluster A100

`G01_static` is the executable local level. Its cluster counterpart is
`G01_CLU_static`, in the same `levels/G/` directory. Cluster ablations use
`<family>_CLU_<variant>.yaml` beside the local level, and concrete job-specific
New PPO variations belong in `cluster/manifests/chain-efficiency/fresh-sweep.json`; historical manifests are in `cluster/manifests/historical/**/*.json`. Do not create `levels/CLU/`.

Both levels use **exactly the current G03 reward block**, current G03 voting
and observation settings, and the single authored `B02c_static_map_noSpawn`.
G01's obsolete reward names and old commented migration block were replaced.
No building generation, held-out validation or random base is involved.

## Matched hardware comparison

Run locally from the repository in the normal Linux/WSL training environment:

```bash
uv run swarmecho-train level=G01_static
```

Submit from the Windows repository root:

```powershell
.\cluster\submit.ps1 -Run clu-g01
```

These use the same task, seed and PPO tuple **4000 envs / 100 rollout steps /
4 epochs / 20 minibatches**. Only hardware, source level/run labels and output
location differ. This is the direct same-configuration race. Start each from
scratch with the same source revision and no competing GPU training.

## Cluster configuration ablation

The focused larger-minibatch candidate is a separate named submission:

```powershell
.\cluster\submit.ps1 -Run clu-g01-wide-minibatch
```

It changes only `training.num_minibatches=5` (4000/100/4/5). It compares the
cluster configuration against the local baseline on the same curriculum, not
hardware alone. Its all-seven-active E1 stress probe used about 49 GiB, so
**keep the local 4090 on 20 minibatches**. Do not run the two cluster variants
concurrently on the same GPU. Neither manifest uses the tuning controller,
pruning, 30-minute startup timeouts or the old six-hour search budget.

Use `-DryRun` on either submission to inspect/package it without submitting.
No dependency-image rebuild is needed for these source/config changes.

## Names derived from PPO settings

Both cluster manifests use `"run_name_template": "clu-g01_{ppo}"`. The `{ppo}`
part compares the effective YAML settings plus manifest overrides with the
current `G03_legacy_random_buildings` values for `num_envs`, `num_steps`, `num_epochs`
and `num_minibatches`. Matching all four produces `baseline`. Otherwise only
differences appear, in that order, joined with underscores:

| Effective change from G03 | Configuration label |
|---|---|
| None (currently 4000/100/4/20) | `baseline` |
| 8000 environments | `numEnvs8k` |
| 200 rollout steps and 2 epochs | `numSteps200_numEpochs2` |
| 8000 environments and 5 minibatches | `numEnvs8k_numMinibatches5` |

The resulting cluster name is, for example,
`CLU_clu-g01_numEnvs8k_numMinibatches5-<UTC>-<id>`. Determined, the W&B group
and output/tracking folders use that unique submission name. The W&B display
name is `CLU_clu-g01_numEnvs8k_numMinibatches5`, without the automatic timestamp
or submission ID. W&B keeps separate runs using its own run IDs. The preset filename
does not determine the configuration label: changing a YAML parameter or a
manifest override changes the label on the next submission. Manifest overrides
take precedence over YAML values. Timestamps and unique IDs remain for repeats.

Local G01 defaults to `logging.run_name=G01_static_{ppo}`, with its existing
timestamp suffix for output folders and no timestamp in the W&B display name.
If supplying a custom local name, include `{ppo}` to retain
automatic labeling; a literal name remains literal. The naming logic changes
no training parameters. Cluster preparation reads only the four explicit
integer YAML scalars and imports no training dependencies.

## Spawn-filter curriculum, with ordinary continuous training

The only progression setting is:

```yaml
training:
  agent_count_curriculum:
    enabled: false
  adaptive_target_spawn:
    enabled: true
    curriculum_start_agents: 2
```

The existing adaptive spawn sampler accepts targets requiring at most its
current budget, starting with targets requiring one or two agents and progressing
2 -> 3 -> 4 -> 5 -> 6 -> 7. The recruitment cap is always
seven, and each episode starts with one active drone. Set
`curriculum_start_agents=null` to disable progressive filtering while retaining
ordinary adaptive cell sampling.

PPO, Adam state, recurrent state, ongoing episodes, run name and W&B run remain
ordinary continuous training. Budget changes affect subsequent episode resets.
There are no training stages, checkpoint handoffs, extra lifecycle status files
or per-budget step allowances. Classic population curricula retain their
original behavior and configuration; they cannot be stacked with this filter.

## Progression source and comparable metrics

The sampler follows `inspection_evaluation.early_exit_source`, using the same
success threshold, hold count and optional success-length requirement:

| Configured source | Current-budget progression metric |
|---|---|
| `inspection_evaluation` | `eval/spawn_curriculum_success_rate` |
| `validation` | `validation/spawn_curriculum_success_rate` |

Both G01 files select `inspection_evaluation.target_sampling: training`.
Ordinary `eval/success_rate` now measures targets drawn with training's live
adaptive cell probabilities and current virtual budget. Every inspection case
therefore belongs to the current budget; `eval/spawn_curriculum_success_rate`
matches that success fraction and `spawn_curriculum_eligible_episodes` is 500.
The policy can recruit all seven agents throughout. `eval/target_budget` records
the measured budget, while `spawn/target_budget` records the sampler setting on
the training step axis. Progression reports both measured and next budgets.

Validation alone retains independent uniform **full solvable task** sampling.
Its progression score, when selected as the curriculum source, uses the subset
of its full-task cases whose required-agent count is at most the virtual budget.
That subset may contain fewer than 500 cases; an empty subset cannot advance.
Neither measurement adds a separate GPU pass for progression scoring.

With the default inspection progression source, current-budget success at 95%
twice raises the spawn filter. **Full-level early stopping requires validation
success at 95% twice**, using validation's own cadence and hold counter. A high
score on the current training distribution cannot stop the run. Selecting
validation as the progression source makes it control both advancement and
stopping. To restore the former independent full-task inspection distribution
and inspection-based stopping, set `inspection_evaluation.target_sampling=independent`.

Both pipelines use 500 deterministically seeded episodes per pass and G03's PPO-update
cadence: inspection evaluation every 100 updates and validation every 50
updates. No environment-step interval overrides either schedule. Finalization
still performs the final measurements. The selected source's measurement
cadence also determines when the spawn curriculum can advance; its 95% threshold
and two-check hold remain unchanged. Training's adaptive cell probabilities
affect inspection, while validation remains independent.
Inspection targets can change as its sampler weights or budget change.
Validation uses fixed uniform full-task cases across the entire run, with
`validation.mode=uniform_full_task`. This is a same-map measurement, not a
held-out-map generalization claim. G03's existing validation defaults to
`held_out_maps` and keeps its original behavior.

### WSL smoke check

Run in the normal WSL/GPU environment:

```bash
uv run swarmecho-train level=G01_static \
  logging.run_name=G01_inspection_sampling_smoke logging.wandb_mode=disabled \
  training.num_envs=16 training.num_steps=8 training.num_minibatches=4 \
  training.num_epochs=1 training.total_timesteps=128 \
  inspection_evaluation.eval_parallel_envs=16 inspection_evaluation.eval_freq=1 \
  inspection_evaluation.eval_video=false \
  validation.num_episodes=16 validation.parallel_envs=16 validation.val_freq=1
```

This checks one PPO update and both evaluation paths, not learning quality.
The new run's `metrics.json` should report `eval_target_budget: 2` and
`eval_spawn_curriculum_eligible_episodes: 16`; training startup should still
report seven recruitment slots. `evaluation_history.jsonl` should identify
`target_sampling: training` and `stop_source: validation`. Validation's metadata
should retain the seven-agent stage and `uniform_full_task` mode. Repeat with
`inspection_evaluation.target_sampling=independent` to exercise the previous
inspection behavior. A standalone `swarmecho-evaluate checkpoint=<saved path>`
uses that checkpoint's sampler for a training-sampling run.

## Limits, persistence and results

Both levels use `training.total_timesteps=1000000000` (1B) as a safety ceiling
and `training.max_wall_seconds=3600` as the primary run budget. The clock starts
when training setup begins and includes initialization/JIT, PPO updates and
periodic evaluations/validation. It is checked between complete updates; an
in-flight update or evaluation can overrun the deadline. Training then saves
the actual completed update and produces final validation/checkpoint/media,
which can add time beyond the hour. There is no external process kill.
The normal 95% full-task success gate can still stop a run earlier.
Reaching either budget below that gate is a completed attempt, not a convergence
result. `metrics.json` and the W&B summary record `stop_reason` (`time_limit`,
`step_limit` or `early_exit`) and training wall seconds. Resuming starts a fresh
per-invocation time allowance; the step ceiling remains run-wide.
Other levels default to `max_wall_seconds=null` (no wall-time limit).
There is no additional curriculum budget setting.
Gamma remains 0.99 and GAE lambda 0.95. OOM auto-reduction is disabled so the
compared environment count cannot silently change.

There is **one W&B run and one output directory per configuration**. Target
budget and progression hold are stored in the existing
`adaptive_spawn_state.json` when an ordinary checkpoint is written. Budget
advancement itself creates no checkpoint. No `virtual_target_curriculum.json`
or `target_curriculum.json` is created. Existing non-curriculum sampler
checkpoints remain readable with the new optional setting disabled.

Intermediate evaluations produce scalar curves; final checkpoint evaluation
produces the existing final artifacts. Validation creates no media or best
checkpoint writes. The usual `config.yaml`, `training_history.jsonl`,
`evaluation_history.jsonl`, `validation/history.jsonl`, `metrics.json` and
checkpoint sampler state are sufficient for analysis and retrieval. Standalone
inspection restores that checkpoint's cell weights and target budget. Resuming
after changing inspection sampling preserves the budget and resets its
progression hold.
Cluster retrieval remains `.\cluster\fetch.ps1`.

Total elapsed time includes setup, training, evaluation and final artifacts.
The machines can have numerical differences despite identical settings/seeds;
this is one observed comparison, not a universal hardware or PPO optimum.

The historical no-curriculum study remains
`ablations/cluster/ppo-tuning/g01_capacity_review_20261029/2026-10-29_exp54837_G01_no-curriculum_capacity-and-PPO-review.md`.
Legacy `clu-g01-probe`, `clu-g01-search` and `clu-g01-explore` explicitly disable
the progressive spawn filter and are not the commands for this comparison.
