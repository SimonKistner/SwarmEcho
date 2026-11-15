# G03 CLU path auxiliary experiment

The feature defaults to `training.geo_probe_auxiliary.enabled: false`.
The existing `cluster/manifests/chain-efficiency/fresh-sweep.json` is unchanged. These configs train fresh
policies using the existing four-hour G03 CLU training budget, task settings,
held-out validation cases and checkpoint selection. Source changes are packaged
by the submitter; the dependency image does not need rebuilding for this feature.

## Submit manually

Run from the repository root in PowerShell. First submit one real sweep member:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\geo-auxiliary\geo_auxiliary_test.json
```

This is lambda=0.1, five snapshots, seed=456789. It has the full experiment budget,
so it remains a valid member of the comparison. Check its log for `Path auxiliary
ready`, the first rollout and optimizer completing, and update 1 progressing.
Informed samples can be zero early in training. The path auxiliary progress line
shows their count and effective lambda; lambda remains zero for the first ten
updates containing valid supervision. Let this trial continue as one of the 24.

Once it compiles and training starts, submit the other 23:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\geo-auxiliary\geo_auxiliary_remaining23.json
```

The latter explicitly excludes the exact test variation/seed. Alternatively,
submit all 24 together with the following command **instead of** either command
above; it does not deduplicate experiments already submitted:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\geo-auxiliary\geo_auxiliary_all24.json
```

Append `-DryRun` to prepare and inspect any batch without connecting or submitting.
Submission receipts and generated source contexts live in `cluster/.submissions/`.
The shared workspace capacity remains 24 slots, including other running jobs.
For the November 2 quota failures and the prepared six-combination retry batches,
see [the investigation and retry guide](GEO_AUXILIARY_RETRY_20261102.md).

## Variants

Each row uses seeds 456789 and 456790. Map-selection, validation and inspection
seeds keep their existing fixed settings.

| Policy path weight | Snapshots per agent slot per rollout |
| --- | --- |
| Disabled: policy-only control | No auxiliary head |
| 0.01 | 1, 5 |
| 0.03 | 1, 5 |
| 0.1 | 1, 2, 5 |
| 0.3 | 1, 5 |
| 1.0 | 1, 5 |

Twelve variants times two seeds give 24 independent runs. Larger snapshot counts
do not multiply the loss coefficient. Lambda=1 is a unit coefficient, not equal
gradient influence or equal task/path performance importance.

## What trains

The decoder reads updated GRUs from active agents who knew the target before
that actor step. K eligible steps are sampled distinctly, one from each ordered
time band of the current rollout. An auto-reset preserves each frame's own
episode/target identity. The differentiable PPO replay reconstructs these GRUs;
labels never become policy observations. Loss averages snapshots per agent,
agents per represented episode, then episodes **within each PPO minibatch**.
Long episodes may still appear in several updates.

The head learns at its independent learning rate using unweighted BCE. Only its
gradient into the actor is multiplied by the scheduled lambda. Both optimizers
clip separately. Supervised updates 1..10 train the head while PPO continues;
updates 11..110 linearly ramp policy pressure from 1% to 100% of lambda. Updates
without labels do not advance this counter. Labels reuse the offline probe's
125-cell base-relative canonical shortest-path implementation in
`src/swarmecho/analysis/path_targets.py`.

## Memory recovery

All three configs enable `training.oom_recovery.parameter=optimizer_microbatches`.
An optimizer OOM increases execution microbatches to a valid divisor of the
original 1,000-environment PPO minibatch: 1 -> 2 -> 4 -> 8 -> 20, etc. Gradients
accumulate with original minibatch denominators and one clipping/Adam step.
PPO diagnostics also use these chunks. Rollout OOM instead lowers execution
lanes; validation OOM lowers its existing parallel_envs. Full effective rollout
size, horizon, PPO epochs/minibatches, validation population, K and lambda stay
fixed. Per-lane RNG keys are retained. Floating-point kernel/reduction differences
can still occur when execution shapes change.

Recovery exits the failed worker and resumes a fresh process from the latest
complete checkpoint. Model, both Adam states and the supervised-update counter
are saved together. Execution settings and failure phase are journaled under
`oom_recovery.jsonl`; work after the last checkpoint is repeated. Retries have
no arbitrary count limit, but stop at the smallest execution shape or the
remaining time budget. This handles allocation failures; it cannot guarantee
that every hardware failure or an oversized single sequence will recover.
`oom_recovery_clock.json` retains the first worker's post-setup training clock:
failed training and subsequent retry setup consume the same four-hour budget.

## Diagnostics

Eight new W&B series use `geo_probe_auxiliary/`:

| Suffix | Practical interpretation |
| --- | --- |
| `train/path_similarity_pct` | Path-cell overlap at 80%; missing and extra cells reduce it. |
| `train/soft_path_similarity_pct` | Same idea using probabilities directly, without a cutoff. |
| `validation/path_similarity_pct` | Thresholded overlap on existing held-out validation episodes. |
| `validation/soft_path_similarity_pct` | Probability overlap on those same episodes. |
| `validation/exact_path_accuracy_pct` | Percentage of masks exactly correct at 80%. |
| `data/supervised_samples_per_update` | New valid selected snapshots, excluding PPO epoch repeats. |
| `gradient/aux_to_ppo_ratio` | Scheduled auxiliary gradient norm relative to PPO on the shared actor representation. |
| `gradient/cosine_similarity` | Whether those gradients align (+), are unrelated (0), or oppose (-). |

Train scores attach to the **existing completed-episode sliding window**: first
average each agent's selected snapshots, then agents in that episode, then eligible
episodes in the current window. Episodes without informed states stay in the
window but have no decoding score. No older samples replace them. Missing/empty
scores are unavailable rather than zero accuracy. Live-episode accumulators are
retired on completion; they are not an extra historical window.

Validation uses one deterministic random-priority reservoir sample per informed
agent from each normal validation episode. Scores average agents, then episodes;
there is no extra validation population, rollout, temporal smoothing or new probe
fitting. Sampling keys are independent of policy action RNG. Gradient diagnostics
reuse `training.diagnostics_every`, using one existing minibatch. They replace its
usual backward pass with two passes and therefore have a real compute cost.

Both similarity scores ignore correctly empty cells. Thresholded similarity is
`2 * correct path cells / (predicted path cells + reference path cells)`; soft
similarity replaces predicted membership with cell probability. Unthresholded
BCE remains the training objective, stored in local training history for debugging.
Configuration constants live in W&B config, not extra flat charts. Missing-route
counts and examples are local artifacts; the final missing-reference percentage
is a run-summary value. Task success, KL/clip metrics and throughput reuse their
existing charts. Compare task success against both steps and wall time across seeds.

## Local verification (WSL)

```bash
.venv/bin/python -m pytest ablations/path-probe/check_auxiliary.py tests/test_training.py ablations/path-probe/check_offline.py tests/test_validation.py -q
.venv/bin/python ablations/path-probe/runtime_smoke.py
```

Equivalence checks use full matrix precision; production keeps existing precision.
The runtime smoke uses two lanes and a tiny network on the real authored G01
environment, exercises chunked rollout/PPO and normal validation, disables W&B,
and removes its temporary artifacts. It is a local wiring check, not a cluster
benchmark or evidence of auxiliary benefit. Independent offline probes remain
manual post-training analyses.
