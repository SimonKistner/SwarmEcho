# Cluster G01 tuning

Experiment 54837 completed the broad exploration but produced no 95%-converged
configuration. Its unequal time budgets must not be treated as a completed
convergence comparison. The downloaded study's detailed review and proposed
two-run follow-up are in
`cluster/docs/g01_capacity_review_20261029/2026-10-29_exp54837_G01_no-curriculum_capacity-and-PPO-review.md`.
The historical exploration commands below remain documented for reproducibility.

For the current curriculum comparison, use
[G01 local/cluster curriculum comparison](09_G01_local_cluster_curriculum_comparison.md).
`G01_static` and `G01_CLU_static` now enable the same virtual target curriculum,
with all seven recruitment slots available throughout. The legacy
probe/search/explore workers below explicitly disable it; do not use those
jobs for the curriculum race.

## Single six-hour exploration job

The broad capacity-first experiment is submitted from `cluster/` with:

```powershell
.\submit.ps1 -Run clu-g01-explore
```

This job needs no previous probe. Its original six-hour deadline covers both
stages and persists across controller restarts. It spends up to one hour on
capacity discovery and uses the remaining time for fresh learning trials.
Capacity candidates get up to eight minutes for startup/compilation and two
complete PPO updates; their W&B logging is disabled. Initial probes span env
counts up to 64,000, horizons from 25 to 1,400 and several minibatch sizes.
Successful fits expand toward 128,000 envs if time permits; actual OOMs bracket
smaller candidates, with midpoint refinement when a lower fit is known.
Timeouts never imply OOM. Near-limit fits also trigger lower operating points
to verify room for production training.

The learning domain includes 500–128,000 envs, horizons 25/50/100/200/400/700/1400,
epochs 1/2/4/8/16/32 and minibatches 5/10/20/40/80/160/320. Divisibility and
measured capacity constrain the valid combinations. Learning candidates require
10% headroom within the same 85% JAX allocator budget. Shapes with no successful
capacity evidence are not presented as established fits. Exact OOM boundaries
for every shape are not guaranteed within the stage budget.

The original 4090 tuple is one learning reference. Six startup learning trials
use at most 30 minutes each; subsequent adaptive trials may use up to 90 minutes.
Up to sixteen learning configurations are attempted within the shared deadline,
with fresh initialization, common screening seed, unchanged gamma/lambda and the
100M-step/confirmed-95% stop. Fewer trials may finish, and a budget-limited run
is not a convergence result. This version uses continuous trials and fresh
bounded retries, not paused/promoted weights-only continuations. The original
two-job probe/search workflow below remains available for smaller experiments.

An episode still ends by step 700. A 1,400-step rollout continues across resets,
collecting several episode fragments/episodes; terminal masks prevent GAE and
recurrent state from carrying credit across episode boundaries. Horizons beyond
700 are optional throughput/sequence-length experiments, not longer episodes.
Batch size (`num_envs × num_steps`) varies freely. Matched-batch comparisons are
controls rather than a requirement of the search.

## Failure handling and restart recovery

The CPU supervisor owns study state; every GPU attempt runs in its own process
group and has a watchdog. Failed CUDA contexts are killed and reaped before
launching another attempt. Retries never silently change the requested tuple.

| Event | Action |
| --- | --- |
| OOM | Record failure/last memory measurements; terminate context; test another configuration |
| Timeout | Preserve completed updates and curves; keep separate from memory exhaustion |
| NaN/Inf in core PPO losses | Record numerical failure and skip this configuration |
| Unknown worker error/crash | Retry once from fresh weights within the original trial deadline; then skip |
| Three consecutive unresolved worker errors | Stop with partial results to avoid repeating a shared failure |
| Missing/unavailable W&B | Use local JSON curves; a W&B worker error gets one fresh attempt without W&B |
| Dashboard reporting failure/stall | Best-effort bounded background reporter; GPU supervision continues |
| Setup/hardware mismatch, unusable storage or worker that cannot be reaped | Stop GPU work and retain receipts where storage permits |
| Controller/allocation interruption | Up to two Determined restarts; recover accepted results and incomplete-attempt evidence |

Atomic, flushed JSON snapshots and an append-only flushed `events.jsonl` journal
retain attempt boundaries and study decisions. Results are stored before
dashboard reporting. On restart the controller checks settings/source/GPU class,
retains completed trials, restores sampling state and its original absolute
deadline, and classifies any unaccepted in-flight attempt as interrupted.
Interrupted candidates may be selected again from fresh weights; this is study
recovery, not exact optimizer/RNG training continuation. A file lock prevents
two controllers from owning the same study concurrently. Finished studies are
not re-executed just because final reporting needs a retry.

Every retry receives a new `_a00`, `_a01` output suffix. Retry overhead counts
toward the original trial budget and time-to-threshold. `time_to_95_seconds`
observations stay separate from overall process status; an OOM/error cannot
win the recommended-run ranking. Successfully completed qualifying trials alone
produce `best_run.json`. Unknown process deaths are not guessed to be OOMs.
Useful capacity evidence from a timeout after two complete optimizer updates
can qualify if memory statistics also establish the configured headroom.

Permanent PFSS failure or loss of the entire storage service cannot be made
durable by this runner. It stops rather than continuing unrecorded GPU work,
and attempts to print its last summary to the job log. No failure path promises
unlimited recovery or a guaranteed converged model within six hours.

## Execute phases 1 and 2

From PowerShell in `Q:\_0_Projects\000_SwarmEcho\SwarmEcho\cluster`:

```powershell
.\submit.ps1 -Run clu-g01-probe
```

Wait until that experiment finishes successfully, then:

```powershell
.\submit.ps1 -Run clu-g01-search
```

The search reads the latest successful probe directly from PFSS. There is no
results copy or manual candidate editing between phases. Run these in sequence,
not concurrently: submitting search while the probe is running may find no
probe or select an older successful one. The printed Determined URLs show
progress. No jobs are submitted by adding these files or running `-DryRun`.

Search uses online W&B. The existing `WANDB_API_KEY` environment variable or
owner-only PFSS `.wandb_api_key` file must already be available as documented in
[cluster submission](../../cluster/README.md#persistence-and-wb). Credentials
are never included in model snapshots. `setup-wandb.ps1` securely installs an
existing local key through a separate SSH connection when needed; the probe
needs no W&B key.
Both jobs use the current dependency image; no rebuild or new package install
is needed for the controller.

Preparation-only commands:

```powershell
.\submit.ps1 -Run clu-g01-probe -DryRun
.\submit.ps1 -Run clu-g01-search -DryRun
```

Dry runs validate packaging, run structure and controller settings using system
Python. They do not initialize JAX, load/validate the full level, exercise the
GPU worker or verify remote credentials and hardware. Repository rules prohibit
using the local WSL-created `.venv` for runtime verification.

## Phase 1: bounded capacity probe

The controller has a 900-second budget including hardware inspection and child
startup/compilation. It reserves ten seconds for cleanup and summaries. Worker
process groups have independent watchdog deadlines; each candidate gets at
most 180 seconds. Container startup, queueing and Determined resource teardown
are outside this controller clock.

The job records GPU model/UUID, driver, visible devices/MIG listing, CPU/RAM and
cgroup limits. Each candidate verifies exactly one JAX-visible GPU. It starts
with `(4000, 100, 4, 20)` for `(envs, steps, epochs, minibatches)`, then probes
larger env counts, 200/400-step horizons and different minibatch counts. OOMs
trigger smaller candidates; timeouts remain incomplete timing evidence. The controller stops at its deadline
with partial results rather than promising an exact OOM boundary.

Every candidate runs three complete rollout/advantage/optimizer updates in a
fresh subprocess. The first is cold; subsequent updates provide steady timing.
There are no evaluations, checkpoints, media or W&B sessions in this phase.
All seven drones are active from reset and decommissioning is suppressed for
the probe, so inactive slots cannot understate memory pressure. Its SPS is a
capacity diagnostic; phase 2 measures the actual recruitment task's learning.

Both phases explicitly use JAX preallocation with an 85% memory fraction. The
probe records JAX peak bytes and allocator limit separately from `nvidia-smi`
reservation/utilization and host RSS. A search-eligible fit must complete all
three updates (or two completed updates before a timeout) with at least 15% of the allocator limit still unused at peak.
This leaves additional space beyond the allocator's physical VRAM reservation.
Unknown memory statistics are reported but cannot establish a safe envelope.

Successful results are published atomically through
`<PFSS project>/tuning/G01_CLU_static/latest_probe.json`. A failed probe does not
replace a previous successful pointer. Candidate-local errors have bounded
retries; shared setup failures and repeated unresolved errors stop the study.

## Phase 2: adaptive joint learning search

Defaults in `cluster/archive/runs/clu-g01-search.json`:

| Budget | Default |
| --- | --- |
| Fresh configurations | At most 16 |
| Per-configuration wall clock | At most 1 hour, including startup/finalization |
| Overall controller wall clock | At most 6 hours |
| Environment steps per configuration | At most 100M, rounded down to complete PPO updates |
| Success stop | 95% on two consecutive same-map evaluations |
| Evaluation population | 500 parallel episodes, independent of training envs |
| Evaluation cadence | After crossing each 2M-step boundary |

Six hours is a hard total budget, so the job may finish fewer than 16 trials.
Edit the JSON budgets before submitting if desired. Only one trial runs on the
allocated GPU at a time; there is no competing GPU workload from this search.
Every configuration starts from fresh weights with the same screening seed,
gets a unique output folder and W&B run, and shares its parent job's W&B group.
Gamma remains 0.99 and GAE lambda remains 0.95, with runtime guards against
changing them.

The finite starting domain is:

| Parameter | Candidate values |
| --- | --- |
| num_envs | 1000, 2000, 4000, 8000, 16000, 32000 |
| num_steps | 50, 100, 200, 400 |
| num_epochs | 2, 4, 6, 8 |
| num_minibatches | 5, 10, 20, 40 |

Only divisible recurrent minibatches are allowed. The measured envelope bounds
env count, horizon, total rollout size and transitions per minibatch. This is a
conservative inference for unmeasured shapes, not proof of their feasibility;
unexpected OOMs are recorded and exclude dominated larger candidates. Source
fingerprints, GPU class/driver, worker runtime and allocator limits must match
the probe, including the effective memory available to a MIG slice. A mismatch
fails early and requires re-probing rather than guessing at capacity.

Search starts with the original baseline if it has verified headroom, otherwise
the nearest safe alternative. Five startup trials include space-filling
alternatives. Later suggestions use a compact joint Parzen/KDE density ratio
over log envs, horizon, epochs and recurrent sequences per minibatch, with 25%
random exploration. This is a dependency-free TPE-style implementation, not an
Optuna integration. All four settings remain variable throughout the search.

The relevant couplings at the baseline are:

| Quantity | Formula | Baseline |
| --- | --- | --- |
| Environment transitions per PPO update | envs × steps | 400,000 |
| Recurrent sequences per minibatch | envs / minibatches | 200 |
| Environment transitions per minibatch | envs × steps / minibatches | 20,000 |
| Optimizer steps per PPO update | epochs × minibatches | 80 |

Pruning checks success-versus-wall-time curves at 10, 20 and 40 minutes. It
requires at least ten updates, 1,000 completed episodes, two evaluations and
three comparable prior curves. A candidate is pruned only when materially
below the lower quartile of peers that already show meaningful progress. The
first baseline trial is protected. Zero success at the first evaluation alone
is never a pruning criterion. Pruned runs close W&B cooperatively at the next
training boundary, with process-group cleanup if they cannot stop promptly.

Converged trials rank by time to the second qualifying 95% evaluation, including
cold startup and scalar evaluation time. Final media time is recorded in total
trial time but not added to time-to-threshold. Non-converging curves are useful
for suggestions/pruning and are never presented as achieved convergence. No
weights-only checkpoint promotion is used. All screening seeds are shared;
repeat finalists across seeds in phase 3 before claiming a reliable winner.

## Level and artifacts

`G01_CLU_static` uses G01's single `B02c_static_map_noSpawn` authored map with
G03's current signed rewards, observations, voting and recurrent architecture.
Ordinary `clu-g01` training uses the same virtual target curriculum as local `G01_static`.
Only these legacy tuning workers override the curriculum off and start with
one drone that can recruit up to seven from the first update. Geometry stays
static while targets retain adaptive sampling. There is no random-building
generation, adaptive map sampling or held-out validation.

Intermediate evaluation produces scalar curves only. Final checkpoint evaluation
creates heatmap data/manifests and representative success/failure replay media
through the existing artifact pipeline. Intermediate checkpoints are also
suppressed. Pruned or hard-timed-out runs retain their curves and receipts;
they do not perform expensive final evaluation/media generation. Artifact
creation is best effort in the existing pipeline and is subject to the trial's
hard deadline. The standalone `clu-g01` run launches only the updated baseline
with the same final-only media settings and online W&B.

Evaluation step boundaries are independent of PPO update count. Large batches
can overshoot a boundary; the actual step is logged. Adaptive-target smoothing
still uses update counts as in G03, and its configured value is held fixed.
The updated task/rewards mean the old one-hour G01 result is context, not a
measured baseline for this level version.

## Results and retrieval

For each parent job, PFSS contains:

```text
outputs/<printed CLU run name>/
  tuning_run.json
  metrics.json                    # Written on controller completion
  tuning/
    hardware.json
    summary.json                  # Incrementally updated after each trial
    results.csv
    events.jsonl
    best_run.json                 # Search only, if a threshold was observed
    trial_000/
      trial.json
      result.json
      attempt_00/
        trial.json
        progress.json
        progress.jsonl
        telemetry.jsonl
        result.json
        worker.log
outputs/<parent>_t000_n..._h..._e..._m..._a00/
  config.yaml
  training_history.jsonl
  evaluation_history.jsonl
  checkpoints/                    # Normally completed search trials only
  artifacts/
```

The WebUI receives scalar tuning progress in addition to the final completion
metric. W&B uses environment steps as the ordinary x-axis, matching normal
training. `perf/wall_seconds` remains available for optional timing comparisons;
the tuning worker does not redefine the train/eval/PPO/critic axes. This change
does not rewrite existing runs' metric definitions or saved dashboard panels.
JSON/CSV receipts persist independently of W&B.

Use existing incremental retrieval without manually finding remote files:

```powershell
.\fetch.ps1
```

This imports pending tracked jobs and their candidate stages. To select only
one job use `-Choose` or `-Run <printed CLU run name or experiment ID>`. Parent
tuning summaries are included even while a job is still running. Inspect the
parent's local `tuning/results.csv` and `tuning/summary.json`. A generated
`best_run.json` is a reusable normal run definition for later confirmation;
it does not launch phase 3 automatically.

Before production G03 training, confirm finalists across seeds and repeat a
short capacity/throughput check on G03's actual map-bank/evaluation workload.
Single-map G01 results cannot establish G03's memory envelope or convergence.

## Method references

- [Parzen-based hyperparameter search](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)
- [Model-based search with adaptive budgets](https://proceedings.mlr.press/v80/falkner18a.html)
- [JAX GPU allocation behavior](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
- [W&B custom metric axes](https://docs.wandb.ai/ref/python/experiments/run/)
