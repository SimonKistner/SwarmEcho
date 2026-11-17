# G03 automatic deployment / curriculum sweep

Prepared 2 November 2026. Manifest: `cluster/manifests/target-curriculum/g03_curriculum_auto7_21.json`.

Shared sweep settings are defaults in `G03_CLU_random_buildings.yaml`; the
manifest only varies curriculum mode, auxiliary lambda and seeds. A plain G03
CLU run now uses no curriculum, auxiliary lambda 0.03, K=5, automatic deployment,
differentiable base replay, independent target-cell eligibility preparation and
execution-microbatch OOM recovery. The manifest now uses `curriculum_v3` to keep
these runs separate from the earlier `curriculum_v2` sampler.

## Runs

| Target schedule | Policy-only, detached probe | Auxiliary 0.03 | Auxiliary 0.3 | Auxiliary 1 |
| --- | --- | --- | --- | --- |
| None: full budget within eligible cells | 456789 | 456789, 456790 | 456789, 456790 | 456789, 456790 |
| Forward: <=2, <=3, ... <=7 | 456789 | 456789, 456790 | 456789, 456790 | 456789, 456790 |
| Reverse: 7, >=6, ... >=2 | 456789 | 456789, 456790 | 456789, 456790 | 456789, 456790 |

Total: **21 independent runs, one GPU slot each**. Every run starts from fresh
weights. The existing 12-hour training budget, PPO 6000/50/3/6, map split,
validation cases, noise, rewards unrelated to voting, and decommission settings
are retained. Execution microbatch OOM recovery preserves the scientific batch.

## Behavior and parameters

- `training.target_curriculum.mode`: `none` (default), `forward`, `reverse`.
  The new controller is independent of adaptive target/map weighting. Both
  adaptive weighting toggles are false in this sweep. Positions are sampled
  uniformly within retained cells' allowed legal, certified target volume,
  rather than balancing required-agent categories. Stage-specific histograms
  shortlist cells and exclude unsupported maps before reset rejection sampling.
  Map retries reuse that filtered map CDF. Bounded exhaustion is an error,
  never an easier-target fallback.
- `training.target_spawn_eligibility.enabled=true`: run preparation independently
  of both success-based adaptive weighting toggles. `samples_per_map=800`
  collects accepted, uniformly proposed full-budget targets for each map;
  `minimum_cell_fraction=0.01` retains cells with at least eight observations.
  Save each cell's required-agent histogram and observed minimum/maximum. A
  curriculum shortlist uses actual observed groups, rather than assuming every
  group between the minimum and maximum exists. Proposal weights are retained
  spawnable volumes, not probe frequencies; each fresh position still passes
  exact certification and the current stage's requirement filter.
  These are empirical support masks: an unobserved small region can be excluded,
  so the training distribution is uniform within retained support, not all legal
  volume. Validation and independent inspection keep their full-task distributions.
  Increase `samples_per_map` if the measured support is insufficient; no stage
  silently admits easier targets.
- Bank runs share the preparation under the pool's `target_spawn_eligibility/`
  directory. An advisory lock reduces duplicate probe work where supported;
  publication uses a unique temporary filename per writer, so correctness does
  not depend on cross-node locking. Completed map entries survive interruptions.
  Identities include map geometry,
  full-budget feasibility settings, preparation seed/settings and relevant source.
  Training seeds, auxiliary lambda and curriculum order do not change the cache.
  Each run and checkpoint saves `target_spawn_eligibility.json`, freezing support
  and histograms for resume. An enabled adaptive target controller can reuse
  the same measurements and apply its success weights within that support.
- `training.target_curriculum.success_threshold=0.75`: validation advances one
  stage when the exact newest group's success is >=75%. Forward checks 2, 3,
  ...; reverse checks 7, 6, ... . Empty groups do not advance. Validation always
  samples the full task independently of training. Final full-task early stop
  remains 99%. The older target-budget gate also now checks the exact current
  group; G01 and G03 target-curriculum levels use validation as their source.
- `adapt_size.recruitment_enabled=false` (default is true): reserve all seven
  agents at reset, release automatically every five steps, and permanently
  retire decommissioned lifetimes. Queued initial launches prevent empty-swarm
  termination. There is no vote action/head, vote observation, vote transport,
  reinforcement/rejection reward, or replacement recruitment in this mode.
  `adapt_size.enabled=true`, `initial_agents=7`, and existing cleanup behavior
  remain in effect. Checkpoint contracts distinguish the changed action semantics.
- `training.base_memory_replay=differentiable` (default is `stored`): record
  the base cache's selected sender/write events and environment episode ends.
  PPO replay reconstructs the cache from current sender emissions, including
  reset clearing. The rollout's initial cache remains detached. This restores
  sender gradients through base delivery within the existing 50-step unroll;
  it does not extend backpropagation into preceding rollouts. Reception masks
  and write decisions remain the recorded physical communication events.
- `training.geo_probe_auxiliary.enabled=true` for every run, with fixed K=5.
  Coefficients are 0, 0.03, 0.3, 1. At zero the head trains on detached policy
  features with its own optimizer, giving the policy-only control validation
  similarity metrics without auxiliary policy gradients. Labels for head fitting
  come from training, not validation. Existing warmup/ramp and head settings stay.
- Validation additionally logs success/count per exact required-agent group for
  all variants. Existing `peer-informed-pct` / `base-informed-pct` metrics remain.

Training still requires chain holding and surplus cleanup. Validation still scores
chain holding; its progression threshold does not claim cleanup mastery.

## Commands

From the repository root in PowerShell:

```powershell
# Submit all 21 when ready.
.\cluster\submit.ps1 -Config cluster/manifests/target-curriculum/g03_curriculum_auto7_21.json
```

The earlier v2 dry run expanded and packaged 21 experiments successfully. The
v3 matrix retains those 21 combinations; all 21 effective configurations load
with independent eligibility enabled, and changed Python files compile. No v3
runs have been submitted, and already running jobs are unaffected by local
source changes. Runtime behavior and throughput have not been benchmarked.

Full runtime testing is left to the user. Focused contracts cover the sweep,
volume weights, group masks, map CDFs and frozen eligibility persistence:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q ablations/target-curriculum/check_sweep.py ablations/target-curriculum/check_eligibility.py
```

An earlier broader check completed 16 tests and encountered stale existing
expectations: `test_all24_partition_exactly_into_one_test_and_remaining23` expects
the former four-hour budget; `test_train_base_memory.py` loads a default level
with retired reward names. That check was stopped at three failures. The new
focused file was interrupted before completion; it is not reported as passing.

## One-map replay comparison, 3 November 2026

The six screenshot groups (none/forward/reverse, policy-only/auxiliary 0.3)
use the six completed experiments below, with training seed 456789 for every
group. Each run's
`checkpoints/best_val_checkpoint`, saved configuration and first frozen inspection
map have been fetched into its tracked `outputs/CLU_G03_auto7_*/` folder. The
reverse auxiliary runs use the completed `curriculum_v3_cachefix` replacements;
the earlier v3 auxiliary attempts 55412/55413 failed. The canceled reverse v2
attempts are excluded.

| Variant | Training seed | Experiment | Best update | Saved validation success |
| --- | ---: | ---: | ---: | ---: |
| None, policy-only | 456789 | 55374 | 2100 | 77.87% |
| None, auxiliary 0.3 | 456789 | 55377 | 2300 | 79.23% |
| Forward, policy-only | 456789 | 55381 | 1650 | 87.42% |
| Forward, auxiliary 0.3 | 456789 | 55384 | 1800 | 86.90% |
| Reverse, policy-only | 456789 | 55409 | 2200 | 36.14% |
| Reverse, auxiliary 0.3 | 456789 | 55418 | 2300 | 63.09% |

The scores are from the downloaded checkpoints' `recovery_ready.json`, measured
on the original 400-map validation task. They are not one-map replay results.

Run [the comparison script](../../../cluster/eval_g03_curriculum.py) from the
repository root in the existing WSL environment:

```bash
# Evaluate all six sequentially and generate heatmaps / required-agent replays.
uv run python cluster/eval_g03_curriculum.py
```

The script checks that all six targeted imports are complete and that
`random_eval_0000.yaml` has identical bytes across runs. The fetched map's SHA256
is `8b44e8feda9dddd3aad2c2fdf9d4f37f6c8ad0481fb8cc6c30becf32af635297`.
It explicitly selects `map_suite_map_id=random_eval_0000`, so both metrics and
replays use exactly one map. Each run evaluates 1,000 independent full-task
targets with inspection seed 456789, then generates one replay per observed
required-agent class. Selection includes failures, as in the existing
required-agent replay workflow.

The runner prepares a loader-compatible copy of each saved configuration under
that run's `.cluster/replay_eval/`, retaining its settings and the original
`config.yaml`. It disables W&B and labels the local evaluation
`curriculum_best_one_map`. Results appear under each run's `artifacts/eval/` and
can be opened with `uv run swarmecho-inspect root=outputs`. A failed eval stops
the sequence. Retrieval and the preparation-only dry run completed; GPU
evaluations are left for the user.

## Continue the six screenshot groups for another 12 hours

The nine proposed continuations are recorded below.


The local preparation completed on 3 November 2026. No jobs were submitted.
Nine jobs request nine GPU slots; allow for other jobs within the shared 24-slot
workspace limit. This helper is specific to the completed runs in the screenshot.

| Curriculum | Variant | Seed | Source experiment | Final checkpoint | Existing W&B ID |
| --- | --- | ---: | ---: | --- | --- |
| None | Policy-only | 456789 | 55374 | `ckpt_002327` | `9bnzl3ou` |
| None | Auxiliary 0.3 | 456789 | 55377 | `ckpt_002350` | `o5mcki0w` |
| None | Auxiliary 0.3 | 456790 | 55378 | `ckpt_002363` | `gtwgu9u2` |
| Forward | Policy-only | 456789 | 55381 | `ckpt_002086` | `68iyfcr9` |
| Forward | Auxiliary 0.3 | 456789 | 55384 | `ckpt_002105` | `6789l8na` |
| Forward | Auxiliary 0.3 | 456790 | 55385 | `ckpt_002123` | `qm968aaz` |
| Reverse | Policy-only | 456789 | 55409 | `ckpt_002434` | `jpwrj02i` |
| Reverse | Auxiliary 0.3 | 456789 | 55418 | `ckpt_002400` | `ov6vd9q8` |
| Reverse | Auxiliary 0.3 | 456790 | 55419 | `ckpt_002439` | `kchkdgan` |

New experiment/output names use, for example,
`CLU_G03_forward_aux_l0p3_seed456789_resume12h`. W&B groups remain
`CLU_G03_forward_aux_l0p3`, etc.; the existing W&B IDs continue the original
curves. `WANDB_RESUME=must` requires those existing runs. Renamed W&B groups
do not rename the old PFSS output folders, so checkpoint paths deliberately
retain `auto7` and the original `curriculum_v2/v3/cachefix` suffixes.

Each job copies its original submission's frozen source and driver, pins its
original dependency image and map pool, and sets explicit checkpoint loading
mode `resume`. The existing trainer restores model/head weights, optimizer
state, update counters, target curriculum and frozen validation state from
the final checkpoint. Episode rollouts start fresh; this is application
checkpoint continuation rather than a suspended process.

The new supervisor starts a fresh `training.max_training_seconds=43200`
budget. Setup precedes that budget; first JIT/evaluation consume it, while final
saving/media can extend job duration. Existing success stopping remains enabled.
The new output history segment receives the original five-billion-step limit
minus the completed steps, retaining the same lifetime update ceiling.

Preparation checks the saved completed-run inventories and original receipts.
It does not contact PFSS or execute GPU restoration. Generated manifests and
configs are under `cluster/.submissions/`; batch receipts use
`batch-g03-resume12h-*.json`. The helper refuses already registered/submitted
continuations and stops on a submission error. Inspect receipts and the WebUI
after a partial/ambiguous submission before retrying.

## Chain-efficiency and retirement sweep, 3 November 2026

The saved `cluster/manifests/chain-efficiency/fresh-sweep.json` describes 12 fresh training runs:

| Target curriculum | Chain-efficiency coefficient | Training seeds | Auxiliary lambda |
| --- | --- | --- | --- |
| Forward | 25 / 50 / 100 | 456789, 456790 for each coefficient | 0.3 |
| Reverse | 25 / 50 / 100 | 456789, 456790 for each coefficient | 0.3 |

`G03_CLU_random_buildings.yaml` enables chain efficiency at 50 and sets
`non_contributor_decommission_reward=3000`. This is the qualified surplus-agent
exit payment; the unconditional `decommission_reward` remains zero. The batch's
common override fixes auxiliary lambda at 0.3. Variations select forward/reverse
curriculum and override only the chain-efficiency reward coefficient for 25/100;
the 50 variants inherit the level default. Both seeds share one W&B group per
variation, giving six comparison groups. Job names include, for example,
`CLU_G03_forward_aux_l0p3_eff25_seed456789_decom3000_v1`.

From the repository root in PowerShell:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\chain-efficiency\fresh-sweep.json
```

Each job starts from fresh weights with the level's 86,400-second (24-hour)
training budget and five-billion-step ceiling. Training still requires chain holding
and surplus cleanup; validation/inspection still measure chain holding alone.
The batch requests 12 GPU slots within the shared 24-slot workspace limit.
Preparation creates local receipts; actual submission is left to the user.

## Replace the fresh efficiency sweep with branches

The replacement workflow targets only initial efficiency experiments
55450–55461, leaving running parents 55442/55443/55445/55446 intact. Each of
the four curriculum/seed parents supplies one latest published numbered
checkpoint shared across all three efficiency coefficients. The helper queries
live PFSS readiness markers, cumulative step history, saved curriculum state
and source dataset; it never substitutes the original pre-resume checkpoint
or a best-validation checkpoint. Checkpoint lookup requires no model/map download.


This is a specific replacement operation, not a generic retry command. The
helper verifies experiment names, project and ownership before changing state,
prepares the complete batch before stopping anything, and refuses previously
submitted branches. Inspect its `batch-g03-efficiency-branches-*.json` receipt
after an interrupted or partial operation before taking further action.

Branch names end in `decom3000_branch_v2`, with six new W&B groups containing
the paired seeds. Model/head weights and cumulative step provenance are loaded
from the running resumed parents. Matching uniform target curriculum stages
are now inherited in branch mode. Missing or incompatible curriculum settings
start a new curriculum; strict resume still rejects missing/incompatible state.
Optimizer and update-based auxiliary warmup/ramp state start fresh in branch mode.

The **24 hours include inherited training**. The level retains 86,400 seconds,
but each branch overrides `training.max_training_seconds` with 86,400 minus
the training time represented by its selected parent checkpoint. Legacy parent
checkpoints lack a cumulative training clock, so the helper combines finalized
ancestor training clocks with the W&B training clock at the selected update,
including checkpoint-publication work from the logged timestamp/readiness mtime.
Those inputs and the remaining allowance are saved in `summary.json` under
`branch_parent`. All three coefficients for a parent use the same allowance.
The W&B `perf/training_seconds` clock remains session-local; add the saved
parent offset when analyzing cumulative training time for these branches.
Cooperative stopping, evaluation, queueing, setup and final media mean exact
wall-clock finishing times can differ.

The replacement submission completed on 3 November 2026. All original sweep
experiments 55450–55461 were confirmed canceled; the four parents remained
running. The live checkpoint selection at 15:18 UTC was:

| Curriculum | Seed | Parent | Selected checkpoint | New jobs (efficiency 25 / 50 / 100) | Remaining training hours |
| --- | ---: | ---: | --- | --- | ---: |
| Forward | 456789 | 55442 | `ckpt_002520` | 55462 / 55464 / 55466 | 9.488 |
| Forward | 456790 | 55443 | `ckpt_002520` | 55463 / 55465 / 55467 | 9.607 |
| Reverse | 456789 | 55445 | `ckpt_002880` | 55468 / 55470 / 55472 | 9.473 |
| Reverse | 456790 | 55446 | `ckpt_002910` | 55469 / 55471 / 55473 | 9.502 |

All four selected saved curriculum groups were 7. Cumulative training clocks
were approximately 14.393–14.527 hours; job allowances complement those clocks
to 24 hours. Exact values, checkpoint-publication timing inputs, source hashes,
replacement target states and successful submission receipts are retained in
`cluster/.submissions/batch-g03-efficiency-branches-20261103T151732Z-82f8f782.json`
and each experiment's `summary.json`.

### Network failure after branch setup

At approximately 16:29 CET on 3 November 2026, all replacement experiments
55462–55473 were confirmed in `STATE_ERROR`. Experiments 55462–55472 restored
their model weights and matching curriculum group 7, then stalled and failed
during W&B initialization before their first rollout/update. Logs show W&B
connection errors and repeated DNS failures for its Sentry telemetry host
across multiple compute nodes. Experiment 55473 failed earlier, in the platform
GPU-slot limiter on cn43, because `login01.ai.tu-darmstadt.de` could not resolve.
The four parent experiments remained `STATE_RUNNING` with advancing updates.

The attempts were successfully submitted and partially initialized, but did
not perform branch training. A pause attempt found the first stalled job had
already terminated; a subsequent check confirmed all twelve had failed, so no
pause or further resubmission was performed. The network diagnosis and states
are recorded under `cluster/.submissions/g03_branch_network_*.json`.
Recovery requires working compute-node DNS/master connectivity before retrying
the saved branch inputs. Offline W&B alone cannot repair the platform limiter's
master lookup. Retrying the unchanged first branch batch is rejected by the
existing submission-identity guard; prepare explicit retry identities after
network recovery rather than rerunning the replacement operation blindly.

The user will execute all subsequent submissions. A single network retry of
55462 is prepared under a new identity, using the same source checkpoint and
remaining training allowance.


Check that this one job returns from W&B
initialization and completes its first optimizer update before retrying the
whole sweep. Existing jobs can retain working connections while new containers
fail name resolution; their continuing W&B reporting does not establish that
fresh container DNS works, nor do these logs alone establish the network fault's
root cause. No per-experiment DNS/network override was introduced in the failed
branch batch, which used the same dependency image as its working parent.

The original retry plan split the twelve runs into three selections:
`-Batch all12`, `-Batch test1`, or `-Batch remaining11`. The single job is
forward/auxiliary 0.3/efficiency 25/seed 456789; the remaining eleven exclude
exactly that job. All three selections share the same static per-run configs
under `cluster/manifests/chain-efficiency/checkpoint-retries/`, retaining the saved parent paths,
paired W&B groups and remaining time allowances. Choose all twelve together,
or submit one followed by the other eleven. The original wrapper submitted the selected runs directly;
the user executes them. No tests or dry runs were performed for this split.

A separate fresh-weight comparison removes checkpoint loading while preserving
the same forward curriculum, auxiliary lambda 0.3, efficiency reward 25, image
and pinned dataset. It has a new output identity and separate W&B group, with a
600-second training budget.


This config has not been submitted, tested or dry-run by the assistant. If it
also fails initializing W&B, checkpoint loading is not required to trigger that
failure. If it succeeds, node/timing differences still need consideration before
attributing the earlier failures to branch loading. A read-only W&B request
confirmed the locally configured API key authenticates; that does not independently
verify the PFSS copy. The observed `No route to host` warning targets Sentry
telemetry, so it is not by itself proof of an authentication failure or the
underlying W&B API connection problem.

The parent group is `CLU_G03_forward_aux_l0p3`; the corresponding branch/retry
group is `CLU_G03_decom3000_branch_v2_forward_aux_l0p3_eff25`, already separate
from its parent. The fresh comparison group is
`CLU_G03_scratch_netcheck_v1_forward_aux_l0p3_eff25`. Groups organize runs;
the W&B run ID determines continuation. Parent run IDs are not injected into
the branch/retry experiment environments.

An unchanged historical control is prepared from successful experiment 55384's
original submission receipt, copying the entire model context and experiment
config. Only the experiment/run name and W&B group are changed. It retains the
original source, driver, dependency image, overrides, level YAML and 12-hour
training budget, without packaging current repository changes.


The frozen copy is under
`cluster/.submissions/CLU_G03_forward_aux_l0p3_seed456789_exact55384_v1/`.
`control_origin.json` records the source receipt and identity-only edits.
The submit wrapper uses this existing copy directly and rejects duplicate
submissions after an experiment ID is recorded. The completed original run was
selected instead of its currently running resumed child, because the child
explicitly injects its parent's W&B ID and `WANDB_RESUME=must`; copying that
identity would write into the parent's W&B run. The historical control needs
only name/group edits and will obtain a separate W&B ID normally. It has not
been submitted or dry-run by the assistant.

### Retry identity collision after connectivity recovered

Trial 77013 / experiment 55486 reached the application and failed in
`ValidationRunner.__init__` with `Validation history exists; resume explicitly
or choose a new run.` It reused the output name ending in
`decom3000_branch_v2_dnsretry1`, whose previous failed attempt had already
created validation history. This failure happened before W&B initialization;
it is a confirmed retry-identity collision rather than a new W&B timeout.

The static retry configs now use `decom3000_branch_v3` output names and
`CLU_G03_decom3000_branch_v3_<variation>` W&B groups. Their selected parent
checkpoints, curriculum settings, scientific overrides and remaining training
allowances are unchanged. Membership of `all12`, `test1` and `remaining11`
remains static. The same commands select the updated files.

The submitter checks names already present in local tracking or in
successful/ambiguous submission receipts before preparing or submitting the
batch. Conflicts prompt `Press Enter to override, or Ctrl+C to abort`. The
static split wrapper checks its full selected set and collects approval for
all conflicts before submitting any member. Enter authorizes replacing local
tracking only after a new experiment ID is returned, archiving the previous
record under `.cluster/tracking_history/`. Existing tracking remains authoritative
during discovery, so older receipts cannot undo that replacement.

Override does not clear PFSS validation history. Another local `-receipt-...`
directory never makes the PFSS output name unique. For a fresh output directory,
select a new `run_postfix`; preserve the old histories rather than deleting
them or switching a branch to strict resume.
No tests, dry runs or cluster submissions were performed for this correction.

### Recover canceled reverse auxiliary seed 456790

Experiment 55446 (`CLU_G03_reverse_aux_l0p3_seed456790_resume12h`)
is canceled, ending at 2026-11-03 16:26:42 UTC. Its last W&B training row
reports update 3155, 946,500,000 cumulative steps and 13,271.67 seconds in
the resumed session. Together with 43,320.78 seconds from the original session,
this is approximately 15.72 training hours. Checkpoint recovery may roll back
to the latest complete published update, so the command calculates its exact
allowance at execution rather than using that last log row as a checkpoint.


The command checks conflicts before creating its temporary zero-GPU metadata
shell. It selects the latest complete numbered checkpoint of experiment
55446, requires optimizer state, verifies the reverse curriculum and original
W&B identity, and budgets `86400 - ancestor clocks - checkpoint session clock`.
An embedded clock is used when available; otherwise it requires the matching
W&B update and publication timestamp. It copies experiment 55446's frozen
source/image and scientific settings into a new `resume24h_recovery1` output
directory, loads in strict resume mode and continues W&B `kchkdgan`.
The original lifetime step limit is also retained. No other training jobs are
stopped, changed or submitted by this command.

The twelve static efficiency branches remain calibrated to 24 cumulative
training hours. Their inherited clocks are 51,814.53–52,297.12 seconds;
the corresponding new-session allowances are 34,102.88–34,585.47 seconds
(9.47–9.61 hours). These budgets stay attached to the pinned parent checkpoints;
later progress in the running parents does not change the branches' inherited
training. They target the same total training time, with different wall-clock
finish times after delays and queueing. Training stops at a loop boundary and
final artifact work can add elapsed time.

The assistant only read existing cluster/W&B status and authored the recovery
command; it did not run the recovery helper, create a shell, test, dry-run or
submit a job.

### Shared-parent continuation-seed experiment

At the user's request, the efficiency experiment now uses one shared trained
parent per curriculum and two continuation seeds. Seed 456789 supplies both
parents. The original independent-history design remains a different experiment;
these replacements isolate continuation randomness from a selected initial policy.

The six seed-456790 original branches were explicitly terminated and confirmed
`STATE_CANCELED`; all six retained seed-456789 branches were confirmed running:

| Curriculum | Efficiency | Retained experiment | Canceled original replacement |
| --- | ---: | ---: | ---: |
| Forward | 25 | 55489 | 55491 |
| Forward | 50 | 55492 | 55493 |
| Forward | 100 | 55494 | 55495 |
| Reverse | 25 | 55496 | 55497 |
| Reverse | 50 | 55498 | 55499 |
| Reverse | 100 | 55500 | 55501 |


The static six-job helper copies each retained experiment's frozen submission.
Forward replacements load `CLU_G03_forward_aux_l0p3_seed456789_resume12h/checkpoints/ckpt_002520`
and receive 34,155.4516851604 seconds of additional training. Reverse replacements
load `CLU_G03_reverse_aux_l0p3_seed456789_resume12h/checkpoints/ckpt_002880`
and receive 34,102.88276382256 seconds. Thus each pair has identical starting
weights, inherited training time, branch-mode optimizer reset, auxiliary
schedule, source/image, rewards, map pool and new-session budget; continuation
training seeds are 456789 and 456790. The total target remains 24 cumulative
training hours, with no wall-clock synchronization requirement.

Replacement outputs use `decom3000_branch_v3_shared789_v1` to avoid existing
PFSS history. Each new W&B run uses exactly the retained counterpart's group:
`CLU_G03_decom3000_branch_v3_<forward|reverse>_aux_l0p3_eff<25|50|100>`.
Canceled originals remain historical records and should be excluded from
paired comparisons. The old `remaining11` wrapper is not this six-job command.

The helper confirms local conflicts before preparing any member, verifies
retained/canceled identities and terminal replacement states, prepares all six
frozen contexts before submission, and records source pairing in each summary
and batch receipt. It contains no termination operation. Termination actions
are recorded in `cluster/.submissions/g03_shared_checkpoint_termination.json`.
Only the user will execute the new submission command; no tests, dry runs or
replacement submissions were performed by the assistant.

### 4 November: preemption diagnosis, five recoveries and eight extensions

The user subsequently authorized the assistant to submit. Live capacity was
11/24 slots occupied, leaving exactly thirteen. Completed original experiments
55438–55445 were extended by twelve additional training hours; failed
shared-parent experiments 55504–55508 resumed their own latest complete numbered
checkpoints, preserving continuation seed 456790 and their original W&B groups/IDs.

The five failures were scheduler preemption followed by a lifecycle error.
Determined requested preemption, forcibly killed allocations after they failed
to yield within one hour, and rescheduled the same trial. Its unchanged branch
command then encountered its own validation history and stopped with
`Validation history exists; resume explicitly or choose a new run.`

The new driver polls `core.preempt.should_preempt(auto_ack=False)`, requests a
checkpoint at a complete update, and acknowledges the signal after the worker
exits with a published checkpoint. It resumes the same output's latest complete
checkpoint after reallocation and deducts portable accumulated training seconds
from its job allowance. Every newly published checkpoint carries that clock.
Preemption checkpoints preserve optimizer/controller/validation state and skip
final evaluation/media. The existing scientific source/settings and image were
retained; only this lifecycle bridge and driver were patched into frozen contexts.

| Source | New experiment | Checkpoint | New allowance |
| ---: | ---: | --- | ---: |
| 55438 | 55511 | ckpt_004530 | 12h |
| 55439 | 55512 | ckpt_004497 | 12h |
| 55440 | 55513 | ckpt_004548 | 12h |
| 55441 | 55514 | ckpt_004184 | 12h |
| 55442 | 55515 | ckpt_004150 | 12h |
| 55443 | 55516 | ckpt_004247 | 12h |
| 55444 | 55517 | ckpt_004845 | 12h |
| 55445 | 55518 | ckpt_004654 | 12h |
| 55504 | 55519 | ckpt_000330 | 7.678h |
| 55505 | 55520 | ckpt_000360 | 7.566h |
| 55506 | 55521 | ckpt_000360 | 7.540h |
| 55507 | 55522 | ckpt_000300 | 7.654h |
| 55508 | 55523 | ckpt_000240 | 8.195h |

The five remaining allowances subtract checkpoint progress from their original
branch allowances, preserving the 24-hour cumulative training target. Legacy
periodic checkpoint timing uses the previous ordinary W&B update plus elapsed
time to checkpoint publication, covering checkpoint/validation work without
confusing a merged validation timestamp with an earlier training clock.

Source inventory: `cluster/.submissions/g03_nov04_resume_sources.json`.
Batch receipt: `cluster/.submissions/batch-g03-nov04-20261104T013739Z-817b0471.json`.
Startup restore evidence is recorded in `cluster/.submissions/g03_nov04_startup_audit.json`.
At 02:43 CET all thirteen new experiments were running, with the expected
checkpoint restores and W&B identities confirmed. The consolidated status and
budget endpoints are in `cluster/.submissions/g03_nov04_verified_status.json`;
first-update/optimizer-entry progress is in `cluster/.submissions/g03_nov04_first_progress.json`.
No tests, compile checks, dry runs or deliberately induced preemptions were run.

The user then requested matching 36-hour coverage for the separately recovered
reverse auxiliary seed 456790. Source experiment 55490 had completed at
02:44 CET, leaving one free workspace slot (23/24 occupied). Experiment 55524
was submitted as `CLU_G03_reverse_aux_l0p3_seed456790_resume36h_v1`, loading
the latest final `ckpt_004812` at 1,443,600,000 cumulative steps with optimizer
and reverse curriculum state intact. It adds 43,200 training seconds and
preserves W&B `kchkdgan`, group `CLU_G03_reverse_aux_l0p3`, the original frozen
source/image and the preemption lifecycle patch. This completes the nine-run
36-hour continuation cohort. Submission used the remaining slot; platform
startup confirmed 24 total allocated slots within the workspace limit.

### Selected next twelve-hour extension: seventeen runs

The user authorized another twelve training hours for the six original groups
and efficiency coefficients 25 and 50, retaining both seeds wherever present.
The original six groups comprise nine runs, including the separately recovered
reverse auxiliary seed 456790; the four efficiency groups comprise eight runs.
All coefficient-100 branches are excluded.

| Group family | Source experiments |
| --- | --- |
| None policy only | 55511 |
| None auxiliary 0.3 | 55512, 55513 |
| Forward policy only | 55514 |
| Forward auxiliary 0.3 | 55515, 55516 |
| Reverse policy only | 55517 |
| Reverse auxiliary 0.3 | 55518, 55524 |
| Forward efficiency 25 | 55489, 55503 |
| Forward efficiency 50 | 55492, 55519 |
| Reverse efficiency 25 | 55496, 55521 |
| Reverse efficiency 50 | 55498, 55522 |

Live W&B inspection confirmed the user's corrected efficiency group names:
`CLU_G03_<forward|reverse>_aux_l0p3_eff<25|50>_decom3000_branch_v3`.
These are deliberately taken from the live runs instead of the older frozen
submission summaries, which contain the previous group-name ordering. Original
group names remain `CLU_G03_<curriculum>_<policy_only|aux_l0p3>`.

The historical continuation helper checked run and seed coverage and read
the latest numbered checkpoints through a temporary zero-GPU shell, preserves
optimizer/controller state and frozen source/image, and continues existing
W&B IDs with the corrected groups. Completed sources receive 43,200 new seconds;
interrupted sources receive that extension plus unused prior training allowance.
The new output postfixes are `resume48h_v1` for originals and
`decom3000_branch_v3_resume36h_v1` for efficiency runs. Preemption handling and
portable allocation clocks are carried forward. Submission requires all source
experiments to be terminal. By default it requires seventeen free workspace
slots; `--allow-queue` explicitly permits submission with fewer immediately free
slots. The nine main runs always precede the eight efficiency branches.

Selection evidence is in `cluster/.submissions/g03_next12h_selection.json`.
Cluster access initially timed out and the user put the task on hold during
maintenance. After access returned on 4 November, all seventeen live W&B groups
were reverified and the latest complete checkpoints inventoried. The nine
originals ended with agent failures and allocation restoration errors reporting
`0 container snapshots`; their portable checkpoints retain approximately
3.3–3.7 hours of unfinished allowance. Their new budgets are approximately
15.3–15.7 hours including the additional twelve hours. All eight selected
efficiency sources completed and receive twelve new hours.

The user authorized submitting all seventeen to exercise scheduler queuing,
with main runs first. Preparation receipt:
`cluster/.submissions/batch-g03-promising-plus12h-20261104T181040Z-159a6b7e.json`.
All seventeen submissions succeeded: main experiments **55550–55558** were
created first, then efficiency experiments **55559–55566**. Observed free
workspace allowance before submission was 21 slots; physical GPU allocation
and scheduler queuing are separate from this workspace limit.
No tests, dry runs or compile checks were run.

### Four selected best-checkpoint replays, 4 November

Fetched one best-validation checkpoint for each requested continuation. Strict
resumes inherit the previous best score, so a new output may not publish a new
`best_val_checkpoint` until it improves. The retrieval follows strict-resume
ancestry, stops at branch boundaries and selects the highest saved validation
score. No latest-checkpoint substitution is made.

| Inspector label | Requested experiment | Best source experiment | Best update | Validation success |
| --- | ---: | ---: | ---: | ---: |
| G03 Forward - aux 0.3 - efficiency 25 | 55559 | 55489 | 1450 | 95.80% |
| G03 Reverse - aux 0.3 - efficiency 25 | 55521 | 55521 | 1700 | 89.68% |
| G03 Reverse - aux 0.3 | 55557 | 55518 | 5950 | 90.46% |
| G03 Forward - aux 0.3 | 55555 | 55555 | 5650 | 95.41% |

All four evaluation imports completed without conflicts/deferred files. Their
frozen `random_eval_0000` map is identical (SHA256
`8b44e8feda9dddd3aad2c2fdf9d4f37f6c8ad0481fb8cc6c30becf32af635297`).
The exact fetched identities and hashes are recorded in
`cluster/.fetch/g03_best_replays.json`. Local `.cluster/inspector.json` aliases
hide seed, resume and branch suffixes while retaining distinct internal IDs.

Run manually from the repository root in WSL:

```bash
uv run python cluster/eval_g03_best_replays.py
```

The runner now selects only the two forward checkpoints (requested experiments
55559 and 55555), evaluated sequentially using their saved configurations,
using 1,000 independent targets on the shared map with evaluation seed 456789.
Inspection success is overridden to `chain_held_and_cleand_up` (hold the chain
and decommission active non-contributors), with eval label
`chain&cleanUp`. The selected checkpoints and their historical validation
scores still use the original validation success condition.
It creates heatmaps and one replay for each observed required-agent class,
including failed episodes, with W&B disabled. Evaluations were not executed by
the assistant; no tests, dry runs or compile checks were run.

### Previous-step path probe feedback

`training.geo_probe_auxiliary.observe_prev_path_probe` defaults to `false`.
When enabled, an informed active drone receives the 125 sigmoid outputs of
its own online path head from the preceding action step. The head decodes the
post-GRU state into a canonical base-relative path grid. The first step of a
lifetime and steps before the first informed prediction receive zeros. The
actor input grows from 141 to 266 values in G03 CLU. The critic uses the 141
environment observations, and PPO reuses the feedback recorded in rollout.

Evaluation, validation and replay use the same head saved with the policy
and the same feedback timing. The validation route labels score the prediction.
Independently fitted offline heads remain available for diagnostic analysis.
Feedback changes the actor checkpoint shape, so the ablation starts from
fresh weights in all arms.

The comparison crosses forward and no target curriculum with feedback off and
on for seeds 456789 and 456790. Run names use `CLU_G03_aux_forward` or
`CLU_G03_aux_none`, then `_obsPathProb` when enabled, then the seed and `_v2`.
The v2 manifest starts all eight runs fresh and explicitly pins the 24-hour
training allowance for allocation recovery.
All eight runs share the original map pool,
fixed held-out validation cases, auxiliary weight 0.3, chain-efficiency
coefficient 25 and qualified decommission reward 3000. The G03 CLU defaults
allow up to 5 billion environment steps and 24 training hours, with the same
validation early-exit rule in every arm. Compare validation at matched steps
reached by both members of each pair.

Run the proof of concept from the repository root in WSL.

```bash
uv run pytest -q ablations/path-probe/check_feedback.py
```

Submit the eight v2 experiments from the repository root in PowerShell.

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\path-probe-feedback\g03_path_probe_feedback.json
```
