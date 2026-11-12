> Historical workflow notes. For new runs use [the current workflow](../README.md) and `cluster/config.json`. Wave scripts and manifests in this folder reproduce past experiments; they create fresh jobs when invoked.

# Cluster submission

For the complete command reference, see [cluster/docs/COMMANDS.md](../docs/COMMANDS.md).
Its PowerShell examples consistently start at the repository root.

## Parameter tuning reports

See the [report index](../docs/README.md). The [four-wave G03 PPO refinement report](../docs/g03_refinement_analysis_20261031/G03_refinement_report.html) is the **current source of truth**, updated **1 November 2026**; earlier reports are marked historical.

Run from PowerShell in `Q:\_0_Projects\000_SwarmEcho\SwarmEcho\cluster`.
`submit.ps1` uses standard-library Python locally (3.7+), without importing
training code or using the project `.venv`. It selects the first system `python`
on PATH; override it with `-Python 'C:\path\to\python.exe'` if needed.

## Capacity limit: 24 slots

**Planning constraint, confirmed on 2026-10-31: workspace 129 has a shared
allocation limit of 24 GPU slots.** The trial logs explicitly report
`Workspace id: 129 has slots limit: 24`. This is a workspace allocation ceiling,
not the cluster's total physical capacity or 24 slots reserved for this user.
Count every occupied slot in the workspace, including other users' jobs and
jobs outside the current experiment batch. With the current
`slots_per_trial: 1`, at most 24 one-slot trials fit at once; multi-slot trials
consume their requested slot count. Zero-GPU result-transfer shells consume no
GPU slots.

Design every future submission wave around the free capacity below this limit.
Trials can receive experiment IDs and then be killed by the allocation limiter
before training starts; do not assume excess jobs will queue.
A 32-run comparison needs staged dispatch. A 24-then-8 split is only an upper
bound when the workspace is otherwise empty; use the actually free workspace
capacity in practice. If the initial pool-building training run is still
active, it occupies one slot and leaves at most 23 slots for the first remainder batch, further
reduced by other workspace allocations. Wait for confirmed workspace capacity
before dispatching more jobs, allowing headroom for concurrent allocations.

Track rejected or never-started submissions separately from OOMs and learning
outcomes. Check successful experiment receipts before retrying a submission so
an uncertain response does not create a duplicate. Existing bulk-loop wrappers
do not enforce this limit or automatically replenish slots.
Sequential calls to `det experiment create` still launch overlapping trials;
they do not serialize training or make an oversized wave safe.

## Settings and runs (step 1)

`settings.json` holds the master, CLI version, workspace/project labels, project
ID 367, `42_Compute`, one slot, PFSS mount and persistent directory. The project
ID selects the destination; workspace/project names are descriptive labels.
If moving projects, update the ID as well as the labels.

- `runs/g03-smoke.json`: small generated-map G03 run, W&B disabled.
- `runs/g03-full.json`: current full G03 level, no training overrides, W&B online.
- `runs/clu-g01.json`: matched local/cluster G01 virtual target curriculum,
  seven recruitment slots throughout, PPO 4000/100/4/20, full-task evaluation
  and uniform validation, plus a sampler-budget success score from the selected
  early-exit source; W&B online and final-only replay rendering. A one-hour
  budget includes setup/JIT and periodic evaluation, with a 1B-step ceiling;
  final saving/media follow after a complete update.
- `runs/clu-g01-wide-minibatch.json`: the same curriculum with five recurrent
  minibatches instead of twenty; the same one-hour budget and 1B-step ceiling,
  without search/pruning or external trial termination.
  See [the comparison guide](../../docs/02_guide/09_G01_local_cluster_curriculum_comparison.md).
- `runs/clu-g01-probe.json`: automated capacity probe, at most 15 controller
  minutes, W&B disabled.
- `runs/clu-g01-search.json`: joint search of all four PPO settings using the
  latest successful probe, up to 16 fresh W&B runs, 1 hour per run and 6 hours
  overall. See [cluster tuning](../../docs/02_guide/08_cluster_G01_tuning.md) for
  execution, result retrieval, budgets and failure handling.
- `runs/clu-g01-explore.json`: single six-hour job: up to one hour of broad
  capacity discovery followed by learning trials across the measured feasible
  region. Includes expected OOMs, bounded recovery and durable partial results.

Cluster ablation levels live beside their local family in
`src/swarmecho/curriculum_config/levels/G/G01_CLU_static.yaml`; use
`<family>_CLU_<variant>.yaml`, not a separate `levels/CLU/` directory. The
probe/search/explore jobs are the older no-curriculum workflow and explicitly
override the current level's curriculum off.

To add a run, copy a JSON file and change its level/overrides. Unspecified values
come from the level source YAML. The wrapper owns `level`/`levels`,
`logging.run_name`, `logging.log_dir`, `logging.wandb_mode`, `logging.wandb_group`
and `logging.use_timestamp_postfix`; do not also put them in overrides.
Don't put credentials in configs. Each submission receives a unique `CLU_...`
run name. PFSS stage folders and imported stage folders use that same name
with the usual stage suffixes; the W&B group is the parent run name. W&B display
names omit automatic timestamps and submission IDs, retaining configuration
and stage labels. W&B's own IDs distinguish repeated runs.

Single-level non-tuning manifests can opt into PPO-derived names with
`"run_name_template": "clu-g01_{ppo}"` (enabled for both current G01 presets).
`{ppo}` is `baseline` when all four PPO counts match current G03, otherwise
only the changed values appear, e.g. `numEnvs8k_numMinibatches5`. YAML edits and
manifest overrides are both reflected; overrides win. The `CLU_` prefix and
unique timestamp/ID remain for Determined/output identities; W&B displays only
the descriptive name, such as `CLU_clu-g01_baseline`.
See [the naming examples](../../docs/02_guide/09_G01_local_cluster_curriculum_comparison.md#names-derived-from-ppo-settings).

### G03 refinement: Wave 3, 17 missing runs with a four-hour learning budget

With at least 17 free one-GPU slots in the shared workspace, run:

```powershell
.\cluster\archive\submit-g03-refinement-wave3.ps1
```

The loop submits 17 separate experiments: seven original-grid neighbors with
both seeds, the additional 75-step check with both seeds, and only seed 456790
for the current 8000/50/3/5 candidate. It reuses that candidate's completed
seed 456789 and both completed 6000/50/3/5 neighbors rather than repeating them.

| Purpose | Envs | Steps | Epochs | Minibatches | New seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| Candidate confirmation | 8000 | 50 | 3 | 5 | 456790 |
| Higher environment count | 10000 | 50 | 3 | 5 | 456789, 456790 |
| Shorter rollout | 8000 | 25 | 3 | 5 | 456789, 456790 |
| Longer rollout | 8000 | 100 | 3 | 5 | 456789, 456790 |
| Fewer epochs | 8000 | 50 | 2 | 5 | 456789, 456790 |
| More epochs | 8000 | 50 | 4 | 5 | 456789, 456790 |
| Fewer minibatches | 8000 | 50 | 3 | 4 | 456789, 456790 |
| More minibatches | 8000 | 50 | 3 | 8 | 456789, 456790 |
| Additional rollout check | 8000 | 75 | 3 | 5 | 456789, 456790 |

The shared source level `G03_CLU_random_buildings.yaml` explicitly sets
`training.max_training_seconds: 14400`, `training.max_wall_seconds: null` and
`training.oom_recovery.enabled: false`. The learning clock starts after map,
bank and model preparation, at tracking initialization. It includes first JIT,
training, periodic inspection/validation and checkpoint work. The runner stops
cooperatively after the active update/evaluation; final validation, checkpoint
saving and media can follow the deadline. This is a learning-time budget, not
a hard four-hour allocation kill. The 2B-step and normal success gates remain.

All manifests also explicitly disable OOM recovery, so a failed cell cannot
silently change its PPO parameters. Their other overrides are only the four
PPO counts and training seed. Inspection/validation cadence and task/data settings
remain in the level YAML. The existing compatible PFSS pool is reused.

New pairs share `G03_CLU_refinement_wave3_N..._T..._E..._M...` groups. The single
candidate confirmation retains `G03_CLU_refinement_wave2_N8000_T50_E3_M5` to join
existing seed-A run `qmcncz8f`; unique submission/output names still say Wave 3.
The prior 8000/50/2/5 seed-B attempt was incomplete and is restarted with fresh
weights and a new output identity.

This is a raw submission loop, not a capacity-aware queue. The 24-slot ceiling
includes other users: 17 submissions fit only when 17 slots are actually free.
No pool-building prerequisite run is needed for this already prepared dataset.

### G03 refinement: Wave 2 axial checks, one training seed

With at least six free workspace slots, run from the repository root:

```powershell
.\cluster\archive\submit-g03-refinement-wave2.ps1
```

This submits only the six missing one-parameter neighbors of 6000/50/3/5:
4000/50/3/5, 8000/50/3/5, 6000/100/3/5, 6000/50/2/5,
6000/50/3/4 and 6000/50/3/6 (envs/steps/epochs/minibatches). Each uses training
seed 456789 and a fresh run identity in a `G03_CLU_refinement_wave2_N...` W&B
group. Previously measured center and worse neighbors are not repeated.

All shared task, dataset, evaluation and budget settings come from
`G03_CLU_random_buildings`. The only overrides are the four PPO counts,
training seed and `training.oom_recovery.enabled=false`; OOMs remain feasibility
outcomes for the exact requested cell. The existing compatible PFSS pool is
reused. This raw loop does not inspect free slots or automatically queue excess
jobs, so check workspace capacity before launching it.

### G03 refinement: one combined wave, two training seeds

From the repository root, first submit one run to prepare the shared PFSS pool:

```powershell
.\cluster\submit.ps1 -Run clu-g03-refine-wave1-leader-seed456789
```

When its log reports `Loaded 3600 precompiled random buildings`, submit the
remaining 31 runs:

**Capacity note:** the remainder script below is a raw 31-run loop. Under the
24-slot limit it must be dispatched in capacity-aware subsets, not all at once:
at most 23 additional one-slot jobs while the initial run remains active, then
the remaining 8 as slots become free (fewer if other user jobs occupy slots).

```powershell
.\cluster\archive\submit-g03-refinement-wave1.ps1
```

The initial run can continue training while the others start. The remainder
script excludes that initial configuration/seed pair, so the full wave still
contains exactly 32 runs. Use the same shared level/source settings for both
submissions so pool compatibility and the comparison remain consistent.

The single wave contains 16 configurations (envs/steps/epochs/minibatches):

| Purpose | Envs | Steps | Epochs | Minibatches |
| --- | ---: | ---: | ---: | ---: |
| Prior winner | 6000 | 50 | 3 | 5 |
| Prior consistent runner-up | 4000 | 100 | 4 | 5 |
| Fewer environments | 4000 | 50 | 3 | 5 |
| More environments | 8000 | 50 | 3 | 5 |
| Fewer epochs | 6000 | 50 | 2 | 5 |
| More epochs | 6000 | 50 | 4 | 5 |
| Shorter horizon | 6000 | 25 | 3 | 5 |
| Longer horizon | 6000 | 100 | 3 | 5 |
| Fewer, larger minibatches | 6000 | 50 | 3 | 2 |
| More, smaller minibatches | 6000 | 50 | 3 | 10 |
| Environment/horizon bridge | 4000 | 100 | 3 | 5 |
| Horizon/reuse bridge | 6000 | 100 | 4 | 5 |
| More environments with less reuse | 8000 | 50 | 2 | 5 |
| Runner-up with larger minibatches | 4000 | 100 | 4 | 2 |
| Longer horizon with less reuse | 6000 | 100 | 2 | 5 |
| More environments, longer horizon, less reuse | 8000 | 100 | 2 | 5 |

Every configuration receives seeds 456789 and 456790, including the two prior
leaders. This is a clean wave after shared non-PPO changes; historical seed runs
do not count toward these pairs. All start from fresh weights. Run names include
`refine-wave1-v2` and the training seed.

Each manifest specifies a `wandb_group`, such as
`G03_CLU_refinement_wave1_v2_N6000_T50_E3_M5`, shared by both seeds. These new
groups keep this wave separate from all earlier runs. This optional manifest
field changes W&B grouping only; unique submission and output identities remain.
Without it, existing submission grouping behavior is unchanged. Earlier W&B
leader runs `1k45t3el` and `bqr5bdp7` retain their historical refinement groups;
they are not moved into the new clean-wave groups.

Shared dataset settings are explicit in `G03_CLU_random_buildings.yaml`:
`training.map_pool_path: null` automatically reuses a compatible PFSS pool or
creates one when the generation/exclusion settings no longer match.
`training.map_selection_seed=456789` fixes the selected pool's 3600/400 split, and
`inspection_evaluation.seed=456789` fixes inspection reset cases.
`validation.seed=456789` fixes validation cases within the selected dataset. Eligibility
support discovery also follows the fixed bank-selection seed. Training model,
rollout and optimizer randomness follows each manifest's `training.seed`.
For an intentionally fixed dataset, set `training.map_pool_path` explicitly;
missing, incompatible or corrupt pinned pools fail rather than replacing it.
Existing submissions and resumed runs retain their frozen dataset selections.

Shared non-PPO settings come from the current cluster level. Each manifest
overrides only the four PPO counts, training seed and
`training.oom_recovery.enabled=false`. This last override is deliberate for the
fixed-parameter comparison: an OOM marks the requested cell infeasible rather
than silently turning it into a different environment-count experiment.
The shared level's normal OOM policy is unchanged. No wall-time, inspection,
validation or artifact overrides are injected by these manifests.

This is a local neighborhood plus interaction checks, not a complete Cartesian
grid or a guarantee of a global optimum. Compare both seeds over the same
elapsed-time window and keep OOM outcomes distinct from learning outcomes.

For multiple levels, replace `"level": "G03_random_buildings"` with, for example,
`"levels": ["G02_static", "G03_random_buildings"]`. The driver uses the existing
curriculum entrypoint. Both level and agent-budget stages remain flat output
folders. Level short names must differ to avoid stage-folder collisions.

## Preparation and submission (step 3)

```powershell
.\submit.ps1 -Run g03-smoke -DryRun
.\submit.ps1 -Run g03-full -DryRun
```

Each command prepares an ignored `.submissions/<unique-run-name>/` directory:

- `summary.json`: settings, resolved image and source size.
- `experiment.yaml`: generated Determined config (JSON syntax is valid YAML).
- `model/`: exact source snapshot to upload.
- `model/cluster/run.json`: level, overrides, W&B mode and unique run name.

Dry runs make no `det` calls, use no network and allocate no GPU. They validate
basic structure, paths, level existence, reserved/duplicate overrides, and package
size. They do not validate training override semantics, remote storage, secrets,
image availability or runtime compatibility. The first smoke job checks those.
An unset image repository is allowed for inspection but blocks real submission.

After publishing the matching image in step 2:

```powershell
.\submit.ps1 -Run g03-smoke
# After smoke succeeds and the W&B key is ready:
.\submit.ps1 -Run g03-full
```

Real submission prepares current source again, checks the installed `det` version,
sets `DET_MASTER` for its child process, and submits to project 367 using the
existing CLI login. The wrapper prints the experiment ID/URL and returns after
submission. It saves `submission.log` and `result.json` alongside the snapshot.
As soon as the CLI reports an experiment ID, it creates a local tracking folder
at `outputs/CLU_<run>/cluster_run.json`, before training output exists. Dry runs
do not create these folders. Keep these folders to retain retrieval history.
If the connection fails, inspect the WebUI before trying again: the job may have
been created. There is no automatic submission retry. Normal runs retain
`max_restarts: 0`; tuning runs allow two controller restarts using their saved
study state and original deadline. Internal training OOM recovery follows the
level config; tuning deliberately uses independent failed attempts instead of
silently reducing an existing candidate's environment count.

If execution policy blocks the script, use a process-scoped invocation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\submit.ps1 -Run g03-smoke -DryRun
```

The old standalone G03 YAML files and hardcoded driver are replaced by named
run configs and generated experiment files. `prepare_context.py` remains available
for manual packaging but is called automatically by submission.

## Validate steps 1 and 3

1. Inspect `settings.json`: project 367, `RAI_WS / swarm_echo`, `42_Compute`, one
   slot, and `/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo`.
2. Run both dry runs above. Expect success and a lowercase GHCR image reference.
   A dry run does not check that this image has been published yet.
3. Inspect each printed `experiment.yaml`: pool, slots, image, mount and entrypoint.
4. Inspect each `model/cluster/run.json`: smoke has the small overrides and
   disabled W&B; full has no overrides and online W&B. Names must differ.
5. Inspect the snapshot's G03 YAML for current edits. The snapshot excludes `.env`,
   `.venv`, existing outputs, W&B data, map pools and the rest of the repository.

This completes local preparation validation without submitting a job.

## Dependency image (step 2)

Follow [IMAGE_BUILD.md](../docs/IMAGE_BUILD.md). `submit.ps1 -ImageInfo` prints the exact
repository, dependency tag and fingerprint expected by submission. Windows CRLF
and Linux LF produce the same fingerprint. Inputs are `cluster/Dockerfile`,
`pyproject.toml`, and `uv.lock`. Changing them requires publishing the new image.
Source and level changes alone require only a new submission.

The image contains dependencies; `det` uploads the current local training source.
Source edits do not need a GitHub push for training. Dependency image builds use
committed files, so push dependency changes through your normal workflow before
building. The image workflow is in `.github/workflows/cluster-image.yml`; it runs
after relevant changes reach the default branch, or on manual dispatch there.
Adding the workflow locally does not itself build or publish an image.

## Persistence and W&B

The agreed PFSS `swarm_echo` directory already exists. The driver uses `outputs/`
and `map_pools/` beneath it. Full G03 generates a compatible map bank if needed.
Map preparation and compilation can delay the first useful SPS reading.

For online W&B, supply `WANDB_API_KEY` inside the trial or an owner-only
`.wandb_api_key` file in that PFSS directory. Local environment secrets are not
uploaded in model snapshots. To configure the PFSS key from the existing local
`WANDB_API_KEY` environment variable or repository `.env`, run:

```powershell
.\setup-wandb.ps1
```

This uses a temporary zero-GPU shell and SSH stdin, writes only
`<persist_root>/.wandb_api_key` with permissions `600`, and never prints the
credential or puts it in command arguments. An existing key is preserved;
`-Replace` explicitly updates it. Ordinary `fetch.ps1` sessions remain read-only.

Alternatively, in a cluster shell, create the file without echoing the key or storing
it in command history:

```bash
PROJECT_DIR=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo
(umask 077; read -r -s -p 'W&B API key: ' wb_key; printf '\n'; printf '%s\n' "$wb_key" > "$PROJECT_DIR/.wandb_api_key")
chmod 600 "$PROJECT_DIR/.wandb_api_key"
stat -c '%a %n' "$PROJECT_DIR/.wandb_api_key"
```

Expect `600`; do not print the file. W&B project/entity can be set through normal
run overrides. Detailed training metrics stay in W&B; tuning jobs also report
scalar candidate progress to Determined. Other runs report completion only.

## Zero-GPU shell

The existing `pfss-shell.yaml` mounts the workspace PFSS directory:

```powershell
$env:DET_MASTER = 'https://login01.ai.tu-darmstadt.de:8080'
det shell start -w RAI_WS --config-file .\pfss-shell.yaml
```

If Windows rejects the temporary SSH key, reconnect to the same shell with
`det shell open YOUR_SHELL_ID --show-ssh-command`. Use `exit` inside the shell to
disconnect, then `det shell kill YOUR_SHELL_ID` in PowerShell to release it.
The WebUI project does not create a PFSS directory or select shell paths.

Current G03 enables adaptive map and target sampling; its older G03 guide is
stale on those switches. Full submission uses the current source YAML.

## One-command retrieval

With VPN connected and the existing Determined login available, from `cluster`:

```powershell
.\fetch.ps1
```

If exactly one tracked experiment has never been fetched, it is selected
automatically. Otherwise, multiple tracked experiments produce a numbered menu
with `all` and `cancel`. A sole previously fetched experiment is selected again
automatically. Successful submissions from the earlier submitter are adopted
from `.submissions/*/result.json`; you do not need to supply their IDs.

Useful optional commands:

```powershell
.\fetch.ps1 -Choose        # Force experiment menu, including previously fetched runs
.\fetch.ps1 -All           # Fetch all tracked experiments
.\fetch.ps1 -SelectStages  # Pick an available curriculum stage, or all stages
.\fetch.ps1 -Status        # Show last observed remote/local progress, without network
```

For a manual evaluation, import one checkpoint and its frozen evaluation maps
instead of the full run and map bank. From the repository root:

```powershell
.\cluster\fetch.ps1 -Run 54978 -Checkpoint ckpt_001050 -EvalMaps 3
```

`-Checkpoint` enables the targeted mode. It imports the complete requested
checkpoint, `config.yaml`, the frozen map selection, and the first `-EvalMaps`
inspection maps (default `3`; match `inspection_evaluation.inspection_maps` in
your evaluation command). It uses the run's saved inspection YAMLs when present;
otherwise, for bank-backed runs, it fetches only the selected source YAMLs from
the referenced bank. The lightweight `random_buildings.json` descriptor is also
included when present. Geometry-probe suites default to `map_source: compatible_pool`:
they reuse a compatible local pool or create one with the checkpoint's generator
version and saved generation/geometry settings, then freeze their own map selection.
The original cluster validation bank is not required for that mode.

Other checkpoints, histories, existing media, validation-only maps, bank records,
and bank caches are excluded. Missing maps or an unpublished checkpoint fail
explicitly; no replacement map split is generated. Generated-map runs require
their inspection YAMLs to have been saved already.

Filtering happens remotely before file hashing, so targeted retrieval also avoids
scanning the full map bank. The summary prints the selected file count and size;
unchanged local imports are skipped as usual. Targeted fetches retain conflict
protection, atomic checkpoint promotion, and interrupted-import recovery. They
appear separately in `-Status` and do not mark a full run as synchronized.
Use ordinary `fetch.ps1 -Run ...` whenever a complete import is wanted. Stop an
existing fetch of the same run before starting another; the run lock prevents
simultaneous imports.

`-Run CLU_<name>` also selects a known experiment explicitly. Normal use requires
no ID. `-SelectStages` selects stages after connecting; without it, all currently
available stages are fetched. Multiple level/agent stages are tracked separately.
Future stages are not inferred to be finished simply because an earlier stage
has a final checkpoint. Agent and virtual target curriculum journals are retained,
including mastered flags and the maximum budget.

The fetcher starts one temporary zero-GPU shell per master/workspace/mount,
mounts PFSS read-only, transfers through Determined's SSH tunnel and kills only
its own shell afterward. No registry login or training environment is required
locally. OpenSSH and the existing `det` installation must be on PATH. On Windows,
the CLI's retained-key mode handles key permissions; no private key is added to
the repository. Connection diagnostics are retained in ignored `cluster/.fetch/`.
If cleanup reports an error, use the printed shell ID to stop that shell.

## Incremental imports and curriculum state

The parent `outputs/CLU_<run>/` contains the experiment record; imported stages
are ordinary sibling folders, e.g. `outputs/CLU_<run>_agents_2/` for population
curricula. Spawn-filter curricula keep one `outputs/CLU_<run>/` folder and one
W&B run; their budget is part of the existing adaptive spawn state. The inspector
and diagnostics use their existing flat discovery layout. Open/refresh the usual
local inspector after fetching. A parent with no training artifacts is not an
inspectable stage by itself.

`cluster_run.json` tracks the experiment ID/state, remote observation timestamp,
remote/local stage updates, validation updates, checkpoint inventories, mastery,
missing files and conflicts. `synced_to_observation` means the selected stage and
its banks match that particular remote inventory, not that a running job has
stopped producing new results. Use another fetch to advance the local snapshot.
`-Status` is explicitly last-observed state, not a live cluster monitor.

Each file has a remote and imported-content checksum. Unchanged imports are
not downloaded again. New artifacts/checkpoints are added; changed cluster-owned
metadata and growing histories are refreshed only when their local copy still
matches the previous import. Locally edited files and locally generated
diagnostics are preserved and reported as conflicts. Remote deletions do not
delete ordinary local output files. The fetcher skips W&B's local cache directory;
the online run remains available through W&B itself.

Checkpoints without readiness markers and pending/temp files are excluded.
Checkpoint members are staged and verified together before replacing an imported
checkpoint directory. Changing source files are deferred to the next fetch.
Interrupted imports retain completed work and receipts. Concurrent imports of
the same experiment are blocked by a process lock. The next fetch resumes
comparison; it does not create a second copy of a stage.

Known absolute PFSS references in JSON/YAML metadata are rewritten to portable
repository-relative output/pool references in local copies. Remote files are
unchanged; the record retains the path mapping and original remote checksums.
The inspector and frozen-bank readers resolve these references in Windows or
WSL. Referenced original map banks are fetched automatically into the normal local
pool storage, with a shared `CLU_` bank name, and reused across stages/imports.
Large banks can make the first fetch substantial. Missing remote banks are
reported rather than silently regenerated or marked synchronized.

Older already-submitted runs keep their original remote/W&B identity; their local
imports receive a `CLU_` alias. Newly submitted runs use matching `CLU_` names
everywhere. No changes to past W&B runs are made during retrieval.
