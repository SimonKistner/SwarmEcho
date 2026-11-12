# Cluster workflow command register

Updated **1 November 2026**. Commands match the repository wrappers and the
installed **Determined CLI 0.35.0**. This is a reference; commands are not run
automatically. Examples with experiment IDs refer to existing runs, while
submission commands always create new experiments.

## Terminals and common variables

Run the PowerShell commands from the **repository root**, not from `cluster/`:

```powershell
Set-Location -LiteralPath 'Q:\_0_Projects\000_SwarmEcho\SwarmEcho'
$Py = 'C:\Users\Simon\AppData\Local\Programs\Python\Python310\python.exe'
$env:DET_MASTER = 'https://login01.ai.tu-darmstadt.de:8080'
$Run = 'g03-smoke'       # Historical manifest under cluster/archive/runs, without .json
$ExperimentId = 55264    # Replace when inspecting another experiment
```

Cluster wrappers use **system Python**, not the WSL-created project `.venv`.
The later `uv run` commands belong in the existing **WSL environment** at
`/mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho`.

Current destination: **RAI_WS / swarm_echo**, workspace **129**, project **367**,
resource pool **42_Compute**, normally **one GPU slot per trial**. The shared
workspace ceiling is **24 slots**, including other users and other experiments.
Submission does not check free capacity or wait for training to end. The new
config-driven submitter rejects batches requesting more than 24 slots total.
An accepted experiment ID does not guarantee that the allocation limiter lets it
start. Zero-GPU transfer shells do not consume GPU slots.

## 1. Install/check tools and log in

Connect the VPN before commands that contact the cluster.

```powershell
Get-Command python, det, ssh
& $Py -m pip install --user 'determined==0.35.0'
det --version
det user login
det user whoami
```

`det user login YOUR_USERNAME` selects an explicit account. `det user logout`
ends the retained login. Repository wrappers supply their configured master to
child processes; the `DET_MASTER` variable above does the same for direct CLI use.

Inspect destination settings and available run manifests:

```powershell
Get-Content -LiteralPath .\cluster\settings.json
Get-Content -LiteralPath .\cluster\config.json
Get-ChildItem -LiteralPath .\cluster\archive\runs -Filter *.json | Select-Object Name
Get-Content -LiteralPath ".\cluster\archive\runs\$Run.json"
```

## 2. Resolve and publish the dependency image

Print the exact image/tag/fingerprint expected by submission, without network:

```powershell
& .\cluster\submit.ps1 -ImageInfo -Python $Py
```

Publish through **GitHub Actions → Build cluster dependency image → Run workflow**
on the repository's default branch, or its normal dependency-change trigger.
The image must be publicly pullable with the current submitter. See
[image setup](IMAGE_BUILD.md) for package visibility and CI details.

For a manual Docker build/push instead, using the same dependency-only context:

```powershell
$ImageInfo = (& .\cluster\submit.ps1 -ImageInfo -Python $Py | ConvertFrom-Json)
$ImageContext = Join-Path $env:TEMP ('swarmecho-image-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $ImageContext
Copy-Item -LiteralPath .\pyproject.toml, .\uv.lock, .\README.md -Destination $ImageContext
docker login ghcr.io
docker build --platform linux/amd64 -f .\cluster\Dockerfile -t $ImageInfo.image $ImageContext
docker push $ImageInfo.image
```

Changes to `cluster/Dockerfile`, `pyproject.toml` or `uv.lock` change the dependency
fingerprint. Source/level edits alone are uploaded with submission and do not
require an image rebuild. GitHub CLI is not required by this workflow.

## 3. Configure W&B on PFSS

Use an existing local `WANDB_API_KEY` environment variable or repository `.env`:

```powershell
& .\cluster\setup-wandb.ps1 -Python $Py
& .\cluster\setup-wandb.ps1 -Replace -Python $Py
```

The first command preserves an existing remote key. `-Replace` updates it.
The helper transfers the key over SSH stdin and stores it as
`<persist_root>/.wandb_api_key` with mode `600`; it does not print the key or add
it to the source snapshot. Do not put credentials in run manifests.

## 4. Prepare and inspect a submission

```powershell
& .\cluster\submit.ps1 -DryRun -Python $Py
```

A dry run prepares `.submissions/<unique-name>/` and checks packaging/basic
config structure. Inspect each experiment's `summary.json`, `experiment.yaml` and
`model/cluster/run.json` at the printed path. It does not submit, allocate a GPU,
check the published image, or prove runtime compatibility.

Shared task, evaluation cadence, artifact and budget settings belong in the
source level YAML. `cluster/config.json` selects the level, seeds and deliberate
variations; the default `seeds: 1`, empty overrides and empty variations submit
the level unchanged. `seeds: 2` uses its training seed and the next integer;
an explicit integer list chooses specific seeds. Each variation's seed runs
share one W&B group, fresh for each invocation. See [configuration examples](../README.md).
The G03 CLU comparison level currently declares `max_training_seconds: 14400`
and OOM recovery disabled. `g03-full` selects the local G03 level; it is not an
alias for the G03 CLU comparison configuration.

If PowerShell execution policy blocks a wrapper, the process-scoped form is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cluster\submit.ps1 -Run g03-smoke -DryRun -Python $Py
```

## 5. Submit the configured run or seed/variation batch

```powershell
& .\cluster\submit.ps1 -Python $Py
```

Edit `cluster/config.json` before this command. Its checked-in defaults submit
one `G03_CLU_random_buildings` experiment, using the level's settings. Each seed
and variation creates a separate experiment. All use the same source snapshot;
each has a unique output folder. The printed batch receipt lists prepared and
submitted experiments. A submission error stops the batch. Repeating the command
creates fresh experiments; inspect saved receipts/WebUI before retrying an
uncertain response so an already-created job is not duplicated.

Before the first submission, all selected names are checked for existing local
tracking or experiment receipts. On a conflict, press Enter to override or
Ctrl+C to abort. Tracking is replaced only after the new experiment ID is
returned, with the previous record archived in `.cluster/tracking_history/`.
This leaves remote training history intact; use a new `run_postfix` when a
fresh remote output directory is required. The static G03 retry wrapper prompts
once for its entire selected set and forwards the approval automatically.

Use another config or select a historical manifest explicitly:

```powershell
& .\cluster\submit.ps1 -Config .\path\to\config.json -Python $Py
& .\cluster\submit.ps1 -Run $Run -Python $Py
```

Retry a slot-rejected cell with its same manifest, for example:

```powershell
& .\cluster\submit.ps1 -Run clu-g03-refine-wave4-n10k-t25-e3-m5-seed456790 -Python $Py
```

This is a historical retry example; that Wave 4 cell has already been retried
successfully. Choose the actual rejected manifest for future retries. Keep slot
rejections separate from OOMs; an OOM is an observed failure of the PPO cell.

Recover canceled experiment 55446 alone, preserving its frozen source, rewards,
optimizer, curriculum and existing W&B curve with only the remaining allowance
to 24 cumulative training hours:

```powershell
.\cluster\submit-g03-recover-reverse.ps1 -Python $Py
```

At execution this starts and releases a temporary zero-GPU metadata shell,
selects the latest complete checkpoint and calculates the remaining training
time before submitting one continuation. It targets 24 total training hours;
it does not synchronize wall-clock finish times with other jobs.

Submit the six shared-parent efficiency replacements only:

```powershell
.\cluster\submit-g03-efficiency-shared-parent.ps1 -Python $Py
```

These run continuation seed 456790 from the same seed-456789 parent checkpoint
and training allowance as each retained efficiency counterpart. The original
frozen source/image and existing per-variation W&B group are preserved. New
output names end in `decom3000_branch_v3_shared789_v1`. This command does not
submit the retained six or use the old `remaining11` selection.

The authorized 4 November batch uses a separate preparation/submission helper:

```powershell
& $Py -X utf8 .\cluster\tools\resume_g03_nov04.py --prepare
# Use the exact BATCH_READY path printed above; inspect receipts before retries.
& $Py -X utf8 .\cluster\tools\resume_g03_nov04.py --submit-batch .\cluster\.submissions\batch-g03-nov04-<timestamp>-<id>.json
```

It extends completed originals 55438–55445 by 12 hours and recovers failed
shared-parent continuations 55504–55508 with their remaining original allowances.
It installs the checkpoint/preemption lifecycle patch into each otherwise frozen
source/image context. The 4 November submission created experiments 55511–55523;
repeating this batch is not a continuation of those new experiments.

## 6. Archived multi-experiment submission loops

For new experiments use `config.json` and the command in section 5. The wrappers
below are archived for reproducing historical waves. All create fresh experiments; they
are not commands for resuming or fetching the historical waves.

| Command from repository root | Submissions | Scope |
| --- | ---: | --- |
| `.\cluster\archive\submit-g01-five.ps1` | 5 | Original G01 candidate batch |
| `.\cluster\archive\submit-g01-uniform.ps1` | 8 | Uniform G01 variants; baseline is separate |
| `.\cluster\archive\submit-g03-ablation.ps1` | 10 | Original G03 ablations; baseline is separate |
| `.\cluster\archive\submit-g03-transfer-five.ps1` | 5 | G03 transfer variants |
| `.\cluster\archive\submit-g03-refinement-wave1.ps1` | 31 | Wave 1 remainder, excluding initial pool-building run |
| `.\cluster\archive\submit-g03-refinement-wave2.ps1` | 6 | Wave 2 axial checks |
| `.\cluster\archive\submit-g03-refinement-wave3.ps1` | 17 | Wave 3 confirmations/boundaries |
| `.\cluster\archive\submit-g03-refinement-wave4.ps1` | 16 | Wave 4 base batch |
| `.\cluster\archive\submit-g03-refinement-wave4-addon.ps1` | 5 | Only the five additional main-grid seed-B runs |

Wrappers call `submit.ps1` with system `python` on PATH; they do not expose a
`-Python` parameter. They submit sequentially but training overlaps. **The 31-run
Wave 1 remainder exceeds the 24-slot ceiling even in an empty workspace** and
must be dispatched in smaller subsets. Any other batch also needs its requested
number of actually free slots.

The original Wave 1 prerequisite was:

```powershell
& .\cluster\submit.ps1 -Run clu-g03-refine-wave1-leader-seed456789 -Python $Py
```

Its log message `Loaded 3600 precompiled random buildings` indicated that the
shared pool was ready for the remaining consumers. A compatible existing PFSS
pool is reused automatically according to the level's map-pool settings.

## 7. Capacity, live state, logs and configuration

These direct CLI commands use the master configured in the terminal variables.

```powershell
det workspace list
det workspace describe RAI_WS --json
det workspace list-pools RAI_WS
det job list --resource-pool 42_Compute --pages all --json
det task list
det experiment list --all --show_project
det experiment describe $ExperimentId --json
det experiment describe 55264,55270,55271 --json
det experiment config $ExperimentId
det experiment list-trials $ExperimentId
```

Use workspace allocations in the WebUI together with these listings to assess
free capacity. Counts of jobs/experiments alone are not counts of occupied GPU
slots. The workspace limit is shared, not a reservation for this project.

For convenient logs of the experiment's first trial:

```powershell
det experiment logs $ExperimentId --head 100
det experiment logs $ExperimentId --tail 150
det experiment logs $ExperimentId -f
det experiment wait $ExperimentId
```

For an exact trial, take its ID from `describe`/`list-trials`:

```powershell
$TrialId = 76791
det trial logs $TrialId --head 100
det trial logs $TrialId --tail 150
det trial logs $TrialId --level ERROR
det trial logs $TrialId --json
det trial logs $TrialId -f
det trial logs $TrialId --tail 200 | Select-String -Pattern 'Allocations slots limit reached', 'RESOURCE_EXHAUSTED', 'Out of memory'
```

`Allocations slots limit reached - Limit: 24` is a quota rejection.
`RESOURCE_EXHAUSTED: Out of memory` is an OOM. The latter can occur after an
initially successful update. Stop a local follow/wait with `Ctrl+C`; it does not
cancel the remote training job.

## 8. Stop, pause, reactivate and archive

Each command below changes the named experiment's state:

| Command | Purpose |
| --- | --- |
| `det experiment cancel $ExperimentId` | Request cancellation |
| `det experiment kill $ExperimentId` | Force termination |
| `det experiment pause $ExperimentId` | Pause scheduling/execution |
| `det experiment activate $ExperimentId` | Activate/unpause |
| `det experiment archive $ExperimentId` | Archive the experiment in Determined |
| `det experiment unarchive $ExperimentId` | Restore it to the ordinary listing |

Scheduler activation is distinct from application-level checkpoint recovery.
For SwarmEcho training recovery, configure the explicit checkpoint as below;
do not assume that an allocation restart automatically restores PFSS state.

## 9. Resume, branch or initialize training from a checkpoint

The specific replacement workflow for fresh efficiency experiments 55450–55461
is `cluster/replace-g03-efficiency-with-branches.ps1`. It branches from the
latest checkpoints of their four running resumed counterparts, retains matching
curriculum stages and allocates only the remaining time to 24 cumulative
training hours. See [replacement details](../../docs/02_guide/10_G03_auto7_curriculum_sweep.md#replace-the-fresh-efficiency-sweep-with-branches).

For the nine completed G03 curriculum runs in the 3 November screenshot, use
the prepared continuation batch from PowerShell at repository root:

```powershell
& .\cluster\submit-g03-resume12h.ps1 -DryRun -Python $Py
& .\cluster\submit-g03-resume12h.ps1 -Python $Py
```

This starts new jobs from the final PFSS checkpoints, each with another 12-hour
training budget and the same W&B run ID. It uses each original source snapshot
and dependency image. See the [exact run/checkpoint mapping and budget semantics](../../docs/02_guide/10_G03_auto7_curriculum_sweep.md#continue-the-six-screenshot-groups-for-another-12-hours).

Create/edit a named run manifest and add explicit training overrides:

```json
"training.checkpoint_path=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo/outputs/OLD_RUN/checkpoints/final_checkpoint",
"training.ckpt_loading_mode=resume"
```

Then submit that manifest through the same wrapper:

```powershell
& .\cluster\submit.ps1 -Run YOUR_RESUME_MANIFEST -Python $Py
```

The path must be the **remote PFSS path** when training on the cluster. Use a
published checkpoint with its recovery metadata and matching training contract.
`resume` continues counters, `branch` carries progress into a branch, and `init`
loads weights only. New output identity and compatible dataset settings still
matter. Budget/PPO changes should be explicit in the level/manifest.

## 10. Fetch outputs or a single evaluation checkpoint

| PowerShell command | What it does |
| --- | --- |
| `& .\cluster\fetch.ps1 -Python $Py` | Default selection/menu of tracked runs |
| `& .\cluster\fetch.ps1 -Choose -Python $Py` | Force the experiment menu |
| `& .\cluster\fetch.ps1 -Run $ExperimentId -Python $Py` | Full import of one tracked experiment |
| `& .\cluster\fetch.ps1 -Run 'CLU_FULL_RUN_NAME' -Python $Py` | Select by tracked local run name |
| `& .\cluster\fetch.ps1 -All -Python $Py` | Full imports of all tracked experiments |
| `& .\cluster\fetch.ps1 -SelectStages -Python $Py` | Select available curriculum stages |
| `& .\cluster\fetch.ps1 -Status -Python $Py` | Last locally recorded fetch status; no network |

The normal full import includes the run's referenced banks and can be large.
For local inspection/evaluation, use the targeted import instead:

```powershell
& .\cluster\fetch.ps1 -Run 55264 -Checkpoint best_val_checkpoint -EvalMaps 3 -Python $Py
& .\cluster\fetch.ps1 -Run $ExperimentId -Checkpoint ckpt_001050 -EvalMaps 3 -Python $Py
```

`-EvalMaps` requires `-Checkpoint`; it defaults to three. A checkpoint name is one
directory name, not a full path. Targeted mode imports the checkpoint, saved
configuration/selection, available generator descriptor and requested inspection
maps. It does not import the full validation/training bank. Existing imports
are reused by checksum and local edits are preserved. It has its own completion
record; it does not mark the whole run synchronized.

The helper starts a read-only **zero-GPU** transfer shell and releases it.
`Ctrl+C` interrupts retrieval; completed local imports remain. Rerun the same
fetch to continue. If cleanup prints an owned shell ID, release that shell using
the commands in the next section.

Python equivalents, using the same system interpreter:

```powershell
& $Py .\cluster\tools\submit.py
& $Py .\cluster\tools\submit.py --config .\path\to\config.json
& $Py .\cluster\tools\submit.py --run $Run --dry-run
& $Py .\cluster\tools\submit.py --run $Run
& $Py .\cluster\tools\submit.py --image-info
& $Py .\cluster\tools\setup_wandb.py
& $Py .\cluster\tools\setup_wandb.py --replace
& $Py .\cluster\tools\fetch.py --run 55264 --checkpoint best_val_checkpoint --eval-maps 3
& $Py .\cluster\tools\fetch.py --status
```

`det experiment download`/`list-checkpoints` concern Determined-registered
checkpoints; this training workflow stores its application checkpoints on PFSS,
so use the repository fetcher for them.

## 11. Manual zero-GPU PFSS shell and cleanup

```powershell
det shell start -w RAI_WS --config-file .\cluster\pfss-shell.yaml
det shell list
$ShellId = 'REPLACE_WITH_SHELL_UUID'
det shell open $ShellId
det shell show-ssh-command $ShellId
det shell logs $ShellId
```

The manual shell configuration mounts PFSS read-write. Inside that Linux shell:

```bash
PROJECT_DIR=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo
ls -lh "$PROJECT_DIR/outputs"
ls -lh "$PROJECT_DIR/map_pools"
du -sh "$PROJECT_DIR/outputs" "$PROJECT_DIR/map_pools"
stat -c '%a %n' "$PROJECT_DIR/.wandb_api_key"
exit
```

`exit` disconnects; release the allocation from PowerShell afterward:

```powershell
det shell kill $ShellId
```

Kill only the shell you own/intend to release. `show-ssh-command` provides the
reconnect command when Windows SSH key handling needs inspection.

## 12. Evaluate a fetched checkpoint in WSL

From the repository root in WSL, the registered M6 seed-B evaluation command is:

```bash
RUN="/mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho/outputs/CLU_clu-g03-refine-wave4_numEnvs6k_numSteps50_numEpochs3_numMinibatches6_seed456790-20261031T202546Z-69ced70c"

uv run swarmecho-evaluate \
  checkpoint="$RUN/checkpoints/best_val_checkpoint" \
  level=G03_CLU_random_buildings \
  training.num_envs=6000 training.num_steps=50 \
  mode=parallel \
  replay_after=true replay_by_required_agents=true \
  inspection_evaluation.training_heatmap_creation=true \
  inspection_evaluation.dynamic_max_geo_distance=true \
  inspection_evaluation.target_sampling=independent \
  inspection_evaluation.eval_video=true \
  inspection_evaluation.map_suite_replay_limit=1 \
  eval_name=best_val_650_required_agents
```

That checkpoint is update **650**, not a directory named `ckpt_001050`.
The command evaluates the saved three-map inspection suite and produces
required-agent replays on one map. `map_suite_replay_limit=1` limits replay maps,
not the maps used for metrics. This is not the original 400-map validation score.
Its `independent` sampling override is intentional. Replay labels include their
required-agent cluster and map ID, for example `required_agents_3_random_eval_0000`.

For metrics only, omit replay/heatmap overrides:

```bash
uv run swarmecho-evaluate checkpoint="$RUN/checkpoints/best_val_checkpoint" level=G03_CLU_random_buildings training.num_envs=6000 training.num_steps=50 mode=parallel
uv run swarmecho-inspect root=outputs
uv run swarmecho-inspect root="$RUN" port=8765 open=false
```

For the six curriculum comparison groups fetched on **3 November 2026**, run
six best checkpoints (training seed 456789 in every group) on the same frozen map:

```bash
uv run python cluster/eval_g03_curriculum.py --dry-run
uv run python cluster/eval_g03_curriculum.py
```

The first command only prepares configurations and prints commands. The second
runs sequential one-map metrics, heatmaps and required-agent replays. See the
[selected runs and checkpoint scores](../../docs/02_guide/10_G03_auto7_curriculum_sweep.md#one-map-replay-comparison-3-november-2026).

## 13. Geometry-probe suite and local compatible pools (WSL)

```bash
uv run python geo_experiments/suite.py plan=true
uv run python geo_experiments/suite.py
uv run python geo_experiments/suite.py resume=geo_experiments/outputs/YOUR_SUITE_FOLDER
```

`probes.yaml` selects the policy/checkpoint and experiments. The configured
`compatible_pool` mode reuses a matching complete local pool or creates one;
it does not need the original cluster validation maps. Selection and map hashes
are frozen under the checkpoint run's `probe_maps/`. Compatibility includes the
recorded generator version and generation/geometry settings.

**`plan=true` may prepare a missing CPU map pool** while resolving the plan;
it does not execute policy rollout or decoder fitting. The second command does
execute the suite. Resume uses the saved suite plan/configuration.

Explicit map-source and reporting commands:

```bash
uv run python geo_experiments/suite.py policies.g03_agents_7.map_source=compatible_pool plan=true
uv run python geo_experiments/report.py run=geo_experiments/outputs/YOUR_SUITE_FOLDER/runs/YOUR_READOUT
uv run python geo_experiments/reproduce.py run=geo_experiments/outputs/YOUR_SUITE_FOLDER/runs/YOUR_READOUT
uv run python geo_experiments/auditing.py run=geo_experiments/outputs/YOUR_SUITE_FOLDER/collections/YOUR_COLLECTION
```

`map_source=validation` instead requires the exact original validation metadata
and map files from a full import. It is not an alias for the compatible-pool mode.

Manual pool construction, normally unnecessary for the suite's automatic mode:

```bash
uv run python -m swarmecho.curriculum_config.maps.random_building_bank.generate_bank --level G03_CLU_random_buildings --workers 4
```

That manual command builds/appends the generator's default batch (currently
2,000 maps); the probe's automatic preparation requests its own required count.
Use saved-config-compatible settings for the checkpoint being probed. See the
[probe guide](../../geo_experiments/README.md) for sampling and provenance.

## 14. Fetch tuning scalars and rebuild the refinement report

These are the existing analysis helpers for the completed G03 refinement study,
not the general checkpoint downloader. From PowerShell at repository root:

```powershell
$Analysis = '.\outputs\g03_refinement_analysis_20261031'
& $Py "$Analysis\fetch_results.py" --wave4
& $Py "$Analysis\fetch_results.py" --wave4 --cluster
& $Py "$Analysis\classify_outcomes.py"
& $Py "$Analysis\analyze.py"
& $Py "$Analysis\build_report.py"
& $Py "$Analysis\finalize_wave4_data.py"
Invoke-Item -LiteralPath "$Analysis\G03_refinement_report.html"
```

The first fetch gets W&B metadata and complete histories. `--cluster` gets
Determined states/logs and PFSS scalar journals through a zero-GPU transfer.
`--wave3` is also supported; future wave selectors require updating these
study-specific helpers. Classification keeps quota, OOM and learning outcomes
separate. Analysis verifies saved maps/configuration/validation journals before
the report is rebuilt.

The maintained [published report index](README.md) lives in `cluster/docs/`.
The build command writes the working analysis directory under `outputs/`; it
does not publish those files automatically. When adopting a rebuilt report,
refresh its linked charts/data in the published directory and update the index.

## Artifacts and command discovery

| Location | Contents |
| --- | --- |
| `cluster/settings.json` | Destination, resource pool, CLI version, image repository |
| `cluster/config.json` | Default level, seed count/list, common overrides and named variations |
| `cluster/archive/runs/*.json` | Historical run manifests |
| `cluster/tools/` | Python implementation and cluster runtime |
| `cluster/.submissions/batch-*.json` | Per-invocation prepared/submitted experiment receipts |
| `cluster/.submissions/<name>/` | Source snapshot, generated experiment config, submission log/receipt |
| `outputs/CLU_<name>/cluster_run.json` | Experiment identity and incremental fetch tracking |
| `cluster/.fetch/` | Transfer diagnostics and temporary staging |
| PFSS `swarm_echo/outputs/` | Training outputs and published application checkpoints |
| PFSS `swarm_echo/map_pools/` | Shared cluster map pools |
| Local `src/swarmecho/curriculum_config/maps/random_building_bank/pools/` | Local/generated/imported pool files |
| `cluster/docs/` | Published cluster reports and this command register |

For additional flags, read the installed command's help:

```powershell
det --help
det experiment --help
det trial logs --help
det shell --help
& $Py .\cluster\tools\submit.py --help
& $Py .\cluster\tools\fetch.py --help
```

Related references: [cluster workflow](../README.md),
[dependency images](IMAGE_BUILD.md), [reports](README.md),
[local command guide](../../docs/02_guide/03_commands.md).
