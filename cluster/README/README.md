# Cluster workflow

From the repository root in PowerShell:

```powershell
.\cluster\submit.ps1
```

This reads [g03-default.json](../manifests/g03-default.json) and submits one
`G03_CLU_random_buildings` run with the level's training seed and settings.
Local `swarmecho-train` uses the same level.

Use `-Config` for a different recipe. The saved
[chain-efficiency sweep](../manifests/chain-efficiency/fresh-sweep.json) and
[checkpoint retries](../manifests/chain-efficiency/checkpoint-retries/) record
earlier experiments. See the [G03 experiment record](../../docs/legacy-ablation/02_guide/10_G03_auto7_curriculum_sweep.md)
for run IDs, lineage, and results.

File-preparation progress updates on one line in an interactive terminal. When
output is redirected, only the final file/size summary is printed. The full CLI
output is retained in each submission's `.submissions/<run>/submission.log`.
Repeated preparation uses a new local receipt directory, retaining previous
snapshots even when the experiment display name has no timestamp.

## Check allocated GPU slots

```powershell
.\cluster\capacity.ps1
```

This read-only command uses the workspace/project/resource pool in `settings.json`
and detects the signed-in `det` user. It reports `X/24` allocated GPU slots,
then lists GPU allocations with their types, owners, projects, names and IDs.
The zero-GPU shell count appears beneath the table. Requested/queued slots and
zero-GPU shells consume no allocated GPU slots. Unknown ownership is marked
instead of guessed.
`-Json` returns sanitized machine-readable data, including the overlapping
outside-project and outside-user slot counts.
No raw shell connection data is printed or saved. The result is a snapshot, not
a reservation of the displayed free slots.

## One config for runs, seeds and variations

Only `level` is required; the other fields below show their defaults:

```json
{
  "level": "G03_CLU_random_buildings",
  "run_postfix": "",
  "seeds": 1,
  "wandb_mode": "online",
  "overrides": [],
  "variations": {}
}
```

- `seeds: 1` submits the level unchanged, including its training seed.
- `seeds: 2` submits the level's training seed and the next integer. For the
  current G03 CLU level these are `456789` and `456790`. Only `training.seed`
  varies; map-selection and evaluation seeds stay as defined in the level.
- `seeds: [123, 456]` selects explicit training seeds. Use this field instead of
  a `training.seed` override for multi-seed or explicit-seed submissions.
- `overrides` contains deliberate common `key=value` changes. Leave it empty to
  use level settings. Submission adds only the existing cluster output/logging
  settings, plus a training-seed override when requested.
- `variations` maps names to additional override lists. Each listed variation
  receives every seed; a variation replaces common overrides with the same key.
  An empty mapping submits the base configuration. When variations are listed,
  include a named empty list if you also want the base configuration.
- `exclude_trials` optionally lists exact `{"variation":"name","seed":456789}`
  pairs to omit. Unknown pairs and duplicates are rejected. The auxiliary sweep
  uses this to leave its initial test trial out of the remaining 23.

The ready-to-submit 24-run path auxiliary comparison, its one-run test and the
remaining 23 are documented in [GEO_AUXILIARY.md](../../ablations/cluster/path-probe/GEO_AUXILIARY.md).

The [five-seed G03 CLU baseline](../manifests/g03-baseline-5-seeds.json) uses
seeds 456789–456793. With `training.seed_map_pool=true`, each seed gets a
separate pool on shared PFSS storage. The manifest sets
`training.map_selection_seed=null` so the training/validation split also follows
the run seed. A later run can reuse the pool for the same seed and compatible
generation settings.

For example, four experiments, with two W&B groups containing two seeds each:

```json
{
  "level": "G03_CLU_random_buildings",
  "seeds": 2,
  "variations": {
    "baseline": [],
    "longer_rollout": ["training.num_steps=75"]
  }
}
```

The run prefix comes from the selected level YAML's `logging.run_name`, preserving
its case. Any `{ppo}` token is resolved using the level's PPO settings plus the
common and variation overrides. Optional `name` overrides this prefix; optional
`description` describes the experiments. Optional `run_postfix` appends your text after the
variation and seed, separated by an underscore. For example, `"run_postfix":
"experimentA"` with the current level's `run_name: G03` produces
`CLU_G03_seed456789_experimentA`.
An empty or omitted postfix adds nothing.

Run names have no UUID. The selected level YAML's
`logging.use_timestamp_postfix` controls whether a UTC timestamp such as
`-20261101T143025Z` is appended after the postfix. The current G03 CLU level sets
this to `false`. For historical multi-level manifests, the first level controls
this setting. W&B display names omit the timestamp and keep your postfix.

Seed variants share a W&B group per variation. The default group includes the
run prefix, optional postfix and optional timestamp, without the seed or a UUID;
variation names are appended when present. An explicit `wandb_group` sets the
comparison group instead. With timestamps disabled, change `run_postfix` to
distinguish submissions. Batch receipt filenames retain an internal random ID.

All experiments in one invocation use the same frozen source snapshot. They
are submitted sequentially and train independently. On a submission error the
batch stops; inspect its printed receipt before retrying, because a lost response
can still mean an experiment was created. Repeating the command starts a fresh
batch, not a continuation or automatic retry. Before preparing or submitting the
batch, the submitter checks every selected name against local tracking and
successful/ambiguous submission receipts. Conflicts prompt `Press Enter to
override, or Ctrl+C to abort`. All confirmations happen before the first job
is submitted. Enter permits replacing local tracking after a new experiment ID
is returned; the old `cluster_run.json` is archived under the run folder's
`.cluster/tracking_history/`. Aborting leaves tracking unchanged. 

Override retains PFSS training history. With timestamps disabled, use a new
`run_postfix` for a fresh training directory; overriding local tracking alone
does not resolve `Validation history exists` in an existing remote directory.
The local `-receipt-...` suffix protects local snapshots but does not isolate
the remote training directory. Discovery keeps existing tracking authoritative
so older receipts cannot replace an explicitly overridden record.

## Capacity and prerequisites

The workspace has a shared 24-slot limit. Check allocated capacity before
launching a batch. The submitter validates the requested slot count. Actual
allocation depends on the cluster.

The driver handles preemption by saving a complete checkpoint after an update.
On allocation restart it resumes the latest complete checkpoint and applies
the remaining training allowance.

[settings.json](../settings.json) holds the cluster destination, resource pool,
PFSS paths, CLI version, and dependency image. Use system Python and the
configured Determined CLI. Pass `-Python` to select the system executable.

## Other commands

For the four selected G03 auxiliary checkpoints (forward/reverse baseline and
efficiency 25), `tools/fetch_g03_inspection.py` fetches each published
`best_val_checkpoint` plus one frozen inspection map. It follows strict resume
ancestry when an inherited best has not been superseded, stops at branch
boundaries, and records the exact requested/source identities and hashes in
`.fetch/g03_best_replays.json`. It also sets short local inspector display aliases.

The replay runner selects only the two forward checkpoints (baseline and
efficiency 25, requested experiments 55555 and 55559). Run them sequentially in the existing WSL
environment from the repository root:

```bash
uv run python cluster/eval_g03_best_replays.py
```

This generates heatmaps and one replay per observed required-agent class on the
same saved map, using 1,000 independent targets and a common evaluation seed.
Each run uses its saved settings with inspection success explicitly set to
`chain_held_and_cleand_up`: hold the chain and decommission active
non-contributors. Local evaluations are labeled `chain&cleanUp`;
W&B is disabled. Checkpoints remain selected by their saved validation scores.
Open results with `uv run swarmecho-inspect root=outputs`.

```powershell
.\cluster\submit.ps1 -Config .\path\to\another-config.json
.\cluster\submit.ps1 -ImageInfo
.\cluster\fetch.ps1 -Choose
.\cluster\setup-wandb.ps1
```

The [command register](COMMANDS.md) covers setup, monitoring, cancellation,
checkpoint retrieval, local evaluation/probes and report updates. See the
[image guide](IMAGE_BUILD.md) and [report index](../../ablations/cluster/README.md).

## Folder layout

| Path | Purpose |
| --- | --- |
| `manifests/g03-default.json` | One G03 CLU run with the level defaults |
| `settings.json` | Stable cluster infrastructure settings |
| `submit.ps1`, `fetch.ps1`, `setup-wandb.ps1` | Main entry points |
| `tools/` | Python implementation and uploaded runtime driver |
| `README/` | Workflow, command reference, and image build guide |
| `manifests/` | Saved batches grouped by study, with G01 and G03 historical recipes |
| `scratch/` | Ignored local recipes for one-off submissions |
| `.submissions/`, `.fetch/`, `.context-*/` | Generated receipts, transfer state and snapshots |

Historical manifests are under `manifests/historical/`. See the
[cluster experiment reports](../../ablations/cluster/README.md) for the recorded results.
