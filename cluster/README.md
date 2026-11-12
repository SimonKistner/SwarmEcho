# Cluster workflow

First connection from macOS: [Mac SSH quickstart](docs/MAC_QUICKSTART.md)
(VPN, installation, login, zero-GPU shell and cleanup).

From the repository root in PowerShell:

```powershell
.\cluster\submit.ps1
```

This reads [config.json](config.json), submits one independent experiment per
seed/variation, and prints the experiment links. The checked-in config selects
`G03_CLU_random_buildings` with two seeds for each of six variations:
forward/reverse target curriculum crossed with chain-efficiency rewards
25/50/100, all using auxiliary lambda 0.3. This creates 12 fresh one-GPU jobs
with a 24-hour training budget each. The level enables chain efficiency at 50
and qualified non-contributor decommission reward at 3000; variations override
the efficiency coefficient for 25/100 and select the curriculum. Other PPO,
evaluation, artifact, OOM and time-budget settings come from the level YAML.
See the [sweep details](../docs/02_guide/10_G03_auto7_curriculum_sweep.md#chain-efficiency-and-retirement-sweep-3-november-2026).

The initial fresh efficiency experiments 55450–55461 were canceled and replaced with
matched branches from running parents 55442/55443/55445/55446. The specific
replacement entry point is `replace-g03-efficiency-with-branches.ps1`; it reads
the latest published parent checkpoints, inherits matching curriculum stages,
and subtracts inherited training time from the 24-hour total. Its dry run
contacts the cluster/W&B; add `-Offline -DryRun` to inspect cached metadata.
The ordinary `submit.ps1` command still describes the initial fresh sweep.
The submitted replacement experiments are 55462–55473; do not submit another
copy to follow their progress.
These attempts subsequently failed during compute-node network initialization:
55462–55472 timed out initializing W&B, and 55473 failed resolving the master
in the platform GPU-slot limiter. None reached a training update. The four
parent runs continued training. Restore compute-node DNS before retrying the
saved branch configurations; offline W&B cannot bypass a pre-entrypoint
master-resolution failure.

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
remaining 23 are documented in [GEO_AUXILIARY.md](docs/GEO_AUXILIARY.md).

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
`.cluster/tracking_history/`. Aborting leaves tracking unchanged. The static
G03 retry wrapper checks its entire selected set and prompts once for all its
conflicts, forwarding that approval to child submitters with `-OverrideExisting`.

Override retains PFSS training history. With timestamps disabled, use a new
`run_postfix` for a fresh training directory; overriding local tracking alone
does not resolve `Validation history exists` in an existing remote directory.
The local `-receipt-...` suffix protects local snapshots but does not isolate
the remote training directory. Discovery keeps existing tracking authoritative
so older receipts cannot replace an explicitly overridden record.

## Recover the canceled G03 continuation

To recover only canceled G03 reverse auxiliary seed 456790 (experiment 55446),
continuing its existing W&B curve to 24 cumulative training hours:

```powershell
.\cluster\submit-g03-recover-reverse.ps1
```

This command first checks the new local identity, then uses a temporary zero-GPU
shell to select the source's latest complete numbered checkpoint. It computes
the remaining allowance from all ancestor training clocks plus the saved
checkpoint's session clock, freezes the original source and dependency image,
and submits one strict continuation. Optimizer, curriculum, map pool, rewards
and original W&B identity are retained; the output folder is new. The metadata
shell is released before the training job is submitted. W&B clock lookup must
succeed when a periodic checkpoint has no embedded session clock.

The static efficiency branches also target 24 cumulative training hours using
their pinned checkpoints and remaining allowances. Neither workflow guarantees
the same wall-clock finish: queueing, downtime, setup and final saving differ.

## Shared-parent efficiency replacements

The six original seed-456790 efficiency branches (55491, 55493, 55495,
55497, 55499, 55501) were canceled at the user's request. Seed-456789 branches
55489, 55492, 55494, 55496, 55498 and 55500 remain the retained counterparts.
Submit only the six replacements with:

```powershell
.\cluster\submit-g03-efficiency-shared-parent.ps1
```

Each replacement copies its retained counterpart's frozen source/image,
checkpoint, map pool, reward settings, training allowance and exact W&B group.
The continuation seed becomes 456790, and the new output postfix is
`decom3000_branch_v3_shared789_v1`. Forward pairs share seed 456789's
`ckpt_002520` and 34,155.4516851604 new training seconds. Reverse pairs share
seed 456789's `ckpt_002880` and 34,102.88276382256 seconds. Inherited plus new
training targets 24 hours per run. Fresh W&B run IDs join the existing six
`CLU_G03_decom3000_branch_v3_<curriculum>_aux_l0p3_eff<coefficient>` groups.

The helper checks all local identities before preparation and verifies the
old replacement targets are terminal before submitting. It neither stops jobs
nor starts a metadata shell. The older `remaining11` command selects a different
batch and is not the replacement command. Exclude the canceled original runs
from W&B comparisons so each group represents the intended two continuations.

## Capacity and prerequisites

The authorized 4 November continuation batch is prepared/submitted with
`tools/resume_g03_nov04.py`: eight completed experiments 55438–55445 receive
another 12 training hours, and failed shared-parent experiments 55504–55508
resume their own latest checkpoints with their remaining original allowances.
Fresh output identities continue the existing W&B run IDs and groups. Each
frozen source snapshot receives only the checkpoint-clock/preemption lifecycle
patch, keeping its original scientific configuration and dependency image.

The driver polls Determined's preemption context and asks training to publish
a complete model/optimizer/controller checkpoint after a finished update. It
acknowledges preemption after the worker exits. On allocation restart, the
driver restores this output's latest complete checkpoint in strict resume mode
and subtracts its portable accumulated training clock from the job allowance.
The allowance comes from the original submission override or the resolved
level default in the frozen source. Each restart subtracts the cumulative
clock from that original allowance.
Preempted checkpoints skip final media/evaluation work. Periodic checkpoints
also carry the portable clock, covering abrupt allocation loss. This behavior
is packaged into the new jobs; previously running frozen jobs are unaffected.

Preparation reads the latest checkpoint inventory and records the five clock
calculations. Submission separately verifies source identities and live free
capacity before creating all thirteen jobs. No automatic duplicate submission
is attempted after a CLI failure; inspect the batch receipt first.

The separately recovered reverse auxiliary seed 456790 (55490) completed at
02:44 CET on 4 November and was subsequently extended through experiment 55524,
`CLU_G03_reverse_aux_l0p3_seed456790_resume36h_v1`. It loads that recovery's final
`ckpt_004812`, adds twelve hours, continues W&B `kchkdgan` in
`CLU_G03_reverse_aux_l0p3`, and receives the same preemption lifecycle patch.
`tools/resume_g03_reverse36h.py` records this one-job operation; repeating it
is not an extension of experiment 55524.

The next promising-run continuation selection contains all six original groups
(nine runs) plus forward/reverse efficiency 25 and 50 with both continuation
seeds (eight runs), for seventeen jobs. Efficiency 100 is excluded. The helper
`tools/extend_g03_promising.py` separates live checkpoint inventory, package
preparation and submission. It reads the live W&B identities and corrected groups
recorded in `cluster/.submissions/g03_next12h_selection.json`, verifies complete
seed coverage, copies each source's frozen scientific configuration/image and
continues its own latest complete checkpoint in strict resume mode. Every run
receives twelve additional training hours; interrupted sources also retain any
unused prior allowance. Original targets become approximately 48 cumulative
hours and efficiency targets approximately 36 hours.

The corrected branch group pattern is
`CLU_G03_<forward|reverse>_aux_l0p3_eff<25|50>_decom3000_branch_v3`.
The helper verifies these live groups before inventory and submission. By default
it requires seventeen free workspace slots. After maintenance, the user authorized
submitting the entire batch with `--allow-queue`, even if physical GPUs are busy.
This flag records the observed capacity and queue authorization in the batch
receipt. The nine main runs are submitted before the eight efficiency branches.
The 4 November batch submitted main experiments **55550–55558**, followed by
efficiency experiments **55559–55566**, preserving existing W&B curves/groups.

Workspace **129 / RAI_WS has a shared limit of 24 GPU slots**, including other
users and already running jobs. The submitter rejects a batch requesting more
than 24 slots, but does not query free capacity or queue until slots are free.
For the seventeen-run continuation helper, `--allow-queue` explicitly permits
submitting beyond the immediately free capacity. An experiment ID does not
guarantee that allocation succeeds.

[settings.json](settings.json) holds the stable destination, resource pool,
PFSS paths, CLI version and dependency image. Routine experiment changes belong
in `config.json` and the level YAML. Use system Python and the configured
Determined CLI, with VPN/login, a published dependency image and the PFSS W&B
key already available. The wrapper never uses the WSL-created project `.venv`.
Select system Python explicitly with `-Python` if necessary.

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
.\cluster\submit.ps1 -DryRun
.\cluster\submit.ps1 -ImageInfo
.\cluster\fetch.ps1 -Choose
.\cluster\setup-wandb.ps1
```

The [command register](docs/COMMANDS.md) covers setup, monitoring, cancellation,
checkpoint retrieval, local evaluation/probes and report updates. See the
[image guide](docs/IMAGE_BUILD.md) and [report index](docs/README.md).

## Folder layout

| Path | Purpose |
| --- | --- |
| `config.json` | Editable run/variation/seed configuration |
| `settings.json` | Stable cluster infrastructure settings |
| `submit.ps1`, `fetch.ps1`, `setup-wandb.ps1` | Main entry points |
| `tools/` | Python implementation and uploaded runtime driver |
| `docs/` | Command reference, image setup and published reports |
| `archive/` | Historical wave scripts, 113 manifests and previous workflow notes |
| `.submissions/`, `.fetch/`, `.context-*/` | Generated receipts, transfer state and snapshots |

Historical manifests remain accessible with `submit.ps1 -Run <old-name>`.
Archived scripts use the same entry point. Historical outputs, receipts and
transfer state are retained; the archive is for reproducing old experiments,
not the normal way to configure new ones.
