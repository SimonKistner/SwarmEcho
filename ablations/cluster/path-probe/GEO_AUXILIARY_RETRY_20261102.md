# Auxiliary sweep: quota investigation and retries, November 2, 2026

The live workspace allocation check found 18 sweep experiment allocations and a
separate three-GPU shell: 21 allocated slots against the workspace limit of 24.
The quota is shared by everyone in RAI_WS, including shells. It is not an
experiment count or a limit reserved for this user's runs.

The three-GPU allocation is shell `arc`, owner `sk89sino`, started October 30:

- Shell/task ID: `2a81d562-22a6-4719-8c76-7dfe3afd3aae`.
- Job ID: `b7050121-1398-4b26-94dd-a14018dd9fa8`.
- Workspace: 129 / RAI_WS; resource pool: 42_Compute.

No older active experiment allocations were found in this workspace: its active
experiments were the 18 surviving sweep runs. Other zero-GPU shells do not consume
GPU slots. An experiment's training timeout does not release an unrelated shell.
The reported 27-slot startup count is consistent with the 24-run sweep plus this
three-slot shell. Multiple concurrent startup checks can reject more allocations
than the excess before the rejected allocations have finished releasing resources.

## Confirmed failures

| Experiment(s) | Lambda | Snapshots | Seed(s) | Cause |
| --- | --- | --- | --- | --- |
| 55311; earlier duplicate 55310 | 0.1 | 5 | 456789 | Existing validation history in reused run directory. |
| 55330 | 0.3 | 5 | 456790 | Workspace allocation quota. |
| 55331, 55332 | 1 | 1 | 456789, 456790 | Workspace allocation quota. |
| 55333, 55334 | 1 | 5 | 456789, 456790 | Workspace allocation quota. |

The earlier attempt 55308 was canceled. These are six distinct missing
combinations, not seven: both errored test submissions refer to the same
lambda=0.1/K=5/seed=456789 cell. The 23-run batch has 18 surviving runs.

All five quota failure log tails were checked. The test's error was:
`Validation history exists; resume explicitly or choose a new run.` Reusing that
original name would also collide with local experiment tracking. Retrying uses
fresh names/directories with postfix `geo_aux_v1_retry1`; original W&B variation
groups are explicitly retained, as are seeds, PPO settings, lambda and K.

## Submit the missing combinations

No experiments were stopped or submitted during this investigation. The configs
below were expanded locally to verify their combinations, and the first batch
was prepared with `-DryRun`.

From the repository root in PowerShell, with at least three slots free:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\geo-auxiliary\geo_auxiliary_retry_now3.json
```

This submits lambda=0.1/K=5/seed=456789 and lambda=1/K=1 for both seeds.
With the allocation observed during investigation, this fills 21 -> 24 slots.

Wait until at least three slots are free again, then:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\geo-auxiliary\geo_auxiliary_retry_later3.json
```

This submits lambda=0.3/K=5/seed=456790 and lambda=1/K=5 for both seeds.
The first and second batches are disjoint and cover all six missing combinations.
Do not stop another user's shell; coordinate its release with the owner if needed.

Alternatively, if at least six slots are free, submit all six together **instead
of** either split batch:

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\geo-auxiliary\geo_auxiliary_retry6.json
```

The submitter does not deduplicate a retry already submitted from another config.
Do not rerun the original test/remaining23 configs: that would duplicate surviving
combinations and reuse existing output directories.

Check current workspace GPU occupancy without printing shell connection data:

```powershell
$env:DET_MASTER = (Get-Content .\cluster\settings.json -Raw | ConvertFrom-Json).master
$jobs = det job list --resource-pool 42_Compute --pages all --json | ConvertFrom-Json
$used = ($jobs.jobs | Where-Object { $_.workspaceId -eq 129 } | Measure-Object allocatedSlots -Sum).Sum
"$used allocated; $(24 - $used) free"
```

The quota is enforced during allocation startup rather than reliably queuing the
excess experiments. Recheck capacity before each submit because other users can
allocate slots concurrently. Verification receipts and log tails are under
`outputs/verification/geo_auxiliary_capacity_20261102/`.
