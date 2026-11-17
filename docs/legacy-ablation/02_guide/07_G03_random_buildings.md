# G03 random buildings

## Shared local and cluster default

Both local training and the cluster launcher use
[G03 CLU](../../../src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml).
The local names `G03` and `G03_random_buildings` resolve to that preset.

```bash
uv run swarmecho-train
```

| Setting | G03 CLU |
| --- | --- |
| Training task | Hold the chain for 30 consecutive steps and retire active non-contributors |
| Deployment | Seven slots reserved at reset, with launches five steps apart. Recruitment is disabled. |
| PPO | 6,000 lanes, 50 steps per update, three epochs, six minibatches |
| Budget | Five billion environment steps or 24 hours of training |
| Training maps and targets | 3,600 bank maps, frozen target-cell eligibility, uniform sampling, no target curriculum |
| Held-out validation | 400 maps, 10,000 fixed cases, every 50 updates and on finalization. Success means chain holding. |
| Inspection | Three held-out maps, 1,000 cases per map, independent target sampling, final heatmaps and replays |
| Path auxiliary | Enabled at λ 0.03. Previous-step probe feedback is off in the preset. |

See the [validation reference](../01_reference/12_validation.md) and
[cluster guide](../../../cluster/README/README.md).


## Pool layout and identity

Explicit generation settings match the old pool's layout: length/width 15 m,
three storeys, 5 m cells, wall/tile thickness 0.25 m, 2-4 rooms per storey, extra
door probability 0.15, window probability 0.2 and at most two staircases.
Roadmaps retain full mode, density 0.5, merged wall ends, zero corner bonus,
drone radius 0.25 m, planning clearance 0.1 m and target wall buffer 0.1.

Layout matching does not bypass source fingerprints. The old 5000-map pool has
an older compiler fingerprint; current startup will build a compatible pool of
at least 4000 maps. Reusing the exact old pool requires a validated migration.
The random-building feature defaults to base-cell-only target exclusions.
G03 CLU and the legacy local G03 profile both enable
an expanded version of G02's exclusion heuristic:

```yaml
random_buildings:
  advanced_exclustion_base_LOS: true # Enabled for local and cluster G03; global default false.
```

This excludes the base cell and every cell whose representative point is visible
from the base within `env.comm_radius_base + env.visual_radius`. Each representative
is the centre of that cell's largest legal target spawn box, using the configured
target wall buffer. Segment tests against compiled solids handle diagonal walls,
door/window openings and stairs. The nearest hidden cell sharing a face with,
and visible from, one of those directly visible cells is also excluded if it lies
within the same range. If no such corner exists, only the direct exclusions apply.
The rule runs once per map generation, with no random position sampling or
agent simulation. It may leave some single-agent-solvable targets; it does not
inherit G02's sampled two-agent starter screening. A layout with no remaining
legal target volume fails explicitly rather than falling back to base-only.

The flag and its range/buffer settings participate in map-bank compatibility,
so enabling it uses a compatible bank instead of old base-only spawn boxes.
Local and cluster G03 use `training.map_pool_path: null`, allowing startup to
reuse a sufficient compatible pool or create a new one when none matches. Cluster
pools persist under the shared PFSS `map_pools` directory. An explicit pool path
pins the dataset and fails on incompatibility instead of replacing it. Existing
runs retain their frozen selections; changing these settings requires a fresh run.
Training, generated validation and inspection share the same rule. Existing
authored G02 maps retain their authored exclusions.

Set `training.seed_map_pool=true` to generate and cache a separate pool for
each `training.seed`. The pool directory and `metadata.json` record the seed.
Startup can reuse a completed pool with the same seed, compatible generation
settings, and enough maps. Set `training.map_selection_seed=null` to also use
the training seed for the disjoint training and held-out split. The
[five-seed CLU baseline](../../../cluster/manifests/g03-baseline-5-seeds.json)
enables both options for 3,600 training and 400 validation maps per seed.

## Historical local G03

[`G03_legacy_random_buildings`](../../../src/swarmecho/curriculum_config/levels/G/G03_legacy_random_buildings.yaml)
preserves the earlier local profile. It starts with one drone, uses vote-based
recruitment and progression by geometric agent count, and enables adaptive
map and target sampling. It uses 4,000 training lanes and 500 million steps
per stage. The saved historical launch recipes that used this profile now
refer to it by its explicit legacy name.

The [G03 CLU workflow checks](../../../tests/test_g03_workflows.py) use a small
generated split to exercise training, checkpoint evaluation, and held-out
validation without a prebuilt pool.


## G04 CLU 4×4×4 comparison

`G04_CLU_random_buildings` uses 20 × 20 × 20 m buildings with 5 m cells,
four floors, four to seven rooms per floor and one or two staircases per
floor transition. The agent-count curriculum starts at five automatically
deployed drones and advances through six, seven and eight. Each stage samples
targets feasible with its own population. Held-out validation success of at
least 99% triggers the next stage, which starts from the previous checkpoint.

The manifest expands three probe variants across two training seeds. All six
trials share the same map-bank selection and stage-specific validation cases,
efficiency reward 25, decommission reward 3000, a 24-hour training budget per
stage and OOM recovery. The two probe variants use auxiliary weight 0.3.
G04 uses a base-relative `7 × 7 × 7` path grid to cover offsets −3 through
+3 along every axis. Probe feedback increases the actor input from 141 to
484 values. The critic uses 141 in all three arms.

The first trial to acquire the shared pool lock builds a compatible G04 pool
if needed. The other trials reuse it. Each trial uses one GPU slot and runs its
stages sequentially, with separate W&B runs and checkpoints for each stage.
Automatic cluster recovery currently looks for checkpoints under the base
run name, while these stages save under names ending in `_agents_<count>`.
Submit the batch from the repository root in PowerShell.

```powershell
.\cluster\submit.ps1 -Config .\cluster\manifests\path-probe-feedback\g04_path_probe_feedback.json
```
