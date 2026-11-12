# G03: a 4000-map training/validation dataset

`G03_random_buildings` uses 15 x 15 x 15 m random buildings. It starts 4000
parallel training lanes over 3600 persistent training maps, reserving 400 maps
for independent statistical validation. Three of those held-out maps also form
the existing inspection suite. See the [validation reference](../01_reference/12_validation.md).

The cluster variant `G03_CLU_random_buildings` selects
`env.success_condition: chain_held_and_cleand_up`: its 100-step uninterrupted
chain hold must be complete and every remaining active drone must contribute
before successful termination. A chain break during cleanup resets the hold
counter normally. It also uses an unnecessary-call penalty of -500 per
responsible qualified caller, gated by the base already knowing the target.
Normal G03 retains `chain_held` and its -200 unnecessary-call penalty.

## Counts, scheduling and diagnostics

| Setting | G03 |
| --- | --- |
| `training.num_envs` | 4000 parallel lanes, reducible after OOM |
| `training.num_maps` | 3600 training maps, fixed across lane reductions/stages |
| `validation.num_maps` | 400 held-out maps, disjoint from training |
| `validation.num_episodes` | 10000 total IID uniform map-target cases per pass |
| `validation.parallel_envs` | 500 lanes per execution batch |
| `validation.val_freq` | Every 50 PPO updates |
| `validation.fixed_targets` | Enabled; cases frozen within each agent stage |
| `validation.val_on_finalization` | Enabled |
| `validation.save_per_map_stats` | Enabled; compact local results table |
| `validation.save_best_checkpoint` | Enabled; actor and critic saved together |
| `inspection_evaluation.inspection_maps` | Three fixed maps from the validation pool |
| Inspection episode budget | 1000 per map, retained |
| Inspection metrics/video frequency | Every 100 updates, matching G02 |
| Inspection replay limit | Five-map upper limit; all three inspection maps |
| Training map selection | Uniform at episode reset; adaptive map sampling disabled |
| Map success window | Configured as 100; inactive while map adaptation is disabled |
| Cell success window | Configured as 500; adaptive target spawning disabled |
| Eligibility discovery | Configured as 800 per map; inactive while adaptive target spawning is disabled |
| Individual training cell/map diagnostics | Both disabled |

The training budget remains 500 million steps per agent stage. Validation controls
progression at 85% intermediate and 99% final thresholds; inspection
results are diagnostic and disable the stage reachability filter. Validation
continues to use the filtered training task. Uniform map choice and the frozen
3600/400 split are independent of adaptive weighting. No map warmup or adaptive
cell discovery runs with the current two adaptation toggles disabled.

If adaptive map sampling is enabled, a complete map warmup needs 360000 completed episodes;
at the 700-step cap these completed episodes represent 252 million steps, plus
unfinished episodes. Early exit can still precede map warmup. If adaptive target
spawning is enabled, support discovery requires about 2.88 million accepted targets per stage, or 17.28 million over
six stages, before rejected proposals. Disabling reporting does not remove
sampling or discovery work.

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
Both `G03_random_buildings` and `G03_CLU_random_buildings` explicitly enable
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
authored G02 maps are unchanged. A launch override is:

```bash
uv run swarmecho-train level=G03_random_buildings random_buildings.advanced_exclustion_base_LOS=true
```

## Differences from current G02

Both local `G03_random_buildings` and cluster `G03_CLU_random_buildings` require
`adapt_size.quorum_hold: 3`: three consecutive environment steps of base quorum.
Their `env.spawn_delay: 5` now gates every base deployment, as in all levels,
including reinforcements. Every actual newborn is registered at the base as NO
before its first GRU policy decision; the inspector displays these true records.

G03 now follows current G02's latest-decision voting, vote holding 1 (inactive
with this protocol), unanimity, base-vote/population observations and enabled
privileged vote rewards (-1 for qualified withheld votes, -200 for unnecessary
calls). Both use agent budgets two through seven with one spare slot, capped
at seven deployable agents. The final success threshold and inspection
metrics/video cadence also match G02. G03 currently uses an 85% intermediate
validation threshold and validates every 50 updates.

G03 retains its 500-million-step budget per stage versus G02's 1.3 billion,
disabled map/cell adaptation (its stored 100-episode map window is inactive),
random-map split, independent validation,
bounded inspection budget and disabled individual training diagnostics.
The observation/protocol differences from the earlier G03 config have been
removed; historical G03 checkpoints from that config remain incompatible with
the new observation/protocol contract. G02's authored five-map reporting and
stopping behavior remain unchanged.

## Runtime checks still required

The statistical validation pipeline, inspection aliases, independent map counts,
fixed cases, best-checkpoint state and local inspector view are implemented.
Before a large run, verify in WSL that a small fresh random run can reset, execute
both pipelines, save/resume checkpoints, hand off stages and retain its split
through OOM recovery. Validate target acceptance and decommission zones, including
all held-out maps. Profile preparation, discovery, padded geometry, compilation,
throughput and checkpoint size/time at increasing bank sizes.

A small generated-map smoke configuration avoids building a large disk pool:

```bash
uv run swarmecho-train level=G03_random_buildings logging.run_name=G03_validation_smoke random_buildings.load_maps_from_bank=false training.num_envs=20 training.num_maps=18 training.num_steps=8 training.total_timesteps=320 validation.num_maps=2 validation.num_episodes=40 validation.parallel_envs=10 validation.val_freq=1 inspection_evaluation.inspection_maps=2 inspection_evaluation.eval_parallel_envs=20 inspection_evaluation.eval_video=false logging.wandb_mode=disabled
```

Use a new name for each fresh smoke run. This checks the generated path; the
saved-bank path and the full 3600/400 workload still require runtime verification.
The intended full launch is:

```bash
uv run swarmecho-train level=G03_random_buildings logging.run_name=G03_random_4k_v1
```

## G04 CLU 4×4×4 comparison

`G04_CLU_random_buildings` uses 20 × 20 × 20 m buildings with 5 m cells,
four floors, four to seven rooms per floor and one or two staircases per
floor transition. Nine drones deploy automatically. The forward curriculum
opens target difficulty groups from two through nine required agents.

The three runs share a seed, the same map-bank selection and validation
cases, efficiency reward 25, decommission reward 3000, the 24-hour training
budget and OOM recovery. The two probe arms use auxiliary weight 0.3.
G04 uses a base-relative `7 × 7 × 7` path grid to cover offsets −3 through
+3 along every axis. Probe feedback increases the actor input from 141 to
484 values. The critic uses 141 in all three arms.

The first run to acquire the shared pool lock builds a compatible G04 pool
if needed. The other runs reuse it. Submit the three runs from the
repository root in PowerShell.

```powershell
.\cluster\submit.ps1 -Config .\cluster\g04_path_probe_feedback.json
```
