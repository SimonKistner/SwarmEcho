# G02: five static buildings

G02 now uses the [signed individual reward defaults](../01_reference/11_individual_rewards.md):
success +3000 and target delivery +125 per active drone, time pressure -0.75 per
acting drone per step, and a -3000 individual adjustment for each useless drone
at chain success. Local coverage, sender and soft-cluster attribution remain.
G01 retains an unsupported legacy reward block with a commented replacement.

`G02_static` uses the migrated signed individual rewards, replays all five
maps, sets `evaluation.final_early_exit_success_rate: 0.99`, and enables
`training.adaptive_map_sampling` with a one-update probability half-life.
It selects `training.dynamic_max_geo_distance.mode: global_candidates` with
`candidate_spacing_fraction: 0.5`. Production and the fraction experiment use
the same seeded continuous-uniform sampler: 216 accepted random positions on
each 15 m cube, plus legal roadmap/launch anchors. There is no grid sampler.
All five buildings are 15 x 15 x 15 m, with 5 m cells and three storeys. Their
geometry is frozen in ordinary map YAMLs, ready for inspection and editing in
the maze builder. The original random bank is unchanged.

## Map selection and noSpawn regions

`tools/prepare_g02_maps.py` shuffles the complete matching 15 m bank with seed
456789 and takes the first five candidates that satisfy the noSpawn rule and
have a sampled two-agent starter placement. The selection is therefore seeded
and screened, rather than an unbiased sample of every bank entry. Two candidates
were skipped. Source paths, SHA256 hashes, representative LOS points, skip reasons
and starter placements are in
[`G02_static_selection.json`](../../src/swarmecho/curriculum_config/maps/G02_static_selection.json).

| Static map | Source bank map | Three base-LOS noSpawn cells | Adjacent non-LOS noSpawn cell |
| --- | --- | --- | --- |
| G02_static_map_01 | 03054 | (2,0,0), (2,1,0), (2,2,0) | (1,1,0) |
| G02_static_map_02 | 01807 | (0,2,0), (0,1,0), (0,0,0) | (1,1,0) |
| G02_static_map_03 | 02953 | (2,2,0), (1,2,0), (2,1,0) | (1,1,0) |
| G02_static_map_04 | 04456 | (2,2,0), (2,1,0), (2,0,0) | (1,1,0) |
| G02_static_map_05 | 02628 | (0,0,0), (0,1,0), (1,0,0) | (1,1,0) |

Coordinates are zero-based XYZ cells. The three direct-LOS cells **include the
base cell**. The fourth shares a face with one of those cells and is visible
from that cell, while blocked from the base. LOS is checked against compiled
solids using the centre of each cell's largest legal target box, avoiding stair
treads. This is a representative-point rule: it does not claim every point in
an excluded cell has the same visibility, or exclude every possible base-LOS
point elsewhere. Inspect these regions before starting the experiment.

CPU checks confirmed valid map geometry, the four exclusions, a clear return/
decommission zone above each base and a physical two-agent chain example to an
allowed target. These examples reduce the risk of an empty initial stage, but
are **not** certificates from the production JAX relay calculator.

The preparation tool refuses to overwrite differing maps by default. Its
`--replace-generated` flag explicitly permits replacing generated G02 files,
including manual edits that retain the generation header; do not use it after
editing the noSpawn regions unless regeneration is intended.

## Training and feature behavior

```bash
uv run swarmecho-train level=G02_static logging.run_name=G02_global_individual_v1
```

Use a new run name for the estimator/reward change. A fresh run has no
checkpoint path by default. Loading old weights should use a new branched run
with fresh adaptive support; strict resume rejects a changed relay identity.

The static-suite bank initially assigns balanced lane IDs: at 4000 training
lanes this means 800 per building. Enabled adaptive map sampling can switch a
lane's building when its episode finishes; ongoing episodes continue normally.
Each lane uses its building's base, solids, roadmap and exclusions. A shared
policy trains across all buildings; there is no new actor map-ID observation.

- **Adaptive size:** votes, recruitment, base state, reference rewards and
  decommission zones remain independent per environment. G02 sets
  `adapt_size.quorum_fraction: 1.0`, requiring qualified approval from every
  currently active drone at the base before a recruitment attempt.
- **Reinforcement vote shaping:** G02 enables
  `reward.priveleged_vote_reward.enabled` with latest YES/NO decisions.
  Informed connected withholding costs -1 per communication tick; responsible
  YES sources on unnecessary grants receive -200 instead of +10. The estimator
  budget is tightened by the smallest population that has formed a valid chain,
  with decision-time provenance protecting delayed previously necessary votes.
  Set the toggle to false for the previous reward behavior. See
  [eligibility and timing](../01_reference/06_adaptive_swarm_size.md#optional-reinforcement-vote-shaping).
- **Adaptive map sampling:** each map has a 500-outcome window; adaptation starts
  after all five have filled their 500-episode windows. A 30% uniform floor and one-update
  probability half-life move new episode assignments toward unsuccessful maps.
  This uses probabilities, not lane quotas. All parameters are grouped under
  `training.adaptive_map_sampling`; default-off globally, enabled here. See the
  [map sampling reference](../01_reference/10_adaptive_map_sampling.md).
- **Agent curriculum:** geometric budgets run from two through seven. With one
  spare and a hard cap of seven, population caps are 3, 4, 5, 6, 7, 7. Targets
  remain certified against the geometric budget, not the spare population.
- **Max geo distance:** the global-candidate relay filter queries cached hop
  labels for the current map, both in training and evaluation. A failed
  construction or a count above the stage budget is rejected; there is no
  unfiltered fallback. Cache records are reused across agent stages and runs.
- **Adaptive target spawning:** each building has its own cell support, volume
  baseline, rolling success estimates, floors, smoothed probabilities and
  warnings. Eight of that map's 800 accepted discovery targets are required
  to retain a cell, independently of live lane counts. That sampled support
  stays fixed until the next stage.
  See the [spawning reference](../01_reference/08_adaptive_target_spawning.md).
- **Soft cluster credit:** each environment scores its own current chain using
  its map's obstacle geodesics; no reward allocation is shared between maps.

Resets index a shared padded `[5, 63]` float32 weight table, totaling 1260 bytes.
Changing its values between PPO updates does not change its shape. CPU updates
process 115 cell categories in total. Separate map journals live under
`adaptive_spawn_maps/map_0000/adaptive_spawn/` through `map_0004`; the existing
spawn dashboard lists each as `run / source_map`. One checkpoint stores all map
controllers, windows and masks together, rejecting changed map order on resume.

After inspecting/editing maps, use a **new run name**. Source hashes and cached
evaluation map copies protect an existing run from silent map changes. G1 policy
weights can be used for a branch with a compatible population/observation
contract; G1 adaptive statistics are not transferable to different buildings.

## Inherited settings worth checking

The level preserves G1's settings apart from map selection, adaptive map sampling,
the replay limit and the final-stage success threshold:

- **Evaluation cost:** 4000 episodes per map, hence 20,000 per evaluation call,
  evaluated sequentially. OOM reductions also lower the per-map evaluation count.
  Adaptive training weights never affect evaluation proposals.
- **Mastery:** evaluation early exit uses the equally weighted **mean** map
  success at 95% for intermediate stages and 99% for the final stage, with one
  qualifying evaluation. `evaluation.final_early_exit_success_rate: 0.99`
  overrides the normal 95% threshold only when the geometric target budget and
  population cap both reach seven, sampling targets reachable by up to seven
  agents across the maps. The six-agent target stage retains 95% even though its
  spare slot already raises the population cap to seven.
  Four maps at 100% and one at 95% can pass the final mean gate.
  `eval/worst_map_success_rate` and
  `eval/map/G02_static_map_01/success_rate` through `05` expose that imbalance;
  the progression criterion remains mean success.
- **Replays:** `evaluation.map_suite_replay_limit: 5` records every building
  per scheduled replay pass. Evaluation heatmaps remain separate per map.
- **Training exposure:** 500 million steps is a budget per agent-curriculum
  stage, with early exit; it is shared over five maps, not multiplied by five.
  Adaptive map sampling changes each building's share of environment steps.

The standard global success/return/useless-agent charts pool the suite. Use the
new per-map success charts and separate replay/heatmap artifacts when diagnosing
one difficult building.

## Verification boundary

Windows CPU checks passed for geometry, the original settings copy, probability
isolation, normalization, padded weights, state restoration and dashboard journal
discovery/resume. The full level loader needs OmegaConf, unavailable in the
standalone Windows interpreter. No WSL/GPU training, JAX compilation or multi-map
reset execution was performed from this editing session.
Additional CPU checks passed for adaptive-map warmup/floors, packed rings,
checkpoint continuation and a 4000-map controller. The JAX map-switch reset
path and large-bank throughput still require project-environment verification.

The current host audit (`outputs/g02_preflight/host_audit.json`) checked actual
production random preparation and disk-cache reloads on all five maps, plus
512 uniformly sampled legal targets per map with independently reconstructed
physical witnesses. It found 367, 371, 367, 369 and 401 candidates respectively,
zero unresolved/invalid sampled witnesses, and at least one <=2-agent starter
target on every map. Maps 02 and 05 had 107/512 and 26/512 sampled counts above
seven, respectively; the training filter excludes those targets. These counts
are constructive upper bounds and do not prove that seven-agent solutions
are impossible. Fixed-base lazy hop labels and reachability also matched the
stored randomized-base graph searches, and denser/coarser point prefixes agreed
on every map. Map 04's points/hops/parents matched the prior random experiment.
This host audit did not run the full OmegaConf loader or JAX.

In the project WSL environment, this command compares the selected production
JAX estimator to the original CPU simplifier and certifies its placements on
all five maps. Its reset check also covers every selected map separately with
adaptive spawn/size combinations. It does not exercise the multi-map probability
table or perform a PPO update:

```bash
uv run python max_geo_distance/validate_integration.py --level G02_static --maps G02_static_map_01 G02_static_map_02 G02_static_map_03 G02_static_map_04 G02_static_map_05 --targets 32 --runtime-batch-size 32 --geometry-details 0 --reset-smoke
```

The first normal G02 startup/rollout still needs runtime verification for the
shared adaptive table and all five startup eligibility masks.
The signed reward audit found the configured coefficients consistent with the
new runtime recipients, per-agent PPO rewards/lifetime boundaries and running
value normalization. No reward coefficients were retuned. Recurring gap shaping
remains enabled (`gap_reward_uses_change: false`), and the terminal success and
useless-agent terms cancel for a useless active drone; earlier and reference
rewards still remain. These are intentional settings to inspect in the first
run, not evidence of equal total returns across population sizes.

## Later: 4000 maps

[`G03_random_buildings`](07_G03_random_buildings.md) now copies G02 with explicit
15 m random-bank generation settings and bounded evaluation/map warmup budgets.
Its guide lists the corrected journal dispatch, old-pool compatibility
and runtime verification work; configuration alone does not establish readiness.

The dedicated [road to 4k](../road-to-4k.md) records the concerns, proposed
implementation and verification requirements. These are future work.

Independent map/cell distributions remain practical in principle. With similar
geometry, a `[4000, 63]` float32 box-weight table is about 1 MB. Even
`4000 x 23 x 500` outcomes occupy about 46 MB if stored as packed byte arrays.
The distributions themselves are not the main scaling obstacle.

Enabled adaptive map sampling now provides fixed per-map discovery budgets,
packed map/cell windows, grouped cell outcomes, bounded large-bank histories and
stable bank identities across lane reductions. Persistent random banks can use
both adaptive features. This is an implementation path to 4k, not a measured
GPU throughput result; discovery cost, geometry padding and existing per-cell
warning/checkpoint overhead still need profiling.
