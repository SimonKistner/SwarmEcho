# Road to 4k maps

Status: core adaptive map sampling is implemented, with CPU controller checks;
the full 4000-map GPU workload remains unbenchmarked. The grouped, default-off
`training.adaptive_map_sampling` feature supports authored suites and persistent
generated/saved random banks, and is enabled in G02. See the
[configuration and runtime reference](01_reference/10_adaptive_map_sampling.md).

Implemented decisions:

- Direct episode-selection probabilities, not lane quotas; 30% uniform floor,
  500-outcome per-map windows, all-map warmup until every window is full and three-update
  probability smoothing. No extra regression detector or refresh mechanism.
- Map changes at natural episode endings, refreshing geometry and recurrent state.
- Independent accepted-target discovery budgets (800 per map), packed map/cell
  windows, grouped cell outcomes and bounded large-bank journal output.
- Frozen random-map selections across stage handoffs and OOM lane reductions,
  checkpointed controller state, and ten held-out evaluation maps by default.

The concerns below explain these choices and remaining profiling work.

## What must remain true

Each persistent map must own its target-cell identity, legal-volume baseline,
eligibility mask, success statistics and probability distribution. A cell in
one building must not change a cell's probability in another building, even
when their XYZ indices match. Probability floors, failure prioritization and
smoothing normalize over cells within one map. Map allocation is a separate
decision; equal lanes do not imply equal completed episodes.

Map IDs must remain tied to immutable geometry/base/exclusion identities across
resets, population stages and checkpoint resume. Adaptive size and relay-budget
queries already use each environment's state and geometry. Scaling their banks
and workloads still needs measurement; existing map-local semantics alone do
not establish acceptable runtime or memory use at 4000 maps.

## 1. Startup discovery is the first correctness problem

With adaptive map sampling disabled, the relay-filtered adaptive sampler discovers support from the accepted
uniform startup reset batch. A cell is retained if it receives at least 1% of
its map's startup targets. The mask stays fixed for that agent-budget stage.

- G02: 4000 lanes / 5 maps = 800 targets per map; threshold eight.
- 4000 maps with 4000 lanes: one target per map; threshold one. Almost every
  other cell receives zero probability for that stage, even if feasible.
- Current balanced persistent assignment requires at least one lane per map.
  OOM recovery below 4000 lanes would also prevent hosting all 4000 maps with
  that assignment.

Computing probabilities faster cannot fix these issues. Discovery needs a
budget independent of live training lanes, or the support rule needs redesign.
The 1% rule is a sampled frequency cutoff, not a proof of cell feasibility;
more samples do not make it include cells whose true accepted frequency is
below 1%.

The enabled feature implements the following approach; alternatives remain possible:

1. Run bounded, chunked discovery batches per map before policy training,
   reusing certified accepted samples where practical. Measure cost at every
   agent-budget stage; a G02-like 800 samples per map would mean 3.2 million
   accepted samples per stage before rejection costs.
2. Admit newly discovered feasible cells during training, with an explicit
   probing distribution so zero-probability cells can actually be discovered.
   Specify how a changing baseline affects floors, windows and smoothing.
3. Select the next map probabilistically at episode reset, allowing more bank
   maps than live lanes while retaining statistics by persistent map ID.
   This is implemented, including fixed bank membership across lane reductions.

Do not silently keep the one-sample mask or treat unresolved relay queries as
proof that a map/cell is impossible.

## 2. Probability arrays are small; Python histories are not

Illustrative storage for 4000 maps like G02, assuming 23 spawnable cells,
63 padded free-volume boxes and a 500-outcome window per cell:

| Data | Approximate raw storage |
| --- | ---: |
| One float32 box-weight table `[4000, 63]` | 1.0 MB |
| One float32 cell array `[4000, 23]` | 0.37 MB |
| Byte outcome buffers `[4000, 23, 500]` | 46 MB |

These are decimal MB estimates, not total runtime memory estimates. Masks,
counts, cursor arrays, warning history, checkpoints and geometry add more.
The disabled legacy path's Python deques store references rather than packed outcome bytes;
references alone for 46 million entries would occupy roughly 368 MB on a
64-bit runtime, before deque/container overhead.

Implemented in the enabled path: packed per-cell ring buffers with write cursors, valid counts
and success sums. Replacing an old outcome updates its sum in constant time.
Preserve the exact latest-N-episodes semantics, including multiple completions
for one cell in a rollout. Measure checkpoint size and serialization time too;
JSON expands numeric arrays substantially.

## 3. Batch updates without mixing maps

The legacy small-suite wrapper scans completed outcomes separately for each map and
uses Python cell windows. Repeating a full outcome scan for every map scales
with map count times completed outcomes. The enabled path groups outcomes by map
once, updates packed windows and keeps the existing per-map probability formula
on the host. Per-cell plateau histories and JSON checkpoint metadata remain
profiling concerns. A full GPU probability controller has not been introduced.

Group or scatter outcomes by a composite map/cell ID, keeping deterministic
ordering when several episodes update one window. Probability calculations can
then use padded `[map, cell]` arrays and reduce only along the cell axis.
JAX broadcasting/vmap can parallelize failure weighting, floors, normalization
and smoothing independently for every map. Mask invalid and ineligible cells,
and handle empty support and all-success rows explicitly.

Ring-buffer writes need collision-safe treatment: a naive scatter overwrite
loses outcomes when several completions hit one cell. Design batching/order
before moving state to the GPU. Benchmark packed CPU updates against a device
implementation; JAX is an option, not an assumed speedup. Avoid copying entire
window buffers between host and device every PPO update.

## 4. Geometry and rejection may dominate the sampler

The current immutable bank pads solids, target boxes, roadmap nodes and distance
arrays to the largest map. Pairwise roadmap data can be much larger than the
probability table, and one unusually large map increases padding for the bank.
Measure preparation, device transfer, first compilation, warm reset throughput
and peak memory separately. Consider grouping maps by geometry shape or using
compact/chunked records only if profiling justifies the added complexity.

The relay filter certifies actual proposed targets during reset. Early stages
may accept a very small region in some buildings, causing expensive rejection
or exhausting the configured attempt bound. A physical two-agent example does
not guarantee a usable acceptance rate or that the production calculator finds
it. Report per-map attempts, rejection/exhaustion and retained support; preserve
the existing rule against unfiltered fallback.

## 5. Diagnostics and mastery need their own scale plan

Legacy per-map JSON journals, warning histories and dashboard entries are
reasonable for five maps. At 4000 maps they multiply file operations and history
size, and thousands of individual plots are hard to inspect. Keep compact
statistics for every map, but use sampled/coarser history and selectable
diagnostic detail. The enabled path now bounds large-bank journals and W&B
series, while preserving complete controller state in checkpoints. Per-cell
plateau metadata remains an overhead to measure.

G02 evaluates 4000 episodes per building. Applying that unchanged to 4000 maps
would mean 16 million episodes per evaluation call. Define a representative
evaluation budget and periodic broader audit rather than multiplying that
setting blindly. The enabled feature now reserves ten held-out evaluation maps
for random runs, while authored G02 retains its five-map evaluation. Report
the mean plus worst-map/low-quantile or failing-map
summaries so an aggregate does not hide systematic failures. Decide the mastery
criterion explicitly: current G02 progression uses 98% mean success, not 98%
on every map. Likewise, replay a useful subset rather than every map each time.

## Suggested order and acceptance checks

1. Establish G02 runtime correctness and profile the five-map baseline.
2. Define discovery coverage, map scheduling and OOM behavior independently of
   lane count. Measure acceptance rates on a larger sample of buildings.
3. Benchmark the implemented packed windows and grouped host updates against
   a possible JAX implementation without changing statistical semantics.
4. Measure compact persistence and bounded diagnostic/evaluation output.
5. Scale through intermediate bank sizes before 4000 maps.

At each step verify map isolation, normalized distributions, probability floors,
padding masks, exact outcome-window ordering, stage-specific eligibility and
checkpoint continuation. Include map reordering/identity rejection and OOM
recovery. Check actual per-map exposure and support coverage as well as speed;
a fast run that trains on one cell per building is not a successful scale-up.

Related implementation and behavior:
[G02 guide](02_guide/06_G_curriculum.md),
[adaptive spawning reference](01_reference/08_adaptive_target_spawning.md),
[relay-budget integration](../max_geo_distance/integration.md).
