# Relay estimator modes and resolution

The target feasibility filter has two implementations:

```yaml
training:
  dynamic_max_geo_distance:
    enabled: true
    mode: global_candidates # simplifier | global_candidates
    candidate_spacing_fraction: 1.0
    max_candidates_per_map: 4096
```

`simplifier` is the default when mode is omitted. It uses the existing
construct-and-simplify code, record preparation, range margins and adaptive
sampler identity. Its numerical algorithm is unchanged. The resolution and
candidate-limit settings have no effect in this mode. Existing levels remain
on the simplifier; G02 explicitly selects `global_candidates` with fraction 0.5
(216 accepted random points on its 15 m cube, plus legal roadmap/launch anchors).

`global_candidates` uses a complete prepared candidate graph, with physical
radio LOS and separate body-clearance flight connectivity. A fixed map/base
has cached minimum-hop labels and parents, independent of agent budget. A
target query selects the cheapest existing finder with visual contact or
appends a legal, flight-reachable finder at the target. A legal target within
base radio range is a one-drone construction. Counts include the finder.

The acceptance test remains `certified and required <= geometric_agent_budget`.
The filter is a constructive geometric upper bound: it does not prove the
continuous minimum or successful execution by the policy, account for mission
time limits, or turn a discovery/delivery task into a different objective.

## Resolution

All legal existing roadmap positions are retained, including corner/opening
features and the launch position. Additional positions are continuous-uniform
random samples of legal free space, filtered by drone-body/planning clearance
and uniqueness. This is the sole sampler for `global_candidates`; there is no
grid option or fallback. Relay positions may occupy target noSpawn cells:
noSpawn restricts targets, not drone flight.

The retained setting `candidate_spacing_fraction` controls an exact density
budget, not a physical point spacing:
`random_count = ceil(world_volume / (cell_size_m * fraction)^3)`.
On a 15 x 15 x 15 m building with 5 m cells, fractions 1.0, 0.5 and 0.2 request
27, 216 and 3375 accepted random points, respectively, plus fixed legal anchors.
Larger buildings get proportionally more samples at the same fraction.

Sampling uses a fixed reproducible seed sequence `[456789, 104729]`, independently
of training RNG, targets, lanes, agent budget and worker order. For a given map,
denser pools retain every coarser pool's points as a shared sequence prefix.
Radio range constrains communication edges; it does not cap the sample budget.
Logs report accepted random and fixed-anchor counts. Finite sampling still
does not guarantee every continuously feasible target will be found.

The valid launch position is included. Body-inaccessible candidate positions
are discarded. Only flight-reachable positions can belong to a certified
radio chain; radio visibility through a narrow opening alone does not make
a disconnected room reachable by a drone.

`max_candidates_per_map` is a preparation guard, not a truncation rule. If
the pool exceeds it, preparation fails with the actual count and guidance to
increase the fraction or the explicit limit. Oversized budgets are rejected
before drawing points, rather than silently returning fewer samples. Finer resolution increases preparation
cost and query work; the experimental value of 1.0 is a starting setting,
not a demonstrated universal optimum.

## Maps, bases and caches

Immutable records belong to maps, not lanes. A 4,000-environment run reuses
records by map ID, including after adaptive map selection. Adaptive target
weights and agent-budget changes do not rebuild graph data.

With `randomize_base=true`, the map's radio graph and flight-component labels
are cached. Base-specific flight reachability, radio attachments and BFS
run in JAX for each base proposal. A legal launch position is added to that
query's graph. The fixed-base benchmark's speedup does not measure this path.

Derived records are saved separately from immutable geometry:

- Loaded random pool: `<pool>/relay_cache/global-candidates-v2-random/`.
- Authored maps and generated maps without a pool:
  `outputs/relay_cache/global-candidates-v2-random/`.

Pool workers populate the cache when generating maps with the global mode
enabled. Existing pools/new settings populate missing records at run
initialization. Cache keys cover geometry, roadmap coordinates, world bounds,
base/mode, clearance, ranges, resolution and implementation identity. Files
are replaced atomically. Changing radio settings rebuilds derived data without
regenerating buildings. Records are padded and uploaded once, with integer
hop/parent arrays and boolean adjacency preserved. Fixed-base preparation uses
the same lazy visibility/radio BFS as the fraction benchmark, generating only
the edges it needs and storing reachability/hop/parent arrays. Randomized-base
records retain radio adjacency and flight-component labels for JAX searches.
The version and sampler identity prevent reuse of old grid caches; their first
replacement is rebuilt and persisted automatically.

Current restrictions remain: relay filtering requires a target task with
static geometry and positive communication/visual ranges. Coverage tasks and
per-reset generated obstacles are excluded. Adaptive target spawning retains
its existing fixed-base restrictions.

## Checkpoints and migration

Keep `mode: simplifier` to retain the old estimator and sampler identity.
Switching mode, changing global resolution, or changing the global estimator
implementation changes target support. Such a checkpoint must use
`training.ckpt_loading_mode=branch` and a new run name. Weights are retained;
adaptive target support, map/cell statistics and progress gates start fresh.
An incompatible `resume` raises an explicit error. New global identities are
also included in adaptive controller and checkpoint metadata.

Training and filtered evaluation use the same selected mode and resolution.
Evaluation can still disable the filter through its existing boolean override.

## GPU experiments

From the repository root in the WSL training environment:

```bash
uv run python outputs/g02_map04_spawn_investigation/benchmark_resolution.py --targets 4000
uv run python outputs/g02_map04_spawn_investigation/benchmark_resolution.py --targets 10000
```

The test runs **only the original saved G02 map 04**, with fixed base and
uniform legal target positions respecting noSpawn regions. The 4k target
set is the prefix of the 10k set. Every method receives identical positions.

The experiment uses the production continuous-uniform sampler and fixed-base
host BFS helpers directly. Legal roadmap and launch positions remain
as fixed anchors. Fractions come from the editable `FRACTIONS` tuple in the
script and are displayed separately without grouping or renaming.

For this random sampler, a fraction defines a density budget rather than
physical grid spacing: `random_count = ceil(world_volume / (cell_size * fraction)^3)`.
Exactly that many unique random points are accepted after clearance checks,
in addition to the fixed anchors. Smaller fractions receive longer prefixes
of the same seeded sequence, so dense pools retain all coarser candidates.
Radio range constrains communication edges; it does not cap the random sample
budget. Very small fractions can therefore request large, expensive pools.
Candidate generation is independent of target generation and target count.

The prepared records use the production global-candidate JAX query. Host graph
search generates visibility edges as needed instead of storing a quadratic
adjacency matrix. The benchmark passes these records through the calculator's
map-bank interface, so it does not regenerate the pool or apply the
production `max_candidates_per_map` guard to experimental density sweeps. Saved
summaries include the sampling seed, budget rule and per-fraction random budgets.
Console output remains one target-count table: agent counts
as rows and requested fractions as columns; the simplifier is not run. Unresolved or
invalid cases get a row only if present. Witnesses are physically checked.
Batch size remains 32. Below the distribution, `Candidates` and `Error`
appear first, followed by a separator and the timing rows:

- `Error`: sum of absolute category-count differences from the smallest requested fraction, used as the reference regardless of column order: `sum(abs(count[k] - reference_count[k]))`. This is an integer histogram difference, not a per-target mismatch percentage. Moving one target between two categories contributes two. Unresolved outcomes are included as a category; invalid witnesses still fail the run. Per-category differences are also saved in the summary.
- `Prep (s)`: cold host graph preparation/cache writing plus synchronized GPU upload and calculator setup for that fraction.
- `Compile(s)`: query compilation for that fraction.
- `Warmup (s)`: its first query execution, excluded from the warmed timing.
- `Time (s)`: one warmed GPU count pass over all sampled targets.
- `Check (s)`: separate witness reconstruction/validation, including compilation; benchmark-only work.
- `Total (s)`: all fraction-specific work, including preparation, query passes, validation and result transfer. Shared work and final result-file saving are separate.
- `Time (%)`: total fraction-specific time divided by the reference's total time, times 100. The reference is 100%; lower values mean less total time. This replaces the former query-only `Cost (%)` row and follows `Total (s)`.

Shared preparation (including JAX initialization, saved-map loading, target
generation/upload and shared random candidate sampling), other/saving time,
and wall total are printed once below the table. Wall timing starts at entry
to `main` and ends before the final table/summary write. The sum of shared,
other and fraction totals accounts for that interval. Sampling is shared
across fractions, so its cost is not charged to each fraction separately.
Cold preparation, compilation and warmup are startup costs; warmed query time
is the recurring estimator cost. Production can reuse cached graph records,
and actual training resets also include sampling, retries and map selection.

No recommendations, map-suite runs or smoke
tests are printed or run by this distribution test. Numeric records and the
console table are saved under
`outputs/g02_map04_spawn_investigation/invest_results/<timestamp>_map04_distribution_<N>/`.
