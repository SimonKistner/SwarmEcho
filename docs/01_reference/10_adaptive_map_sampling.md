# Adaptive map sampling

An independent, default-off training feature selects a building for each new
episode. It works with authored static suites, saved random-building banks and
buildings generated once at startup. Buildings retain stable IDs; they are not
regenerated at each episode. G02 explicitly enables it.

## Configuration

All parameters are grouped under `training.adaptive_map_sampling`:

```yaml
adaptive_map_sampling:
  enabled: false
  min_probability_fraction: 0.30
  success_window_episodes_per_map: 500
  probability_smoothing_half_life_updates: 3.0
  eligibility_samples_per_map: 800
  held_out_maps: 10
```

These are also the config dataclass defaults; G02 uses the same values with
`enabled: true`. The enabled toggle must be boolean. When false, other known
parameters are neither validated nor used, no map controller/journal is created,
and the block is omitted from saved config snapshots. The original fixed lane
assignment, reset key handling, evaluation selection and adaptive-cell behavior
are retained. Misspelled/unknown config keys remain errors.

Start a new G02 run with `uv run swarmecho-train level=G02_static
logging.run_name=G02_adaptive_maps`. To retain the old fixed map allocation, add
`training.adaptive_map_sampling.enabled=false`; the remaining map-sampling
parameters then have no effect. Use a new run name or explicit branch when
changing the feature state on an existing model.

With the feature enabled, window/sample/evaluation counts must be positive;
the floor must be in [0,1], and smoothing
half-life must be positive and finite. Use evaluation-based early exit: the old
training gate assumes fixed map ownership of lanes.

## Success windows, warmup and probabilities

Every map has a separate latest-500-completed-episodes success/failure window.
Outcomes are recorded against the **terminal map ID**, before a lane switches
maps. Packed byte ring buffers retain chronological order, including multiple
completions for the same map in one rollout. Fast maps cannot evict slow maps'
measurements. A partly filled window uses its available outcomes.

Startup live lanes use the original balanced assignment where possible. Until
**every** map has filled one complete success window (500 completed episodes by
default), new assignments favor maps with
underfilled warmup buckets, accounting for episodes currently in flight. If all
remaining deficits are already covered by live episodes, assignments use the
remaining completion deficits. The floor applies to failure-based sampling
after warmup; warmup may temporarily assign zero probability to filled buckets.
There is no additional forgetting detector or periodic refresh mechanism.

After warmup, let `b_i=1/M`, `s_i` be window success, `d_i=1-s_i`, and
`m=max(d_i)`. For at least one imperfect map:

```text
q_i = b_i*d_i / sum_j(b_j*d_j)
p_desired_i = f*b_i + (1-f)*((1-m)*b_i + m*q_i)
blend = 1 - 2**(-1/h)
p_next = (1-blend)*p_applied + blend*p_desired
```

This is the adaptive-cell rule with a uniform map baseline. When every map is
perfect, desired probabilities return to uniform. Equal success across maps also
produces uniform probabilities. Updates occur once per PPO iteration, using
completed episodes from that rollout. For `f=.30`, the lower bound is 6% per map
with five maps and 0.0075% with 4000 maps. The controller reserves 30% collectively
for uniform coverage; it does not give each map 30% or reserve permanent lanes.

Map probabilities allocate **new episodes**. Expected simulation-step shares
also depend on episode lengths: in a stationary regime they are proportional
to probability times mean episode length. Long struggling episodes naturally
occupy more lanes. Their success estimate updates only when they finish.

## Episode-boundary reset and JAX

The device receives one fixed-shape map CDF. On episode completion, a separate
random key selects the next map through a CDF search. Reset fetches that map's
base, solids, roadmap, coverage mask and target distribution. Sparse resets
refresh geometry/map ID fields that were invariant in the disabled path.
Actor/critic recurrent state and communication/base memory reset through the
existing episode-done masks. No active episode is interrupted.

Only values and indices change, not lane count or map-bank array shapes. Changing
probabilities does not request a new JIT shape. Geometry preparation, relay
rejection and large-bank throughput still require GPU profiling.

## Interaction with adaptive cell spawning

The two toggles are independent. Map selection decides **which building** to
sample; adaptive cells decide **where inside that building** to propose a target.
Both use distinct windows and floors. Enabling map sampling permits the existing
adaptive-cell feature on persistent random maps, with fixed bases and its other
existing restrictions still enforced.

For relay-filtered adaptive cells, support discovery uses exactly
`eligibility_samples_per_map` accepted uniform targets per map, independent of
live lane count. It reuses startup targets when enough are available; otherwise
it performs bounded grouped reset probes. With the default 800 samples, cells
need eight accepted targets to pass the existing 1% mask rule. This remains a
frequency cutoff, not a proof that all excluded cells are impossible. Exhausted
proposal attempts fail explicitly, naming the map; there is no unfiltered
fallback and warmup does not silently wait forever on a failed discovery.

Cell windows use packed storage in the enabled path, and completed outcomes are
grouped by map before updating cells. Up to 32 maps retain ordinary per-map
dashboard journals. Larger banks use one bounded history with eight sampled
maps per update, capped console warnings and complete checkpoint state. The
small-suite dashboard does not display this large-bank aggregate journal.
Existing per-cell plateau histories still have host storage/CPU costs that
must be profiled at 4k.

## Stable banks, checkpoints and evaluation

Random training maps are initially selected/generated using the configured lane
count. Their count and ordered identities are then frozen independently of later
lane reductions. `adaptive_maps/selection.json` persists the bank choice or
generation count, and is copied into checkpoints. Later curriculum stages reuse
the selection through the incoming checkpoint. OOM recovery can therefore retain
4000 maps with fewer than 4000 live lanes. Missing/corrupt frozen banks fail
instead of silently replacing the dataset.

`adaptive_maps_state.npz` stores packed windows, counters, probabilities and
warmup progress. It is published with the model and adaptive-cell state before
the checkpoint readiness marker. Resume checks ordered map geometry identities,
stage identity and enabled configuration. New agent-budget stages branch into
fresh windows/warmup. Older checkpoints without map state start fresh warmup
with an explicit warning. Use a new run name when enabling this feature on an
existing model; `ckpt_loading_mode=branch` keeps weights and accumulated steps.
The former `warmup_episodes_per_map` setting is ignored when loading legacy
configs/checkpoints. Saved outcomes are retained, but a resumed partial window
must finish filling on every map before adaptation resumes.

Random-map runs use `held_out_maps` fixed evaluation buildings, default ten.
Saved-bank selection uses disjoint training/evaluation IDs; the generated path
uses separate evaluation seeds. Authored G02 continues to evaluate all five
authored maps. Evaluation is unweighted and never uses adaptive map/cell
probabilities. Its per-map episode budget remains an evaluation setting.

## Reporting and verification

W&B receives:

- `train/map_balanced_success_rate`: mean map-window success over observed maps;
  watch observed/warmup fractions until all maps have measurements.
- `train/worst_map_success_rate` and `adaptive_maps/ready`, `warmup_fraction`,
  `observed_fraction`, `min_probability`, `max_probability`.
- For up to 32 maps, `adaptive_maps/<map>/probability`, `success_rate`, cumulative
  `step_share` and `episode_start_share`.

The existing pooled `train/success_rate` is preserved for comparison. Map
metadata and append-only history live in `adaptive_maps/`; resume sessions mark
history rollback boundaries. Large banks log summaries and the ten most probable
maps rather than thousands of W&B series. Checkpoints retain every map.

CPU checks passed for inactive-parameter isolation, enabled validation, packed
ring ordering, warmup, floors, all-success normalization, compact-cell equivalence,
map/cell checkpoint roundtrips, journal resume, map-order rejection and a
4000-map controller with 2 MB of outcome bytes. No compilation check, WSL/JAX
reset run or GPU training benchmark was performed from this Windows session.
First verify a small G02 run in the project environment, checking map switches,
reset memory, both cell-toggle combinations and checkpoint/OOM continuation,
before a large-bank run. See [road to 4k](../road-to-4k.md).
