# Adaptive target spawning

Relay-filtered support uses the selected [relay estimator mode and resolution](11_global_relay_candidates.md).
The default `simplifier` retains the existing sampler identity; changing to
`global_candidates` requires fresh support discovery in a branched run.

This opt-in training curriculum allocates more target samples to cells with lower
recent mission success. It preserves continuous target positions, valid geometry
and evaluation's ordinary free-volume distribution. It is independent of the geo
probe/auxiliary-loss experiments and of adaptive swarm population control.

Implementation: [controller and local journal](../../src/swarmecho/training/adaptive_spawn.py),
[environment resets](../../src/swarmecho/env/environment.py),
[training integration](../../src/swarmecho/training/train.py), and the
[independent dashboard](../../src/swarmecho/analysis/adaptive_spawn_dashboard.py).

## Scope and disabled behavior

The implementation supports static authored 3D target maps, including small
multi-map suites, a fixed base per map, no generated obstacles and no
minimum-geodesic-separation filtering. Persistent random-map banks are supported
with either uniform or [adaptive map sampling](10_adaptive_map_sampling.md).
Stable map IDs and lane-independent eligibility discovery do not depend on
adaptive map weighting. The existing `eligibility_samples_per_map` field supplies
the per-map support-discovery budget in both modes.
Enabling it with unsupported settings fails explicitly, before training.
Coverage-only missions have no target and are unsupported.

`enabled=false` is the global default. The controller and journal are not created.
Training uses the original volume weights, random-key splits and initial/dense/
sparse reset paths. It adds no per-cell statistics, files or W&B metrics; disabled
configuration snapshots omit this new block. Evaluation never receives adaptive
weights, even when the training feature is enabled. `G01_static` explicitly enables
the feature and declares all its parameters; `G02_static` copies those settings
for five maps. Other levels inherit the disabled default.

When combined with the relay filter, eligibility discovery and sampler identity
use the geometric stage budget. Curriculum `additional_agents` adds deployable
slots without admitting harder targets or expanding the discovery distribution.
See [relay-budget integration](../../max_geo_distance/integration.md) for the
startup support mask and spare-agent stage semantics.

## Configuration

All controls live under `training.adaptive_target_spawn`:

```yaml
training:
  adaptive_target_spawn:
    enabled: false
    min_probability_fraction: 0.20
    success_window_episodes_per_cell: 500
    probability_smoothing_half_life_updates: 3.0
    plateau_warning_enabled: true
    plateau_warning_horizon_updates: 50
    plateau_warning_improvement_threshold_pp: 3.0
```

Boolean toggles are strict. Episode-window size and warning horizon are positive
integers. The minimum fraction is in `[0,1]`, the probability half-life is finite
and positive, and the improvement threshold is in `[0,100]` percentage points.
Three percentage points means a change from 40% to above 43%, not a relative 3%
increase of 40%.

```bash
uv run swarmecho-train level=G01_static training.adaptive_target_spawn.enabled=true logging.run_name=G01_adaptive_spawn
```

## Cell identities and baseline volume

Each spawnable XYZ building cell is its own category. There are no path-distance
categories, curriculum walls or changed exploration rewards. Cell ordering is
Manhattan distance from the fixed base cell, then ascending XYZ to break ties.
Distance is for display ordering only; it does not enter the probability rule.

The existing sampler partitions allowed space into disjoint boxes after target
exclusions, wall/floor clearance and buffered solid subtraction. For cell `i`,
let `V_i` be the sum of its boxes' volumes. Define:

```text
b_i = V_i / sum_j(V_j)
minimum_i = f * b_i
```

Here `f` is `min_probability_fraction`. Stair cells get the mass appropriate to
their actual remaining free space; there is no assumed universal 50% multiplier.
The number of boxes in a cell cannot increase its probability. Given assigned
cell mass `p_i`, a box `k` in that cell has mass `p_i * V_k / V_i`, followed by a
uniform continuous XYZ draw inside that box. Existing geometry validation and
fallback remain; the enabled fallback also uses adaptive weights.

With the current default target buffer, `B02c_static_map_noSpawn` has 27 interior
cells, four excluded cells, 23 spawnable cells and 51 free-volume boxes. Baseline
cell shares are approximately 1.28%–5.87%. The 20% minimum is therefore specific
to each cell, not a universal 0.87%.

## Measuring success

Every cell has a separate bounded window of its latest completed training
episodes, including failures. The default is 500 outcomes per cell. Its estimate
`s_i` is the fraction of successful missions in that window. While the window
fills, use available outcomes; an empty window initializes to zero. With no new
completion, retain the cell's existing window and estimate.

An outcome is counted once at episode completion, with its terminal target cell.
Episodes crossing rollout boundaries stay underway until they complete. The
controller's windows are independent of the existing general logging window,
whose current size is `training.num_envs`. Easy, fast episodes cannot evict
outcomes from a slower cell's window. Window smoothing is by completed episode
count, not by elapsed training steps; a rarely visited cell can retain an older
estimate.

## Desired distribution

Let `d_i = 1-s_i` and `m = max_i(d_i)`. For at least one imperfect cell:

```text
q_i = b_i*d_i / sum_j(b_j*d_j)
p_desired_i = f*b_i + (1-f)*((1-m)*b_i + m*q_i)
```

The minimum pool `f` stays distributed according to baseline volume. The fraction
`(1-f)*m` prioritizes failures, weighted by baseline volume. As the worst success
rate rises, `(1-f)*(1-m)` returns to ordinary volume-based sampling. Equal success
rates in every cell reproduce the baseline. If all cells reach 100%, set the
desired distribution directly to the baseline, avoiding division by zero.

For illustration only, assume 23 equal-volume cells and `f=0.20`. When 22 cells
are perfect and one remains imperfect:

| Remaining cell's success | Its desired probability | Each other cell |
| ---: | ---: | ---: |
| 0% | 80.87% | 0.87% |
| 50% | 42.61% | 2.61% |
| 90% | 12.00% | 4.00% |
| 100% | 4.35% | 4.35% |

These are equilibrium/desired values; actual probabilities approach them through
application smoothing. They are not G01's unequal-volume probabilities.

## Applying probability smoothing

Initialize `p_applied=b`. Once per PPO update, after collecting its completed
episode outcomes, use half-life `h` measured in PPO updates:

```text
blend = 1 - 2**(-1/h)
p_next = (1-blend)*p_applied + blend*p_desired
```

For `h=3`, move approximately 20.6% of the remaining difference per update; with a
constant desired distribution, half the initial difference remains after three
updates. This convex interpolation preserves total probability and each minimum.
It does not require another measurement EMA: the per-cell episode window is the
first smoothing stage, and this interpolation is the second.

One shared fixed-shape float32 box-weight array is passed as a dynamic argument
through the compiled rollout to initial/automatic resets. Array values change
between rollouts; sampler shape and compiled functions do not change. Each lane
has independent random keys. No Python callback or disk operation runs inside
the JAX scan. New weights govern targets sampled for new episodes; existing
episodes are not moved or reset when the distribution changes.

## Multiple static maps

Each map owns an independent instance of the same cell controller: cell IDs,
baseline volume, success windows, eligibility, probability normalization,
smoothing and plateau warnings never pool across buildings. A failing cell in
map A cannot take probability from map B. Lane IDs select the map;
the initial balanced assignment gives G02 800 lanes per map at 4000 lanes.
Enabled adaptive map sampling can change these IDs at episode boundaries.
Equal lane counts do not imply equal completed episodes, because episode lengths
can differ between maps.

Without adaptive map sampling, relay-budget filtering groups the accepted uniform startup batch by
map **before** its 1% eligibility threshold is applied. At 800 lanes per map,
the threshold is eight accepted targets in that map's cell, not 40 across the
whole suite. Discovery targets remain the initial episodes. A cell below this
sampled threshold stays excluded for that stage, even if some points in it are
feasible. The mask is refreshed at the next stage, as in the single-map case.

The device receives one padded `[map, spawn_box]` table. Resets index their map's
row, mask padding to zero, then retain ordinary geometric rejection. G02's table
is `[5, 63]`, only 1260 bytes in float32. No new per-physics-step computation or
separate training rollout per map is added. Probability updates remain on the
CPU once per PPO update. Evaluation uses ordinary legal-volume proposals and
the current relay budget, never this adaptive table.

Enabled adaptive map sampling uses a separate fixed discovery budget (default
800 targets per map), packed cell windows and grouped outcomes. It avoids the
one-startup-target-per-map support problem on a 4000-map bank. Larger banks use
bounded aggregate journals instead of thousands of per-map history writes;
their GPU reset and preparation costs remain unbenchmarked. See the
[map sampler](10_adaptive_map_sampling.md), [G curriculum guide](../02_guide/06_G_curriculum.md)
and [road to 4k](../road-to-4k.md).

## Plateau warnings

With horizon `H` and threshold `delta` in percentage points, compare each cell's
window success at interval start `s_i(t-H)` with every measured value during the
following `H` PPO updates. A warning requires:

1. Some cell has previously improved above the threshold in this run.
2. The candidate has new completed episodes during the interval.
3. The starting value has room for an improvement above the threshold.
4. No measured success in the interval exceeds the starting value by more than
   `delta/100`; reaching exactly the threshold still counts as stalled.
5. The same continuing plateau has not already been reported.

The output names each cell, starting/current/peak success, average **applied**
spawn probability across the interval and the number of new completed episodes.
A significant gain rearms that cell's warning. The warning does not alter the
sampler or halt training. It identifies a condition to investigate, not proof of
an impossible task. No maximum-probability cap beyond the baseline floor is added.

## Persistence, resume and visualization

`logging.adaptive_cell_diagnostics` defaults to `true`. Set it to `false` to
keep adaptive target selection and checkpoint state while skipping all cell
dashboard journals/metadata, per-update diagnostic rows and individual-cell
plateau reports. G03 disables it by default. There are no per-cell W&B series
in either mode. Startup prints one support total instead of one line per map
when detail is disabled. This does not reduce the discovery sample budget or
discard the outcome windows used by the sampler. Existing journal files remain
readable as historical data; no new rows are appended while detail is off.

With diagnostics enabled, the following journal behavior applies.

An enabled single-map run saves `adaptive_spawn/metadata.json` and
`adaptive_spawn/history.jsonl`. A suite saves one ordinary journal per map under
`adaptive_spawn_maps/map_0000/adaptive_spawn/`, `map_0001/adaptive_spawn/`, etc.
The dashboard discovers these as separate entries named `run / source_map`.
This preserves the existing cell charts without combining unrelated geometries.
One compact journal row is appended after each PPO update using outcomes already
transferred to the host. Writes are closed for immediate visibility, without a
forced disk `fsync` per update. Metadata and checkpoint state use atomic replacement.

Rows record probabilities actually applied during the selected rollout, window-
smoothed success measured at its end, window/completion counts and smoothed next-
rollout probabilities. This timing distinction matters: the outcome estimates
from a rollout cannot retroactively change the weights that sampled its targets.
An episode already underway may have been spawned under an earlier distribution.

Checkpoints save `adaptive_spawn_state.json`: bounded windows, next probabilities,
counts, warning history and sampler identity. Explicit `resume` requires matching
controller configuration and spawn geometry. An older checkpoint without state
prints a warning and starts a new baseline controller. `branch`/`init` start new
windows and should use a new run name. Resume session markers preserve the original
journal but remove its post-checkpoint portion from the dashboard's displayed
history before new records are added.

For suites, the same checkpoint filename contains all child states and their
ordered map names. The combined `adaptive_spawn/eligibility.json` also preserves
map order and each map's startup mask. Resume rejects reordered or changed maps;
the existing static-suite source hashes additionally protect geometry identity.
Single-map checkpoint formats and journal locations remain compatible.

```bash
uv run swarmecho-spawn-dashboard root=outputs port=8767 open=false
```

Open `http://127.0.0.1:8767`. The independent dashboard polls only appended data
every two seconds, tolerates partial final writes, and shows skipped corrupt
records. Its timeline/playback controls update aligned probability/success bars,
a large thin-line probability plot and three individual cell plots per row.
Colours and Manhattan ordering remain consistent. No per-cell W&B logging or
training-time video generation is used. See the
[dashboard guide](../02_guide/02_analysis_and_tools.md#adaptive-spawning-dashboard).

## Focused verification before a training run

```bash
uv run pytest tests/test_adaptive_spawn.py -q
```

This checks volume-derived baseline/floors, redistribution, independent episode
windows, probability half-life, checkpoint continuation, the original keyed
disabled sampler, dynamic JIT weight changes without retracing, and both dense
and sparse automatic-reset application. It trains no policy.

During implementation, separate CPU-only checks passed for the controller,
checkpoint continuation, plateau warnings and interrupted journal recovery.
The JAX integration tests require the project's WSL environment and have not been
executed from the Windows editing session. Browser validation is left to the user.
