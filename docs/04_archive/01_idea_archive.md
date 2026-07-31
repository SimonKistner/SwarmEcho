# Idea Archive

This file preserves the motivation and outcome of removed experimental features
without keeping their implementation in the active project.

## Pre-covered base communication area

The environment could initially mark cells within base-station communication
range as covered. The idea was that agents might learn to spread out more
robustly when the area around the base no longer produced new-coverage
information.

The feature was removed because enabling it did not make agents learn to spread
faster. The coverage probe was later disabled in the maintained 2D levels,
which removed the main reason for keeping this special reset behavior. The
probe implementation itself remains temporarily available as a learning
scaffold for early 3D development.

## Memory T-maze diagnostic suite

The memory testing suite introduced isolated T-maze levels for checking
recurrent actor memory and TarMAC communication. It used fixed per-agent target
and anti-target positions, masked observations, communication-transparent mesh
walls, and multi-target diagnostic metrics.

The suite was created when TarMAC was introduced. It provided useful insights
and assurance that memory communication worked as expected, so it fulfilled its
purpose. The diagnostic levels, terminal memory samples, and the online linear
probe that predicted target cells from saved base communication were removed
once the normal training levels became the maintained workflow.

Communication-transparent mesh walls were removed with the suite because their
only purpose was to separate movement occlusion from communication inside those
diagnostics. Normal levels use one wall model consistently for movement,
visibility, and communication.

## Optional base and target counts

The configuration exposed `num_bases` and `num_targets`, although the maintained
task uses exactly one base station and one target. Supporting zero or multiple
entities added branches and misleading configuration options without serving
the normal workflow.

The parameters were removed and the environment now assumes exactly one base
and one target.

## Runtime spawn-policy overrides

The environment previously allowed level configuration to choose random or
fixed base and drone spawns, and to replace map-defined target sampling with
ring or outside-base samplers. These controls were introduced for early
B-series curricula.

They were removed because spawn policy belongs to the map definition. Base,
drone, and target positions now always come from their respective map spawn
zones. Fixed positions remain expressible as zero-area zones, and rectangular
or circular target no-spawn areas are also map data.

## Adaptive target spawning

The adaptive spawn curriculum grouped valid target cells by path difficulty.
Its soft gate changed sampling probabilities according to recent success,
while its hard gate progressively opened categories and temporarily blocked
inactive regions with walls and exploration-reward masks.

The feature worked as designed, but it did not improve training speed. It was
removed so normal training uniformly samples the valid target positions defined
by the map.

## Architectural Streamlit map builder

The architectural map builder provided a Streamlit editor for rooms, hallways,
custom wall segments, and rectangular spawn zones.

It was removed because its output path no longer preserved the current target
exclusion fields and it duplicated map-authoring tooling. The browser-based
2D grid maze builder remains because it supports the maintained core maze
workflow and provides a useful reference for future 3D tooling.

## Legacy configuration compatibility aliases

Older runs could use `resume_update`, `video_freq`, `eval_episodes`,
`async_video`, and a shared renderer fallback instead of the maintained
checkpoint-loading, evaluation, and renderer behavior.

These aliases were removed because they duplicated current parameters, obscured
which settings actually controlled a run, and were no longer used by maintained
level configurations.

## Per-frame observation CSV logging

Evaluation videos could save every agent observation at every rendered step to
a companion CSV. The writer assigned semantic column names using the former
57-dimensional observation layout, including a fixed coverage block and radar
offset.

The feature was removed because the maintained observation layout is
configurable and currently 37-dimensional, so the exported labels were
structurally wrong. Raw per-frame observations were also unnecessary for the
maintained evaluation workflow. Compact evaluation outcome CSVs remain the
supported data export.

## User-controlled adjacency storage

The environment exposed a configuration switch that requested storage of the
direct communication adjacency matrix in every state. It was useful for early
diagnostics and connection-matrix video overlays.

The switch was removed because adjacency availability is an internal runtime
requirement, not a training choice. The maintained communication and
wall-aware reward-routing workflow now retains one direct matrix in the
current state. Reachability remains temporary, and evaluation reuses the
transferred matrix rather than storing or reconstructing another graph.

## Legacy A- and B-series levels

The A-series provided early open-field, warehouse, and complex-maze tasks. The
B-series provided two square relay curricula using the same map with slightly
different training seeds.

They were removed after the M-series became the maintained map and level
family. Their geometry files, generated visibility caches, snapshot, default
curriculum references, and documentation examples were removed with them so
the active repository no longer presents two competing curriculum systems.

## Global mean critic

The global mean critic encoded all agent observations, applied self-attention,
and pooled them into one shared team value. It was retained as an ablation
alternative to the per-agent agent-centric critic.

It was removed because no maintained configuration used it. Keeping it forced
the rollout buffer, GAE calculation, and PPO trainer to support both scalar and
per-agent value shapes without benefiting the maintained workflow.

## Matplotlib video renderer

The slow renderer used Matplotlib to produce high-resolution scientific-layout
rollout videos as an alternative to the OpenCV implementation.

It was removed because maintaining two video backends duplicated rendering
logic and threaded renderer selection through training, evaluation, previews,
and dashboards. The faster OpenCV renderer covers the maintained workflow with
substantially lower rendering time and memory overhead.

## Reward-routing experiments

The reward system could switch between target-side-only or two-front chain
progress, local or globally averaged rewards, locally retained exploration and
safety rewards with globally shared chain rewards, shortest-route or
whole-component chain credit, and one deterministic route or every equally
short route.

These switches supported reward-credit ablations, but newer maintained levels
converged on one behavior: progress counts both the base-connected and
target-connected fronts, the dynamic chain reward is local to agents on one
deterministically selected shortest route, and non-contributing agents receive
the maximum gap penalty. Already shared target-found and success terms remain
divided by swarm size. The alternatives were removed because they obscured the
actual maintained objective and made the reward, logging, and renderer carry
branches that were no longer being compared.

## Proximity reward shaping

Three optional shaping terms tried to guide motion directly: a penalty for
drones being near each other, a bonus for being near the base, and a bonus for
being near a known target.

The maintained levels set all three terms to zero. They were removed because
collision handling and the communication-chain objective already provide the
relevant constraints, while explicit distance shaping adds an intuition about
the desired policy that was not shown to improve the maintained training
workflow.

## Post-delivery return to target

An experimental task split the finder bonus between delivering target
information to the base and subsequently returning to the target. A dedicated
toy corridor level exposed this behavior for diagnosis.

The mechanic and toy level were removed because returning to the target after
delivery is not part of the maintained relay-chain objective. The diagnostic
had served its narrow testing purpose and had also become stale relative to
the normal level and map configuration.

## Training target-spawn heatmap

The training loop used to retain the target position of every completed rollout
episode and periodically plot their spatial distribution. It showed which
map-defined target cells happened to be sampled during recent training.

It was removed because target sampling is now uniformly determined by static
map spawn and exclusion zones. Outcome-linked parallel evaluation CSVs and
heatmaps provide more useful spatial diagnostics without carrying an extra
training-rollout collection and rendering path.

## Sequential and selectable metric evaluation

Evaluation could either run episodes sequentially or as one parallel JAX batch.
The sequential path also supported a first-come selective renderer: it simulated
up to a configured maximum and filled fixed success and failure video buckets,
including a special closest-to-corners success selection. A CSV loader could
replay selected failed target coordinates.

These paths were removed because they duplicated evaluation semantics, coupled
metric calculation to video selection, and selected coarse first-arriving
examples rather than the most informative episodes. Parallel evaluation is now
the only metric path, while an evaluation video is one independently collected
episode. A future CSV-ranked renderer is recorded in `../03_roadmap/02_future_features.md`.

## Duplicate early-stopping controls

Training previously had separate recent-training-window, parallel-evaluation,
and curriculum success thresholds, plus configurable `success` versus
`target_found` metrics and `train` versus `eval` modes.

They were consolidated because multiple stopping authorities could disagree
about when the same run or curriculum level was complete. The maintained rule
is one optional early exit based on parallel evaluation success. Reaching it
saves a handoff checkpoint and returns from training.

## Spatial failure clustering

Evaluation tooling could group failed target positions with either a
distance-based breadth-first connected-components algorithm or HDBSCAN. It
rendered colored failure clusters, outliers, and one representative target
position per cluster.

The feature was removed because it duplicated substantial analysis code and
was not useful enough to justify maintaining it during the 3D transition.
Evaluation CSVs and ordinary failure heatmaps remain. Spatial clustering may
be reconsidered after the 3D workflow is working, using the maintained CSV as
its input rather than coupling clustering to simulation.

## Legacy M-series map alternatives

The 2D curriculum accumulated overlapping maze and open-arena variants with
inconsistent names. This included a 216 m grid maze without maze-cell metadata,
an unused alternative tiny-maze topology, and an experimental open arena that
used the coverage-observation scaffold.

They were removed to leave one explicit progression from open space through
large, medium, and small mazes. The modern cell-aware mazes were retained
because the maintained discrete finder-path task depends on their cell
metadata; the normal core-validation level remains as a separate run profile.

## Heuristic VRAM safeguard

Configuration loading could estimate a supposedly safe minibatch count, warn,
or terminate the process based on fixed transitions-per-GiB constants. The
training banner used a similar constant to print an estimated allocation.

These checks were removed because memory use also depends on agent and
observation dimensions, network architecture, recurrent state, optimizer
state, and compiled execution details. The constants described one historical
setup rather than a generally valid 2D or future 3D limit. Minibatch count is
now explicit; real peak-memory measurements and CUDA out-of-memory errors are
the authority when tuning a new workload.

## Heatmap point-table CSVs

Heatmap generation previously wrote separate `*.points.csv` files containing
repeated aggregate rates for each plotted coordinate. They were a partial,
ambiguous view of the same evaluation batch and could drift from the heatmap
that generated them.

They were replaced with one always-written per-episode evaluation CSV. Each
row records target coordinates, realized target-to-base distance, and one
terminal stage: `not_found`, `visually_found`, `found_and_delivered`, or
`chain_success`. Heatmaps and the analysis dashboard now filter that canonical
record, leaving one inspectable source of truth for spatial outcomes.
