# Idea Archive

This file preserves the motivation and outcome of removed experimental features
without keeping their implementation in the active project.

## Pre-covered base communication area

The environment could initially mark cells within base-station communication
range as covered. The idea was that agents might learn to spread out more
robustly when the area around the base no longer produced new-coverage
information.

The feature was removed because enabling it did not make agents learn to spread
faster. Coverage observations were later removed altogether, which also removed
the main reason for keeping this special reset behavior.

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
including a special closest-to-corners success selection. A CSV loader and the
failure-clustering pipeline could replay selected failed target coordinates.

These paths were removed because they duplicated evaluation semantics, coupled
metric calculation to video selection, and selected coarse first-arriving
examples rather than the most informative episodes. Parallel evaluation is now
the only metric path, while an evaluation video is one independently collected
episode. A future CSV-ranked renderer is recorded in `future_features.md`.

## Duplicate early-stopping controls

Training previously had separate recent-training-window, parallel-evaluation,
and curriculum success thresholds, plus configurable `success` versus
`target_found` metrics and `train` versus `eval` modes.

They were consolidated because multiple stopping authorities could disagree
about when the same run or curriculum level was complete. The maintained rule
is one optional early exit based on parallel evaluation success. Reaching it
saves a handoff checkpoint and returns from training.
