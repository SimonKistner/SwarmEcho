# Path predictor: report and model design

The user confirmed prototype implementation. The path predictor, binary loss and
nested viewer are implemented in shared blocks; see README.md for runnable
settings and the exact prototype conventions. Recurrent/inspector extensions
remain deferred. This document retains the design rationale.

## Report structure and metrics

Reuse the current report document shell, navigation/comparison controls, run and
checkpoint metadata, dataset distributions, training curves and collapsible
required-agent/storey/map breakdowns. Add a mask-specific report block rather
than reusing categorical cell confusion matrices or cell-distance error bins.

The user confirmed that “target spawn clusters” means the **existing groups by
target-required-agent count**, not newly computed spatial clusters. This count
is the sampler's estimator metadata; do not relabel it as a proven global minimum.
Zero denotes missing/not-estimated metadata, not a zero-agent mission.

Keep the overview compact: one row per input variant, aggregated across all
detail groups, with **exact-path accuracy (%)**, **average path similarity (%)**
and a similarity histogram of imperfect predictions. Required-agent groups appear only after opening
that input row's existing Details disclosure, followed by a further collapsed
example viewer within the relevant group. Do not promote required-agent groups
to top-level report sections. Keep secondary counts/seed variation unobtrusive.

Exact-path accuracy is the percentage of predictions whose selected cells match
the target mask completely. Path similarity measures agreement on path cells,
penalizing both missing and extra cells; correctly empty space earns no credit.
Display this plain-language name, not IoU/F1/Dice terminology. For implementation
use symmetric overlap `2*shared/(predicted_count+target_count)` (Dice overlap),
with identical empty masks defined as 100%. The reference path includes its base
and target cells and is normally nonempty. Include the full saved mask, without
adding a separate origin-excluded score to the prototype. This is cell overlap,
not a claim about route order, navigability or distance between disjoint paths.

Average each sample's similarity with the agreed snapshot → agent → episode
aggregation. The histogram uses **path similarity**, in percentage
points, for incorrect predictions only. Use simple fixed 10-point bins; show
percentages of incorrect predictions using the same episode/agent balancing.
Perfect predictions are represented by the exact-path percentage. Explain that
higher similarity further right means closer agreement. Use the same two
scores/histograms within detail groups, with numbered axes and clear bin labels.

Do not add precision/recall tables, IoU labels, Brier scores, prevalence baselines,
calibration plots or extra technical diagnostics to the first prototype. Potential
future uses are recorded in BACKLOG.md. Retain ordinary training/validation loss
curves in the existing training section, not as new headline evaluation metrics.

Compute metrics using snapshot → agent → episode means for each input's eligible
population. Default the main comparison to common episodes across every input
and seed. Offer all available episodes per input as an explicit alternative, with
availability counts, since missing events change each input's population. Provide
a report-wide threshold selector (50/75/80/90/95%, default 80%) that explicitly
recomputes all aggregate scores and histograms from saved probabilities. Do not
optimize thresholds or representatives on held-out performance. Keep training
curves independent of this display threshold and give them axes/legends, two per row.

## Nested sample viewers

Keep this hierarchy: input overview → Details → required-agent/storey/map group
→ closed-by-default `Representative prediction (3D)`. Add that disclosure inside
each relevant detail group, including each populated required-agent group.
The group retains its ordinary numeric results; the inner disclosure owns the
sample viewer. Reuse the same component in storey/map details where appropriate.
This is not a single global scene at the bottom of the report.

The viewer contains:

- Rotatable building geometry, base/target markers and the actual saved shortest
  route as a distinct polyline. Adapt the inspector's existing Plotly building,
  path, camera and coverage-mesh primitives; avoid embedding the whole inspector
  server/UI or recomputing a path when the disclosure opens.
- Coverage-style translucent cell cubes for predicted membership. Default to
  probability color/intensity with hover values; offer thresholded prediction and
  an error overlay labelled “correctly selected,” “extra,” and “missed” cells.
  Keep the reference route distinguishable from every overlay.
- Original building coordinates only, with the inspector's upright turntable
  mouse/touch controls and matching wall strips, floor tiles, outlines and lighting.
  Use the saved inverse transform so all 125 prediction positions,
  including predictions outside the actual building, are aligned correctly.
  Do not silently remove invalid-cell predictions using privileged geometry.
- Input selector for all five variants. Keep the same episode, reference route
  and camera while switching. Missing input samples show an explicit unavailable
  state rather than silently switching to another episode.
- Agent selector for the all-knowing groups and a snapshot/time-since-information
  selector for the repeated group. Show agent ID, capture step, information age,
  decoder seed and the selected example's scores. Finder/base events can occur
  at different times; expose that difference instead of implying matched timing.
- Include a clearly labelled **probability threshold slider** (0–100%) and its
  current value. Moving it immediately changes selected prediction cells and
  the example's exact-match/similarity indicators, while keeping the true route
  visible. Show the rule “select cells at or above this probability.” Provide
  reset-to-report-threshold. This local slider does not change the aggregate
  overview/histograms; the explicit held-out selector controls those together.
- Optional controls for floors/wall transparency, predicted-cell opacity,
  reference path and ground-truth cell mask.

Use one deterministic held-out representative per group, selected using saved
IDs/geometry independently of model predictions. Freeze the episode choice and
reuse it across inputs/seeds. A representative is illustrative; group scores
remain aggregate results over all eligible test episodes. Display unavailable
or empty groups explicitly. Default the seed selector to one recorded seed;
do not present an average of seeds as if it were a single fitted decoder.

Embed bounded example data (geometry reused by map, route once by episode,
predictions per selected input/agent/time/seed) and the JS dependency once per
report, retaining offline standalone HTML use. Instantiate a scene only when its
inner disclosure opens, resize after visibility changes, and release/reuse WebGL
resources when collapsed. Many closed groups must not allocate many scenes.

## Loss and probability interpretation

Existing location classification uses softmax cross-entropy over mutually
exclusive cells. For path membership, use 125 logits, sigmoid probabilities and
numerically stable binary cross-entropy with logits. Multiple cells can be on the
path, so a single softmax over 125 cells does not express this target.

This is already probabilistic: each output estimates cell membership conditional
on the provided frozen-policy feature. The 125 marginal probabilities are not a
normalized distribution over complete valid routes, and a thresholded mask may
be disconnected or branched. Label uncertainty also differs from uncertainty
over all physically admissible alternative routes: version one labels one
canonical shortest reference route.

Propose ordinary per-cell BCE for the initial baseline, with the agreed
episode/agent sampling. Keep any positive-cell
class weighting as an explicit ablation/config choice: it changes the probability
optimum and the precision/recall tradeoff. It is different from balancing episode
or agent contributions. Avoid adopting Dice/focal mixtures before establishing
the probabilistic baseline. Calibration and other technical diagnostics are
potential later investigations, not prototype report sections. Any later fitted
calibration uses validation only. The chosen similarity score gives no credit for
correctly empty cells, avoiding a score dominated by the large zero-padded region.

## Mask versus sequence: interpretation and future comparison

There is no demonstrated universal superiority of the mask. It is the proposed
first diagnostic because the question is which route cells can be decoded from
one GRU/message, with a small fixed output and no arbitrary output ordering.
It shares a useful representation family with the planned own-cell predictor.

A direction/waypoint sequence represents traversal order and can model coupled,
multimodal route choices. It introduces variable lengths, stop tokens, ordering
and corner-discretization conventions, plus autoregressive decoding errors and
a more powerful decoder. Those tradeoffs require an empirical comparison on the
same episode splits and controlled decoder capacities. Neither an unconstrained
mask nor an unconstrained token sequence guarantees physical route validity.

Persist the reference polyline/transform alongside labels now so a future
sequence probe can reuse exactly the same episodes. Do not implement it in the
first prototype or imply that good cell overlap proves a navigable route.

## Belief updates over an agent's lifetime

The prototype report viewer selects among already retained snapshots; it does
not provide full-episode playback. A later inspector feature could run the same
stateless trained probe at every alive timestep for a selected replay agent and
show its changing probabilities in a second rotatable building view. This is
recorded separately in BACKLOG.md and is outside the prototype scope.

A future recurrent decoder could maintain a separate state from the sequence of
GRUs/messages and elapsed times. It could accumulate or retain evidence that is
no longer recoverable from the current policy GRU alone. It therefore measures
information available across feature history, a different question from the
single-snapshot probe. It also requires ordered sequence training, temporal masks,
clear reset semantics and controlled capacity comparisons. Keep it a separate
later experiment; do not add its memory to the first prototype. Both this idea
and full-episode inspector belief playback are deferred, not implementation work
authorized by the current report refinement.
