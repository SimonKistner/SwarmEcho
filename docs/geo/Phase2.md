# Phase 2 — long-term ideas and open questions

Status: **unscoped backlog**, 2026-10-13. These ideas preserve the discussion;
they are not approved experiments or features and have no fixed implementation
schedule. [Phase 0](Phase0.md) is the original frozen baseline; [Phase 1](Phase1.md)
records the visual baseline, scoped own-cell diagnostic and proposed static
target-cell auxiliary intervention.

## Intended longer-term progression

The research vision discussed so far is:

```text
static baseline [done]
  → static target-cell auxiliary supervision [Phase 1 proposal]
  → continuous exact target-location estimation
  → compare learning speed and attainable skill
  → repeat with freshly trained checkpoints, with/without loss, on five random maps
  → design map encoding/decoding, ideally integrating target position
  → investigate own-position representation, possibly with a separate head
```

The steps after cell supervision are retained here as future ideas, rather than
being bundled into Phase 1. Progress depends on what the audit and paired static
experiments actually show. On 2026-10-14 the user scoped the detached static
**own-cell diagnostic** earlier; its protocol is recorded in [Phase 1](Phase1.md#own-cell-diagnostic-scoped-2026-10-14).
Own-position auxiliary supervision and generalization remain future ideas here.

## Position precision beyond cells

| Idea | Question | Candidate first design / unresolved choice |
|---|---|---|
| Continuous target location relative to base | Can supervision retain within-cell precision, and does it help coordination? | Compare a simple XYZ regression control with a cell distribution plus within-cell offset/uncertainty. No representation selected. |
| Uncertainty rather than a single point | How should indistinguishable positions be represented? | Coordinate distributions or mixtures may represent alternatives; a mean under MSE can land between valid locations, including inside walls. Decide proper scoring and spatial metrics. |
| Learning speed / skill ceiling | Does spatial supervision help learn faster or solve harder cases? | Paired fresh training, fixed budgets and per-target outcomes; final success alone has little headroom on the present map. |
| Later target snapshots | Does precision emerge after additional observations? | Explicitly compare first sighting, first outgoing packet, later informed states and base receipt. Keep original Phase 0 timing as its baseline. |

The frozen-vector mean-coordinate residuals in Phase 0 are a reference for the
present snapshots, not a fundamental limit after training. Binary visibility
without bearing/range may leave genuine uncertainty. Auxiliary supervision cannot
make indistinguishable histories identify an exact target coordinate, although a
changed policy might gather more informative observations.

## Five maps and generalization

- Preserve the idea of **five randomly generated layouts** and freshly trained
  checkpoints with and without the same losses. First decide whether this means
  five separate static-map comparisons or one policy trained on a five-map bank.
- Five separate tasks test repeatability across layouts. Joint training on five
  fixed maps can still permit memorization; it does not establish performance in
  a genuinely unknown building.
- For an unknown-map claim, separate training and held-out layouts/generator
  seeds. Consider a larger bank or procedural stream later, only if necessary.
- Preserve fixed base/world-axis label semantics where possible. Random base
  positions, orientations, starts and variable building extents need explicit
  reference-frame and label contracts; they are separate changes.
- Broader sampling or action-noise controls might produce more distinct internal
  states than noise-free static evaluation. Do not silently change the baseline
  protocol or assume increasing episode count alone will do this.

## Building representation and shared spatial memory

The initial question remains open: **how should geometry be represented so that
a head can reconstruct it from the latent of a drone which found and delivered
the target?** Related variants are reconstruction from the finder packet, a base
network predicting the scene or base-to-target region, and aggregation of agents'
local estimates at the base.

| Design question | Ideas retained, without selection |
|---|---|
| Unknown extent and dimensions | A base-centred open/growing representation instead of assuming a fixed building-size output cube; sparse cells/blocks, hierarchical grids or a coordinate-query field are candidates. |
| What geometry is predicted? | Occupancy/free space, walls, floors, stairs and openings; the 27 room-cell labels alone do not encode traversability or detailed geometry. Full building versus locally observed area versus base-to-target region remains open. |
| Where does the belief live? | Implicit GRU state, explicitly stored per-agent spatial estimate, base-aggregated memory, or a shared reconstruction. Feeding explicit estimates back to agents would be a separate architectural stage. |
| What is learned from observation versus prior? | Mark observed/unknown regions. Supervision of unseen geometry can teach layout priors, but cannot make uniquely unobservable details known. |
| Unified target and self-location | Put occupancy and target/self probability in the same base-relative coordinate space, with uncertainty and temporal updates; decide separate heads versus a shared representation. |
| Message role | Communicate local map evidence or latent features; distinguish individual outgoing packets, actually stored base packets and aggregated attention/context. |
| ML-assisted local/shared SLAM | Investigate spatial memory and joint localization/mapping rather than assuming a decoded map is already correct. |

Relevant architectural references to investigate include
[Neural Map](https://arxiv.org/abs/1702.08360),
[Active Neural SLAM](https://arxiv.org/abs/2004.05155), and
[Occupancy Networks](https://arxiv.org/abs/1812.03828). They offer different memory,
mapping and coordinate-query ideas; none is a selected drop-in solution for this
3D drone communication task. A static-map decoder might merely output a constant
building template. Varied and held-out layouts are needed to test actual mapping.
There is no scientifically necessary “90% target-cell accuracy first” threshold;
the reason to defer geometry is scope and interpretability.

## Evaluating positions when the predicted building is wrong

This was an important open question: if the predicted layout is incorrect, how
can a target or own position within it be rewarded or penalized?

Proposed direction: use a common base-relative metric coordinate frame for
geometry and locations. Score target/self location against the **true position**
and score the geometry separately against the **true building**, then report their
consistency. An incorrect predicted topology does not redefine ground truth.

- Continuous position can use metric-coordinate error and, for probabilistic
  predictions, a proper distribution score. Discrete predictions can use the true
  map's cell graph when they correspond to defined cells.
- Report probability or point estimates inside true solids/invalid cells; do not
  silently snap them to a valid location before scoring.
- If a predicted route is blocked in the true map, distinguish invalidity from
  endpoint error. If a geodesic is undefined, specify its handling rather than
  treating it as a convenient finite distance.
- Separate localization and mapping errors. Oracle true-pose/true-map diagnostic
  controls could isolate which component fails. Learned-map coordinates without
  a shared frame would need an alignment/correspondence protocol first.
- A soft consistency loss could later encourage agreement between predicted free
  space and location beliefs. Hard-masking targets with an incorrect predicted
  map could suppress the true position; do not make that the default.

Mapping and localization can be jointly estimated rather than requiring a fully
correct map before any position estimate. Velocity/history can provide some
position cues independently of a completed map. The exact losses, uncertainty
representation and alignment rules remain unchosen.

## Own-position probes and heads

- Own-cell and own-storey prediction are **separate tasks**, both distinct from
  target-cell and target-storey prediction.
- Include **all drones equally**; finder or reporter role is not the own-position
  label definition. Proposed sampling was every **30 steps**, with current GRU
  and communicated packet as inputs and that drone's current true cell as label.
- Define lifetime/reset masking, episode splits and weighting so long episodes
  or particular agents do not dominate; avoid treating temporally adjacent
  snapshots as independent episodes.
- First detached diagnostics, then consider an auxiliary head if useful. A
  separate head versus unified spatial belief remains an architectural choice.

## Base-to-target paths

Retain path reconstruction as a future target-relative diagnostic. A predicted
representation must determine an endpoint/path from the base without ambiguity.
Direction tokens in fixed building axes, e.g. +X, -Y, +Z, are one candidate; “left”
cannot depend on whichever heading the drone happens to have.

The user's accepted evaluation idea is **not exact agreement with one canonical
shortest route**: follow the predicted route, check whether it is valid in the
provided true map and ends at the target, and measure valid graph distance from
its endpoint to the true target when wrong. Distinct valid routes can be correct.
Position offset specifies an endpoint but not obstacle-avoiding connectivity.

If variable-length token prediction is used, distinguish EOS/STOP termination
from padding: stop at the first EOS, mask unused padded positions in the loss,
and do not execute padding as actions. Maximum length, loops, invalid moves and
teacher/training loss remain open. A canonical route could be an optional training
teacher, but it is not a required evaluation label or chosen implementation.

The earlier simplification suggestion meant removing **the entire door/window
wall segment into free space**, retaining other walls/stairs. Simply deleting a
door/window annotation while retaining its wall would close the opening and is
the opposite change. No simplification was made for the current G01 baseline;
any such map must be a separately named geometry/control with its own evaluation.

## Replay inspector and temporal beliefs

Ideas retained for later inspection:

- A selectable drone's decoded spatial belief evolves alongside the replay; a
  base view could show aggregation over time. A full map reconstruction requires
  a trained geometry decoder, which Phase 0 does not have.
- For target beliefs, the preferred idea evolved to a probability histogram with
  the true target cell marked, ordered spatially so nearby cells tend to be nearby.
  A per-storey XY landscape with probability as height could preserve more
  neighbourhood structure; flattening 3D to one order cannot preserve every
  adjacency. Decide layout only when this is scoped.
- Show valid predicted paths and endpoints against a supplied map where relevant.
  Clearly distinguish supplied geometry from inferred geometry.
- Cell colouring/3D heatmap overlays were **scrapped for the near-term report**.
  Probability-example sections were also removed from that report. Do not add
  them back automatically because they remain in this idea backlog.
- A softmax output is a model probability, while the current error histogram is
  the frequency of incorrect outcomes. They are different quantities; calibration
  is needed before reading a decoder distribution as a drone's literal belief.

## Other conditional follow-ups

- Direct outgoing-message or base-message auxiliary losses if a GRU gain does
  not reach communication; preserve the original detached, no-gradient second-net
  diagnostic as the baseline.
- A storey auxiliary control, though storey decoding is already near ceiling.
- Event-wise diagnostics including found-but-unsuccessful missions, to separate
  sensing, transmission and chain failure. Matched successful-only Phase 0 remains
  unchanged; a finding event itself does not imply eventual mission success.
- Readout tuning at the base if its optimization gap matters, or controlled head
  capacity comparisons if justified; no large default hyperparameter sweep.
- Geometry-aware target losses, dense supervision over later informed states,
  action-noise controls and changed sensing only as separately identified variants.

Before promoting an idea, specify its question, labels/reference frame, inputs,
timing, controls, evidence and limits using the
[record-keeping practice](Phase0.md#record-keeping-for-all-subsequent-geo-experiments).
Move a scoped design into the relevant phase record while retaining its history
here. Do not imply a backlog entry has already been run.
