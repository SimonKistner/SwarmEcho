# Potential directions beyond the first prototype

These are ideas only; the current prototype scope does not authorize their
implementation. See REPORT_PLAN.md for the deliberately compact report.

## G03 CLU auxiliary policy training (implemented, opt-in)

Train a path head alongside a fresh G03 CLU policy, allowing a lambda-weighted
path loss to shape informed-agent GRUs during recurrent PPO replay. Reserve a
clean policy-only variant with the same two seeds in the 24-slot sweep. Vary
lambda and snapshots per agent across twelve configurations. Fresh
independent probes remain manual post-training suite runs. See the [feature design](../../../docs/legacy-ablation/03_roadmap/06_G03_CLU_path_auxiliary_loss.md)
for sampling, gradient flow and open scientific questions, and the
[execution guide](../../../ablations/cluster/path-probe/GEO_AUXILIARY.md) for the 24-run sweep.
This changes policy training and is separate from the implemented offline probe.

## Inspector: one agent's path belief over a full replay

Select a replay agent and a trained probe whose frozen-policy checkpoint identity
matches the replay. At every timestep the agent is alive, obtain its actual GRU
and run the stateless path probe. In a separate panel/popup, display a second,
independently rotatable building with cell intensity showing predicted membership
probability, synchronized to the main replay timeline. Include the reference path
and a threshold slider. Handle inactive periods and unsupported probe inputs
explicitly. Mark pre-information predictions as outside the informed-snapshot
training population. This requires per-step inference/GRU capture, not merely the
prototype's stride-and-cap report samples. No extra recurrent probe is needed.

## Recurrent probe / explicit belief memory

Explore a separate decoder that accumulates GRU/message history and elapsed time
into its own recurrent state. It answers what is recoverable from history rather
than the current policy GRU. Requires ordered sequence training, reset semantics,
episode splits and matched-capacity controls. Do not add it to the first prototype.

## Sequence representations and multiple reference routes

Compare occupancy masks with ordered cells/waypoints using identical episodes and
controlled decoder capacity. Preserve saved reference polylines for that purpose.
Explore alternative-route supervision with explicit semantics; avoid merging masks
and pretending their union is one valid traversable route.

Eventually replace the single shortest-route teacher with reward-inspired
supervision over multiple valid base-to-target chain routes, giving shorter or
more efficient routes higher preference without treating every alternative as
wrong. The representation, weighting and scoring are open research questions;
they are deferred and do not change the first auxiliary prototype. Resolving
missing sparse-roadmap attachments remains a separate label-coverage issue.

## Optional technical diagnostics

- Probability calibration: check whether, for example, cells assigned 80% are
  actually path cells about 80% of the time. Reliability plots or Brier scores
  could support this later, if needed.
- Cell-prevalence baseline: test how much a decoder improves on predicting only
  which relative cells are usually on paths in training, without reading the GRU.
- Precision/recall and origin-excluded diagnostics: investigate whether failures
  mainly add extra cells, miss path cells, or exploit the always-positive base.

These diagnostics may help research/debugging, but are not prototype user-facing
metrics. The first report exposes exact-path accuracy, average path similarity
and similarity distributions in plain language.
