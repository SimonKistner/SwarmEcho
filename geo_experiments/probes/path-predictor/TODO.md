# Path predictor: implementation contract

[`probe.yaml`](probe.yaml) is active. Acquisition, target construction, training
and reporting live in `shared/`; this folder stays declarative. The policy
remains frozen. See README.md for runnable settings and tested conventions.

## Agreed target representation

- Predict the base-to-target reference route as a binary cell-membership mask.
- Use a base-cell-centred `5×5×5` mask: 125 outputs, offsets `−2..+2` on each
  axis for a `3×3×3` building. Logical `(0,0,0)` is array index `[2,2,2]`.
- Make labels invariant to whole-scene horizontal rotations of 90°, 180° and
  270°, with geometry, base and target rotated together. Keep gravity/up fixed;
  do not fold reflections or vertical flips into the equivalence.
- Canonicalize base-anchored geometry and endpoints, with deterministic symmetry
  and equal-cost-route handling. Transform cells using exact integer quarter
  turns; handle boundary coordinates explicitly. Save the inverse transform and
  ordered cell offsets so reports can display masks in the original building.
- A global rigid transformation must preserve the label. Moving the base alone
  can legitimately change the relative route. The frozen GRU/predictions need
  not themselves be rotation-invariant.

## Route source and reward alignment

Current code does **not** save an ordered route per target in probe archives.
The collector saves endpoint coordinates, map ID and target-required-agent count.
`EnvState` retains roadmap geometry/distance matrices. Spawn acceptance may
construct a relay witness, but normally returns only count and validity. A relay
witness certifies feasibility; it is not a guarantee of globally shortest length.

Reuse existing geometry/planner code. Reuse an already persisted route when its
endpoint/geometry/planner/cost provenance matches the chosen reference and its
tie handling satisfies the rotation-invariance contract. Otherwise reconstruct
and cache one route per map/base/target/layout/request identity. Do not solve it
again for each input, agent or time snapshot.

The reward's geodesic reference uses ordinary `planning_distances`, via
`chain_distance_matrix`. The corner-penalized matrix is used in spawn selection;
it must not be silently substituted for the reward reference. Check reconstructed
route length against the reward reference within a defined numerical tolerance.

`chain_efficiency_reward` can credit multiple **live agent relay chains**. Their
efficiency is `clip(reference_length / chain_length, 0, 1)`; with redundancy
enabled each agent receives its best eligible chain efficiency. These are not a
stored set of alternative optimal roadmap polylines.

Version one proposes one shortest reference route and a hard bool mask. Leave
multi-route supervision for a separate version. A future extension can score
against the best admissible route or use explicitly weighted route membership;
unioning paths can imply a route that never existed. A cell mask alone also
does not specify traversal order, so reward-like route efficiency needs an
explicit path reconstruction/evaluation rule.

## Agreed input/readout groups

Each group gets a separate decoder; all rows for an episode use the same target.

| Input group | Snapshot rule |
|---|---|
| `finder_gru` | Existing updated finder GRU at the first communication tick after discovery |
| `finder_message` | Existing finder message at that same tick |
| `all_knowing_after_info` | Each active informed agent's GRU once, at the first actor update that has processed its target information |
| `all_knowing_till_end` | Each active informed agent's GRU at its first informed snapshot, then every configurable 10 environment steps until the episode ends or the agent becomes inactive |
| `base` | First current message stored by the base, potentially from a relay |

The two all-knowing groups are interpreted as **GRU** inputs. “All” includes the
finder. The periodic grid is relative to each agent's first informed snapshot,
not the episode's global step zero, and does not require a communication tick.
Verify the state/actor-update ordering before implementing first-information
capture, so a knowledge flag never selects a hidden state from before receipt.

Store episode, map, agent, capture step, time since information, event and input
group for each row. Share the episode split across every input group. Missing
base snapshots must not invalidate available GRU samples under the proposed
all-informed-episodes eligibility rule; record per-input availability separately.

## Agreed candidate selection and balanced training

At seven informed agents and 500 remaining environment steps, stride 10 yields
about 350 trajectory rows per episode, versus seven first-information rows.

- Keep `every_steps: 10` configurable: it determines the candidate grid relative
  to each agent's first informed snapshot. Keep a separate configurable cap of
  **20 retained trajectory snapshots per agent per episode**.
- Select at most that many candidates with indices evenly spread from the first
  to last eligible stride-grid snapshot. Keep both endpoints when at least two
  slots/candidates exist. Keep all candidates if fewer than the cap exist; never
  pad by duplicating rows. A cap of one keeps the first candidate. Save candidate
  and retained counts plus exact selected steps for reproducibility.
- The last eligible candidate can precede decommission/termination by up to one
  stride. The stride is a real acquisition constraint, while the cap controls
  retained data volume. Document both beside the actual implementation/config.
- Exact evenly spaced selection requires knowing the candidate count at the end.
  The retained-row cap alone does not bound capture working memory. Use bounded
  rollout batches/candidate buffers, or deterministic replay at selected times
  if needed. Do not silently replace this with an approximate random reservoir.
- Budget independent episodes/maps first. Configure per-readout row limits
  separately if needed; do not let temporal duplicates silently reduce map or
  episode coverage. Split by episode before any train/validation/test thinning.
- Train each input's decoder by uniformly selecting an eligible episode, then
  uniformly selecting one represented agent in that episode, then uniformly
  selecting one retained snapshot for that agent. For finder/base inputs, there
  is just one row per eligible episode. This gives each episode equal influence
  and each represented agent equal influence within the episode, even when the
  stored row counts differ. It does not force equal counts across input variants
  or duplicate data in the saved dataset.
- With `E` eligible episodes, `A_e` represented agents in episode `e`, and `K_ea`
  snapshots for agent `a`, each draw has probability `1/(E*A_e*K_ea)`. Draw with
  replacement for training; define/document a fixed update/sample budget because
  an epoch is no longer a single uniform pass through the stored rows.
- Evaluate by averaging snapshots per agent, then agents per episode, then
  episodes. Report episode/agent/row counts and time-since-information breakdowns.
  Input availability differs, so default to matched episodes and offer each
  input's full eligible-episode results as an explicit population choice.
  No episode crosses data partitions.
- Store each target mask/route once per episode and reference it from input rows.
  Never duplicate mask construction for the repeated feature rows.

## Prototype conventions

- Use one shortest route and informed samples from eventual failures as well as
  successes; delivery is not a prerequisite for GRUs.
- Include the base cell and target endpoint cell, rasterize segment interiors,
  and reject map extents above three cells per axis. Canonical coordinates use
  six decimal places; reference lengths are checked at 1e-4 m absolute / 1e-5
  relative tolerance. See path_geometry.py and README.md for tie conventions.
- Episode budgets and optimizer settings are in probe.yaml. The report uses exact-path
  accuracy, average path similarity (shared cells with missing/extra penalties),
  and similarity histograms of imperfect paths. Reports offer 50/75/80/90/95%
  cell thresholds (default 80%); no technical metric panels in version one.

The [report and model discussion plan](REPORT_PLAN.md) describes the proposed
nested 3D sample viewers, metrics, probabilistic output and deferred model ideas.

See [the shared architecture](../../CONCEPT.md). Full-episode inspector playback,
additional recurrence and alternative-route supervision remain backlog items.
