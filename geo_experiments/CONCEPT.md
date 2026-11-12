# Target-defined probes composed from shared blocks

## Definition boundary

The target determines the family. `target_location/` has three subprobes: shared
coordinates, per-map decoder, and joint map-and-cell. `own_cell/` predicts each
agent's position. `path-predictor/` predicts membership in an optimal roadmap path.
Changing finder GRU to finder message is an input choice, not a new target family.

Each `probe.yaml` declares these independent pieces:

| Piece | Responsibility |
|---|---|
| `target` | The ground-truth question, source state and label semantics |
| `inputs` | Named feature requests: representation, agent selection and capture event |
| `sampling` | Sample unit, backend and group used for train/validation/test splitting |
| `model` | Decoder architecture and feature normalization |
| `output` | Encoding, vocabulary/shape, loss and how inputs form readouts |
| `settings` | This probe's collection budgets, optimizer, diagnostics and report choices |
| `status` | `active`, `planned`, or `archived`; only active definitions can run |

The suite specifies checkpoints and repetitions. It contains no network,
input-selection, label, dataset-budget or rendering settings. The current runner
compiles definitions to the established archive layout so shared coordinates
keeps the same report and saved-model contracts.

## One acquisition service

`shared/inputs.py` resolves feature requests to collector capabilities.
`shared/rollout.py` runs the frozen policy and captures events; it is also reused
by archived workflows. `shared/collector.py` owns quotas, eligibility, metadata,
target construction, splits and saving. There is no collector in a probe folder.

The active `episode_events` backend aligns the requested finder/base events by
episode. `project_event_inputs` returns only requested features and validates only
the requested events. Delivery/success filtering remains independent of capture
completeness. The rollout currently tracks the established event state internally;
adding new requests does not require copying its evaluation loop into a probe.

The path backend emits one row **per selected agent at a sampled event**.
Selecting all agents does not concatenate a whole team into
one GRU vector. Store `episode_id`, `map_id`, `agent_id`, capture step and event with
each row. Agent-event rows must remain grouped by episode when splitting, even
when many agents and steps produce samples. Different sampling/feature layouts
use distinct request identities and collections.

The acquisition boundary is named feature tensors plus row identity/metadata and
ground-truth state. Feature dimensions are discovered and saved with the data.
A future structured/ragged input needs an explicit layout and model adapter;
the vector MLP must not silently flatten or truncate it. Aligned multi-input
readouts likewise need an explicit join/concatenation rule. The implemented
example uses independent readouts, one per requested input.

## Targets and output blocks

`shared/targets.py` separates ground-truth construction from supervised output
encoding. Shared coordinates builds target-position labels and encodes the
pooled vocabulary of absolute XYZ cells. `shared/fitting.py` iterates requested
inputs and seeds; `shared/training.py` owns the vector-MLP training block.

The implemented path encoding is `binary_cell_mask`. Save the ordered cell list
alongside the mask; vector index must have a stable meaning on every sample/map.
Its contract is a sigmoid per cell with binary cross-entropy, with per-cell,
whole-mask and topology-aware metrics chosen by the report block. The label
cardinality is independent: path labels can contain many ones; own-cell labels
contain exactly one. Do not force either into a categorical cell-ID interface.
Nonmatching map dimensions require an explicit padding/masking or rejection rule.

The definition schema records those future contracts but does not claim their
implementations exist. Planned capabilities fail before checkpoint rollout or
directory creation. Promoting a YAML to active is insufficient until its input,
sample layout, target builder, encoder, loss and renderer are registered.

## Reporting and compatibility

`shared/reports/dispatch.py` selects report content by explicit renderer ID.
Reusable report shell, dashboard, navigation, numeric comparison metadata and
artifacts are shared across output types. Location-specific accuracy/distance
charts remain a renderer block, rather than a mandatory interface for masks.
Future renderers must identify metric semantics/versions so unrelated numeric
outputs are not compared just because a table column has the same name.

`archive/` preserves older implementations, configs and artifact interpretation.
The per-map, joint map-and-cell and historical own-cell classifier are not being
rewritten to satisfy the new schema. Existing reports/refits remain readable.
Historical suite plans use the archived adapter. Strict source identity checks
may require a fresh collection after code changes; they are not bypassed.

## Implemented path-predictor contract

1. The agreed output is a base-to-target, base-cell-centred `5×5×5` binary mask
   (125 values), invariant under whole-scene yaw quarter turns with vertical up
   preserved. Implement deterministic canonicalization and symmetry/tie handling.
2. Reuse the reward's unpenalized roadmap reference and existing geometry/planner
   utilities. Probe archives currently save no ordered per-target route; reuse a
   compatible artifact or reconstruct/cache one canonical route per sampled pair.
   Initially propose one shortest route; defer multi-route supervision.
3. Add shared mixed event acquisition for finder GRU, finder message, each
   informed agent's first GRU, informed-agent GRUs every configurable 10 steps,
   and the first stored base message. Preserve common episode splits and save
   the mask once per episode. Retain at most 20 trajectory candidates per agent,
   evenly spaced over its stride-10 candidate grid, including first/last when
   possible. Train using a uniform episode → agent → snapshot sampler and
   evaluate with matching nested means. Document candidate stride, retained cap,
   capture working-memory limits and training update budget separately.
4. Implement the shared binary-mask encoder, sigmoid/BCE training output and
   mask metrics. Rasterize the actual route segments, with explicit boundary
   rules. Verify reference length, endpoint inclusion, rotation invariance and
   unsupported-map behaviour. Reward chain efficiencies apply to live agent
   chains, not to an already persisted set of alternative label masks.
5. Keep one overview row per input with exact-path accuracy, average path
   similarity and an error histogram. Nest rotatable representative examples
   inside the existing detail groups, with a local probability-threshold slider.
   Full-episode inspector belief playback and an additional recurrent decoder
   are backlog ideas. Budgets/optimizer and eligibility live in the definition.
   The path predictor is active after regression checks and a deterministic
   collector-to-report smoke run.

The [path-specific requirements and open decisions](probes/path-predictor/TODO.md)
record the agreed input groups and the remaining route choices. The
[report plan](probes/path-predictor/REPORT_PLAN.md) specifies nested rotatable
sample viewers and explains probabilistic masks versus future sequence/recurrent
decoders. Those additional decoder/inspector ideas remain deferred.

## Implementation plan: own-cell predictor

1. Choose sampling cadence, pre/post-update timing, balancing/exclusions and
   whether any discovery phase restricts sampling. All active agents supply GRUs.
2. Reuse `agent_events`, without the path predictor's target-knowledge filter.
3. Build each selected agent's position mask in the saved map-cell order, exactly
   one `1` per row. Set the output size from the map rather than assuming 27 cells.
4. Reuse the binary-mask encoder/trainer, enforce one-hot label validity and add
   own-position mask metrics/views. Evaluate grouped episodes without allowing
   other agents or nearby steps from the same episode into another split.
5. Keep its sampling, training and report config separate from path prediction;
   activate after its own checks. Historical classification remains archived.
