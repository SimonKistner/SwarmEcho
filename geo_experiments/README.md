# Frozen geo probes

A probe family is defined by **what is predicted**. Inputs, decoder structure and
output encoding are independent building blocks. Probe folders contain YAML
definitions and notes; acquisition, fitting and report code live in `shared/`.
The policy remains frozen.

```text
geo_experiments/
  probes.yaml                         # Checkpoints, probe IDs, repetitions, run metadata
  probes/
    target_location/
      shared-coordinates/probe.yaml   # Active example of the shared pipeline
      per-map-decoder/probe.yaml      # Archived definition
      joint-map-and-cell/probe.yaml   # Archived definition
    own_cell/cell-mask/               # Planned agent-position mask predictor
    path-predictor/                   # Active optimal-roadmap-path mask predictor
  shared/                             # One collector, runner and reusable blocks
  archive/                            # Previous implementations/configs
  outputs/                            # Saved collections, fits and HTML reports
```

See [the architecture and extension plan](CONCEPT.md) for the contracts and the
implementation steps for path and own-cell prediction.

## Definitions and scheduling

The categorical example is
[`probes/target_location/shared-coordinates/probe.yaml`](probes/target_location/shared-coordinates/probe.yaml).
The active [path predictor](probes/path-predictor/README.md) is selected by the
checked-in suite schedule. Both definitions specify target ground truth, inputs,
MLP, output encoding/loss,
sampling unit and all dataset/training/audit/report settings.

`probes.yaml` contains only run name/output root/timezone, named training runs
and checkpoints, and ordered jobs:

```yaml
execute:
  - {policy: g03_agents_7, probe: path-predictor, repeats: 3}
```

`repeats: 3` uses decoder seeds `[0, 1, 2]` on one frozen collection. The path
predictor's five input groups produce **15 fits**; the shared-coordinate example's
three inputs produce nine fits. Repetitions do not repeat policy rollout.

From the repository root in the existing WSL environment:

```bash
uv run python geo_experiments/suite.py
uv run python geo_experiments/suite.py execute.0.repeats=5
uv run python geo_experiments/suite.py resume=geo_experiments/outputs/<suite-folder>
```

`config=<yaml>` selects a different suite schedule. Checkpoint overrides retain
the form `policies.<name>.checkpoint=<path>`. `latest` chooses the highest completed
numeric checkpoint; an explicit checkpoint directory is also accepted. Choices
and fully resolved definitions/settings are frozen in the suite's `run.json`.
Resume uses that frozen plan without reloading current definition files. Completed
reports are skipped; an unfinished fit restarts. Resume accepts no overrides.

**Shared coordinates and path prediction are active**. The other two
target-location subprobes are archived. The new own-cell mask predictor remains
planned and cannot be scheduled. Historical own-cell classification stays in the archive;
it is not the planned binary map-mask output.

## Shared collection and outputs

The shared collector receives input requests. The current event backend supports
finder GRU/current message at the first communication tick after discovery, and
the first current message stored by the base (possibly from a relay). Only
requested feature arrays are returned. Sample completeness depends on the
requested events. Episode eligibility is a separate rule, currently `delivered`.
The three-input example retains the established matched-snapshot semantics.

The default budgets remain **10,000 train + 1,000 validation + 5,000 test samples
total per checkpoint**, divided into exact per-map quotas. Coordinate support is
pooled across maps; the same maps occur in every split. This does not measure
transfer to maps excluded from decoder training. Map identity is never an input.

`settings.policy.map_source: compatible_pool` reuses or creates a compatible local
CPU map pool from the saved checkpoint generator settings; the saved validation
map count is retained. `validation` requires the original complete frozen bank
and verifies its hashes without regenerating replacements. Use `training` for
authored-map runs. Saved sensors, population and target feasibility rules apply.

The collection identity includes input, target, output and sampling definitions,
checkpoint/runtime/map hashes and shared source hashes. Changing input requests
or label semantics cannot silently reuse another collection. New definitions use
new collection keys; historical reports and refits retain their saved semantics.

The implemented outputs are categorical absolute XYZ coordinates and sigmoid/BCE
binary path masks. Path collection uses the one shared rollout engine with
episode and informed-agent events. Planned own-cell masks need their own agent
sampling and target builder before they can run.

## Reports, audits and saved data

Report content is selected explicitly by renderer ID. The reusable location
renderer keeps existing tables, learning curves, confusion matrices and distance
diagnostics, using the shared document shell, dashboard and comparison tools.
The path renderer keeps one overview row per input with exact-path accuracy,
average path similarity and an error histogram. Nested detail examples contain
rotatable offline 3D scenes, input/agent/time selectors and a local threshold slider.
Open `outputs/index.html` or a self-contained `report.html`.

```bash
uv run python geo_experiments/report.py run=geo_experiments/outputs/<suite>/runs/<readout>
uv run python geo_experiments/reproduce.py run=geo_experiments/outputs/<suite>/runs/<readout>
uv run python geo_experiments/auditing.py run=geo_experiments/outputs/<suite>/collections/<key>
```

Refitting uses archived features, labels and splits without policy rollout. Probe
training settings and report DPI may be overridden. Report generation uses saved
predictions and dispatches historical artifacts to their original renderers.

The current location-collision audit block requires the three named event inputs.
Disable it for a different input subset. Its conditional screen checks exact
latent duplicates; packet diagnostics use saved vectors, while only conflicting
finder GRUs can require observation-history replay. Low duplication does not rule
out approximate ambiguity. The standalone audit command never loads a policy.

Artifacts remain non-pickled `data.npz` arrays and `run.json` manifests. Collections
store features, geometry, labels, splits and episode metadata once. Fit archives
refer to their collection through `shared_data`. Move whole suite folders together.
No W&B run is created; the console uses training-style timestamps.

The [archive](archive/README.md) contains the older protocols and reproduction
commands. Root compatibility commands remain available. No full experiment run is
part of this restructuring.
