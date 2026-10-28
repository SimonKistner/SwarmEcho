# Frozen spatial-information probes

The policy stays frozen. Small supervised decoders investigate target location
in finder GRUs, finder messages and the first message stored by the base.
Historical own-cell and single-map protocols remain available under `legacy/`.

## Configure and run

**Edit only [probes.yaml](probes.yaml) for the active suite.** It contains the
run name, policy/checkpoint selections, common defaults, decoder definitions and
ordered `execute` list. The suite does not inherit a historical G01 YAML file.

From the repository root in the existing WSL environment:

```bash
uv run python geo_experiments/suite.py plan=true
uv run python geo_experiments/suite.py
```

`policies.<name>.run` identifies the trained run. `checkpoint` accepts `latest`
or a full repository-relative/absolute checkpoint directory, including
`checkpoints/best_val_checkpoint`. `latest` selects the highest completed numeric
checkpoint with its metadata, training contract and recovery-ready marker. The
choice is frozen at suite startup; it does not chase new checkpoints during a run.
`best_val_checkpoint` is loaded from the directory selected at startup and its
checkpoint hash is recorded; changing it invalidates snapshot reuse.
The checked-in policy selection is user-maintained; inspect the plan for its
current value instead of relying on a checkpoint mentioned in historical docs.

Example override, using the policy key currently declared in `probes.yaml`:

```bash
uv run python geo_experiments/suite.py policies.g03_agents_7.checkpoint=outputs/G03_random_4k_v1_agents_7/checkpoints/best_val_checkpoint plan=true
```

Use `defaults.dataset.*`, `defaults.probe.*` and other `defaults.*` overrides for
suite-wide settings. `config=<path>` selects another complete suite YAML. A saved
suite's `config.yaml` can start a fresh execution with those frozen choices.

Resume a suite with its original plan and checkpoint choices:

```bash
uv run python geo_experiments/suite.py resume=geo_experiments/outputs/<suite-folder>
```

Completed experiments are skipped. Completed collection is retained before an
optional audit, and reused only after checkpoint, map, runtime and collector
identity checks. An unfinished fit restarts; incomplete collection is not treated
as a complete dataset. Resuming does not accept configuration overrides.

## Readouts and sampling

| Decoder | Question |
|---|---|
| `shared_cell` | Can one decoder predict absolute XYZ cells across buildings? |
| `map_specific` | Can independent decoders predict cells when map identity selects the expert? |
| `map_cell` | Can one decoder infer both building identity and target cell? |

Only entries in `execute` run. Definitions under `experiments` are alternatives,
not an implicit execution list. Modern suite jobs are target probes; own-cell
collection remains a historical separate workflow. Map IDs never enter feature
vectors. `defaults.protocol.target_storey` toggles the additional storey head;
storey breakdowns still appear when that head is off.

Omit `map_source` for authored-map runs such as G02. Set
`policies.<name>.map_source: validation` for a generated-building run's complete
frozen validation bank. The loader checks selection order, stage and map YAML
hashes against the saved validation metadata, and fails on missing/changed maps.
It never regenerates replacements. Saved population, sensors, model and target
feasibility rules are retained; inspection overrides and adaptive training weights
do not substitute another sampling protocol. Existing target exclusions remain.

The standard budgets are **10,000 training + 1,000 decoder-validation + 5,000
held-out evaluation samples total per policy**, shared across compatible jobs.
Keep an episode after delivery to the base only when all feature snapshots exist;
eventual chain-held success is an analysis attribute. Maps receive exact split
quotas. Cells and target-required-agent groups are not balanced.

Large validation banks use a mixed-map collector, avoiding one compilation per
map and excluding execution padding from counts. Shared-coordinate splits require
observed coordinate coverage in the pooled training set, not every small map
subset. Maps are held out from policy training but occur in every probe split:
this is not transfer to maps excluded from decoder training.

## Audits

The conditional screen fingerprints up to 5,000 retained latent samples. Below
1% duplicates for each source, results appear only in the footer. Triggered finder
message/base-message analyses inspect all saved vectors; they do not replay
observations. Only conflicting finder-GRU groups trigger observation-history
replay. This tests exact equality, not approximate ambiguity. `audit.enabled=false`
disables both stages. The historical full G01 audit remains separate.

Refresh saved collection diagnostics without policy rollout or decoder fitting:

```bash
uv run python geo_experiments/auditing.py run=geo_experiments/outputs/<suite>/collections/<key>
```

If finder replay is needed, resume the suite in WSL. Snapshot-only diagnostics
are saved as `audit.html` and included in each decoder's report.

## Results, reports and reproduction

```text
outputs/<suite>_<timestamp>/
  config.yaml, run.json
  collections/<key>/
    config.yaml, policy_level.yaml, run.json, data.npz
  runs/<readout>/
    config.yaml, run.json, data.npz, report.html
```

Features, geometry and splits are stored once per compatible collection. Child
archives contain decoder fits and use `run.json.shared_data` to locate collection
data. **Move whole suite folders together.** Standalone modern compatibility
commands use the same machinery, with a `collection/` subfolder. No W&B run is
created. Console output uses the training timestamp formatter.

Open `outputs/index.html` for navigation and comparisons, or an individual
self-contained `report.html`. Cards are newest first, with a decoder title, run
name and checkpoint/map/sample/date details. Training runs share cyclic theme
colors. Comparison choices are slim one-line title/run entries, allow different
runs and map counts, and annotate only matching numeric metrics. Random validation
banks with 20 or more maps omit map-ID details; required-agent/storey groups remain.

```bash
uv run python geo_experiments/report.py run=geo_experiments/outputs/<suite>/runs/<readout>
uv run python geo_experiments/reproduce.py run=geo_experiments/outputs/<suite>/runs/<readout>
```

Reporting uses saved predictions. Reproduction fits new decoders on the exact
archived features, labels and splits, without loading the policy. Only probe
settings and report DPI may be overridden in a refit. Historical single-map
labels retain their original semantics. Do not reinterpret them as modern labels.

Artifacts are `run.json` manifests, non-pickled `data.npz` arrays and HTML reports.
Archive prefixes include `dataset/`, `episodes/`, `geometry/`, `labels/`, `splits/`,
`audit*/` and `probes/<task>/<source>/seed_<n>/`. Probabilities follow stored class
values; prediction rows follow `splits/test`. Source hashes and provenance are
retained. `outputs/archive/` is omitted from the active browser and records
superseded runs; standalone archived reports and whole-suite relative links remain.

## Code and historical compatibility

- `suite.py` plans jobs; `execution.py` owns shared collection/fitting orchestration.
- `collection.py`, `target_collection.py`, `policy.py` and `validation_maps.py`
  capture snapshots; `labels.py`, `geometry.py`, `probes.py` and `experiments.py`
  define labels, distances and readouts.
- `audit_common.py`, `auditing.py`, `collisions.py` and `collision_report.py` serve
  current audits. `report_components.py` and `report_layout.py` hold shared UI;
  `target_report.py`, `dashboard.py`, `dashboard.js` and `naming.py` present results.
- `legacy/` preserves G01/own-cell execution, single-map rendering, full observation
  audits and old console imports. Its YAMLs live in `legacy/configs/`. Root
  `run.py`, `own_cell.py`, `own_collection.py`, `audit_observations.py` and
  `audit_results.py` are compatibility forwarders. Old explicit config paths used
  through `run.py` are mapped to the relocated templates.

See [historical commands](legacy/README.md), [G02 protocol](../docs/geo/G02_probes.md),
[G03 protocol](../docs/geo/G03_probes.md), and the [Phase 0](../docs/geo/Phase0.md),
[Phase 1](../docs/geo/Phase1.md) and [Phase 2](../docs/geo/Phase2.md) research records.
GPU collection and fitting belong in the existing WSL environment. No experiment
execution is part of source cleanup or report-navigation refresh.
