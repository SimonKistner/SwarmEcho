# Frozen spatial-information probes

The maintained research record is in [docs/geo/Phase0.md](../docs/geo/Phase0.md),
with [Phase 1](../docs/geo/Phase1.md) proposals and an unscoped
[Phase 2](../docs/geo/Phase2.md) idea backlog. Update these records when planning,
running or interpreting subsequent geo experiments. The scoped G02 implementation
is recorded in [the multi-map protocol](../docs/geo/G02_probes.md).

This folder contains the experiment code, configuration and generated outputs.
The policy stays frozen. Small supervised MLPs test whether its recorded features
allow target localization on one or several authored buildings, and historical
own-cell readout on one static building. No W&B session is created by the probes.

## G02: three readouts, shared collection

The maintained sequential execution plan is **[`probes.yaml`](probes.yaml)**.
It schedules only target probes: three alternatives on the latest completed
`G02_unanimity_YesNo_solo` checkpoint, then the same three on the latest completed
`G02_unanimity_YesNo_agents_7` checkpoint. It does not schedule own-cell probes.

```bash
# Inspect the plan without loading JAX or collecting episodes.
uv run python geo_experiments/suite.py plan=true
# Execute in the existing WSL environment.
uv run python geo_experiments/suite.py
```

`policies` selects trained runs/checkpoints; `experiments` describes the readouts;
`execute` selects their sequential order. `defaults` holds shared collection,
decoder, audit, distance and report settings. `latest` is resolved once at startup
from completed checkpoints with their recovery-ready marker; the exact selected
paths and resolved experiment configurations are snapshotted before execution.
At the 2026-10-24 implementation check, these selected `ckpt_004230` (solo) and
`ckpt_001666` (final curriculum). Future fresh suites may select newer checkpoints.

| Decoder | Output | Interpretation |
|---|---|---|
| `map_specific` | One independent cell decoder per map, input and seed | Known map identity routes to an expert; report per-map and combined held-out results. |
| `map_cell` | One joint map + absolute XYZ-cell class space (135 classes in G02) | Both map and cell must be correct; wrong-map errors have a distinct histogram bucket. |
| `shared_cell` | One absolute XYZ-cell decoder across maps (27 classes in G02) | No map identity is predicted; true-map geometry is used only for evaluating coordinate errors. |

Map IDs are never appended to latent inputs. Target spawn exclusions restrict
actual samples, not the full interior prediction class space. Storey prediction
is **off** in this plan; `defaults.protocol.target_storey=true` enables it without
changing the cell readouts. Storey *breakdowns* remain present with this head off.

The 10k/1k/5k budgets are **totals per policy**, divided equally across maps in
every split (2k/200/1k per map in G02). Each map is sampled independently with
map-specific PRNG namespaces; its retained episodes are split by target cell.
All alternatives use exactly those same retained episodes and splits. Experts
receive only their own map's subset; their input normalizers are fitted on that
map's training subset. Joint normalizers use the pooled training subset.
With the storey head disabled and three seeds, this is 45 specialist fits plus
9 joint map+cell and 9 shared-coordinate fits per policy (126 fits in the default
six-experiment plan).

Collection retains a target episode when the base received the target and all
three feature snapshots exist. Eventual `chain_held` success is **not** required.
The rollout executes the normal movement and discrete recruitment-vote action.
It records each attempted episode's map, delivery, chain success, target cell,
storey and cached `target_required_agents` count. This count comes from the
production relay estimator; zero means unavailable, not a zero-agent solution.
It is not the deployed population or a proved global optimum.
New collections also record the active population at finder capture and first
base-packet receipt. Older saved collections remain usable; missing population
fields are explicitly marked “Not recorded,” never inferred from agent indices.

The audit defaults to `mode: conditional`. On a seeded subset of up to 5,000
retained episodes, it cheaply fingerprints GRUs, finder packets and base packets.
“Same latents” is the percentage of samples belonging to an exact repeated-vector
group, including the first member. If **every input has <1%**, only a footer gives
the rates to three decimals and sample count. Thresholds are evaluated separately
for each source:

- **Base message / finder message:** inspect exact collision groups in **all retained
  snapshots**, without loading the policy or replaying episodes. Reports distinguish
  repeated samples from samples with conflicting labels. Tables compare map/cell,
  storey, target-required agents, sender/finder index, reporter role, chain outcome
  and available capture-time population. Expand groups for category counts, target
  spatial spread, reporting-time ranges and distinct upstream GRUs/messages. Pair
  agreement is compared with a full-dataset random-pair baseline; these are
  descriptive diagnostics, not causal compression claims or held-out probe scores.
- **Finder GRU:** saved-snapshot profiling also runs when triggered. Only screened
  GRU groups with conflicting labels justify observation-history analysis. Replay
  only those episodes, stop at finder capture, and compare raw/encoded/received
  context history fingerprints. Packet duplicates never trigger this replay.

`audit.enabled=false` skips both stages. For modern target probes, `mode: full`
forces saved-data profiles for all inputs, but histories still require conflicting
finder-GRU labels. The historical G01 `mode: full` retains its original full audit.
The screen tests exact equality, not approximate ambiguity. Detailed replay reports
snapshot reproduction because GPU batch shapes can change floating-point trajectories.
Every analysis uses the selected readout's label contract: expert conflicts are
within a map, joint conflicts include map identity, and shared-coordinate conflicts
ignore identity when coordinates agree.

Refresh a stopped suite collection's audit directly from saved arrays:

```bash
uv run python geo_experiments/auditing.py run=geo_experiments/outputs/<suite-folder>/collections/<collection-key>
```

This writes `audit.html` and updates the cached analysis without JAX or MLP fits.
If finder history replay is needed, it remains pending for suite resume in WSL.
Otherwise the collection becomes ready for fitting. The same section is embedded
in each variant's normal results page, with its own ambiguity-label interpretation.
All experiment console output uses the normal training `[HH:MM:SS]` formatter.

Each alternative has its own self-contained `report.html`. Distribution charts
default to target storey versus **Target-required agents**, with count/share,
outcome/split and chain-success highlighting controls. Fresh single-map target
reports additionally offer cell distribution as a secondary left-chart view.
Held-out details list required-agent groups before storeys, then maps; overall
accuracy separates the storey block on the right. Confusions adapt to the readout:
a map-grouped 135-class matrix, separate expert-map matrices, or a shared 27-class
matrix. Validation curves explain epochs, loss, accuracy and checkpoint selection.
New multi-map target reports compare only with other multi-map target reports.
Report titles and comparison choices use the full investigated training-run name
and a plain description of the decoding task. Sidebar subtitles identify the
checkpoint, sample count and run date. Existing report labels can be refreshed
without refitting or changing frozen run IDs; new suite folders also use the
training-run name and readout description instead of policy/experiment aliases.

Outputs are grouped as follows:

```text
outputs/<suite>_<timestamp>/
  config.yaml, run.json              # frozen plan, checkpoint choices and progress
  collections/<collection-key>/
    config.yaml, policy_level.yaml, run.json, data.npz
  runs/<policy>_<experiment>/
    config.yaml, run.json, data.npz, report.html
```

Collection features, per-map geometry, labels, splits and audit data are stored
**once per policy/collection configuration**. Child numeric archives store fitted
readouts, predictions and histories; `run.json.shared_data` locates their shared
collection relative to the report folder. Keep the complete suite folder together
when moving it. HTML reports remain independently viewable. Source YAML definitions,
runtime/collector hashes and exact checkpoint identities are recorded. Model
settings come from the trained run's saved `config.yaml`, not today's G02 level.

Resume an interrupted suite with its frozen checkpoint choices:

```bash
uv run python geo_experiments/suite.py resume=geo_experiments/outputs/<suite-folder>
```

Completed experiments are skipped. Completed primary collection is retained even
if the optional deep audit fails. Valid shared data is reused after identity
checks; audit changes can refresh analysis without recollection. Explicit, hash-pair
migrations permit the reviewed logging/additive-population update while preserving
strict checkpoint, runtime, map, selection and other source checks. Original
collection provenance is retained. Interrupted fits in an unfinished experiment
restart. An incomplete primary
collection saves partial data and fails explicitly instead of shrinking quotas.
No GPU execution has been performed from the Windows editing session.

The single-readout entry point also accepts these decoders, for example:

```bash
uv run python geo_experiments/run.py policy.level=G02_static policy.checkpoint=outputs/G02_unanimity_YesNo_solo/checkpoints/ckpt_004230 protocol.decoder=shared_cell protocol.eligibility=delivered protocol.target_storey=false audit.mode=conditional
```

`labels.position_labels(..., position_key=..., task=...)` and `labels.class_space`
separate position truth from readout choice for a future own-cell adapter. The
existing own-cell collector/report remain available; multi-map own-cell collection
and dashboards are not enabled by the suite.

## Historical G01 protocol and reproduction

`config.yaml` retains `protocol.decoder=legacy`, successful-episode eligibility,
both target heads and the original full audit. `own_cell.yaml` retains its original
whole-episode sampling/filtering. With `policy.from_run_config=true`, fresh
collection loads the checkpoint's saved sensor/model/evaluation settings. The
frozen loader accepts the known historical reward payout names (these probes do
not compute rewards), while retaining supported rollout behaviour switches.

To reproduce the original detached experiment on its **exact saved features,
labels and split**, without relying on current simulator behaviour or loading a
policy, use:

```bash
uv run python geo_experiments/reproduce.py run=geo_experiments/outputs/visual_target_probes_20261014_141641_127517
# Optional new decoder settings, with the same archived samples/split:
uv run python geo_experiments/reproduce.py run=geo_experiments/outputs/visual_target_probes_20261014_141641_127517 probe.seeds=[0]
```

This works for historical target/own-cell archives and new suite readouts. It makes
a separate refit folder and preserves the source. Re-rendering `report.py` still
needs neither JAX nor the checkpoint. Exact numerical recollection across changed
simulator/JAX versions is not promised; the original arrays remain the reference.

The remaining sections describe the original single-map protocol unless noted.

| Prediction | Finder GRU | Finder outgoing message | Base stored message |
|---|---|---|---|
| Target cell relative to base | Independent MLP | Independent MLP | Independent MLP |
| Target storey relative to base | Independent MLP | Independent MLP | Independent MLP |

There are six combinations. The default configuration repeats each with three
initialization seeds, making 18 sequential fits. Set `probe.seeds=[0]` for six
fits. Every combination uses exactly the same episodes and train/validation/test
split. No W&B logging is initialized.

## Own-cell experiment

Run the new whole-episode diagnostic with the checkpoint used by the visual-target
report (configured in `own_cell.yaml`):

```bash
uv run python geo_experiments/own_cell.py run.name=visual_own_cell_probes
```

Override `policy.checkpoint=<path>` for another matching checkpoint. This entry
point overlays `own_cell.yaml` on `config.yaml`; common MLP, sample-budget and report
settings are shared. You can also use `run.py config=geo_experiments/own_cell.yaml`.

It predicts **only current own cell**, separately from **agent GRU** and **agent
outgoing message**, for **all active agents**, irrespective of success, target
discovery or base delivery. There is no base-message input. Snapshots pair the
current pre-motion position with the updated actor state/current token; a token
need not be transmitted on that tick. The policy remains frozen.

Sampling interval is `ceil((cell_size_m / max_speed + sampling.buffer_seconds) / dt)`.
The default 0.2 s buffer gives **7 steps / 0.7 s** in G01. This is a spacing heuristic,
not a guarantee the drone reaches a new cell. Step-zero and subsequent grid ticks
sample active agents; outside-interior snapshots are excluded and counted.

The budget remains **10k train + 1k validation + 5k evaluation snapshots**, rather
than episodes. Collection fills separate episode pools in that order; one episode
never contributes to two splits. By default, `sampling.uniform_cells=true` divides
each split's budget into dedicated buckets for **eligible authored interior cells**.
Each cell gets the integer floor or ceiling of the split budget divided by the
cell count. Seeded rotating remainders also balance the combined dataset to within
one sample per eligible cell. `sampling.exclude_cells` specifies excluded **absolute
map-grid XYZ coordinates**, validated against the selected map. The current G01
config excludes ten slow cells from the collection logs: `[0,2,1]`, `[0,0,2]`,
`[2,2,1]`, `[1,1,2]`, `[0,1,1]`, `[2,0,0]`, `[0,1,0]`, `[0,1,2]`, `[1,1,1]`,
`[0,0,0]`. With 17 eligible cells, budgets give 588/589 training, 58/59 validation
and 294/295 evaluation samples per cell (941/942 combined). Split totals remain
10k/1k/5k. There is no collection-time cap or adaptive cell removal.
Set `sampling.exclude_cells=[]` for the original all-27-cell protocol, or replace
the list when changing maps. Exclusions also apply with uniform buckets disabled.
Candidate coverage retains all cells, while excluded candidates are counted
separately and never claim input-history ownership. All original class coordinates
and the full obstacle-aware distance graph remain in place; the decoder can still
make an incorrect prediction of an excluded cell. Reports name the subset, so
overall accuracy refers only to eligible actual cells, not the whole building.
Reject a sample immediately if its cell's bucket is full; rejected samples do not
claim prefix ownership. Continue collecting until every bucket is filled.
`sampling.uniform_cells=false` reproduces the earlier visitation-weighted protocol.

Within each pool, retain one exact complete
combined-input-history/cell pair. Preserve different cell labels with the same
history so genuine ambiguity is not erased. Later pools exclude histories already
retained in earlier pools, regardless of label. This keeps the common prefix once
without assuming all pre-discovery behaviours are identical or removing every
pre-discovery sample. Eligible samples are shuffled within each episode before
capping quotas. This tests new histories with equal cell weighting, not a raw
time-weighted distribution. Rare cells may need more episodes. At the safety cap,
save partial data and report each underfilled cell; do not move its quota to a
common cell. Per-split cell limits, retained counts, full-bucket rejection counts,
eligible mask and excluded-cell rejection counts are saved in `data.npz` under
`coverage/`; the report adds rejection totals and the excluded coordinates.
This setting is loaded at startup; a running collection does not pick up YAML
edits. Start a newly named run to apply changed eligibility.

Complete raw, encoded and encoded+received-context prefixes are fingerprinted at
every input step with reset flags. History accumulation restarts on policy reset
or inactivity. Temporary team histories use the small `dataset.eval_batch_size=4`
batch and are discarded on the host after filtering. `audit.every=1` is required;
`audit.batch_size` belongs to target collection, while the own collector uses
`dataset.eval_batch_size`. Cached dataset reuse is not yet supported for own-cell;
saved reports can still be regenerated without collection or retraining.

The revised audit samples up to `audit.samples` retained snapshots across all
splits using a fixed seed. For GRU and message separately, it reports exact-state
groups with conflicting own cells and whether those conflicting pairs have
identical raw, encoded or combined-input histories. Same-cell repetition is not
counted as ambiguity. This minimal audit has no approximate-history comparison.

The same dark report/dashboard contains two result rows, expandable actual-agent-
storey details, incorrect-prediction histograms, one accuracy chart, coverage/split
distributions, validation curves, confusion matrices and audit/filter tables.
Measure defaults to **sample counts**. Sources and task identities keep these
metrics distinct from the target reports in comparisons. Original snapshot
features, position, episode/agent/step IDs, discovery flags, SHA-256 history
fingerprints, split membership, source counts/bucket limits and probe fits are in `data.npz`;
the manifest and HTML remain the only companions. No W&B or separate figure files.

Protocol and interpretation belong in [Phase 1](../docs/geo/Phase1.md).

## Run

Run from the repository root in the existing WSL environment:

```bash
uv run swarmecho-train level=G01_static logging.run_name=G01_static_visual
uv run python geo_experiments/run.py run.name=visual_target_probes policy.checkpoint=outputs/G01_static_visual/checkpoints/ckpt_NNNNNN
```

Use the checkpoint path printed by training, or set it in the experiment config.
Policy training is a normal run with W&B enabled; the example uses
`outputs/G01_static_visual/` to retain the original Phase 0 run.
The geo probe runner saves its own artifacts under `geo_experiments/outputs/`
and does not initialize W&B.

Edit `geo_experiments/config.yaml` or supply training-style overrides:

```bash
uv run python geo_experiments/run.py run.name=pilot dataset.train_samples=3000 dataset.validation_samples=500 dataset.test_samples=1000 probe.seeds=[0]
```

Each run creates `geo_experiments/outputs/<name>_<timestamp>/`, using Europe/Berlin
time by default. Open **`geo_experiments/outputs/index.html`** when it finishes:
the left sidebar chooses a report and its **Compare to** dropdown selects another
report. No server is needed. Each self-contained `report.html` also remains usable
directly and links back to All reports.
The experiment configuration is saved in `run.json`. Output paths
must stay inside `geo_experiments/`. Relative paths resolve against the repository.

`policy.level` selects a normal level definition, initially `G01_static`.
It supplies the map, sensors, model, episode limit and evaluation spawn/success
rules directly through the same loader used by training. G01_static uses
`B02c_static_map_noSpawn` and a 700-step episode limit.
Other level settings are inherited as defined, including defaults.

Current level defaults differ from historical G01 sensors/population. The original
Phase 0 `ckpt_000330` has **136 inputs** and cannot be loaded with the current
visual/adaptive settings substituted. The default `policy.from_run_config=true`
restores the checkpoint's own saved settings and validates its training contract.
Historical reports/data remain valid and can be plotted without a policy.

Set `policy.checkpoint` to a checkpoint trained with this level. By default its
saved training config supplies frozen policy settings. Set `policy.from_run_config=false`
only to use the explicitly selected normal level instead. In this legacy entry point there is no
automatic checkpoint selection and no generated policy configuration or separate
map/horizon override. Original authored level and map YAML are embedded in
`run.json` for provenance without expanding defaults. Future experiments select another
normal level with `policy.level=<name>` and its matching checkpoint.

## Collection and labels

- One sample means one **successful episode with all three feature snapshots**.
  Failed episodes and successful episodes missing a snapshot are counted in the
  collection summary, but excluded from probe learning.
- The finder is the first active drone directly seeing the target; simultaneous
  sightings use the smallest agent index. Capture its updated GRU and outgoing
  signature+value at the first communication tick after discovery, including the
  discovery tick if communication is due then.
- The base feature is the first signature+value packet it actually stores under
  the existing evaluation protocol. Its sender can be a relay. Both target
  labels refer to the target, regardless of which drone sends that packet.
- Target-cell classes are interior cells, encoded as XYZ cell offsets from the
  fixed base cell in fixed building axes. Storey classes are signed Z offsets
  from the base storey. Report groups use the actual target's building storey,
  numbered 1, 2, 3 from bottom to top.
- Features are normalized using training data only. Both heads are independent
  cross-entropy classifiers with hidden layers `[256, 128]`, AdamW and early
  stopping on validation loss. Their inputs are recorded features only.
- A seeded cell-stratified episode split is shared by all fits. Split sizes are
  exact, with per-cell allocations rounded using largest remainders and at least
  one training example per observed cell. Distribution charts show coverage.

The default collects 16,000 matched episodes: exactly **10,000 training, 1,000
validation and 5,000 held-out evaluation**. Validation selects the early-stopping
checkpoint; evaluation is excluded from training and model selection. More data
reduces noise but does not guarantee the features contain target information.

The collector stops at `dataset.max_episodes` if it cannot obtain enough matched
successes. It saves partial data and reports an explicit failure instead of
silently proceeding with fewer episodes. `run.json` tracks progress.

## Distance: valid passages, never solid walls

Target-cell distance is the number of transitions on the **shortest valid
six-neighbour cell route between predicted and true target cells**. It is an
obstacle-aware discrete Manhattan distance: two neighbouring cells connect only
if their shared face has an opening clear of the compiled walls, floors and stair
solids. The default clearance is drone radius plus planning clearance. A solid
wall blocks the transition; a detour through valid openings increases the error.

This graph abstracts movement within each authored cell. It measures cell
topology, rather than continuous flight distance. Its edges, free portal points,
clearance and all-pairs distances are saved under `geometry/` in `data.npz`.

A storey-only head does not predict XY. Report its exact storey accuracy and
absolute storey error. The cell head's wrong-prediction histogram uses valid cell
transitions; the storey head's histogram uses absolute storeys off. Disconnected
cell predictions, if any, appear in a separate "No route" histogram bucket so
they remain included in the error total.

## Report and comparable figures

- A dark theme with consistent blue/gold/mint colours for the three inputs across
  tables, comparisons, learning curves and confusion matrices.
- Six overall result rows: probe/input, exact accuracy and a discrete error
  histogram. Each row has collapsed per-storey details. Accuracy is rounded to
  one decimal percent and averaged over initialization seeds.
- Error histograms exclude correct predictions. The shared Measure selector defaults
  to counts and also switches these histograms to shares totaling 100% of errors.
  Counts are incorrect predictions pooled across seeds, rather than unique episodes.
  Hover a bar for its count and share. Storey
  errors have bins 1 and 2 on this three-storey building. Cell bins share the same
  scale across inputs and storey details.
- Two accuracy charts comparing the three inputs overall and on storeys 1/2/3,
  with shared 0–100% scales and initialization standard deviations.
- Storey and cell distribution charts compare all spawned targets, successful
  missions and collected samples. A selector switches to training/validation/
  evaluation splits; Measure defaults to episode counts and switches between
  group-normalized shares and counts, including the incorrect-prediction histograms.
  Every interior cell is visible, including cells with zero samples. Cells are
  ordered along X, then Y, then storey, labelled with base-relative coordinates.
- Validation accuracy/loss curves, aligned by epoch with shared scales.
- Confusion matrices with a common 0–100% row-normalized scale.
- Ambiguity-audit tables: exact GRU groups/conflicting cells, raw/encoded/combined
  input history matches, reference similarity and target separation.

The audit is on by default: `audit.samples=5000`, `audit.every=1`,
`audit.batch_size=32`. It uses the **first matched episodes of the main collection**,
capped at the dataset size. History batches are small; regular collection resumes
after the audit quota. No separate 5k evaluation is required. Set
`audit.enabled=false` to omit it. Full floating histories are discarded; reference
history storage is bounded at 128 MiB. Exact matching still uses fingerprints if
a reference is evicted; unavailable nonidentical reference comparisons are counted.

When comparison is enabled, only matching **numeric table cells** receive a delta,
such as `78.1% (−0.2)`. Percentage deltas are percentage points, not relative
percentage change. Positive deltas are green, negative deltas red, and rounded zero
deltas neutral. Matching uses actual section heading, table name, row, column,
unit and metric version, independently of layout/ordering. Missing sections, N/A,
text, graphs and histograms receive no deltas. Differences do not imply statistical
significance or better/worse performance.

Missing groups show N/A. Seed variation is not a statistical confidence interval.
Charts are embedded in the single HTML report; no separate figures are exported.
These are discovery/delivery snapshots; the audit compares histories **up to**
finder capture, rather than displaying belief evolution. Own-position probes,
path prediction, map reconstruction and replay
inspector extensions are outside this experiment.

## Reuse data or regenerate figures

### Minimal observation-variability audit

```bash
uv run python geo_experiments/audit_observations.py
```

This standalone command remains console-only. Normal experiment runs now include
the same audit automatically. The standalone defaults are 5,000 matched successful episodes, every step up to the finder
snapshot, batches of 32, checkpoint/level/seed from `config.yaml`. Groups identical
finder GRUs and compares raw observations, encoder outputs and combined GRU inputs
(encoding + received context). Prints exact different-cell pair counts, reference
similarity and target separation. No files, plots or probe fits. `--every 5`
subsamples if needed; `--checkpoint`, `--level`, `--samples`, `--seed` and
`--batch-size` override defaults. Always includes capture, ignores padding, and
counts unequal history lengths separately. The completed 5k console audit and
its interpretation are recorded in [Phase 0](../docs/geo/Phase0.md).

### Reuse probe data

Repeat probe learning without collecting the same episodes again:

```bash
uv run python geo_experiments/run.py run.name=retry dataset.path=geo_experiments/outputs/NAME_TIMESTAMP/data.npz
```

The cache must match the configured sample count, checkpoint, normal level
definition, map, runtime/collector hashes, collection seed and horizon.
Its collection arrays are saved in the new timestamped run. Probe settings and
split seed can change independently; requested split sizes must still add up to
the cached dataset size.
With auditing enabled, cached data must also contain an integrated audit with the
requested sample count and sampling interval. Otherwise collect fresh data or
disable auditing explicitly; imported console summaries are not treated as cached
histories from the main collection.

Regenerate static figures from a completed run, without loading the policy:

```bash
uv run python geo_experiments/report.py run=geo_experiments/outputs/NAME_TIMESTAMP
```

This also refreshes the report browser. To attach a historical console audit:

```bash
uv run python geo_experiments/report.py run=geo_experiments/outputs/NAME_TIMESTAMP audit_console=geo_experiments/outputs/AUDIT_TIMESTAMP/console.txt
```

Import verifies checkpoint/map hashes, preserves the supplied console in the
manifest, and labels it as a separate audit execution. Different level/runtime
versions are retained as provenance. The supplied 2026-10-13 5k audit is already
attached to the latest 16k report; no histories or unreported values were invented.

### Adding future comparison tables

Use `dashboard.numeric_cell()` for numeric values inside a named report section,
with semantic row/column labels and units. A table can set `data-compare-table` for
an additional named scope. Change its metric `version` when the definition changes;
reordering rows/sections or changing visual layout requires no migration. The
generator reads actual headings and table metadata, so images are never compared
and ambiguous duplicate keys fail generation. Older/unannotated reports remain
viewable; missing comparison values are ignored.

## Saved source data

Historical compact runs contain three files; new runs also snapshot configuration
YAML, and suite reports refer to their shared collection as described above.
The numeric archive can be read with
`numpy.load(path, allow_pickle=False)`; no pickled objects are required.

| Artifact | Contents |
|---|---|
| `report.html` | Self-contained report with embedded charts and collapsed storey details |
| `run.json` | Config, collection summary, ambiguity-audit summary/provenance, snapshot semantics, fitted-model metadata, progress, source hashes, original level/map definitions and report settings |
| `data.npz` | Features, labels, all attempted episodes, shared splits, geometry, predictions, learning curves, fitted weights and compact integrated-audit source arrays |

One shared `outputs/index.html` catalogs the reports; it does not duplicate their
figures. Report comparison values are embedded in each report rather than exported
as another per-run file.

Archive keys are grouped by prefix: `dataset/`, `episodes/`, `labels/`,
`splits/`, `geometry/` and `probes/<task>/<source>/seed_<n>/`.
Integrated audit arrays use `audit/`: episode IDs, hidden-group indices, SHA-256
history fingerprints (episode × raw/encoded/combined × 32 bytes), reference episode
pairs, cross-cell/flag masks, reference statistics (comparison × stage × max-abs,
RMS, equal-component share, equal-step share), and target separation in metres.
These retain source data for alternative audit tables/distributions without saving
large raw history tensors. An imported old console audit has summary data only.
The probe prefix contains `predicted`, `probabilities`, `history` and `model/`
arrays. History columns are listed in `run.json`; model arrays include weights,
biases, training normalization and class values. Prediction rows follow
`splits/test`, which indexes the shared dataset; `dataset/episode_id` joins to
`episodes/episode_id`. Probability columns follow the model's class values.
Distances and grouped metrics are computed from these arrays, without redundant
exports. New plots can be produced without recollecting or retraining.

The checkpoint and source code remain in their original locations. SHA-256
hashes identify the versions used; source files are not copied into runs.
Regenerating a completed report from the original output layout also consolidates
that run into these three files, retaining its original split and learned data.
