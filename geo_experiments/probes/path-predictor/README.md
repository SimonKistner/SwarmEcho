# Base-to-target path predictor

The active [`probe.yaml`](probe.yaml) uses shared collection, target construction,
binary MLP training and report blocks. The checked-in suite selects it against
the existing frozen checkpoint; no policy training is performed.

```bash
uv run python geo_experiments/suite.py plan=true
uv run python geo_experiments/suite.py
```

Edit this definition for scientific settings. The suite contains checkpoint,
probe and repetition choices only. Default budgets are 10k/1k/5k independent
episodes across all maps, not input rows. Three decoder seeds give 15 fits.
Use the existing suite resume/report/reproduce commands for archived data.

## Labels and inputs

The output is a 125-cell binary mask in a base-cell-centred `5×5×5` grid, with
offsets −2..+2 in each axis. Canonical geometry/endpoints remove yaw quarter turns
while preserving vertical up. Base-on-grid-face ambiguity is resolved jointly
with orientation; the chosen base cell is always marked. Segment-interior cells
and the target endpoint cell are marked. Mere corner contact adds no extra cell.
Maps larger than three cells on any axis fail explicitly.

The target is one deterministic shortest visibility-roadmap route, using the
reward's **unpenalized** distances and checkpoint planning clearance. Compatible
pool distance matrices are reused. Route reconstruction checks physical length
against the reward matrix within `atol=1e-4 m, rtol=1e-5`. Canonical coordinates
are rounded to six decimal places and equal-cost successors use canonical node
order. The route, inverse transform and mask are saved once per episode.

Spawn feasibility can use a denser relay graph than the reward's sparse roadmap.
A valid sampled mission can therefore have no finite reward-reference route.
These unlabelable episodes are logged in `reference_route_failures.jsonl` with
map/episode IDs, coordinates and attachment counts, then replaced to fill the
same per-map budgets. Collection metadata and the report show the rejection
count; results apply to episodes with a finite reference route. Other route
construction errors still stop the run with saved diagnostics.

Five separate readouts use finder GRU, finder message, each informed agent's first
updated GRU, informed-agent GRUs until inactivity/end, and the first base message.
First-information capture uses a knowledge flag observed before the actor update
and stores the updated hidden state. The finder is included in all-knowing groups.
Episodes with an informed agent are retained even if delivery/success later fails;
an absent base message does not remove available GRU samples.

`every_steps: 10` sets each agent's candidate grid from its first informed update.
`repeated_snapshot_cap_per_agent: 20` limits retained rows, evenly spaced over
that grid, including first/last when possible. Short histories are not duplicated.
The final stride candidate can precede decommission/end by less than one stride.
Both parameters have comments next to them. `eval_batch_size` requests an initial
4,096 parallel episodes (or fewer when the entire collection/map budget is smaller).
There is no fixed memory cap. Startup prints the candidate-buffer memory estimate;
the policy and other rollout buffers need additional memory. A catchable device or
host allocation failure during rollout/transfer releases temporary buffers and JIT
caches, halves the batch, and retries the same episode IDs. Failed attempts append
no samples and advance no counters. Previously accepted samples survive, and the
reduced batch stays in use for the rest of collection. Failures at batch one and
unrelated runtime errors propagate. Recovery cannot handle a process killed by the
OS or an unusable GPU context. It finds a batch that fits, not necessarily the
fastest one. No separate benchmark rollouts are performed.

Metadata records the requested, starting and final batches and every OOM reduction;
progress records reductions immediately. Historical `capture_memory_mib` settings
are ignored with an explicit message. The retained snapshot cap remains independent.
Progress logs separate rollout/device-copy time, route-label time and row-assembly
time. The first rollout includes compilation. Existing suites retain their frozen
batch size; start a new suite to use changed defaults. An interrupted collection
before its final save must be recollected.

## Training and scores

Each decoder draws an episode uniformly, then a represented agent uniformly,
then one of its retained snapshots uniformly. This is implemented by the exact
product probabilities, drawing with replacement. Standardization and validation
use those same weights. An epoch has `updates_per_epoch` optimizer updates,
independent of the number of stored rows. Validation BCE selects the best epoch;
test episodes are untouched until final predictions. Missing input splits are
reported as unavailable. All input/agent/time samples from an episode share a split.

Training uses stable sigmoid binary cross-entropy. The definition's threshold
controls training progress scores (default 80%). Reports provide explicit
50/75/80/90/95% display thresholds, starting at 80%, using the saved probabilities
without retraining. No threshold is automatically optimized on held-out results.
Reports show:

- Exact-path accuracy: percentage of completely matching selected-cell masks.
- Path similarity: shared cells, penalized by missing and extra cells; empty
  background earns no credit. Internally this is symmetric Dice overlap.
- Histogram: path similarity among imperfect predictions only, using balanced
  percentages and 10-point bins. Higher similarity appears further right.

The main comparison defaults to episodes available for every input/seed. Switch
to all available episodes per input to retain events from episodes with missing
base/finder messages; each input shows its availability count. All inputs use one
shared episode split. Within each input scores average snapshots, then agents,
then episodes, and across seeds. Missing events never cause synthetic input rows.
The report explains the metrics with a small cell-count example. Loss curves have
numbered epoch/loss axes, legends and selected-epoch markers, two per row.

Terminal training progress shows validation exact-path accuracy and average path
similarity as percentages, using the definition's threshold and episode/agent
balancing. It also prints the selected epoch's validation scores.
The first progress line is after one full epoch, not an untrained baseline. BCE
remains in saved loss histories and selects the best epoch/early stopping; its
absolute value is hard to interpret because most cells are off-path. A falling
BCE alone does not establish that the complete path or its overlap improved.
Reported final results use the untouched test split and can differ from these
validation scores.

## Inspectable examples

Overview → input Details → required-agent/storey/map group → representative 3D
example. All disclosures start closed. A deterministic held-out episode is chosen
independently of predictions, then held fixed across input/seed switches.

The self-contained HTML embeds Plotly once and needs no inspector server. Scenes
show the original building, reference route and predicted cells. Upright turntable
mouse/touch controls, gray striped walls, dark floor tiles, outlines, lighting,
outer-wall/roof visibility and default camera match the inspector. Prediction
cells always map back from canonical labels to original world coordinates; there
is no coordinate selector. The camera pose persists when changing inputs/snapshots.
Future collections embed authored building snapshots; older reports recover the
exact map from verified provenance or fall back to their saved collision boxes.

Select input, agent, retained snapshot and seed; rotate, change transparency and
maximum cell opacity, or show correctly selected/extra/missed cells.
Agent/snapshot selectors appear only when they offer more than one choice; the
current agent and capture step remain visible in the status line. Probability
view uses one blue with opacity = probability × maximum opacity. The local
threshold slider changes only that example; reset uses the current report
threshold. The separate held-out selector updates aggregate scores/histograms.
Missing inputs show an explicit unavailable state. Scenes are
created lazily and purged when their disclosures/ancestors close.

The example selector uses retained snapshots. Full-episode inspector playback,
extra recurrent memory, sequence decoding and multiple-route supervision remain
in [BACKLOG.md](BACKLOG.md).

Validation includes deterministic collector-to-report smoke data, unit tests for
rotation/boundary/timing/balancing, and a DOM/Plotly control-contract test. The
browser tool cannot open local `file:` HTML, so manual visual checking in the
user's browser remains useful. Smoke fixtures are explicitly not research results.
