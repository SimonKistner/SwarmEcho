# Phase 1 — static spatial diagnostics and auxiliary supervision

Status: **sensor and visitation-weighted own-cell runs reported; balanced own-cell
revision implemented**, 2026-10-14.
Auxiliary loss remains **proposed**; auxiliary training has not been run.
Completed baseline and console ambiguity audit: [Phase 0](Phase0.md).
Further ideas: [Phase 2](Phase2.md).

Phase 0 closure changes the premise: all 56,542 audited different-cell pairs
within identical GRUs had identical raw/encoded/combined input histories. The
auxiliary proposal below predates that result. The selected next observation
condition is direct-visible target displacement, implemented as described below;
its new PPO-only baseline comes before auxiliary training.

## Selected sensor-only baseline

The user approved `env.observe_visually_gated_target_displacement`: default off,
enabled for `G01_static`. Add three values, `[dx,dy,dz] / visual_radius`, only while
the drone directly sees the target. Existing direct-sighting range/occlusion and
active-agent rules gate the measurement; persistent target knowledge does not.
Use the existing binary sighting flag for validity and the existing velocity
history for motion integration. Explicit odometry/displacement features were
dropped as redundant engineered inputs; multiple-view bearing-only estimation
is unnecessary for this first bearing-plus-range measurement.

This is an idealized calibrated stereo/depth-camera measurement in the same axes
as velocity, without camera FOV or measurement noise. Realism rationale and
limitations are in the [environment reference](../01_reference/02_environment_and_physics.md#observations).
G01 now has **139 inputs**, so the original **136-input** Phase 0 checkpoint is
not compatible with this enabled sensor. Train a fresh sensor-only checkpoint,
then run the same detached probes with that checkpoint before adding the loss.
The original Phase 0 results remain the historical binary-sighting baseline.
The normal experiment runner now includes a 5k ambiguity audit alongside detached
probes. Its report browser compares matching table values across baseline and
later runs while leaving figures unchanged. No extra standalone audit rollout is
needed for the new sensor-only baseline.

Use a separate training run name to retain those original artifacts:

```bash
uv run swarmecho-train level=G01_static logging.run_name=G01_static_visual
```

Run record: [visual target probes](../../geo_experiments/outputs/visual_target_probes_20261014_141641_127517/report.html),
checkpoint `outputs/G01_static_targetVisual/checkpoints/ckpt_000351`: 16k samples,
16,089/16,320 successful missions (98.58%). The supplied report shows target-cell
accuracy of **98.2% GRU**, **97.9% outgoing message**, **83.4% base message**.
This supports target localization from the new finder features; target cell is
less decodable from the stored base packet with these probes. Transmission,
sender selection and probe fit remain candidate explanations. It does not establish an auxiliary-loss
benefit or isolate policy-seed effects. Source data and provenance remain in the
linked run folder. The integrated audit found **5,000 distinct GRUs in 5,000 samples**,
with no conflicting-cell exact-GRU groups; this removes the observed exact-state
collision from the earlier sample but does not rule out approximate ambiguity.
No auxiliary head,
observation-noise model or camera-FOV change is included in this implementation.

## Own-cell diagnostic (scoped 2026-10-14)

Question: can detached MLPs decode each active drone's **current cell**, from its
updated GRU and current signature+value token, throughout an episode? Target
finding, mission success and base delivery do not gate samples. This is an
additional frozen-policy diagnostic before deciding on auxiliary supervision.

- One cell-class task, two inputs, three decoder seeds; the existing MLP,
  normalization, validation selection, accuracy and valid-route error metrics.
- Regular sampling at `ceil((cell_size / max_speed + buffer_seconds) / dt)`
  simulator steps, starting at step zero. Buffer defaults to 0.2 s: G01 gives
  7 steps / 0.7 s. Sample only active slots, before motion, paired with that
  observation's updated GRU/token; exclude positions outside interior cells and
  report their number. Outgoing tokens need not be transmitted on the sample tick.
- Retain 10k training, 1k validation and 5k evaluation snapshots, collected in
  separate episode pools in that order. Never split snapshots from one episode
  across pools. Episodes include failures and pre-discovery observations.
- Fingerprint complete raw, encoded and encoded+received-context histories
  at every step through each sample, with reset flags and lifetime resets. Dedup
  by exact combined input history **and own cell**, retaining conflicting labels
  within a pool. Histories retained in an earlier pool are excluded from later
  pools irrespective of label, preventing shared-prefix leakage. This does not
  presume every pre-discovery trajectory is identical and does not discard all
  pre-discovery samples. Randomize eligible samples within each episode when
  capping the final episode to the pool quota.
- The revised audit groups a seeded subset of retained snapshots by exact GRU
  or message and counts different-cell pairs plus matching full-history pairs.
  Repeated same-cell states are redundancy, not positional ambiguity. Store
  fingerprints, episode/agent/time identities, positions, discovery flags,
  before-filter cell counts and filter counters as source data.
- Report matches the dark target report: one accuracy chart, own-cell error
  histograms, per-actual-storey details, coverage/split distributions, validation
  curves, confusion matrices and compact audit tables. Semantic metric names
  keep own-cell and target-cell comparisons separate in the shared dashboard.

Limits: held-out accuracy measures new retained input histories on this map,
not the raw time-weighted visitation distribution. Deduplication changes weights;
coverage charts and pre/post-discovery counts expose that selection. Snapshots
within an episode remain correlated. Identical floats and history fingerprints
are exact diagnostics, not a tolerance-based sensor-noise study. No experiment
execution is authorized by this design record alone.

Protocol revision (2026-10-14): the first whole-episode run
[visual_own_cell_probes](../../geo_experiments/outputs/visual_own_cell_probes_20261014_150714_023523/report.html)
used the deduplicated visitation distribution, without cell quotas. The supplied
distribution plot shows strong concentration in a few cells/storeys. Preserve
that run as the visitation-weighted baseline. Source data confirms **16k retained
snapshots from 320 episodes**, with per-cell counts ranging from **0 to 3,223**.
Cell `[0,2,1]` (absolute grid coordinates) has **zero candidate snapshots**, not
just zero after filtering, in that run. No inference that it is permanently
unreachable follows from this finite sample, but strict buckets may stall there.

The user now requests **uniform own-cell buckets**. Future own-cell runs allocate
each split's total across **all authored interior cells**: each cell gets
`floor(total / number_of_cells)` or one extra. A seeded cell order rotates the
remainders across splits, keeping combined counts within one sample as well.
Reject candidates for full buckets before accepting them into the prefix filter;
rejected samples must not claim history ownership. Deduplication and episode-pool
separation remain in force. Sampling does not relocate drones or change policy
behaviour. Persist per-cell limits, retained counts and rejection counts; print
underfilled cells when the episode cap is reached and save partial data rather
than redistributing missing quotas to common cells. Disable this revision with
`sampling.uniform_cells=false` only for a separately named visitation control.

This changes the question to accuracy under an approximately uniform **cell-label
distribution**, not natural visitation frequency. Rare cells may substantially
increase collection cost or prevent completion. Exact-history audit rates are
also conditional on the balancing rule. The all-cell balanced protocol was started:
the user-provided progress log reaches 8,386/10,000 training samples after roughly
1,112 episodes, with 9 cells still underfilled. The slowest include `[0,2,1]`
(6/371) and `[0,0,2]` (8/370). These are interim collection counts, not completed
probe results; historical results must not be relabeled as balanced.

Further revision (2026-10-14): the user proposes limiting buckets to generally
well-visited cells. Add an explicit `sampling.exclude_cells` list, using **absolute
map-grid XYZ indices**, initially the five coordinates printed in the log:
`[0,2,1]`, `[0,0,2]`, `[2,2,1]`, `[1,1,2]`, `[0,1,1]`.
This leaves **22 eligible cells** for this map. Balance each split and the combined
budget over that fixed subset; reject excluded-cell candidates without registering
their histories. Keep all 27 cells and the full route graph for class coordinates,
source coverage and distance errors. Reports explicitly identify excluded cells
and restrict overall accuracy claims to eligible **actual** cells. Predictions
still use the full cell-class space; an excluded-cell guess can be an error.

Exclusions are chosen from collection coverage before the new probe results,
not from accuracy. Preserve the all-cell attempt and the original visitation run.
The user's visual-range explanation is plausible but not verified here; sparse
occupancy does not prove absent map/target information in the GRU. This restricted
protocol answers own-cell decoding only on the selected occupancy-supported subset.
It has not been run. The already-running process will retain its loaded config;
the revised list applies to a newly started run.

Latest selection (2026-10-14): the 22-cell run still has uneven collection rates.
The user's log reaches 7,491/10k training snapshots near episode 400. At the saved
[v3 progress snapshot](../../geo_experiments/outputs/visual_own_cell_probes_v3_20261014_164812_062225/run.json)
read during this discussion, collection had advanced to 7,946 samples / 524
episodes. These are interim counts, not a completed result.

The user explicitly **cancelled time caps and adaptive trimming** and selected
static removal of the five newly printed slow cells: `[2,0,0]`, `[0,1,0]`,
`[0,1,2]`, `[1,1,1]`, `[0,0,0]`. Together with the original five exclusions,
this leaves **17 eligible cells**. Keep fixed **10k/1k/5k** split sizes and uniform
floor/ceil buckets over that subset. No time-cap settings or adaptive selection
remain in the implementation. The earlier occupancy-based scope limitations
still apply, and this 17-cell protocol has not yet been run.

## Scope and question

After the sensor-only baseline, add a target-cell prediction head to the actor
GRU and jointly optimize its prediction loss with PPO. Do we improve:

1. Target information in the GRU and, indirectly, outgoing/base messages?
2. Mission learning speed, completion time, robustness or attainable success?

These are separate outcomes. Phase 0 already has about 98.9% mission success;
there is little absolute headroom on this task. A more precise internal readout
can help learning speed without moving final success, or improve readout alone.
This experiment can answer the static-task question with paired controls; it
cannot establish transfer to unknown layouts or explain the original training
process uniquely.

## First auxiliary loss: proposed design

The user's proposed mechanism is the standard shared-representation auxiliary
pattern: a supervised head reads recurrent state, and a weighted differentiable
loss sends gradients into that state and its upstream network. Established
precedents include shared auxiliary representation learning in
[UNREAL](https://arxiv.org/abs/1611.05397) and joint depth/loop-closure supervision
for recurrent navigation in
[Learning to Navigate in Complex Environments](https://arxiv.org/abs/1611.03673).
These support the mechanism; they do not establish a best head, weight or expected
benefit for SwarmEcho's recurrent PPO and TarMAC setup.

| Choice | First proposal | Reason |
|---|---|---|
| Input | Current actor GRU, 256 dimensions | Directly targets the bottleneck observed in Phase 0 |
| Head | `256 → 128 → 27 logits`, ReLU hidden activation | Simple MLP with the same cell-class output format as the detached probe; architecture need not be identical |
| Classes | Existing 27 interior-cell classes and base-relative XYZ ordering | Preserves the established label and report contract, including four excluded spawn cells |
| Supervision time | First finder communication tick after direct sighting, matching Phase 0 | Makes the initial intervention easy to compare |
| Mask | One valid active finder event per episode; clear event state on reset and retain it across rollout boundaries | Avoids rewarding arbitrary uninformed drones or repeating supervision every timestep; a valid first observation after reset is still eligible |
| Episode inclusion | Every episode with that finding event, including episodes which later fail | Training must not depend on a future successful outcome |
| Loss | Ordinary target-cell cross-entropy | Smooth learning signal for the correct discrete label; no arbitrary geometry shaping in the first trial |
| Weight | Initial pilot **lambda = 0.1**, with event-normalized loss | A starting guess to check gradient scale, not a literature-guaranteed optimum |
| Deployment | Auxiliary head optional for diagnostics; target truth supplied only to training loss | No privileged target coordinates inserted into observations or communicated packets |

For event mask m and true cell y:

```text
p = softmax(aux_head(h))
L_aux = sum(m * -log(p[y])) / max(1, sum(m))
L_total = L_existing_PPO + lambda * L_aux
```

Zero eligible events gives zero auxiliary loss. Normalize by eligible events,
not all agents/timesteps, so event rarity does not silently dilute the weight.
Log eligible counts to detect empty updates or incorrect masks.

This optimizes probability assigned to the true cell. Accuracy and distance of
the argmax guess are evaluation metrics and are not differentiable training
losses. If exact-cell CE improves encoding but produces undesirable error geometry,
a subsequent explicitly separate variant could add expected true-map graph
distance `sum_c p[c] * d_graph(c,y)` with appropriate normalization and handling
of disconnected cells. That variant is not the first experiment.

## How it would fit the current implementation

The current actor computes observation encoding and TarMAC context, updates the
GRU, then produces action distribution and outgoing signature/value. PPO already
replays recurrent minibatches through that actor in
[`training/ppo.py`](../../src/swarmecho/training/ppo.py); its replay currently
discards hidden states from the emitted sequence outputs.

- During rollout, retain the numeric true cell and the finder-event mask alongside
  existing recurrent minibatch data. Carry finder/discovery event state correctly
  across 100-step rollouts, communication ticks and episode resets.
- During **gradient-enabled PPO replay**, compute the head from the freshly
  recomputed hidden state and include CE in the existing objective. The frozen
  `geo_experiments` archive or detached rollout hidden states cannot provide the
  intended gradient into the GRU.
- Gradients reach the head, GRU and its upstream encoder/context computation
  within the existing truncated recurrent replay. The loss does not directly
  supervise action-head layers or the separate critic, and does not differentiate
  through simulator motion or observations. Shared representations can change
  actions and communication indirectly during joint training.
- Initially use only the finder-GRU auxiliary head. Re-run all six **detached**
  Phase 0 probes afterwards to test whether its improvement is transmitted to
  messages and the base. Training-head accuracy alone is not independent evidence.

Once the observation condition is agreed, keep map, sensors, reward, action noise
protocol, agent count, GRU width, horizon and other optimization settings identical
between its PPO-only and PPO-plus-auxiliary conditions. If observations change,
establish a corresponding sensor-only baseline; do not attribute sensor gains to
the loss by comparing only against the original Phase 0 checkpoint.
Normal policy training may use its normal W&B logging for auxiliary loss/count/
accuracy and policy metrics. Geo probe reports remain local and do not use W&B.

## Experiment sequence and decisions

| Step | Question and comparison | Evidence expected | Limits / decision |
|---|---|---|---|
| P1-A: minimal history audit | Observable distinction versus aliased history? 5k finder snapshots, raw/encoded/combined histories within identical GRUs. | **Completed:** all 56,542 different-cell pairs match exactly at all stages; target separation up to 9.818 m. | Ambiguity is present before encoding for these cases. Agree the observation condition before interpreting auxiliary gains. |
| P1-B: short checkpoint pilot | Does the loss run, receive gradients, and preserve task performance? Fine-tune from the baseline checkpoint; use matched PPO-only continuation if attributing gains to the loss. | Nonempty masks, finite CE, appropriate weighted gradient scale, no substantial mission regression | Pilot is a compatibility/signal check. It cannot establish faster learning from scratch. |
| P1-C: paired fresh training | Does supervision change learning speed or final skill? Same normal level and training settings, with/without loss, ideally at least three paired policy seeds. | Success versus environment steps, fixed-budget final evaluations, completion time and per-cell failures | Policy-seed variation matters; three decoder seeds are not three policy runs. More replications if effects are small or unstable. |
| P1-D: fresh detached probes | Does extra information reach GRU, finder packet and base packet? Same two tasks, three inputs, 10k/1k/5k protocol at comparable checkpoints. | Exact accuracy and valid-distance error histograms, collisions and empirical maxima, per-storey results | Do not infer policy benefit from probe accuracy alone. More distinct floats do not by themselves mean better target information. |

For P1-C, use the same training-step budget and checkpoint/evaluation schedule.
The level currently has a mastery early exit; for a fixed-budget comparison it
must be controlled consistently in both conditions. Preserve its original config
in the record and state the comparison override rather than changing G01 silently.
Track steps to stable 90/95/99% success, learning-curve area, final success and
episode duration; wall-clock cost is secondary because the head adds computation.
Define the stability window before running and avoid selecting an isolated noisy
threshold crossing. Use independent evaluation seeds and enough episodes to
resolve changes near the present success ceiling.

Probe protocols should use new data with independent seed namespaces from policy
training and shared source/split rules. Across policies, use a common attempted
target/reset evaluation panel. Changing success or finder identity changes the
conditional successful-only probe dataset: report those changes and, where
needed, compare the intersection of successful episodes as a sensitivity check.
Keep all-attempt mission metrics separate from successful-only probe accuracy.

## What possible results mean

| Observation | Supported interpretation | Next decision |
|---|---|---|
| Better GRU readout and better mission learning | Target supervision is useful in this static setup | Replicate, inspect message transfer, then consider Phase 2's precision/map variation steps |
| Better GRU/message readout, unchanged mission results | Representation can encode more, but measured task benefit remains unproven | Check ceiling and completion-time/speed measures; do not claim improved coordination |
| Better GRU, unchanged finder/base packet readout | Additional information is not yet decoded from communication | Inspect transmission/readout and consider a separately scoped packet auxiliary head |
| No readout gain, identical complete histories | First-event observational ambiguity is a plausible limit | Consider later observation/timing or controlled information gathering; a larger head cannot separate identical inputs |
| No readout gain, relevant history differences exist | Weight, optimization, recurrent credit assignment or feature compression remain candidates | Diagnose gradient scale/timing before adding multiple losses |
| Readout improves while success worsens | Representation objective conflicts with task optimization at this setting | Reduce/adjust weight or reject the intervention; do not promote it based on probe gains |

## Choices still open before implementation

- The visual sensor-only checkpoint/probes are now recorded above. Run the
  scoped own-cell diagnostic and select a matched PPO-only control for auxiliary
  comparisons; policy-seed replication remains open.
- Pilot duration, gradient-scale checks and a small follow-up weight range if
  needed; 0.1 is the first proposed value, not a final tuned choice.
- Paired policy-seed set, fixed budget, evaluation panel and threshold stability
  window. Report the chosen settings before seeing outcomes.
- Exact event-mask storage/recurrent boundaries and checkpoint contract for the
  optional head. Maintain compatibility with ordinary training and frozen probes.

Continuous coordinates, five random maps, map reconstruction, paths and own-position
**auxiliary training heads** remain [Phase 2 ideas](Phase2.md). The detached own-cell
diagnostic is now scoped above, without auxiliary policy gradients. This is an organizational scope choice;
their longer-term ordering is retained there. Add each Phase 1 execution and its
interpretation here using the [record-keeping practice](Phase0.md#record-keeping-for-all-subsequent-geo-experiments).
