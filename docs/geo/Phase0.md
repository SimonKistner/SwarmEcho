# Phase 0 — frozen target-information baseline

Status: **completed**. Baseline probes and the observation-variability audit
(2026-10-13) are complete. [Phase 1](Phase1.md) remains proposed;
[Phase 2](Phase2.md) retains the unscoped long-term ideas.

**Conclusion:** target storey is readily decoded, but target-cell prediction is
limited by observation ambiguity in the audited finder states. Different targets
produced exactly the same complete input histories before the encoder and GRU.
The audit provides no evidence that those distinctions were lost by either stage.

## Experiment design

Question: what target information is already available in the trained policy,
without auxiliary supervision?

- Frozen checkpoint `outputs/G01_static/checkpoints/ckpt_000330`; normal level
  `G01_static`, map `B02c_static_map_noSpawn`, seven agents, fixed base/geometry,
  700-step horizon, `chain_held` success, deterministic evaluation actions.
- Two independent prediction tasks: **target cell** and **target storey** relative
  to the base, in fixed building axes. Report groups are actual storeys 1/2/3.
  The building has 27 interior cells of 5 m; 23 permit target spawning.
- Three inputs: finder GRU (**256 dimensions**), finder outgoing packet and first
  packet stored by the base (**96 dimensions** each: signature 32 + value 64).
- Finder snapshots are taken on the first communication tick after direct
  sighting, including the discovery tick if due. The stored base packet may come
  from a relay; all labels still describe the target.
- One matched sample per **successful episode with all three snapshots**.
  Main split: **10,000 train / 1,000 validation / 5,000 evaluation**, shared across
  all combinations. Collection seed 467789; split seed 20261013.
- Detached MLPs `[256,128]`, cross-entropy, train-only normalization, validation
  early stopping; three decoder initialization seeds per combination, 18 fits.
  No gradient to the policy, no W&B for geo diagnostics.
- Metrics: exact accuracy and histograms of incorrect predictions. Cell distance
  is shortest valid six-neighbour cell transitions **around walls**, through
  clearance-safe openings/stairs; storey distance is absolute storeys off.

Detailed settings, snapshot semantics and archive keys remain in the
[code README](../../geo_experiments/README.md),
[configuration](../../geo_experiments/legacy/configs/g01_target.yaml) and linked run manifests/data.

## Completed runs

| Run | Samples / split | Mission success | Cell accuracy: GRU / finder packet / base packet |
|---|---|---|---|
| [Initial 10k report](../../geo_experiments/outputs/archive/first_target_probes_20261013_173546_856092/report.html) | 10k; 7,997 / 1,001 / 1,002 | 10,131/10,240 = 98.94% | 78.38% / 78.38% / 75.82% |
| [Main 16k report](../../geo_experiments/outputs/target_probes_16k_20261013_192420_964479/report.html) | 16k; 10,000 / 1,000 / 5,000 | 16,207/16,384 = 98.92% | 78.14% / 77.83% / 74.39% |
| [Observation audit console](../../geo_experiments/outputs/observation_audit_5k_20261013_234801/console.txt) | 5k matched; no decoder fitting | 5,009/5,056 = 99.07% | History comparison, below |

All runs had **zero successful episodes missing a snapshot**. The first two
collections overlap and use one checkpoint; they are not independent policy
replications. The earlier custom-policy evaluation was not a controlled baseline;
the final workflow uses normal levels, without `policy_config`.

Main [manifest](../../geo_experiments/outputs/target_probes_16k_20261013_192420_964479/run.json)
and [numeric data](../../geo_experiments/outputs/target_probes_16k_20261013_192420_964479/data.npz)
retain predictions, features, labels, splits and source identities. The full audit
console retains its settings and hashes. Checkpoint/map hashes match the main
run; level/runtime hashes differ, so the audit is recorded as a separate execution,
not an exact replay of the earlier source version.

## Main held-out results

Mean of three decoder seeds; ± is seed standard deviation in percentage points,
not a confidence interval. All fits use the same 5k evaluation episodes; storey
counts are 1,099 / 1,900 / 2,001.

| Task | Input | Overall accuracy ± SD | Storey 1 | Storey 2 | Storey 3 |
|---|---|---:|---:|---:|---:|
| Target cell | Finder GRU | 78.14% ± 0.72 | 67.21% | 73.25% | 88.79% |
| Target cell | Finder outgoing packet | 77.83% ± 0.38 | 65.57% | 73.30% | 88.87% |
| Target cell | Base stored packet | 74.39% ± 1.54 | 66.61% | 75.14% | 77.96% |
| Target storey | Finder GRU | 99.07% ± 0.01 | 98.85% | 98.26% | 99.95% |
| Target storey | Finder outgoing packet | 99.05% ± 0.04 | 98.76% | 98.39% | 99.83% |
| Target storey | Base stored packet | 98.89% ± 0.11 | 99.12% | 98.00% | 99.60% |

Most cell errors are one or two valid transitions off; about 95% retain the correct
storey. Finder GRU/packet performance is similar. Base results reflect later,
usually relayed information as well as decoder optimization; only 2.4% of held-out
episodes have finder = reporter. They do not isolate communication compression.

## Ambiguity evidence

The 16k archive contains **158 exact finder GRU vectors**, and the finder packet
induces the same episode groups. **67 groups / 11,207 episodes** have conflicting
target-cell labels. Their whole-dataset modal-label maximum is **79.5625%**, close
to the GRU probe's 78.14%. Base packets have 576 distinct vectors and an empirical
maximum of 81.14375%. These maxima use all labels and are in-sample diagnostics,
not held-out scores or bounds on a changed policy.

The completed audit ran at **2026-10-13 23:48:01 Europe/Berlin**, using the same
checkpoint/map and seed, **5,000 matched successes**, every step from reset through
finder capture, batch size 32. It compares raw observations, encoder outputs and
encoded observation + received TarMAC context. Timestep alignment and active/reset
flags are included; padding is excluded. Collection took **231.4 seconds**.

| Audit result | Measured value |
|---|---:|
| Exact distinct finder GRUs | 146 |
| Mixed-target-cell GRU groups | 61 |
| Episodes in mixed-cell groups | 3,344 / 5,000 = 66.88% |
| Different-cell episode pairs inside identical GRUs | 56,542 |
| Exact raw-observation history matches for those pairs | 56,542 / 56,542 = **100%** |
| Exact encoded-observation history matches | **100%** |
| Exact combined GRU-input history matches | **100%** |
| All nonreference histories matching their group's first history | 4,854 / 4,854 = **100%** |
| Different-cell reference comparisons | 1,268; **100% exact**, median target separation **4.723 m** |
| Maximum separation of targets with identical complete input histories | **9.818 m**, also across different cells |

All reference maximum-absolute and RMS differences were **zero**; none were
excluded for unequal times/lengths. The 1e-6 and 1e-3 thresholds added no matches
because all were already exact. Pair counts are dependent comparisons, not 56k
independent episodes. Target separation is straight-line metres, not route error.
Script: [audit_observations.py](../../geo_experiments/audit_observations.py).

Reporting update (2026-10-14): the provided console audit is now imported into the
main 16k report, explicitly labeled as a separate 5k execution. The raw console
and original probe data remain intact. The [report browser](../../geo_experiments/outputs/index.html)
selects historical runs and compares matching numeric table values; figures are
unaffected. Future main experiment runs collect this audit during their first
matched episodes and retain compact source arrays.

The report Measure selector defaults to episode counts and also switches incorrect-
prediction histograms between pooled prediction counts and normalized error shares.
Numeric comparison deltas use green for positive, red for negative, and neutral for
rounded zero; colours indicate the sign of the difference.

## Interpretation and limits

- For these frozen-policy capture histories, the ambiguity exists **in the raw
  observations**. The encoder and GRU cannot distinguish target labels from
  identical histories. This weakens the earlier explanation that useful target
  distinctions were present but suppressed by competing learning signals.
- High mission success with imperfect cell readout shows the current task can be
  solved without reliably identifying the exact target cell. It does not prove
  precision is useless for learning speed, other policies or harder buildings.
- Auxiliary supervision can learn a conditional target belief, but cannot make
  these identical histories encode distinct targets. A changed policy could
  produce different observations; the audit is not a universal sensing ceiling.
- Success-only selection limits claims about failures. Continuous spawning is
  not uniform over cells/storeys. Overall success selection is small, but cell
  `[2,2,1]` succeeds only 87.25% and accounts for 96/177 main-run failures.
- 99.74% of held-out finder vectors also occur exactly in training: the probes
  mainly test repeated states with new target positions, not novel internal states.
  No conclusion about unknown-map generalization, explicit mapping, or a GRU
  capacity deficit follows.

**Phase 0 is closed.** Phase 1 must account for the observation ambiguity before
interpreting an exact-cell auxiliary loss as improved information retention.
No sensing change or auxiliary training was included in Phase 0. Its original
G01 checkpoint has 136 inputs; current G01 adds a visual displacement sensor for
the [Phase 1 baseline](Phase1.md#selected-sensor-only-baseline) and has 139 inputs.

## Record-keeping for all subsequent geo experiments

Maintain numbered phase documents under `docs/geo/`. Before execution record the
question/hypotheses, controls, level/map/checkpoint, inputs/labels/timing, seeds,
sample rules and metrics. Afterwards append actual settings/provenance, original
artifact links, results, interpretation, limitations and next decision. Preserve
historical protocols; distinguish evidence from proposals and decoder seeds from
policy replications. Code/raw output remain under `geo_experiments/`; policy
checkpoints stay in normal training outputs. Phase 2 remains an unscoped backlog.
