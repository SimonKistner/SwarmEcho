# Historical observation and probe evidence

These measurements concern older observation layouts and archived readout workflows. The original binary-sighting checkpoint had 136 inputs. Its sensor-only successor had 139 inputs under that revision. Current G03 has a different 140-input contract, explained in the [environment reference](../02_environment_and_physics.md#observations). Historical labels such as Phase 0 and Phase 1 identify those experiments.

The recorded output paths identify the original artifacts. Their availability depends on the local experiment archive. The tables below preserve the reported measurements and their limitations.

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
`geo_experiments/README.md` (code README),
`geo_experiments/archive/legacy/configs/g01_target.yaml` (configuration) and linked run manifests/data.

## Completed runs

| Run | Samples / split | Mission success | Cell accuracy: GRU / finder packet / base packet |
|---|---|---|---|
| `geo_experiments/outputs/archive/first_target_probes_20261013_173546_856092/report.html` (Initial 10k report) | 10k; 7,997 / 1,001 / 1,002 | 10,131/10,240 = 98.94% | 78.38% / 78.38% / 75.82% |
| `geo_experiments/outputs/target_probes_16k_20261013_192420_964479/report.html` (Main 16k report) | 16k; 10,000 / 1,000 / 5,000 | 16,207/16,384 = 98.92% | 78.14% / 77.83% / 74.39% |
| `geo_experiments/outputs/observation_audit_5k_20261013_234801/console.txt` (Observation audit console) | 5k matched; no decoder fitting | 5,009/5,056 = 99.07% | History comparison, below |

All runs had **zero successful episodes missing a snapshot**. The first two
collections overlap and use one checkpoint; they are not independent policy
replications. The earlier custom-policy evaluation was not a controlled baseline;
the final workflow uses normal levels, without `policy_config`.

Main `geo_experiments/outputs/target_probes_16k_20261013_192420_964479/run.json` (manifest)
and `geo_experiments/outputs/target_probes_16k_20261013_192420_964479/data.npz` (numeric data)
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
Script: `geo_experiments/audit_observations.py` (audit_observations.py).

Reporting update (2026-10-14): the provided console audit is now imported into the
main 16k report, explicitly labeled as a separate 5k execution. The raw console
and original probe data remain intact. The `geo_experiments/outputs/index.html` (report browser)
selects historical runs and compares matching numeric table values; figures are
unaffected. Future main experiment runs collect this audit during their first
matched episodes and retain compact source arrays.

## Interpretation and limits

- For these frozen-policy capture histories, the ambiguity exists **in the raw
  observations**. The encoder and GRU cannot distinguish target labels from
  identical histories. This weakens the earlier explanation that useful target
  distinctions were present but suppressed by competing learning signals.
- High mission success with imperfect cell readout shows that historical task could be
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


## Historical sensor-only comparison

The successor added visually gated target displacement to the older layout. Its complete configuration differs from the current preset.

Run record: `geo_experiments/outputs/visual_target_probes_20261014_141641_127517/report.html` (visual target probes),
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
observation-noise model or camera-FOV change was included in that sensor-only run.

## Earlier 9a policy notes

The earlier 9a observation setup exposed peer radar without the later line-of-sight fix. A wall could block communication while peers still appeared in radar. The recorded fix added solid-intersection checks to all three peer groups. Its target observation also preceded the maintained visible bearing/range input. These differences matter when interpreting its relay behaviour.

Its last cited training record averaged about 5,831 reward from recurring partial-chain shaping and 9,306 from success. The accompanying notes proposed that stable partial-chain income could discourage risky rerouting, and that discrete selected-path attribution could change credit between nearly interchangeable drones. Those were diagnostic hypotheses about that policy. The notes provide no controlled intervention result. The run used a 100-step recurrent training segment and gamma 0.99, giving a reward 200 steps later about 13% of its immediate discounted weight.

## Archived own-cell classification attempts

The October 2026 own-cell diagnostic fitted detached classifiers to each active drone's current cell from its GRU or outgoing token throughout the episode. It sampled pre-motion active states at a cadence derived from cell size, maximum speed, a 0.2 s buffer and `dt`. The G01 cadence was seven steps. It separated episode pools for 10k/1k/5k splits and filtered shared input-history prefixes across pools. It retained conflicting labels within a pool for ambiguity analysis.

The completed visitation-weighted collection contained 16k snapshots from 320 episodes. Cell counts ranged from zero to 3,223. Cell `[0,2,1]` had no candidate snapshots in that collection. The source report was `geo_experiments/outputs/visual_own_cell_probes_20261014_150714_023523/report.html`.

A subsequent all-cell balanced collection reached 8,386/10,000 training snapshots at about 1,112 episodes, with nine cells still underfilled. Static exclusions reduced the eligible set first to 22 cells and later to 17. The 22-cell attempt reached a saved progress record of 7,946 snapshots from 524 episodes. Its unfinished output was removed during the October 2026 experiment cleanup. These are collection-progress records, with no completed balanced-probe result established by the old notes.

The first excluded cells were `[0,2,1]`, `[0,0,2]`, `[2,2,1]`, `[1,1,2]`, `[0,1,1]`. The next exclusions were `[2,0,0]`, `[0,1,0]`, `[0,1,2]`, `[1,1,1]`, `[0,0,0]`. Exclusion decisions came from occupancy before the new readout results. Accuracy under those rules would describe the selected cell population. Sparse visitation in a finite sample establishes no permanent geometric inaccessibility.

## Archived map-specific and joint target readouts

The five-map G02 protocol compared five map-specific cell classifiers with one joint 135-class map-and-cell classifier. Specialists received map identity through routing, while the joint classifier had to recover identity and cell together. A shared 27-cell coordinate readout served as another comparison. The maintained shared-coordinate protocol is described in the [geometry experiment documentation](../../geo_experiments/README.md).

One frozen-policy collection supplied matched 10k/1k/5k train/validation/test episodes, equally allocated across five familiar maps. Inputs were finder GRU, outgoing packet and first base-stored packet. Eligibility required base delivery, with chain outcome retained for analysis. Detached `[256,128]` MLPs used train-only normalization, validation stopping and three decoder seeds. Specialist systems had five times the total decoder parameters and less data per model. A wrong-map prediction had no meaningful cross-building route distance.

The saved solo collection used checkpoint `ckpt_004230` and retained 16,000 episodes. Its 5,000-sample screen found zero repeated finder GRUs/messages and 3.940% repeated base messages. The full collection contained 635 samples in 27 repeated base-packet groups, or 3.969%. Sixteen groups had conflicting target cells, affecting 499 samples, or 3.119%. All 27 groups stayed within one map and had distinct finder GRUs and outgoing messages. Eighteen groups agreed on required-agent count and 22 on target storey.

The original audit is `geo_experiments/outputs/archive/G02_target_ablations_20261024_154056_513705/collections/31984fcb8f77c0eb/audit.html`. These saved-snapshot statistics establish repeated packets and conditional label ambiguity. They provide no fitted MLP result or causal localization of communication loss. The original plan also selected curriculum checkpoint `ckpt_001666`. A selected checkpoint establishes a plan entry, with execution and outcomes requiring their own records.

