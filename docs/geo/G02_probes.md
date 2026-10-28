# G02 frozen target readouts

Scoped and implemented 2026-10-24; the solo collection has been gathered in WSL
and analysed from saved snapshots. MLP fitting results remain pending.
This promotes only the frozen five-map target diagnostics from the
[Phase 2 backlog](Phase2.md). No auxiliary loss, policy training or unknown-map
generalization experiment is included. Historical [Phase 0](Phase0.md) and
[Phase 1](Phase1.md) archives remain the reference for their original protocols.

## Question and three alternatives

Can the final policy's finder GRU, outgoing packet and first base-stored packet
support location readout across its five familiar maps? Compare:

1. **Map-specific**: five independent cell MLPs for each input/seed, routed by known
   map identity. This measures within-map decoding with map identity supplied.
2. **Joint map + cell**: one MLP per input/seed over the 135 map–cell classes. Both
   identity and location must be inferred. Additional map-only/coordinate-only
   diagnostic accuracies distinguish these errors.
3. **Shared coordinates**: one MLP per input/seed over the common 27 absolute XYZ
   cells. This tests a common coordinate readout, without requiring map identity.

Five layouts share grid dimensions, but have different bases and traversability.
Labels therefore use absolute grid XYZ for the new alternatives; legacy G01 labels
retain base-relative semantics. Distances use the true map's own cell graph. A
wrong-map prediction has no meaningful cross-building route distance and is
reported separately, not silently reassigned to the true map.

## Default execution and controls

As of 2026-10-26 the checked-in default selects the separate
[G03 validation-map shared-coordinate experiment](G03_probes.md). Existing G02
suite folders retain their frozen execution plans; resume them normally or pass
their saved `config.yaml` to start another suite with the same choices. The
authored-map protocol below is unchanged and does not opt into validation maps.

The original [`geo_experiments/probes.yaml`](../../geo_experiments/probes.yaml) scheduled all
three alternatives for the newest solo YES/NO checkpoint and the newest final
seven-agent curriculum YES/NO checkpoint. The local plan check selected solo
`ckpt_004230` and curriculum `ckpt_001666`. Latest choices are resolved once and
frozen before execution. The plan contains no own-cell experiment. Storey-head
prediction is disabled, with a configuration toggle retained.

One collection per policy supplies 10k/1k/5k total train/validation/test episodes,
equally allocated to maps in each split. All alternatives share target episodes,
features and split membership. Each specialist uses its map's subset of that
same partition; pooled decoders get all maps. Independent `[256,128]` MLPs use
training-only normalization, validation early stopping, and three decoder seeds.
This varies readout architecture/label semantics, not the frozen policy or input
features. Specialists have five times the total decoder parameters, and each sees
less data than a pooled fit; this is not a parameter-matched capacity experiment.

Unlike Phase 0, eligibility requires **delivery to the base**, not eventual chain
success. Discovery/first outgoing packet and first base receipt timing are retained.
Chain outcome remains available for distribution highlights and subsequent analysis.
Adaptive movement/vote execution follows the normal deterministic evaluator.

“Target-required agents” is the production reset-time relay-estimator count for
that episode's target, preserved across terminal transitions. It groups targets
across buildings; it is neither observed deployed population nor a proof of the
minimum possible chain. Zero is labelled “Not estimated.” These categories and
storeys appear in held-out details and accuracy charts.

## Conditional ambiguity analysis

The initial screen hashes a seeded subset of up to 5,000 retained episodes for
each of the three inputs. Its rate counts all members of exact repeated-latent
groups. When every rate is below 1%, deep history replay and the entire audit
results section are omitted; the footer records rates to three decimals and sample
count. A triggered source is profiled across all retained saved snapshots. Base
packet and outgoing-message collisions use this saved-data analysis only. It
reports repeated-vector group sizes, label conflicts under each decoder, categorical
similarities and their dataset baselines, timing/spatial ranges and upstream vector
diversity. Repetition with one task label is redundancy, not localization ambiguity.

Only threshold-triggering **finder-GRU** groups with conflicting labels need raw,
encoded and received-context histories. Replay is restricted to the screened
conflicting episodes and stops at finder capture. Packet collisions never trigger
finder replay. All contracts reuse the same histories when relevant: experts
condition on map, joint labels include map, and shared labels use coordinates only.
Exact reproduction and missing finder captures are disclosed; GPU batch shape can
alter numerical trajectories. No causal relay-loss claim is made from snapshots.

Future collection records active-agent counts at finder capture and base receipt.
The existing solo archive lacks those fields and displays “Not recorded.” It is
reused without generating histories to recover missing population metadata.

## Saved solo collection audit (2026-10-24)

The [actual collection audit](../../geo_experiments/outputs/archive/G02_target_ablations_20261024_154056_513705/collections/31984fcb8f77c0eb/audit.html)
uses 16,000 retained episodes from solo checkpoint `ckpt_004230`. The 5,000-sample
screen found 0% repeated finder GRUs/messages and 3.940% repeated base messages.
Across the full collection, 635 samples (3.969%) belong to 27 repeated base-packet
groups. Sixteen groups contain different target cells, affecting 499 samples
(3.119%); affected does not mean all 499 must be misclassified. All 27 groups stay
within one map and have distinct finder GRUs and finder messages within each group.
Eighteen groups agree on target-required agents and 22 on target storey.
No observation replay is required. This is a saved-snapshot diagnostic, not an MLP
result or proof of where information is lost along communication.

## Interpretation and evidence limits

Known-map specialists, joint identity+position decoding and shared coordinates
have different information conditions; equal raw accuracies do not make them
equivalent. A joint MLP can internally specialize by map. None demonstrates a
geometry reconstruction, causal policy use of location, or transfer to an unknown
building. Decoder-seed variability is not policy-training uncertainty. Delivery
conditioning still excludes episodes that never delivered the target.

The implementation was checked with a hardware-free six-job/two-collection plan,
saved G02 config loading, synthetic renders of all three target layouts, browser
distribution controls, and re-rendering the original G01 target archive. Synthetic
previews are explicitly labelled and are not experimental measurements. The solo
snapshot diagnostics above use actual collected data; MLP results and new GPU
history execution are not claimed by the Windows editing session.

For commands, archive layout, resume/reproduction and report semantics see the
[runner guide](../../geo_experiments/README.md).
