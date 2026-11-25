# Retired runtimes, curricula and evaluation tools

## Earlier environment and map workflows

The project originally used a two-dimensional environment with planar movement and a separate map/runtime path. Earlier M and B level families and a generated C corridor suite supported that development. The C suite varied straight and turning relay corridors. Its generator and presets were retired. The cellar and space-station buildings remain available as authored maps. The [README's early 2D demonstration](../../README.md) preserves the project's visual development history.

A T-maze memory experiment used fixed target/anti-target placement and masked observations to study retained information. An online memory probe accompanied it. That runtime-specific task was retired.

The earlier Streamlit architectural editor and flat-map editor gave way to the current layered building editor. Matplotlib and OpenCV video-rendering paths, per-frame fixed-width observation CSV dumps, an evaluation job queue and the older flat-map target picker were also retired. The old observation CSV labels assumed a fixed feature count and could mislabel later configurations.

## Evaluation alternatives

Noisy ensemble evaluation repeated actions or episodes with an explicit perturbation protocol and summarized robust outcome rates. The live evaluator now uses its deterministic policy mode. Older robust-rate CSVs remain readable by the artifact tools, preserving the meaning of saved experiments.

Earlier stopping selectors used training-rollout or inspection success as the gate. Current run stopping uses validation. Environment-step interval overrides for validation and inspection were also removed. Current scheduling is described in the [evaluation reference](../04_validation_evaluation_and_inspection.md#inspection-protocol-and-scheduling).

First-come success/failure video buckets and corner-case selectors preceded the current evaluation-CSV and replay workflow. Spatial failure clustering, including graph-connected and density-clustering approaches, grouped failure locations for inspection. Those tools did not change the definition of task success.

The unused failure-split configuration field had no runtime consumer in the audited implementation. Its name suggested a distinction between unseen and undelivered targets, but the old records establish no implemented behaviour for that field. Current heatmap categories are documented in the evaluation reference.

## Historical performance records

The office audit recorded about 161 s for rollout/transfer, 11 s for optimizer work per steady update and 239.5 s for final artifacts in `outputs/B01_perf_before/timings.jsonl`. Its old benchmark command refers to retired levels and reward selectors. These are measurements of that historical setup. They establish no speedup for the current branch.

The [historical evaluation recording](historical_evaluation_000572.mp4) is retained from the former documentation assets. Its filename records update 572 and an April 2026 timestamp. Numerical or implementation claims require the producing run's records.
