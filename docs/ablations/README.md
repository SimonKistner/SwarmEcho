# Removed mechanisms and historical experiments

> Start with the [numbered system references](../01_system_overview.md), the live code and the resolved configuration of the run being described. Consult this folder when a specific term, removed feature or historical comparison remains unexplained there. A historical result applies to its recorded source and configuration.

This folder records removed mechanisms and experimental alternatives, including the counting-radar ablation whose implementation remains available. Historical implementations may require the `final-ablation-archive` branch or the experiment's source snapshot. The files are organized by mechanism. Earlier reference guides, configuration catalogs, cleanup plans and duplicate documentation have been retired.

| Search terms | Experiment or historical context |
| --- | --- |
| Population observation, counting radar, completed observation comparisons | [Completed observation ablations](observation_ablations.md) |
| Recruitment, hold action, KEEP/YES/NO, credits, latest decision, quorum, slot reuse, reward by reference, privileged vote reward | [Population voting](population_voting.md) |
| Raw base/target vectors, visual Boolean, clock, coverage probe, vote-ratio observation, XYZ wall rays, always-inform broadcast | [Observation variants](observation_variants.md) |
| Coverage/discovery/delivery success, Euclidean gap, hard credit, redundant paths, discovery bonus, peer reward, hold-record payout, gap-change vesting, idle termination | [Mission and reward variants](reward_variants.md) |
| Simplifier, minimal roadmap, disabled-gate fallback, minimum separation, corner allowance, relay-budget override, inserted obstacles | [Geometry and feasibility variants](geometry_variants.md) |
| Feed-forward actor/critic, global mean critic, privileged graph critic, communication cadence, stored base memory, legacy entropy, raw values, action noise | [Policy and optimization variants](policy_optimization_variants.md) |
| Reverse curriculum, classic population stages, G04, auto7 sweep, previous-probe feedback, historical best-checkpoint comparisons | [Curriculum and feedback experiments](curriculum_experiments.md) |
| Binary-sighting checkpoint, Phase 0, Phase 1, target ambiguity, map-specific/joint readouts, own-cell classification | [Historical observation and probe evidence](observation_ambiguity.md) |
| 2D runtime, M/B/C level families, C corridor generator, Streamlit map editor, MP4 renderer, noisy ensemble, stopping sources, step scheduling | [Retired runtimes and tools](retired_runtimes_and_tools.md) |

Current optional mechanisms are documented in the main references. Success-weighted target/map sampling and randomized bases are explained in [environment](../02_environment_and_physics.md#target-placement-and-base-positions) and [training](../03_marl_and_training.md#success-weighted-cell-and-map-sampling). The completed population-channel and radar comparisons are recorded in [observation ablations](observation_ablations.md).

Original experimental artifacts can live outside the tracked documentation. The probe evidence records their original relative locations as text so missing local reports cannot be mistaken for reproduced results. The [curriculum report](G03_curriculum_auxiliary_report_20261103.pdf) and [historical evaluation recording](historical_evaluation_000572.mp4) are retained assets from the earlier archive. The recording's original filename was `eval_update_000572_20260418_151951_20260418_151952_20260418_151952.mp4`.
