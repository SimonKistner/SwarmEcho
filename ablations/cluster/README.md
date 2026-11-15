# Cluster experiment reports

For cluster setup, submission, and fetch commands, see the [cluster quickstart](../../cluster/README/README.md).

## Latest auxiliary study - interim snapshot, 2 November 2026

The [G03 path auxiliary report](path-probe/g03_auxiliary_analysis_20261102/G03_auxiliary_report.html)
analyzes 21 still-running policies, their W&B diagnostic histories and matching
PFSS validation journals. It identifies insufficient recruitment, weak
hard-category curriculum gating and a detached base-message replay path as the
main issues to investigate. The strongest individual checkpoint (experiment
55329, update 1650, 72.78% validation) has been downloaded for local inspection.
The report distinguishes that single-seed result from paired variant evidence
and provides three focused ablations and a ready-to-run inspection command.

## PPO tuning source of truth - updated 1 November 2026

Start with the [completed four-wave G03 PPO refinement report](ppo-tuning/g03_refinement_analysis_20261031/G03_refinement_report.html). It includes the completed Wave 4 comparison and supersedes the earlier reports for current G03 tuning conclusions. The directory date (`20261031`) marks the analysis start; the latest report content is from **1 November 2026**.

Its current measured recommendation is **6000 environments / 50 rollout steps / 3 epochs / 6 minibatches**. The paired mean best validation reached within four hours is **71.57%**, compared with **70.15%** for five minibatches. The report describes a modest, seed-dependent lead; it does not establish a global optimum. Longer-window comparisons have different seed coverage.

The [interactive PPO landscape](ppo-tuning/g03_refinement_analysis_20261031/experiment_landscape.html) is a companion view of these results. Use the main report for interpretation, comparison windows, OOM distinctions, and limitations.

## Earlier reports - historical context

| Date | Report | Status |
| --- | --- | --- |
| 30 October 2026 | [G03 PPO wall-clock comparison](ppo-tuning/g03_clu_analysis_20261030/G03_PPO_report.html) | Earlier G03 comparison; superseded by the four-wave report for current recommendations. |
| 29 October 2026 | [G01 results and G03 proposal](ppo-tuning/g03_clu_preparation_20261029/G01_results_and_G03_proposal.md) | G01 results and the original G03 candidate proposal, before G03 refinement. |
| 29 October 2026 | [G01 capacity and PPO review](ppo-tuning/g01_capacity_review_20261029/2026-10-29_exp54837_G01_no-curriculum_capacity-and-PPO-review.md) | Historical experiment 54837 capacity and search review; contains an earlier proposal later superseded. |

## File organization

Five report files and their 19 directly linked charts/data files were moved here from `outputs/`, preserving their analysis groups and relative links. Raw run outputs, logs, analysis scripts, and the landscape template remain in `outputs/`. The source groups were:

- `outputs/g03_refinement_analysis_20261031/`
- `outputs/g03_clu_analysis_20261030/`
- `outputs/g03_clu_preparation_20261029/`
- `outputs/CLU_clu-g01-explore-20261029T011735Z-ac60e53d/analysis/` (now `g01_capacity_review_20261029/`)

When newer results are adopted, update this index and the report status notices together so there is one clearly identified current source of truth.
