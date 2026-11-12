# Shared building blocks

- `definitions.py`, `inputs.py`, `axes.py`: definition contracts, acquisition
  capabilities and readout axes. Input choice is independent of prediction target.
- `rollout.py`, `collector.py`, `path_collection.py`: the one frozen-policy acquisition service, event
  capture, eligibility, quotas, target construction and episode partitions.
- `targets.py`, `cell_labels.py`, `path_geometry.py`: target/output blocks,
  canonical shortest routes and cell geometry utilities.
- `runner.py`, `execution.py`, `fitting.py`, `training.py`, `mask_training.py`,
  `sampling.py`: the one execution path
  from definition to collection, supervised MLP fits and reports.
- `reports/`, `diagnostics/`: reusable registered report/diagnostic blocks.
- `report_layout.py`, `report_components.py`, `dashboard.py`, `dashboard.js`,
  `naming.py`: shared presentation shell, navigation and comparisons.
- `common.py`, `artifacts.py`, `cache.py`, `console.py`: paths, archives,
  provenance, strict collection checks and training-style console formatting.
- `policy.py`, `validation_maps.py`, `geometry.py`, `audit_common.py`: policy/map
  loading, cell topology and vector/history comparison helpers.

All current runnable definition code is here. `../probes/` contains definitions
and notes; previous workflows are isolated in `../archive/`. See the full
[contract and extension plan](../CONCEPT.md).
