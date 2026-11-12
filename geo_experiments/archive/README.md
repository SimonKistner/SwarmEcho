# Archived probe workflows

The active pipeline uses [data-only probe definitions](../probes/) and shared
building blocks. `target-location/shared-coordinates` and `path-predictor` are active.

- `target_location/` preserves the previous multi-map adapter, collection,
  categorical fitting and configs. Its per-map and joint map-and-cell variants
  are archived subprobes under the target-location family.
- `legacy/` preserves historical single-map target and own-cell classification,
  observation audits and YAMLs. The planned own-cell map-mask predictor is a
  separate definition under `../probes/own_cell/`.
- Reusable location report/diagnostic blocks now live in `../shared/reports/`
  and `../shared/diagnostics/`. Historical interpretation is retained.

Root `report.py`, `reproduce.py`, `run.py`, `own_cell.py`, `audit_observations.py`
and `auditing.py` remain entry points. Old explicit G01 config aliases still
resolve to `archive/legacy/configs/`. Existing outputs are not relocated.
Strict collection source checks remain in force; refitting saved arrays and
rendering saved predictions do not require accepting a changed collector.
