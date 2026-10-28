"""Regenerate modern or historical reports from saved results; no policy runtime."""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_experiments.artifacts import load_run
from geo_experiments.common import resolve_path
# Preserve existing imports of shared presentation helpers.
from geo_experiments.report_components import (
    BACKGROUND, PANEL, TEXT, MUTED, GRID, COLORS, TASK_LABELS, _legend, _histogram, _table,
)


def generate_report(run_dir, dpi=None, audit_console=None):
    run_dir = Path(run_dir).resolve()
    manifest, arrays, legacy = load_run(run_dir)
    if audit_console is not None:
        from geo_experiments.legacy.audit_console import attach_console_audit
        attach_console_audit(run_dir, audit_console, manifest)
    if "geometry/map_offsets" in arrays:
        from geo_experiments.target_report import generate_target_report
        return generate_target_report(run_dir, manifest, arrays, dpi=dpi)
    from geo_experiments.legacy.report import generate_legacy_report
    return generate_legacy_report(run_dir, manifest, arrays, legacy=legacy, dpi=dpi)


def main():
    arguments = {}
    for argument in sys.argv[1:]:
        key, separator, value = argument.partition("=")
        if not separator or key not in ("run", "dpi", "audit_console") or key in arguments:
            raise ValueError("Use run=<completed run folder>, optionally dpi=<integer> and audit_console=<console file>.")
        arguments[key] = value
    if not arguments.get("run"):
        raise ValueError("Set run=<completed run folder>.")
    dpi = int(arguments["dpi"]) if "dpi" in arguments else None
    if dpi is not None and dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    generate_report(resolve_path(arguments["run"]), dpi,
                    resolve_path(arguments["audit_console"]) if "audit_console" in arguments else None)


if __name__ == "__main__":
    main()
