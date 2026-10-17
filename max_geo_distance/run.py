"""Run selected max-geo-distance inspections and archive their console sections."""

import argparse
from datetime import datetime, timezone
import importlib
from pathlib import Path
from time import perf_counter

from paths import prepare_paths
from reporting import capture_console


INSPECTIONS = ("path_baseline", "corrected_baseline", "corner_patterns", "corner_simplification",
               "remaining_corner_geometry", "doorway_simplification", "lifted_corner_simplification",
               "chain_tightening", "general_corner_simplification", "runtime_chain_hold")
DEFAULT_INSPECTIONS = INSPECTIONS[:3]


def main(default_tests=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", nargs="+", choices=INSPECTIONS,
                        help="Inspections to run; defaults to the original three inspections.")
    parser.add_argument("--targets", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=456789)
    parser.add_argument("--map", default="B02c_static_map_noSpawn")
    parser.add_argument("--level", default="G01_static",
                        help="Source of roadmap, clearance, spawn-buffer, and range settings.")
    parser.add_argument("--pattern-limit", type=int, default=20,
                        help="Maximum consecutive-pattern rows; use a large value to show all.")
    parser.add_argument("--geometry-details", type=int, default=5,
                        help="Number of remaining-pair groups with detailed coordinates and geometry.")
    parser.add_argument("--runtime-device", choices=("gpu", "cpu"), default="gpu",
                        help="Device for runtime_chain_hold; GPU is required by default.")
    parser.add_argument("--runtime-batch-size", type=int, default=128,
                        help="Independent equal-agent-count environments per runtime batch.")
    # Keep previously supplied inspect_paths.py commands working.
    parser.add_argument("--corrected-baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--corner-patterns", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.targets < 1:
        parser.error("--targets must be positive")
    if args.pattern_limit < 1:
        parser.error("--pattern-limit must be positive")
    if args.geometry_details < 0:
        parser.error("--geometry-details must be nonnegative")
    if args.runtime_batch_size < 1:
        parser.error("--runtime-batch-size must be positive")

    selected = list(dict.fromkeys(
        args.tests if args.tests is not None else
        default_tests if default_tests is not None else DEFAULT_INSPECTIONS
    ))
    for name, enabled in (("corrected_baseline", args.corrected_baseline),
                          ("corner_patterns", args.corner_patterns)):
        if enabled and name not in selected:
            selected.append(name)

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    started = perf_counter()
    common, capture = capture_console()
    with capture:
        prepared = prepare_paths(args)
    prepared_at = perf_counter()
    inspections = [importlib.import_module(name).Inspection(prepared, args) for name in selected]

    for points, target in prepared.iter_targets():
        for inspection in inspections:
            inspection.observe(points, target)

    archives = []
    for inspection in inspections:
        number = INSPECTIONS.index(inspection.name) + 1
        archive = results / f"{stamp}_{number:02d}_{inspection.name}.txt"
        with archive.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(common.getvalue())
            _, capture = capture_console(handle)
            with capture:
                try:
                    prepared.report_routes(args.geometry_details)
                    inspection.report()
                except Exception as error:
                    print(f"\nInspection error: {type(error).__name__}: {error}", flush=True)
                    raise
        archives.append(archive)

    footer = (f"\nPreparation: {prepared_at - started:.2f}s; "
              f"target inspection: {perf_counter() - prepared_at:.2f}s\n")
    print(footer, end="", flush=True)
    for archive in archives:
        with archive.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(footer)
    exit_code = max(getattr(inspection, "exit_code", 0) for inspection in inspections)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
