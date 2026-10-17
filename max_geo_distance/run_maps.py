"""Compare the current simplifier across maps, then certify all poses on GPU."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
from time import perf_counter

from paths import MAP_DIR, prepare_paths
from reporting import capture_console
from runtime_chain_hold import Inspection


def numbered_maps():
    paths = [path for path in MAP_DIR.glob("*.yaml")
             if re.fullmatch(r"B02c_static_map_\d+", path.stem, re.IGNORECASE)]
    return [path.stem for path in sorted(paths, key=lambda path: int(path.stem.rsplit("_", 1)[1]))]


def mean(histogram):
    count = sum(histogram.values())
    return f"{sum(k * amount for k, amount in histogram.items()) / count:.3f}" if count else "n/a"


def geometry_table(runs):
    print("\n--- Geometry aggregate: legal corrected baseline versus guarded final rules ---")
    print(" Map                         | Targets | No route | Invalid chain | Baseline mean | Final mean | Max k | Final >7 | Drones saved")
    print("-----------------------------+---------+----------+---------------+---------------+------------+-------+----------+-------------")
    baseline_total, final_total = Counter(), Counter()
    targets_total = no_route_total = invalid_total = 0
    for name, inspection in runs:
        baseline, _, final = inspection.simplifier.histograms
        total, no_route = inspection.args.targets, inspection.simplifier.unreachable
        saved = sum(k * amount for k, amount in baseline.items()) - sum(k * amount for k, amount in final.items())
        over = sum(amount for k, amount in final.items() if k > 7)
        invalid = inspection.simplifier.invalid_assignments
        print(f" {name:27s} | {total:7,d} | {no_route:8,d} | {invalid:13,d} | {mean(baseline):>13s} | {mean(final):>10s} | "
              f"{max(final, default=0):5d} | {over:8,d} | {saved:12,d}")
        baseline_total.update(baseline)
        final_total.update(final)
        targets_total += total
        no_route_total += no_route
        invalid_total += invalid
    saved = sum(k * amount for k, amount in baseline_total.items()) - sum(k * amount for k, amount in final_total.items())
    over = sum(amount for k, amount in final_total.items() if k > 7)
    print(f" {'ALL MAPS':27s} | {targets_total:7,d} | {no_route_total:8,d} | {invalid_total:13,d} | {mean(baseline_total):>13s} | "
          f"{mean(final_total):>10s} | {max(final_total, default=0):5d} | {over:8,d} | {saved:12,d}")
    print("Means exclude unresolved routes and invalid baselines. >7 is descriptive; seven is not a limit on these maps.")

    print("\n--- Final agent-count distribution by map and pooled ---")
    header = " Agents | Corrected total" + "".join(f" | {name:>27s}" for name, _ in runs)
    print(header + " | Final total |   All % | visual")
    print("-" * (len(header) + 38))
    for agents in range(max(max(baseline_total, default=0), max(final_total, default=0)) + 1):
        row = f"{agents:7d} | {baseline_total[agents]:15,d}"
        row += "".join(f" | {inspection.simplifier.histograms[2][agents]:27,d}" for _, inspection in runs)
        percent = 100 * final_total[agents] / targets_total
        bars = "I" * int(round(percent)) or ("." if final_total[agents] else "")
        print(row + f" | {final_total[agents]:11,d} | {percent:6.2f}% | {bars}")
    print(f"No roadmap/finder route: {no_route_total:,}/{targets_total:,}; invalid baseline: {invalid_total:,}; "
          "each I is approximately 1% of all sampled targets.")
    print("Constructed counts are feasible under the CPU geometry model; they are not global minima or runtime results.")


def runtime_table(runs):
    print("\n--- GPU aggregate: actual environment, exactly the proposed k ---")
    print(" Map                         | Targets | Tested | First connected | Hold success | Certified | Failures |  Pass % | GPU seconds")
    print("-----------------------------+---------+--------+-----------------+--------------+-----------+----------+---------+------------")
    combined, reasons, by_agents = Counter(), Counter(), {}
    targets_total = 0
    elapsed = 0.0
    for name, inspection in runs:
        summary = inspection.summary
        totals, requested = summary["totals"], inspection.args.targets
        print(f" {name:27s} | {requested:7,d} | {totals['tested']:6,d} | {totals['first_connected']:15,d} | "
              f"{totals['runtime_success']:12,d} | {totals['certified']:9,d} | {requested - totals['certified']:8,d} | "
              f"{100 * totals['certified'] / requested:6.2f}% | {summary['elapsed_s']:10.2f}")
        combined.update(totals)
        reasons.update(summary["reasons"])
        targets_total += requested
        elapsed += summary["elapsed_s"]
        for agents, _, group in summary["groups"]:
            by_agents.setdefault(agents, Counter()).update(group)
    print(f" {'ALL MAPS':27s} | {targets_total:7,d} | {combined['tested']:6,d} | {combined['first_connected']:15,d} | "
          f"{combined['runtime_success']:12,d} | {combined['certified']:9,d} | "
          f"{targets_total - combined['certified']:8,d} | {100 * combined['certified'] / targets_total:6.2f}% | {elapsed:10.2f}")
    print("Pass percentages include targets without assignments as failures; GPU seconds include first compilation per map/k.")
    print("\n Agents | Tested | First connected | Hold success | Certified | Failures |  Pass %")
    print("--------+--------+-----------------+--------------+-----------+----------+--------")
    for agents, group in sorted(by_agents.items()):
        print(f"{agents:7d} | {group['tested']:6,d} | {group['first_connected']:15,d} | "
              f"{group['runtime_success']:12,d} | {group['certified']:9,d} | "
              f"{group['tested'] - group['certified']:8,d} | {100 * group['certified'] / group['tested']:6.2f}%")
    print("\nPooled failure reasons (may overlap):")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count:,}")
    if not reasons:
        print("  none")
    print("Checks confirm sampled placed assignments only; navigation and policy success remain untested.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", help="Defaults to all B02c_static_map_<digits> YAMLs, numerically ordered.")
    parser.add_argument("--targets", type=int, default=4000, help="Target samples per map.")
    parser.add_argument("--seed", type=int, default=456789, help="Same sampling seed independently on each map.")
    parser.add_argument("--level", default="G01_static")
    parser.add_argument("--geometry-details", type=int, default=5)
    parser.add_argument("--runtime-batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.targets < 1 or args.runtime_batch_size < 1 or args.geometry_details < 0:
        parser.error("Targets and runtime batch size must be positive; geometry details must be nonnegative.")
    maps = list(dict.fromkeys(args.maps if args.maps is not None else numbered_maps()))
    if not maps:
        parser.error("No numbered B02c static maps found.")
    # This command explicitly performs the requested GPU checks; no CPU fallback.
    args.runtime_device = "gpu"
    results = Path(__file__).resolve().parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    geometry_archive = results / f"{stamp}_11_maps_geometry.txt"
    runtime_archive = results / f"{stamp}_12_maps_runtime_chain_hold.txt"
    print(f"Geometry transcript: {geometry_archive}")
    print(f"GPU transcript: {runtime_archive}")
    runs = []
    started = perf_counter()
    with geometry_archive.open("x", encoding="utf-8", newline="\n") as handle:
        _, capture = capture_console(handle)
        with capture:
            try:
                print(f"Maps: {', '.join(maps)}; targets per map: {args.targets:,}; seed per map: {args.seed}; level: {args.level}", flush=True)
                print("Phase 1: corrected legal placements and guarded full simplifier; GPU checking follows the aggregate table.", flush=True)
                for index, name in enumerate(maps, 1):
                    map_started = perf_counter()
                    print(f"\nPreparing map {index}/{len(maps)}: {name}", flush=True)
                    map_args = argparse.Namespace(**vars(args))
                    map_args.map = name
                    prepared = prepare_paths(map_args)
                    inspection = Inspection(prepared, map_args)
                    for points, target in prepared.iter_targets():
                        inspection.observe(points, target)
                    prepared.report_routes(args.geometry_details)
                    inspection.simplifier.report_guard()
                    runs.append((name, inspection))
                    print(f"Completed {name}: {sum(inspection.simplifier.histograms[2].values()):,} assignments; "
                          f"{perf_counter() - map_started:.2f}s", flush=True)
                geometry_table(runs)
                print(f"\nAll-map preparation and simplification: {perf_counter() - started:.2f}s", flush=True)
            except Exception as error:
                print(f"\nGeometry error: {type(error).__name__}: {error}", flush=True)
                raise

    with runtime_archive.open("x", encoding="utf-8", newline="\n") as handle:
        _, capture = capture_console(handle)
        with capture:
            try:
                print(f"Maps: {', '.join(maps)}; targets per map: {args.targets:,}; seed per map: {args.seed}; level: {args.level}", flush=True)
                print(f"Geometry source: {geometry_archive.name}")
                print("Phase 2: GPU checks use the exact targets and chains retained from phase 1; no resampling or rule changes.", flush=True)
                for index, (name, inspection) in enumerate(runs, 1):
                    print(f"\n========== GPU map {index}/{len(runs)}: {name} ==========", flush=True)
                    print(f"Fixed base (m): {inspection.prepared.base.tolist()}; roadmap nodes: {len(inspection.prepared.vertices):,}")
                    inspection.report()
                    # Compiled kernels capture one map's solids. Release them before
                    # compiling the next map rather than retaining every executable.
                    import jax
                    jax.clear_caches()
                runtime_table(runs)
                exit_code = max(inspection.exit_code for _, inspection in runs)
                print(f"\nStatus: {'runtime_assignment_failures' if exit_code else 'all_sampled_assignments_certified'}")
                print(f"Total two-phase time: {perf_counter() - started:.2f}s", flush=True)
            except Exception as error:
                print(f"\nRuntime error: {type(error).__name__}: {error}", flush=True)
                raise
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
