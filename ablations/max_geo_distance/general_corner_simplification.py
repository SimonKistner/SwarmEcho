"""Inspection 9: broaden the existing connector construction to all drone pairs."""

from collections import Counter
import numpy as np

from chain_tightening import prune_relays
from corrected_baseline import corrected_chain
from corner_simplification import simplify_chain
from doorway_simplification import OpeningCatalog, simplify_doorways
from lifted_corner_simplification import lifted_connector, simplify_lifted
from paths import _visible_from
from remaining_corner_geometry import coordinates
from placement_checks import chain_reason, rounded


def scan_general(chain, target, prepared, outcomes, accepted_types, cache, cache_stats):
    route = [prepared.base.copy(), *(point.copy() for point in chain), target.copy()]
    index = 1
    windows = 0
    events = []
    while index + 2 < len(route):
        windows += 1
        window = np.asarray(route[index - 1:index + 3])
        vectors = np.diff(window, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 1e-8):
            index += 1
            continue
        left_is_base = index == 1
        right_is_target = index + 2 == len(route) - 1
        left_range = prepared.cfg.comm_radius_base if left_is_base else prepared.cfg.comm_radius
        right_range = prepared.cfg.visual_radius if right_is_target else prepared.cfg.comm_radius
        if np.linalg.norm(window[3] - window[0]) > left_range + right_range:
            point, reason, intervals, xy = None, "outer endpoints exceed combined ranges", [], None
        else:
            key = (window.tobytes(), left_is_base, right_is_target)
            if key in cache:
                cache_stats["hits"] += 1
                point, reason, intervals, xy = cache[key]
            else:
                cache_stats["evaluations"] += 1
                point, reason, intervals, xy = lifted_connector(window, prepared,
                                                                left_is_base, right_is_target)
                if point is not None:
                    point.setflags(write=False)
                cache[key] = point, reason, intervals, xy
        outcomes[reason] += 1
        events.append({"window": window, "point": point, "reason": reason,
                       "intervals": intervals, "xy": xy,
                       "left_is_base": left_is_base, "right_is_target": right_is_target})
        if reason == "accepted":
            horizontal = np.all(np.abs(vectors[:, 2] / lengths) < 1e-6)
            gap = lengths[1]
            category = "<2m" if gap < 2 else "2-5m" if gap < 5 else "5-15m" if gap < 15 else ">=15m"
            accepted_types[("horizontal" if horizontal else "non-horizontal", category)] += 1
            route[index:index + 2] = [point]
            index = max(1, index - 2)
        else:
            index += 1
    return np.asarray(route[1:-1]), windows, events


class Inspection:
    name = "general_corner_simplification"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total, self.details = args.targets, args.geometry_details
        self.catalog = OpeningCatalog(prepared.building)
        self.histograms = [Counter() for _ in range(3)]
        self.previous = {name: Counter() for name in ("horizontal", "doorway", "lifted", "prune")}
        self.additional = {name: Counter() for name in ("horizontal", "doorway", "lifted", "prune")}
        self.previous_openings, self.additional_openings = {}, {}
        self.previous_roles, self.additional_roles = Counter(), Counter()
        self.outcomes, self.accepted_types, self.cache_stats = Counter(), Counter(), Counter()
        self.cache = {}
        self.guard_outcomes = Counter()
        self.guard_examples = []
        self.invalid_assignments = 0
        self.improved_examples, self.remaining_examples, self.rejected_examples = {}, {}, {}
        self.sample = -1
        self.windows = self.cycles = self.max_cycles = self.unreachable = self.improved = self.max_saved = 0

    def old_rules(self, chain, target, outcomes, openings, roles):
        chain, _ = simplify_chain(chain, target, self.prepared, outcomes["horizontal"])
        chain, _, _ = simplify_doorways(chain, target, self.prepared, self.catalog,
                                       outcomes["doorway"], openings)
        chain, _, _ = simplify_lifted(chain, target, self.prepared, outcomes["lifted"])
        chain, _ = prune_relays(chain, target, self.prepared, outcomes["prune"], roles)
        return chain

    def observe(self, points, target):
        self.sample += 1
        if points is None:
            self.unreachable += 1
            return
        baseline = corrected_chain(self.prepared.base, self.prepared.route_start,
                                   points, target, self.prepared.cfg)
        reason = chain_reason(baseline, target, self.prepared)
        if reason != "accepted":
            self.invalid_assignments += 1
            self.guard_outcomes["invalid baseline: " + reason] += 1
            self.guard_examples.append({"sample": self.sample, "target": target.copy(),
                                        "reason": reason, "chain": baseline.copy()})
            return None
        self.guard_outcomes["baseline accepted"] += 1
        previous = self.old_rules(baseline, target, self.previous,
                                  self.previous_openings, self.previous_roles)
        while len(previous) > 1:
            before = len(previous)
            previous = self.old_rules(previous, target, self.previous,
                                      self.previous_openings, self.previous_roles)
            if len(previous) == before:
                break

        reason = chain_reason(previous, target, self.prepared)
        if reason != "accepted":
            self.guard_outcomes["previous-stage rollback: " + reason] += 1
            previous = baseline.copy()

        current = previous
        cycles = 0
        events = []
        while len(current) > 1:
            cycles += 1
            before = len(current)
            current, windows, new_events = scan_general(current, target, self.prepared,
                                                        self.outcomes, self.accepted_types,
                                                        self.cache, self.cache_stats)
            self.windows += windows
            events.extend(new_events)
            current = self.old_rules(current, target, self.additional,
                                     self.additional_openings, self.additional_roles)
            if len(current) == before:
                break
        self.cycles += cycles
        self.max_cycles = max(self.max_cycles, cycles)
        current = rounded(current)
        reason = chain_reason(current, target, self.prepared)
        if reason != "accepted":
            self.guard_outcomes["final-stage rollback: " + reason] += 1
            current = previous.copy()
        self.guard_outcomes["final accepted"] += 1
        counts = len(baseline), len(previous), len(current)
        for histogram, count in zip(self.histograms, counts):
            histogram[count] += 1
        saved = counts[1] - counts[2]
        self.improved += int(saved > 0)
        self.max_saved = max(self.max_saved, saved)
        cell = tuple(np.floor(target / self.prepared.building.cell_size_m).astype(int))
        improved_key, remaining_key = (cell, counts[1], counts[2]), (cell, counts[2])
        need_improved = saved > 0 and improved_key not in self.improved_examples
        need_remaining = counts[2] > 7 and remaining_key not in self.remaining_examples
        if need_improved or need_remaining:
            example = {"sample": self.sample, "target": target.copy(), "counts": counts,
                       "previous": previous.copy(), "final": current.copy()}
            if need_improved:
                self.improved_examples[improved_key] = example
            if need_remaining:
                self.remaining_examples[remaining_key] = example
        for event in events:
            if event["reason"] == "accepted":
                continue
            prior = self.rejected_examples.get(event["reason"])
            if prior is None or (counts[1] > 7 and prior["counts"][1] <= 7):
                self.rejected_examples[event["reason"]] = {
                    **event, "sample": self.sample, "target": target.copy(), "counts": counts}
        return current

    def print_chain(self, label, chain, target):
        print(f"  {label} ({len(chain)} drones); base={coordinates(self.prepared.base)}")
        previous = self.prepared.base
        for index, point in enumerate(chain):
            radius = self.prepared.cfg.comm_radius_base if index == 0 else self.prepared.cfg.comm_radius
            visible = _visible_from(previous, point[None], self.prepared.physical_solid_lo,
                                    self.prepared.physical_solid_hi)[0]
            print(f"    drone {index + 1}: {coordinates(point)}; "
                  f"incoming={np.linalg.norm(point - previous):.6f}/{radius:g}m; visible={bool(visible)}")
            previous = point
        visible = _visible_from(previous, target[None], self.prepared.physical_solid_lo,
                                self.prepared.physical_solid_hi)[0]
        print(f"    target tip={np.linalg.norm(target - previous):.6f}/"
              f"{self.prepared.cfg.visual_radius:g}m; visible={bool(visible)}")

    def report(self):
        print("\n--- General corner scan versus previous stable chains ---")
        print("Construction unchanged: one forward XY intersection plus analytic allowed-height intervals.")
        print("Eligibility: all nondegenerate drone pairs; no 2m gap cutoff or horizontal/angle-bin restriction.")
        print(" Agents | Corrected | Previous stable | General stable |   Change | Final % | visual")
        print("--------+-----------+-----------------+----------------+----------+---------+----------------------------------------")
        maximum = max(max(histogram, default=0) for histogram in self.histograms)
        for count in range(maximum + 1):
            baseline, previous, final = (histogram[count] for histogram in self.histograms)
            percent = 100 * final / self.total
            bars = "I" * int(round(percent)) or ("." if final else "")
            print(f"{count:7d} | {baseline:9,d} | {previous:15,d} | {final:14,d} | {final - previous:+8,d} | "
                  f"{percent:6.2f}% | {bars}")
        totals = [sum(count * amount for count, amount in histogram.items())
                  for histogram in self.histograms]
        over_seven = [sum(amount for count, amount in histogram.items() if count > 7)
                      for histogram in self.histograms]
        reachable = sum(self.histograms[0].values())
        print(f"\nTargets additionally improved: {self.improved:,}/{self.total:,} "
              f"({100 * self.improved / self.total:.2f}%)")
        print(f"Additional drones saved: {totals[1] - totals[2]:,}; "
              f"maximum additional saving on one target: {self.max_saved}")
        if reachable:
            print("Mean agents (corrected -> previous stable -> general stable): " +
                  " -> ".join(f"{total / reachable:.3f}" for total in totals))
        print("Targets requiring >7 agents: " +
              " -> ".join(f"{amount:,} ({100 * amount / self.total:.2f}%)" for amount in over_seven))
        print(f"Additional targets moved to <=7 agents: {over_seven[1] - over_seven[2]:,}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,}")
        self.report_guard()
        print("Previous accepted horizontal/doorway/lifted/deletion operations reproduced: " +
              " / ".join(f"{self.previous[name]['accepted']:,}"
                         for name in ("horizontal", "doorway", "lifted", "prune")))
        print(f"\nGeneral scan windows (including revisits): {self.windows:,}")
        attempts = sum(self.outcomes.values())
        print(f"General attempts: {attempts:,}; accepted: {self.outcomes['accepted']:,}; "
              f"rejected: {attempts - self.outcomes['accepted']:,}")
        print(f"Unique construction evaluations: {self.cache_stats['evaluations']:,}; "
              f"identical-window cache hits: {self.cache_stats['hits']:,}")
        for reason, amount in self.outcomes.most_common():
            if reason != "accepted":
                print(f"  {reason}: {amount:,}")
        print("\nAccepted general replacements by geometry / middle gap:")
        for (kind, gap), amount in sorted(self.accepted_types.items()):
            print(f"  {kind:14s} / {gap:7s}: {amount:,}")
        print("Additional accepted horizontal/doorway/lifted/deletion operations: " +
              " / ".join(f"{self.additional[name]['accepted']:,}"
                         for name in ("horizontal", "doorway", "lifted", "prune")))
        print(f"General-plus-old-rule cycles, including terminal zero-change cycles: {self.cycles:,}; "
              f"maximum on one target: {self.max_cycles}")

        for title, collection in (("Improved chain examples", self.improved_examples),
                                   ("Remaining >7-agent chain examples", self.remaining_examples)):
            examples = sorted(collection.values(),
                              key=lambda example: (-example['counts'][2],
                                                    -(example['counts'][1] - example['counts'][2]),
                                                    example['sample']))
            print(f"\n--- {title}: up to {self.details} ---")
            for example in examples[:self.details]:
                print(f"Sample {example['sample']} (zero-based); target={coordinates(example['target'])}")
                print("  Agents corrected/previous stable/general stable: " +
                      "/".join(map(str, example["counts"])))
                self.print_chain("Previous stable chain", example["previous"], example["target"])
                self.print_chain("General stable chain", example["final"], example["target"])
            if not examples:
                print("  none")
        print(f"\n--- Rejected construction examples: up to {self.details} ---")
        examples = sorted(self.rejected_examples.values(),
                          key=lambda example: (-(example['counts'][1] > 7), example['sample']))
        for example in examples[:self.details]:
            print(f"Sample {example['sample']}; {example['reason']}; "
                  f"agent counts={example['counts']}; target={coordinates(example['target'])}")
            for name, point in zip("ABCD", example["window"]):
                print(f"  {name}: {coordinates(point)}")
            xy, point = example["xy"], example["point"]
            print(f"  XY: {coordinates(xy) if xy is not None else 'not constructed'}; "
                  f"allowed heights={example['intervals']}")
            print(f"  P: {coordinates(point) if point is not None else 'not constructed'}")
        if not examples:
            print("  none")
        print("\nAll accepted proposals retain the existing full clearance, link-range, visibility and reachability checks.")
        print("No alternate connector is tried; stability is specific to these rules and their order.")
        print("Counts describe constructed geometric chains, not global minima or policy success.")
        print("Previous stages are preprocessing only; earlier inspection reports are not printed.")

    def report_guard(self):
        print(f"Invalid baseline assignments retained as unresolved: {self.invalid_assignments:,}")
        print("Complete rounded-chain guard outcomes:")
        for reason, count in self.guard_outcomes.items():
            print(f"  {reason}: {count:,}")
        for example in self.guard_examples[:self.details]:
            print(f"  Invalid baseline sample {example['sample']}: {example['reason']}; "
                  f"target={coordinates(example['target'])}; chain={example['chain'].tolist()}")
        print("Rejected connector/bypass proposals retain their previous chain; guard failures are never labelled certified.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
