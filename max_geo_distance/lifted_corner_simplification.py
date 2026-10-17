"""Inspection 7: one XY intersection lifted to an analytically permitted height."""

from collections import Counter
import numpy as np

from corrected_baseline import corrected_chain
from corner_simplification import simplify_chain
from doorway_simplification import OpeningCatalog, simplify_doorways
from height_intervals import allowed_connector_heights, nearest_height
from paths import _visible_from
from remaining_corner_geometry import coordinates
from placement_checks import connector_reason, rounded


def eligible_window(window):
    vectors = np.diff(window, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-8) or lengths[1] >= 2:
        return False
    return bool(np.any(np.abs(vectors[:, 2] / lengths) >= 1e-6))


def horizontal_intersection(window):
    a, b, c, d = window
    incoming = b[:2] - a[:2]
    reverse_outgoing = c[:2] - d[:2]
    lengths = np.linalg.norm(incoming), np.linalg.norm(reverse_outgoing)
    if min(lengths) <= 1e-8:
        return None, "outer ray has no horizontal direction"
    incoming, reverse_outgoing = incoming / lengths[0], reverse_outgoing / lengths[1]
    matrix = np.column_stack((incoming, -reverse_outgoing))
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None, "parallel horizontal outer rays"
    first, second = np.linalg.solve(matrix, (c - b)[:2])
    if not np.isfinite([first, second]).all() or min(first, second) < 0:
        return None, "no forward horizontal intersection"
    return b[:2] + first * incoming, "intersection available"


def check_point(point, window, prepared, left_is_base, right_is_target):
    a, b, c, d = window
    if not np.isfinite(point).all() or np.any(point < prepared.lower) or np.any(point > prepared.upper):
        return "outside drone bounds"
    inside = np.all((point >= prepared.solid_lo) & (point <= prepared.solid_hi), axis=1)
    if np.any(inside):
        return "insufficient obstacle clearance"
    cfg = prepared.cfg
    left_range = cfg.comm_radius_base if left_is_base else cfg.comm_radius
    right_range = cfg.visual_radius if right_is_target else cfg.comm_radius
    if np.linalg.norm(point - a) > left_range:
        return "left link exceeds range"
    if np.linalg.norm(d - point) > right_range:
        return "right link exceeds range"
    visible = _visible_from(point, np.stack((a, d)),
                            prepared.physical_solid_lo, prepared.physical_solid_hi)
    if not visible[0]:
        return "left link obstructed in final check"
    if not visible[1]:
        return "right link obstructed in final check"
    reachable = _visible_from(point, np.stack((b, c)), prepared.solid_lo, prepared.solid_hi)
    if not np.any(reachable):
        return "no planning-clear connection in final check"
    return connector_reason(point, window, prepared, left_is_base, right_is_target)


def lifted_connector(window, prepared, left_is_base, right_is_target):
    xy, reason = horizontal_intersection(window)
    if xy is None:
        return None, reason, [], None
    intervals, reason = allowed_connector_heights(xy, window, prepared, left_is_base, right_is_target)
    if not intervals:
        return None, reason, [], xy
    preferred = float((window[1, 2] + window[2, 2]) / 2)
    height = nearest_height(preferred, intervals)
    point = rounded((xy[0], xy[1], height))
    return point, check_point(point, window, prepared, left_is_base, right_is_target), intervals, xy


def simplify_lifted(chain, target, prepared, outcomes):
    route = [prepared.base.copy(), *(point.copy() for point in chain), target.copy()]
    index = 1
    windows = 0
    events = []
    while index + 2 < len(route):
        windows += 1
        window = np.asarray(route[index - 1:index + 3])
        if not eligible_window(window):
            index += 1
            continue
        left_is_base = index == 1
        right_is_target = index + 2 == len(route) - 1
        point, reason, intervals, xy = lifted_connector(window, prepared, left_is_base, right_is_target)
        outcomes[reason] += 1
        events.append({"window": window, "point": point, "reason": reason,
                       "intervals": intervals, "xy": xy,
                       "left_is_base": left_is_base, "right_is_target": right_is_target})
        if reason == "accepted":
            route[index:index + 2] = [point]
            index = max(1, index - 2)
        else:
            index += 1
    return np.asarray(route[1:-1]), windows, events


class Inspection:
    name = "lifted_corner_simplification"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total, self.details = args.targets, args.geometry_details
        self.catalog = OpeningCatalog(prepared.building)
        self.histograms = [Counter() for _ in range(4)]
        self.horizontal_outcomes, self.doorway_outcomes, self.outcomes = Counter(), Counter(), Counter()
        self.doorways = {}
        self.examples = {}
        self.windows = self.unreachable = self.improved = self.max_saved = 0
        self.sample = -1

    def observe(self, points, target):
        self.sample += 1
        if points is None:
            self.unreachable += 1
            return
        baseline = corrected_chain(self.prepared.base, self.prepared.route_start,
                                   points, target, self.prepared.cfg)
        horizontal, _ = simplify_chain(baseline, target, self.prepared, self.horizontal_outcomes)
        doorway, _, _ = simplify_doorways(horizontal, target, self.prepared, self.catalog,
                                         self.doorway_outcomes, self.doorways)
        lifted, windows, events = simplify_lifted(doorway, target, self.prepared, self.outcomes)
        counts = tuple(map(len, (baseline, horizontal, doorway, lifted)))
        for histogram, count in zip(self.histograms, counts):
            histogram[count] += 1
        self.windows += windows
        saved = counts[2] - counts[3]
        self.improved += int(saved > 0)
        self.max_saved = max(self.max_saved, saved)
        for event in events:
            if event["reason"] == "accepted":
                key = ("accepted", tuple(event["window"][1]), tuple(event["window"][2]))
            else:
                key = (event["reason"],)
            previous = self.examples.get(key)
            if previous is None or (counts[2] > 7 and previous["counts"][2] <= 7):
                self.examples[key] = {**event, "sample": self.sample,
                                      "target": target.copy(), "counts": counts}

    def report(self):
        print("\n--- Lifted corner connector versus all previous stages ---")
        print("Rule: one forward XY intersection; choose its nearest allowed height analytically.")
        print("Scope: non-horizontal windows with B-C gap <2m; no turn-angle bins.")
        print(" Agents | Corrected | Horizontal | Doorways | Lifted | New change | Final % | visual")
        print("--------+-----------+------------+----------+--------+------------+---------+----------------------------------------")
        maximum = max(max(histogram, default=0) for histogram in self.histograms)
        for count in range(maximum + 1):
            baseline, horizontal, doorway, lifted = (histogram[count] for histogram in self.histograms)
            percent = 100 * lifted / self.total
            bars = "I" * int(round(percent)) or ("." if lifted else "")
            print(f"{count:7d} | {baseline:9,d} | {horizontal:10,d} | {doorway:8,d} | {lifted:6,d} | "
                  f"{lifted - doorway:+10,d} | {percent:6.2f}% | {bars}")
        totals = [sum(count * amount for count, amount in histogram.items())
                  for histogram in self.histograms]
        over_seven = [sum(amount for count, amount in histogram.items() if count > 7)
                      for histogram in self.histograms]
        reachable = sum(self.histograms[0].values())
        print(f"\nTargets additionally improved: {self.improved:,}/{self.total:,} "
              f"({100 * self.improved / self.total:.2f}%)")
        print(f"Additional drones saved beyond doorway stage: {totals[2] - totals[3]:,}; "
              f"maximum additional saving on one target: {self.max_saved}")
        if reachable:
            print("Mean agents (corrected -> horizontal -> doorways -> lifted): " +
                  " -> ".join(f"{total / reachable:.3f}" for total in totals))
        print("Targets requiring >7 agents (corrected -> horizontal -> doorways -> lifted): " +
              " -> ".join(f"{amount:,} ({100 * amount / self.total:.2f}%)" for amount in over_seven))
        print(f"Additional targets moved to <=7 agents: {over_seven[2] - over_seven[3]:,}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,}")
        print(f"Previous horizontal/doorway replacements reproduced: "
              f"{self.horizontal_outcomes['accepted']:,} / {self.doorway_outcomes['accepted']:,}")
        attempts = sum(self.outcomes.values())
        print(f"\nWindows inspected (including revisits): {self.windows:,}")
        print(f"Eligible lifted-corner attempts: {attempts:,}; accepted: {self.outcomes['accepted']:,}; "
              f"rejected: {attempts - self.outcomes['accepted']:,}")
        for reason, amount in self.outcomes.most_common():
            if reason != "accepted":
                print(f"  {reason}: {amount:,}")

        for accepted, title in ((True, "Accepted connector examples"), (False, "Rejected attempt examples")):
            examples = [example for example in self.examples.values()
                        if (example["reason"] == "accepted") == accepted]
            examples.sort(key=lambda example: (-(example['counts'][2] > 7), example['sample'], example['reason']))
            print(f"\n--- {title}: up to {self.details} ---")
            for example in examples[:self.details]:
                print(f"Sample {example['sample']} (zero-based); {example['reason']}")
                print("  Agents corrected/horizontal/doorways/lifted: " +
                      "/".join(map(str, example["counts"])) + f"; target={coordinates(example['target'])}")
                for name, point in zip("ABCD", example["window"]):
                    role = "base" if name == "A" and example["left_is_base"] else (
                        "target" if name == "D" and example["right_is_target"] else "drone")
                    print(f"  {name} ({role}): {coordinates(point)}")
                xy, point = example["xy"], example["point"]
                print(f"  XY intersection: {coordinates(xy) if xy is not None else 'not constructed'}")
                print(f"  Allowed height intervals (m): {example['intervals']}")
                print(f"  P: {coordinates(point) if point is not None else 'not constructed'}")
                if point is not None:
                    a, _, _, d = example["window"]
                    print(f"  A-P/P-D lengths: {np.linalg.norm(a - point):.6f} / "
                          f"{np.linalg.norm(d - point):.6f}m")
            if not examples:
                print("  none")
        print("\nHeight intervals account for endpoint spheres, point clearance, signal occlusion and local reachability.")
        print("Exactly one resulting point is checked; no placement search or simulation is performed.")
        print("Accepted points independently pass full clearance, range, visibility and reachability checks.")
        print("Counts describe constructed geometric chains, not global minima or policy success.")
        print("Previous stages are preprocessing only; earlier inspection reports are not printed.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
