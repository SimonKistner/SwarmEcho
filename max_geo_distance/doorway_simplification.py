"""Inspection 6: one shared-opening connector, compared with both prior baselines."""

from collections import Counter
from dataclasses import dataclass
import json

import numpy as np

from corrected_baseline import corrected_chain
from corner_simplification import simplify_chain
from paths import _visible_from
from remaining_corner_geometry import coordinates
from placement_checks import SAFETY_MARGIN_M, connector_reason, range_limit, rounded


@dataclass
class Opening:
    label: str
    axis: int
    tangent: np.ndarray
    plane: float
    lo: np.ndarray
    hi: np.ndarray


class OpeningCatalog:
    def __init__(self, building):
        self.openings = []
        extras = json.loads(building.extra_geometry_json)
        cell = building.cell_size_m
        for axis_name, axis in (("x", 0), ("y", 1)):
            along = 1 - axis
            for kind in ("doors", "windows"):
                for face in extras.get(f"{axis_name}_{kind}", []):
                    lo = np.asarray(face, dtype=float) * cell
                    hi = lo + cell
                    plane = face[axis] * cell
                    lo[axis], hi[axis] = (plane - building.wall_thickness_m / 2,
                                         plane + building.wall_thickness_m / 2)
                    lo[along], hi[along] = (face[along] + 0.3) * cell, (face[along] + 0.7) * cell
                    lo[2] = (face[2] + (0 if kind == "doors" else 0.3)) * cell
                    hi[2] = (face[2] + 0.7) * cell
                    self.openings.append(Opening(f"{axis_name}-{kind[:-1]}{tuple(face)}", axis,
                                                 np.asarray((along, 2)), plane, lo, hi))
        self.axes = np.asarray([opening.axis for opening in self.openings], dtype=int)
        self.lo = np.asarray([opening.lo for opening in self.openings]).reshape(-1, 3)
        self.hi = np.asarray([opening.hi for opening in self.openings]).reshape(-1, 3)
        self.low_faces = self.lo[np.arange(len(self.axes)), self.axes]
        self.high_faces = self.hi[np.arange(len(self.axes)), self.axes]

    def match(self, b, c):
        if not len(self.openings) or not 1e-8 < np.linalg.norm(c - b) < 2:
            return []
        first, last = b[self.axes], c[self.axes]
        crossed = (((first <= self.low_faces + 1e-6) & (last >= self.high_faces - 1e-6)) |
                   ((last <= self.low_faces + 1e-6) & (first >= self.high_faces - 1e-6)))
        delta = last - first
        safe = np.where(np.abs(delta) > 1e-8, delta, 1.0)
        for faces in (self.low_faces, self.high_faces):
            fraction = (faces - first) / safe
            face_points = b + fraction[:, None] * (c - b)
            crossed &= np.all((face_points >= self.lo - 1e-6) &
                              (face_points <= self.hi + 1e-6), axis=1)
        return [self.openings[index] for index in np.flatnonzero(crossed)]


def connector(window, opening, prepared, left_is_base, right_is_target):
    a, b, c, d = window
    tangent, axis = opening.tangent, opening.axis
    clearance = prepared.cfg.drone_radius + prepared.cfg.obstacle_planning_clearance_m
    low = np.maximum(opening.lo[tangent] + clearance + SAFETY_MARGIN_M, prepared.lower[tangent])
    high = np.minimum(opening.hi[tangent] - clearance - SAFETY_MARGIN_M, prepared.upper[tangent])
    endpoints = ((a, prepared.cfg.comm_radius_base if left_is_base else prepared.cfg.comm_radius,
                  "left"),
                 (d, prepared.cfg.visual_radius if right_is_target else prepared.cfg.comm_radius,
                  "right"))

    # A face intercept is affine in P because P's normal coordinate is fixed.
    # Its two aperture inequalities produce two exact coordinate intervals.
    for observer, _, _ in endpoints:
        denominator = opening.plane - observer[axis]
        if abs(denominator) < 1e-12:
            continue
        for face in (opening.lo[axis], opening.hi[axis]):
            fraction = (face - observer[axis]) / denominator
            if not 0 < fraction < 1:
                continue
            low = np.maximum(low, (opening.lo[tangent] + SAFETY_MARGIN_M -
                                   (1 - fraction) * observer[tangent]) / fraction)
            high = np.minimum(high, (opening.hi[tangent] - SAFETY_MARGIN_M -
                                     (1 - fraction) * observer[tangent]) / fraction)
    if np.any(low > high):
        return None, "no common visible aperture", low, high

    # Restrict by necessary range bounds, then check the actual radii below.
    for observer, radius, side in endpoints:
        radius = range_limit(radius, construction=True)
        normal_distance = abs(observer[axis] - opening.plane)
        if normal_distance > radius:
            return None, f"{side} endpoint too far from opening plane", low, high
        extent = np.sqrt(max(0.0, radius * radius - normal_distance * normal_distance))
        low = np.maximum(low, observer[tangent] - extent)
        high = np.minimum(high, observer[tangent] + extent)
    if np.any(low > high):
        return None, "range bounds exclude common visible aperture", low, high

    point = (b + c) / 2
    point[axis] = opening.plane
    point[tangent] = np.clip(point[tangent], low, high)
    point = rounded(point)
    if np.any(point < prepared.lower) or np.any(point > prepared.upper):
        return point, "outside drone bounds", low, high
    inside = np.all((point >= prepared.solid_lo) & (point <= prepared.solid_hi), axis=1)
    if np.any(inside):
        return point, "insufficient obstacle clearance", low, high
    for observer, radius, side in endpoints:
        if np.linalg.norm(point - observer) > radius:
            return point, f"{side} link exceeds range", low, high
    visible = _visible_from(point, np.stack((a, d)),
                            prepared.physical_solid_lo, prepared.physical_solid_hi)
    if not visible[0]:
        return point, "left link obstructed", low, high
    if not visible[1]:
        return point, "right link obstructed", low, high
    reachable = _visible_from(point, np.stack((b, c)), prepared.solid_lo, prepared.solid_hi)
    if not np.any(reachable):
        return point, "no planning-clear connection to removed nodes", low, high
    return point, connector_reason(point, window, prepared, left_is_base, right_is_target), low, high


def simplify_doorways(chain, target, prepared, catalog, outcomes, per_opening):
    route = [prepared.base.copy(), *(point.copy() for point in chain), target.copy()]
    index = 1
    windows = 0
    events = []
    while index + 2 < len(route):
        windows += 1
        window = np.asarray(route[index - 1:index + 3])
        matches = catalog.match(window[1], window[2])
        if not matches:
            index += 1
            continue
        if len(matches) != 1:
            outcomes["ambiguous matching openings"] += 1
            index += 1
            continue
        opening = matches[0]
        left_is_base = index == 1
        right_is_target = index + 2 == len(route) - 1
        point, reason, low, high = connector(window, opening, prepared,
                                            left_is_base, right_is_target)
        outcomes[reason] += 1
        per_opening.setdefault(opening.label, Counter())[reason] += 1
        events.append({"opening": opening, "window": window, "point": point,
                       "low": low, "high": high, "reason": reason,
                       "left_is_base": left_is_base, "right_is_target": right_is_target})
        if reason == "accepted":
            route[index:index + 2] = [point]
            index = max(1, index - 2)
        else:
            index += 1
    return np.asarray(route[1:-1]), windows, events


class Inspection:
    name = "doorway_simplification"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total = args.targets
        self.details = args.geometry_details
        self.catalog = OpeningCatalog(prepared.building)
        self.baseline, self.horizontal, self.combined = Counter(), Counter(), Counter()
        self.horizontal_outcomes, self.outcomes = Counter(), Counter()
        self.per_opening = {}
        self.accepted_targets = {}
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
        combined, windows, events = simplify_doorways(horizontal, target, self.prepared,
                                                      self.catalog, self.outcomes, self.per_opening)
        self.baseline[len(baseline)] += 1
        self.horizontal[len(horizontal)] += 1
        self.combined[len(combined)] += 1
        self.windows += windows
        saved = len(horizontal) - len(combined)
        self.improved += int(saved > 0)
        self.max_saved = max(self.max_saved, saved)
        for event in events:
            label, reason = event["opening"].label, event["reason"]
            if reason == "accepted":
                self.accepted_targets.setdefault(label, set()).add(self.sample)
            key = (label, reason)
            previous = self.examples.get(key)
            if previous is None or (len(horizontal) > 7 and previous["counts"][1] <= 7):
                self.examples[key] = {**event, "sample": self.sample, "target": target.copy(),
                                      "counts": (len(baseline), len(horizontal), len(combined))}

    def report(self):
        print("\n--- Doorway connector versus corrected and horizontal-only baselines ---")
        print("Rule: one point on the shared opening plane, projected into its common visible rectangle.")
        print("Matches actual door/window crossings with B-C gap <2m; no angle-bin requirement.")
        print(" Agents | Baseline | Horizontal | Combined | New change | Final % | visual")
        print("--------+----------+------------+----------+------------+---------+----------------------------------------")
        maximum = max(max(histogram, default=0)
                      for histogram in (self.baseline, self.horizontal, self.combined))
        for count in range(maximum + 1):
            before, horizontal, final = self.baseline[count], self.horizontal[count], self.combined[count]
            percent = 100 * final / self.total
            bars = "I" * int(round(percent)) or ("." if final else "")
            print(f"{count:7d} | {before:8,d} | {horizontal:10,d} | {final:8,d} | "
                  f"{final - horizontal:+10,d} | {percent:6.2f}% | {bars}")
        histograms = (self.baseline, self.horizontal, self.combined)
        totals = [sum(count * amount for count, amount in histogram.items()) for histogram in histograms]
        over_seven = [sum(amount for count, amount in histogram.items() if count > 7)
                      for histogram in histograms]
        reachable = sum(self.baseline.values())
        print(f"\nTargets additionally improved by doorway rule: {self.improved:,}/{self.total:,} "
              f"({100 * self.improved / self.total:.2f}%)")
        print(f"Additional drones saved: {totals[1] - totals[2]:,}; "
              f"maximum additional saving on one target: {self.max_saved}")
        if reachable:
            print("Mean agents (baseline -> horizontal -> combined): " +
                  " -> ".join(f"{total / reachable:.3f}" for total in totals))
        print("Targets requiring >7 agents (baseline -> horizontal -> combined): " +
              " -> ".join(f"{amount:,} ({100 * amount / self.total:.2f}%)" for amount in over_seven))
        print(f"Additional targets moved to <=7 agents: {over_seven[1] - over_seven[2]:,}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,}")
        print(f"Existing horizontal replacements reproduced: {self.horizontal_outcomes['accepted']:,}")
        attempts = sum(self.outcomes.values())
        print(f"\nDoorway scan windows (including revisits): {self.windows:,}")
        print(f"Eligible doorway attempts: {attempts:,}; accepted: {self.outcomes['accepted']:,}; "
              f"rejected: {attempts - self.outcomes['accepted']:,}")
        for reason, amount in self.outcomes.most_common():
            if reason != "accepted":
                print(f"  {reason}: {amount:,}")
        print("\n Opening | Attempts | Accepted | Distinct targets improved")
        for label, outcomes in sorted(self.per_opening.items(),
                                      key=lambda item: (-item[1]['accepted'], item[0])):
            print(f" {label:22s} | {sum(outcomes.values()):8,d} | {outcomes['accepted']:8,d} | "
                  f"{len(self.accepted_targets.get(label, ())):8,d}")

        for accepted, heading in ((True, "Accepted placement examples"), (False, "Rejected attempt examples")):
            examples = [example for example in self.examples.values()
                        if (example["reason"] == "accepted") == accepted]
            examples.sort(key=lambda example: (-self.per_opening[example['opening'].label]['accepted'],
                                                example['opening'].label, example['reason']))
            print(f"\n--- {heading}: up to {self.details} ---")
            for example in examples[:self.details]:
                opening = example["opening"]
                print(f"{opening.label}; sample {example['sample']} (zero-based); {example['reason']}")
                print("  Agents baseline/horizontal/combined: " + "/".join(map(str, example["counts"])) +
                      f"; target={coordinates(example['target'])}")
                for name, point in zip("ABCD", example["window"]):
                    role = "base" if name == "A" and example["left_is_base"] else (
                        "target" if name == "D" and example["right_is_target"] else "drone")
                    print(f"  {name} ({role}): {coordinates(point)}")
                print(f"  Opening lo={coordinates(opening.lo)}; hi={coordinates(opening.hi)}")
                print(f"  Tangential bounds low={coordinates(example['low'])}; "
                      f"high={coordinates(example['high'])}")
                point = example["point"]
                print(f"  P: {coordinates(point) if point is not None else 'not constructed'}")
                if point is not None:
                    a, _, _, d = example["window"]
                    print(f"  A-P/P-D lengths: {np.linalg.norm(a - point):.6f} / "
                          f"{np.linalg.norm(d - point):.6f}m")
            if not examples:
                print("  none")
        print("\nEach accepted replacement passes clearance, full link range/visibility, and local reachability checks.")
        print("A failed single point is retained as a rejection; no alternative connector is searched.")
        print("Counts describe constructed geometric chains, not global minima or policy success.")
        print("Prior rules are preprocessing only; no earlier inspection reports are run.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
