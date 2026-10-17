"""Inspection 4: deterministic horizontal corner replacement versus its baseline."""

from collections import Counter
import numpy as np

from corrected_baseline import corrected_chain
from paths import _visible_from
from placement_checks import connector_reason, rounded


def matches_pattern(window):
    """Match the existing XY/same/45-45/1-2m inventory bin, using actual points."""
    vectors = np.diff(window, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-8) or not 1 <= lengths[1] < 2:
        return False
    unit = vectors / lengths[:, None]
    if np.any(np.abs(unit[:, 2]) >= 1e-6):
        return False
    cosines = np.sum(unit[:-1] * unit[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    bins = 5 * np.floor(angles / 5 + 0.5)
    normals = np.cross(unit[:-1], unit[1:])
    return bool(np.all(bins == 45) and normals[0, 2] * normals[1, 2] > 0)


def replacement(window, prepared, left_is_base, right_is_target):
    """Calculate exactly one ray intersection, then check that one placement."""
    a, b, c, d = window
    incoming = (b - a) / np.linalg.norm(b - a)
    reverse_outgoing = (c - d) / np.linalg.norm(c - d)
    matrix = np.column_stack((incoming[:2], -reverse_outgoing[:2]))
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None, "no forward intersection"
    forward_b, forward_c = np.linalg.solve(matrix, (c - b)[:2])
    if not np.isfinite([forward_b, forward_c]).all() or min(forward_b, forward_c) < 0:
        return None, "no forward intersection"
    point = b + forward_b * incoming
    other_intersection = c + forward_c * reverse_outgoing
    if np.linalg.norm(point - other_intersection) > 1e-6:
        return None, "intersection not coplanar"
    point = rounded(point)
    if np.any(point < prepared.lower) or np.any(point > prepared.upper):
        return None, "outside drone bounds"
    inside = np.all((point >= prepared.solid_lo) & (point <= prepared.solid_hi), axis=1)
    if np.any(inside):
        return None, "insufficient obstacle clearance"

    cfg = prepared.cfg
    left_range = cfg.comm_radius_base if left_is_base else cfg.comm_radius
    right_range = cfg.visual_radius if right_is_target else cfg.comm_radius
    if np.linalg.norm(point - a) > left_range:
        return None, "left link exceeds range"
    if np.linalg.norm(d - point) > right_range:
        return None, "right link exceeds range"
    visible = _visible_from(point, np.stack((a, d)),
                            prepared.physical_solid_lo, prepared.physical_solid_hi)
    if not visible[0]:
        return None, "left link obstructed"
    if not visible[1]:
        return None, "right link obstructed"
    reachable = _visible_from(point, np.stack((b, c)), prepared.solid_lo, prepared.solid_hi)
    if not np.any(reachable):
        return None, "no planning-clear connection to removed nodes"
    reason = connector_reason(point, window, prepared, left_is_base, right_is_target)
    return (point if reason == "accepted" else None), reason


def simplify_chain(chain, target, prepared, outcomes):
    # Base and target are protected endpoints. Every other entry is a drone.
    route = [prepared.base.copy(), *(point.copy() for point in chain), target.copy()]
    index = 1
    windows = 0
    while index + 2 < len(route):
        windows += 1
        window = np.asarray(route[index - 1:index + 3])
        if not matches_pattern(window):
            index += 1
            continue
        point, reason = replacement(window, prepared, index == 1, index + 2 == len(route) - 1)
        outcomes[reason] += 1
        if point is None:
            index += 1
        else:
            route[index:index + 2] = [point]
            # Changing P affects the two preceding windows as well as later ones.
            index = max(1, index - 2)
    return np.asarray(route[1:-1]), windows


class Inspection:
    name = "corner_simplification"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total = args.targets
        self.before = Counter()
        self.after = Counter()
        self.outcomes = Counter()
        self.windows = self.unreachable = self.improved = self.max_saved = 0

    def observe(self, points, target):
        if points is None:
            self.unreachable += 1
            return
        baseline = corrected_chain(self.prepared.base, self.prepared.route_start,
                                   points, target, self.prepared.cfg)
        simplified, windows = simplify_chain(baseline, target, self.prepared, self.outcomes)
        original_count, new_count = len(baseline), len(simplified)
        self.before[original_count] += 1
        self.after[new_count] += 1
        self.windows += windows
        saved = original_count - new_count
        self.improved += int(saved > 0)
        self.max_saved = max(self.max_saved, saved)

    def report(self):
        print("\n--- Corner simplification versus corrected baseline (same sampled targets) ---")
        print("Rule: XY / same direction / 45-45 degrees (5-degree bins) / 1-2m middle gap")
        print("One forward ray intersection; sequential replacements with neighbor revisits.")
        print(" Agents | Baseline targets | Simplified targets |   Change | After % | visual")
        print("--------+------------------+--------------------+----------+---------+----------------------------------------")
        maximum = max(max(self.before, default=0), max(self.after, default=0))
        for count in range(maximum + 1):
            before, after = self.before[count], self.after[count]
            percent = 100 * after / self.total
            bars = "I" * int(round(percent)) or ("." if after else "")
            print(f"{count:7d} | {before:16,d} | {after:18,d} | {after - before:+8,d} | "
                  f"{percent:6.2f}% | {bars}")

        reachable = sum(self.before.values())
        before_total = sum(count * amount for count, amount in self.before.items())
        after_total = sum(count * amount for count, amount in self.after.items())
        before_seven = sum(amount for count, amount in self.before.items() if count > 7)
        after_seven = sum(amount for count, amount in self.after.items() if count > 7)
        print(f"\nTargets improved: {self.improved:,}/{self.total:,} "
              f"({100 * self.improved / self.total:.2f}%)")
        print(f"Drones saved across target chains: {before_total - after_total:,}; "
              f"maximum saved on one target: {self.max_saved}")
        if reachable:
            print(f"Mean agents on reachable targets: {before_total / reachable:.3f} "
                  f"-> {after_total / reachable:.3f}")
        print(f"Targets requiring >7 agents: {before_seven:,} "
              f"({100 * before_seven / self.total:.2f}%) -> {after_seven:,} "
              f"({100 * after_seven / self.total:.2f}%)")
        print(f"Targets moved to <=7 agents: {before_seven - after_seven:,}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,}")

        attempts = sum(self.outcomes.values())
        accepted = self.outcomes["accepted"]
        print(f"\nWindows inspected (including revisits): {self.windows:,}")
        print(f"Eligible replacement attempts: {attempts:,}")
        print(f"Accepted replacements: {accepted:,}; rejected: {attempts - accepted:,}")
        for reason, amount in self.outcomes.most_common():
            if reason != "accepted":
                print(f"  {reason}: {amount:,}")
        print("Acceptance checks point clearance, both link ranges/visibility, and local reachability.")
        print("Counts are feasible geometric placements under this model, not global minima or policy success.")
        print("Baseline construction is reused directly; earlier inspection reports are not needed.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
