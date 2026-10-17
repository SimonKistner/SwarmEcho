"""Inspection 3: original-path turn angles and consecutive corner patterns."""

from collections import Counter
import numpy as np


def collect_corners(route, angles, patterns):
    """Describe original consecutive turns without changing or simplifying them."""
    vectors = np.diff(route, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = (lengths[:-1] > 1e-8) & (lengths[1:] > 1e-8)
    unit = vectors / np.maximum(lengths[:, None], 1e-8)
    cosine = np.sum(unit[:-1] * unit[1:], axis=1)
    turns = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    bins = (5 * np.floor(turns / 5 + 0.5)).astype(int)
    normals = np.cross(unit[:-1], unit[1:])
    for index in np.flatnonzero(valid):
        angles[int(bins[index])] += 1
    for index in range(len(turns) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        # B and C are adjacent roadmap nodes in A -> B -> C -> D.
        separation = lengths[index + 1]
        bucket = next((name for upper, name in (
            (0.5, "<0.5m"), (1, "0.5-1m"), (2, "1-2m"),
            (5, "2-5m"), (15, "5-15m"), (np.inf, ">=15m"),
        ) if separation < upper), ">=15m")
        legs = unit[index:index + 3]
        if np.all(np.abs(legs[:, 2]) < 1e-6):
            plane = "XY"
        elif np.all(np.abs(normals[index:index + 2, 2]) < 1e-6):
            plane = "vertical"
        else:
            plane = "3D"
        normal_lengths = np.linalg.norm(normals[index:index + 2], axis=1)
        if min(normal_lengths) < 1e-6:
            sense = "straight"
        else:
            alignment = np.dot(normals[index], normals[index + 1]) / np.prod(normal_lengths)
            sense = "same" if alignment > 0.999 else "opposite" if alignment < -0.999 else "skew"
        patterns[(int(bins[index]), int(bins[index + 1]), plane, sense, bucket)] += 1

def print_corners(angles, patterns, limit):
    total = sum(angles.values())
    print("\n--- Original-path turn angles (0 degrees = straight; 180 = reverse) ---")
    print(" Angle | Occurrences |     in % | visual")
    for angle, amount in sorted(angles.items()):
        percent = 100 * amount / total
        bars = "I" * int(round(percent)) or "."
        print(f"{angle:5d}d | {amount:11,d} | {percent:7.2f}% | {bars}")
    print("Angles are rounded to the nearest 5 degrees; counts include endpoint-adjacent turns.")
    total = sum(patterns.values())
    print(f"\n--- Consecutive turn patterns: top {limit} of {len(patterns):,} groups ---")
    print(" Turn pair | Plane    | Direction | B-C gap | Occurrences |     in % | visual")
    shown = patterns.most_common(limit)
    for (first, second, plane, sense, bucket), amount in shown:
        percent = 100 * amount / total
        bars = "I" * int(round(percent)) or "."
        print(f"{first:3d}/{second:<3d}d | {plane:8s} | {sense:9s} | {bucket:7s} | "
              f"{amount:11,d} | {percent:7.2f}% | {bars}")
    remaining = total - sum(amount for _, amount in shown)
    print(f"Total: {total:,} pattern occurrences; remaining groups: {remaining:,} occurrences.")
    print("Each pattern is A -> B -> C -> D; gap is B-C length. Counts repeat across target paths.")


class Inspection:
    name = "corner_patterns"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.limit = args.pattern_limit
        self.angles = Counter()
        self.patterns = Counter()

    def observe(self, points, target):
        if points is None or not len(points):
            return
        collect_corners(np.vstack((self.prepared.route_start, points, target)),
                        self.angles, self.patterns)

    def report(self):
        print_corners(self.angles, self.patterns, self.limit)


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])

