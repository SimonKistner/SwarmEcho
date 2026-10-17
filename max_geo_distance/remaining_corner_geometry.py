"""Inspection 5: locate remaining short 3D pairs and archive their geometry context."""

from collections import Counter
import json

import numpy as np

from corrected_baseline import corrected_chain
from corner_simplification import simplify_chain
from paths import _visible_from


def coordinates(point):
    return "(" + ", ".join(repr(float(value)) for value in point) + ")"


def window_geometry(window):
    vectors = np.diff(window, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-8) or lengths[1] >= 2:
        return None
    unit = vectors / lengths[:, None]
    if np.all(np.abs(unit[:, 2]) < 1e-6):
        return None
    angles = np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)))
    normals = np.cross(unit[:-1], unit[1:])
    plane = "vertical" if np.all(np.abs(normals[:, 2]) < 1e-6) else "3D"
    normal_lengths = np.linalg.norm(normals, axis=1)
    if min(normal_lengths) < 1e-6:
        sense = "straight"
    else:
        alignment = np.dot(normals[0], normals[1]) / np.prod(normal_lengths)
        sense = "same" if alignment > 0.999 else "opposite" if alignment < -0.999 else "skew"
    bins = tuple(int(value) for value in 5 * np.floor(angles / 5 + 0.5))
    return angles, lengths, (bins[0], bins[1], plane, sense)


def authored_feature_envelopes(building):
    extras = json.loads(building.extra_geometry_json)
    cell = building.cell_size_m
    envelopes = []
    for axis_name, axis in (("x", 0), ("y", 1)):
        along = 1 - axis
        for kind in ("doors", "windows"):
            for face in extras.get(f"{axis_name}_{kind}", []):
                lo = np.asarray(face, dtype=float) * cell
                hi = lo + cell
                lo[axis] = face[axis] * cell - building.wall_thickness_m / 2
                hi[axis] = face[axis] * cell + building.wall_thickness_m / 2
                lo[along] = (face[along] + 0.3) * cell
                hi[along] = (face[along] + 0.7) * cell
                lo[2] = (face[2] + (0 if kind == "doors" else 0.3)) * cell
                hi[2] = (face[2] + 0.7) * cell
                label = f"{axis_name}-{kind[:-1]}{tuple(face)}"
                envelopes.append((label, lo, hi))
    for x, y, z, direction in extras.get("stairs", []):
        lo = np.asarray((x, y, z), dtype=float) * cell
        hi = lo + np.asarray((cell, cell, 2 * cell))
        envelopes.append((f"stairs{(x, y, z)} dir={direction}", lo, hi))
    return envelopes


def distance_to_box(point, lo, hi):
    return float(np.linalg.norm(point - np.clip(point, lo, hi)))


def outer_line_gap(window):
    a, b, c, d = window
    first = (b - a) / np.linalg.norm(b - a)
    last = (d - c) / np.linalg.norm(d - c)
    normal = np.cross(first, last)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return float(np.linalg.norm(np.cross(c - b, first)))
    return float(abs(np.dot(c - b, normal)) / norm)


class Inspection:
    name = "remaining_corner_geometry"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total = args.targets
        self.limit = args.pattern_limit
        self.details = args.geometry_details
        self.features = authored_feature_envelopes(prepared.building)
        self.groups = {}
        self.outcomes = Counter()
        self.sample_index = -1
        self.unreachable = self.over_seven = self.windows = 0
        self.short_occurrences = 0
        self.affected = set()
        self.affected_over_seven = set()

    def observe(self, points, target):
        self.sample_index += 1
        if points is None:
            self.unreachable += 1
            return
        baseline = corrected_chain(self.prepared.base, self.prepared.route_start,
                                   points, target, self.prepared.cfg)
        chain, _ = simplify_chain(baseline, target, self.prepared, self.outcomes)
        over_seven = len(chain) > 7
        self.over_seven += int(over_seven)
        route = np.vstack((self.prepared.base, chain, target))
        for index in range(1, len(route) - 2):
            self.windows += 1
            window = route[index - 1:index + 3]
            geometry = window_geometry(window)
            if geometry is None:
                continue
            self.short_occurrences += 1
            self.affected.add(self.sample_index)
            if over_seven:
                self.affected_over_seven.add(self.sample_index)
            _, _, pattern = geometry
            key = (tuple(window[1]), tuple(window[2]))
            group = self.groups.setdefault(key, {
                "occurrences": 0, "targets": set(), "over_seven": set(),
                "patterns": Counter(), "example": None,
            })
            group["occurrences"] += 1
            group["targets"].add(self.sample_index)
            group["patterns"][pattern] += 1
            if over_seven:
                group["over_seven"].add(self.sample_index)
            example = group["example"]
            if example is None or (over_seven and example["after"] <= 7):
                group["example"] = {
                    "window": window.copy(), "target": target.copy(),
                    "sample": self.sample_index, "before": len(baseline), "after": len(chain),
                    "left_is_base": index == 1, "right_is_target": index + 2 == len(route) - 1,
                }

    def nearby_features(self, midpoint):
        return sorted(((distance_to_box(midpoint, lo, hi), label, lo, hi)
                       for label, lo, hi in self.features), key=lambda item: (item[0], item[1]))

    def report(self):
        ranked = sorted(self.groups.items(),
                        key=lambda item: (-len(item[1]["over_seven"]),
                                          -item[1]["occurrences"], item[0]))
        shown = ranked[:self.limit]
        print("\n--- Remaining short non-horizontal corner geometry ---")
        print("Preprocessing: corrected chain plus the existing horizontal 45/45 replacement.")
        print(f"Existing rule replacements reproduced: {self.outcomes['accepted']:,}")
        print(f"Simplified targets still requiring >7 agents: {self.over_seven:,}/{self.total:,}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,}")
        print(f"Final-chain windows: {self.windows:,}; short non-horizontal occurrences: "
              f"{self.short_occurrences:,}")
        print(f"Distinct affected targets: {len(self.affected):,}; "
              f"affected targets still >7 agents: {len(self.affected_over_seven):,}")
        print(f"Distinct ordered B/C locations: {len(ranked):,}; showing {len(shown):,}")
        print("Filter: positive B-C gap <2m; non-horizontal legs; B and C are drones.")
        print("Priority: distinct >7-agent targets, then occurrences. Feature labels indicate proximity.")
        print("\n Pair | Gap (m) | Occurrences | Targets | >7 targets | Common turns / plane / sense | Nearest feature")
        print("------+---------+-------------+---------+------------+------------------------------+--------------------------")
        for number, (key, group) in enumerate(shown, 1):
            b, c = np.asarray(key)
            midpoint = (b + c) / 2
            features = self.nearby_features(midpoint)
            nearest = f"{features[0][1]} ({features[0][0]:.3f}m)" if features else "none authored"
            pattern, amount = group["patterns"].most_common(1)[0]
            first, second, plane, sense = pattern
            description = f"{first}/{second}d {plane} {sense}"
            print(f"{number:5d} | {np.linalg.norm(c - b):7.3f} | {group['occurrences']:11,d} | "
                  f"{len(group['targets']):7,d} | {len(group['over_seven']):10,d} | "
                  f"{description:28s} | {nearest}")
        unshown = ranked[len(shown):]
        print(f"Unshown groups: {len(unshown):,}; "
              f"occurrences in those groups: {sum(group['occurrences'] for _, group in unshown):,}")

        for number, (key, group) in enumerate(shown[:self.details], 1):
            example = group["example"]
            window = example["window"]
            midpoint = (window[1] + window[2]) / 2
            angles, lengths, _ = window_geometry(window)
            print(f"\nPair {number}: representative sample index {example['sample']} (zero-based)")
            print(f"  Agents: baseline {example['before']} -> simplified {example['after']}; "
                  f"target={coordinates(example['target'])}")
            for name, point in zip("ABCD", window):
                role = "base" if name == "A" and example["left_is_base"] else (
                    "target" if name == "D" and example["right_is_target"] else "drone")
                print(f"  {name} ({role}): {coordinates(point)}")
            print(f"  Turn angles: {angles[0]:.6f} / {angles[1]:.6f} degrees; "
                  f"AB/BC/CD lengths: {lengths[0]:.6f} / {lengths[1]:.6f} / {lengths[2]:.6f}m")
            print(f"  Outer infinite-line separation: {outer_line_gap(window):.6f}m")
            direct = _visible_from(window[0], window[3:4], self.prepared.physical_solid_lo,
                                   self.prepared.physical_solid_hi)[0]
            print(f"  A-D direct segment: {np.linalg.norm(window[3] - window[0]):.6f}m; "
                  f"signal visibility={bool(direct)} (diagnostic only)")
            print("  Common rounded patterns at this B/C location:")
            for pattern, amount in group["patterns"].most_common(3):
                first, second, plane, sense = pattern
                print(f"    {first}/{second}d {plane} {sense}: {amount:,} occurrences")
            print("  Nearest authored feature envelopes (midpoint distance):")
            for distance, label, lo, hi in self.nearby_features(midpoint)[:3]:
                print(f"    {label}: {distance:.6f}m; lo={coordinates(lo)}; hi={coordinates(hi)}")
            if not self.features:
                print("    none")
            print("  Nearest physical solid boxes (midpoint distance, map-local solid index):")
            lo, hi = self.prepared.physical_solid_lo, self.prepared.physical_solid_hi
            distances = np.linalg.norm(midpoint - np.clip(midpoint, lo, hi), axis=1)
            for solid in np.argsort(distances, kind="stable")[:4]:
                print(f"    solid {solid}: {distances[solid]:.6f}m; "
                      f"lo={coordinates(lo[solid])}; hi={coordinates(hi[solid])}")
            if not len(lo):
                print("    none")
        print("\nAll coordinates and bounds are metres; groups use exact ordered coordinate tuples.")
        print("Nearest envelopes are context, not a confirmed doorway/stair classification.")
        print("Only the existing simplification is applied; this step proposes no new drone positions.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
