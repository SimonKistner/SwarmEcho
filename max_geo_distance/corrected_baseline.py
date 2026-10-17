"""Inspection 2: placement counts after repairing segment range limits."""

from collections import Counter
import numpy as np

from reporting import print_histogram
from placement_checks import range_limit, rounded


def corrected_chain(base, route_start, points, target, cfg):
    """Retain roadmap nodes and insert drones along their existing segments."""
    positions = []
    waypoints = list(points)
    if np.linalg.norm(route_start - base) > 1e-8:
        # Include a real drone at the clearance-safe attachment rather than
        # counting a chain that starts from an imaginary raised station.
        waypoints.insert(0, route_start)
    for waypoint in waypoints:
        start = positions[-1] if positions else base
        first_range = range_limit(cfg.comm_radius if positions else cfg.comm_radius_base, construction=True)
        relay_range = range_limit(cfg.comm_radius, construction=True)
        direction = waypoint - start
        length = float(np.linalg.norm(direction))
        extra = int(np.ceil(max(0.0, length - first_range) / relay_range))
        if extra:
            direction /= length
            for index in range(extra):
                positions.append(start + direction * (first_range + index * relay_range))
        positions.append(waypoint.copy())

    start = positions[-1] if positions else base
    direction = target - start
    length = float(np.linalg.norm(direction))
    # Construct farther inside the limit than the rounded acceptance guard.
    advance = max(0.0, length - range_limit(cfg.visual_radius, construction=True))
    if advance > 0 or not positions:
        first_range = range_limit(cfg.comm_radius if positions else cfg.comm_radius_base, construction=True)
        relay_range = range_limit(cfg.comm_radius, construction=True)
        extra = int(np.ceil(max(0.0, advance - first_range) / relay_range))
        unit = direction / length if length else np.zeros(3)
        for index in range(extra):
            positions.append(start + unit * (first_range + index * relay_range))
        positions.append(start + unit * advance)
    return rounded(positions)


class Inspection:
    name = "corrected_baseline"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total = args.targets
        self.histogram = Counter()
        self.original_nodes = 0

    def observe(self, points, target):
        if points is None:
            return
        self.original_nodes += len(points)
        chain = corrected_chain(self.prepared.base, self.prepared.route_start,
                                points, target, self.prepared.cfg)
        self.histogram[len(chain)] += 1

    def report(self):
        print_histogram(self.histogram, self.total,
                        "--- Corrected baseline: agents in the constructed chain ---")
        extra = sum(count * amount for count, amount in self.histogram.items()) - self.original_nodes
        over_seven = sum(amount for count, amount in self.histogram.items() if count > 7)
        print(f"Inserted drones across all reachable target paths: {extra:,}")
        print(f"Corrected chains needing >7 agents: {over_seven:,}/{self.total:,} "
              f"({100 * over_seven / self.total:.2f}%)")
        print("Retains every roadmap node; repairs base/internal links and the visual target tip.")
        print("Constructed placement count, not a minimum; no corner simplification is applied.")
        if np.linalg.norm(self.prepared.route_start - self.prepared.base) > 1e-8:
            print("Includes a drone at the raised planning attachment on each reachable path.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])

