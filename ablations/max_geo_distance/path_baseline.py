"""Inspection 1: original roadmap-node histogram and segment range checks."""

from collections import Counter
import numpy as np

from reporting import range_summary


class Inspection:
    name = "path_baseline"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total = args.targets
        self.histogram = Counter()
        self.unreachable = self.internal_total = self.internal_exceeded = 0
        self.base_exceeded = self.target_exceeded = self.routed = 0
        self.direct_base_exceeded = self.direct_target_exceeded = 0

    def observe(self, points, target):
        if points is None:
            self.unreachable += 1
            return
        cfg = self.prepared.cfg
        self.histogram[len(points)] += 1
        if not len(points):
            physical_length = np.linalg.norm(target - self.prepared.base)
            self.direct_base_exceeded += int(physical_length > cfg.comm_radius_base)
            self.direct_target_exceeded += int(physical_length > cfg.visual_radius)
            return
        self.routed += 1
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.internal_total += len(lengths)
        self.internal_exceeded += int(np.count_nonzero(lengths > cfg.comm_radius))
        self.base_exceeded += int(np.linalg.norm(points[0] - self.prepared.base) > cfg.comm_radius_base)
        self.target_exceeded += int(np.linalg.norm(target - points[-1]) > cfg.visual_radius)

    def report(self):
        cfg = self.prepared.cfg
        print("\nResults (roadmap nodes exclude base and target):")
        print(" Nr Nodes | Amount of targets |     in % | visual")
        print("----------+-------------------+----------+----------------------------------------")
        for count in range(max(self.histogram, default=0) + 1):
            amount = self.histogram[count]
            percent = 100 * amount / self.total
            bars = "I" * int(round(percent))
            if amount and not bars:
                bars = "."
            print(f"{count:9d} | {amount:17,d} | {percent:7.2f}% | {bars}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,} "
              f"({100 * self.unreachable / self.total:.2f}%)")
        print("Visual: each I is approximately 1% of all sampled targets; . is a smaller nonzero bin.")
        print("\n--- Range checks on paths with at least one roadmap node ---")
        range_summary("internal node-node lines", self.internal_exceeded, self.internal_total, cfg.comm_radius)
        range_summary("base-first-node tips", self.base_exceeded, self.routed, cfg.comm_radius_base)
        range_summary("last-node-target tips (visual range)", self.target_exceeded, self.routed, cfg.visual_radius)
        print("Internal lines count occurrences across target paths, including repeated map edges.")
        if self.histogram[0]:
            print("\n--- Direct paths (0 roadmap nodes; one shared endpoint segment) ---")
            range_summary("base-target lines (base comm)", self.direct_base_exceeded,
                          self.histogram[0], cfg.comm_radius_base)
            range_summary("base-target lines (visual)", self.direct_target_exceeded,
                          self.histogram[0], cfg.visual_radius)


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])

