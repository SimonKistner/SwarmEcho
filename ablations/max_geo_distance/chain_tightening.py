"""Inspection 8: bypass existing relays and repeat the prior local rules until stable."""

from collections import Counter
import numpy as np

from corrected_baseline import corrected_chain
from corner_simplification import simplify_chain
from doorway_simplification import OpeningCatalog, simplify_doorways
from lifted_corner_simplification import simplify_lifted
from paths import _visible_from
from remaining_corner_geometry import coordinates
from placement_checks import link_reason


def prune_relays(chain, target, prepared, outcomes, roles):
    route = [prepared.base.copy(), *(point.copy() for point in chain), target.copy()]
    index = 1
    windows = 0
    while index < len(route) - 1 and len(route) > 3:
        windows += 1
        a, _, c = route[index - 1:index + 2]
        right_is_target = index + 1 == len(route) - 1
        if right_is_target:
            radius, role = prepared.cfg.visual_radius, "finder"
        elif index == 1:
            radius, role = prepared.cfg.comm_radius_base, "base-adjacent"
        else:
            radius, role = prepared.cfg.comm_radius, "internal"
        if np.linalg.norm(c - a) > radius:
            outcomes["range exceeded"] += 1
            index += 1
        elif not _visible_from(a, c[None], prepared.physical_solid_lo,
                               prepared.physical_solid_hi)[0]:
            outcomes["direct link obstructed"] += 1
            index += 1
        elif link_reason(a, c, radius, prepared) != "accepted":
            outcomes["rounded bypass rejected"] += 1
            index += 1
        else:
            outcomes["accepted"] += 1
            roles[role] += 1
            del route[index]
            index = max(1, index - 1)
    return np.asarray(route[1:-1]), windows


class Inspection:
    name = "chain_tightening"

    def __init__(self, prepared, args):
        self.prepared = prepared
        self.total, self.details = args.targets, args.geometry_details
        self.catalog = OpeningCatalog(prepared.building)
        self.histograms = [Counter() for _ in range(6)]
        self.old_horizontal, self.old_doorway, self.old_lifted = Counter(), Counter(), Counter()
        self.old_openings = {}
        self.first_prune, self.first_roles = Counter(), Counter()
        self.repeat_horizontal, self.repeat_doorway, self.repeat_lifted = Counter(), Counter(), Counter()
        self.repeat_prune, self.repeat_roles = Counter(), Counter()
        self.repeat_openings = {}
        self.improved_examples, self.remaining_examples = {}, {}
        self.unreachable = self.first_windows = self.repeat_windows = 0
        self.cycles_total = self.cycles_max = self.improved = self.direct_improved = self.max_saved = 0
        self.sample = -1

    def observe(self, points, target):
        self.sample += 1
        if points is None:
            self.unreachable += 1
            return
        baseline = corrected_chain(self.prepared.base, self.prepared.route_start,
                                   points, target, self.prepared.cfg)
        horizontal, _ = simplify_chain(baseline, target, self.prepared, self.old_horizontal)
        doorway, _, _ = simplify_doorways(horizontal, target, self.prepared, self.catalog,
                                         self.old_doorway, self.old_openings)
        lifted, _, _ = simplify_lifted(doorway, target, self.prepared, self.old_lifted)
        pruned, windows = prune_relays(lifted, target, self.prepared, self.first_prune, self.first_roles)
        self.first_windows += windows

        current = pruned
        cycles = 0
        while len(current) > 1:
            cycles += 1
            previous_count = len(current)
            current, _ = simplify_chain(current, target, self.prepared, self.repeat_horizontal)
            current, _, _ = simplify_doorways(current, target, self.prepared, self.catalog,
                                             self.repeat_doorway, self.repeat_openings)
            current, _, _ = simplify_lifted(current, target, self.prepared, self.repeat_lifted)
            current, windows = prune_relays(current, target, self.prepared,
                                           self.repeat_prune, self.repeat_roles)
            self.repeat_windows += windows
            if len(current) == previous_count:
                break
        self.cycles_total += cycles
        self.cycles_max = max(self.cycles_max, cycles)

        counts = tuple(map(len, (baseline, horizontal, doorway, lifted, pruned, current)))
        for histogram, count in zip(self.histograms, counts):
            histogram[count] += 1
        self.direct_improved += int(counts[4] < counts[3])
        self.improved += int(counts[5] < counts[3])
        self.max_saved = max(self.max_saved, counts[3] - counts[5])
        cell = tuple(np.floor(target / self.prepared.building.cell_size_m).astype(int))
        improved_key = (cell, counts[3], counts[5])
        remaining_key = (cell, counts[5])
        need_improved = counts[5] < counts[3] and improved_key not in self.improved_examples
        need_remaining = counts[5] > 7 and remaining_key not in self.remaining_examples
        if need_improved or need_remaining:
            example = {"sample": self.sample, "target": target.copy(), "counts": counts,
                       "lifted": lifted.copy(), "pruned": pruned.copy(), "final": current.copy()}
            if need_improved:
                self.improved_examples[improved_key] = example
            if need_remaining:
                self.remaining_examples[remaining_key] = example

    def print_chain(self, label, chain, target):
        print(f"  {label} ({len(chain)} drones); base={coordinates(self.prepared.base)}")
        previous = self.prepared.base
        cfg = self.prepared.cfg
        for index, point in enumerate(chain):
            radius = cfg.comm_radius_base if index == 0 else cfg.comm_radius
            visible = _visible_from(previous, point[None], self.prepared.physical_solid_lo,
                                    self.prepared.physical_solid_hi)[0]
            print(f"    drone {index + 1}: {coordinates(point)}; "
                  f"incoming={np.linalg.norm(point - previous):.6f}/{radius:g}m; visible={bool(visible)}")
            previous = point
        visible = _visible_from(previous, target[None], self.prepared.physical_solid_lo,
                                self.prepared.physical_solid_hi)[0]
        print(f"    target tip={np.linalg.norm(target - previous):.6f}/{cfg.visual_radius:g}m; "
              f"visible={bool(visible)}")

    def report(self):
        print("\n--- Chain tightening: direct relay deletion and repeated existing rules ---")
        print("Direct deletion checks the bypass link's visibility and applicable range; one finder is retained.")
        print("Repeat horizontal -> doorway -> lifted -> deletion until an entire cycle removes no drones.")
        print(" Agents | Corrected | Horizontal | Doorways | Lifted | Pruned | Stable | Final % | visual")
        print("--------+-----------+------------+----------+--------+--------+--------+---------+----------------------------------------")
        maximum = max(max(histogram, default=0) for histogram in self.histograms)
        for count in range(maximum + 1):
            baseline, horizontal, doorway, lifted, pruned, final = (
                histogram[count] for histogram in self.histograms)
            percent = 100 * final / self.total
            bars = "I" * int(round(percent)) or ("." if final else "")
            print(f"{count:7d} | {baseline:9,d} | {horizontal:10,d} | {doorway:8,d} | {lifted:6,d} | "
                  f"{pruned:6,d} | {final:6,d} | {percent:6.2f}% | {bars}")
        totals = [sum(count * amount for count, amount in histogram.items())
                  for histogram in self.histograms]
        over_seven = [sum(amount for count, amount in histogram.items() if count > 7)
                      for histogram in self.histograms]
        reachable = sum(self.histograms[0].values())
        print(f"\nTargets improved by initial direct deletion: {self.direct_improved:,}/{self.total:,}")
        print(f"Targets improved beyond lifted stage after all tightening: {self.improved:,}/{self.total:,} "
              f"({100 * self.improved / self.total:.2f}%)")
        print(f"Drones saved by initial deletion: {totals[3] - totals[4]:,}; "
              f"additional savings from repeated rules: {totals[4] - totals[5]:,}")
        print(f"Total additional drones saved: {totals[3] - totals[5]:,}; "
              f"maximum additional saving on one target: {self.max_saved}")
        print("Stage order: corrected -> horizontal -> doorways -> lifted -> pruned -> stable")
        if reachable:
            print("Mean agents: " + " -> ".join(f"{total / reachable:.3f}" for total in totals))
        print("Targets requiring >7 agents: " +
              " -> ".join(f"{amount:,} ({100 * amount / self.total:.2f}%)" for amount in over_seven))
        print(f"Additional targets moved to <=7 agents: {over_seven[3] - over_seven[5]:,}")
        print(f"No roadmap route: {self.unreachable:,}/{self.total:,}")
        print(f"Previous horizontal/doorway/lifted replacements reproduced: "
              f"{self.old_horizontal['accepted']:,} / {self.old_doorway['accepted']:,} / "
              f"{self.old_lifted['accepted']:,}")
        print(f"\nAccepted initial direct deletions: {self.first_prune['accepted']:,}")
        print(f"Accepted additional horizontal/doorway/lifted/deletion operations: "
              f"{self.repeat_horizontal['accepted']:,} / {self.repeat_doorway['accepted']:,} / "
              f"{self.repeat_lifted['accepted']:,} / {self.repeat_prune['accepted']:,}")
        roles = self.first_roles + self.repeat_roles
        print("Direct deletions by endpoint role: " +
              ", ".join(f"{role}={roles[role]:,}" for role in ("base-adjacent", "internal", "finder")))
        for label, outcomes, windows in (("Initial deletion", self.first_prune, self.first_windows),
                                         ("Repeated deletion", self.repeat_prune, self.repeat_windows)):
            print(f"{label} windows: {windows:,}; range rejections: {outcomes['range exceeded']:,}; "
                  f"obstructed-link rejections: {outcomes['direct link obstructed']:,}")
        print(f"Repeated cycles including terminal zero-change cycles: {self.cycles_total:,}; "
              f"maximum on one target: {self.cycles_max}")
        print("Attempts include revisits; stable means no reduction by this rule order, not a global minimum.")

        for title, collection in (("Improved chain examples", self.improved_examples),
                                   ("Remaining >7-agent chain examples", self.remaining_examples)):
            examples = sorted(collection.values(),
                              key=lambda example: (-example['counts'][5],
                                                    -(example['counts'][3] - example['counts'][5]),
                                                    example['sample']))
            print(f"\n--- {title}: up to {self.details} ---")
            for example in examples[:self.details]:
                print(f"Sample {example['sample']} (zero-based); target={coordinates(example['target'])}")
                print("  Agents corrected/horizontal/doorways/lifted/pruned/stable: " +
                      "/".join(map(str, example["counts"])))
                self.print_chain("Before tightening", example["lifted"], example["target"])
                self.print_chain("Stable chain", example["final"], example["target"])
            if not examples:
                print("  none")
        print("\nRetained positions and accepted replacement link checks preserve the constructed geometric chains.")
        print("No alternative route or placement search is performed; counts do not prove global minima or policy success.")
        print("Previous stages are preprocessing only; earlier inspection reports are not printed.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
