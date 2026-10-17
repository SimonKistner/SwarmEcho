"""Shared target sampling and shortest-roadmap paths for the inspections."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarmecho.core.compatibility import resolve_level_file, resolve_map_file
from swarmecho.core.config import EnvConfig, LEVEL_DIR, MAP_DIR
from swarmecho.env.buildings import BuildingArrays, load_building, target_spawn_boxes
from swarmecho.env.roadmap_cpu import _visible_from, building_roadmap
from placement_checks import PROTOCOL, SAFETY_MARGIN_M, point_valid, range_limit, rounded, visible32


@dataclass
class PreparedPaths:
    cfg: EnvConfig
    base: np.ndarray
    route_start: np.ndarray
    targets: np.ndarray
    vertices: np.ndarray
    distances: np.ndarray
    paths: list
    solid_lo: np.ndarray
    solid_hi: np.ndarray
    physical_solid_lo: np.ndarray
    physical_solid_hi: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    building: BuildingArrays
    visible_edges: np.ndarray
    components: np.ndarray
    route_counts: Counter = field(default_factory=Counter)
    route_examples: list = field(default_factory=list)

    def finder_attachment(self, target):
        """Attach sighting to reachable nodes/edges, without requiring flight to the target.

        Each edge supplies its one analytic nearest point. Select the cheapest
        valid attachment to the existing shortest-path tree; no new graph search.
        """
        reachable = np.isfinite(self.distances[1:])
        nodes = np.flatnonzero(reachable)
        candidates = [self.route_start[None], self.vertices[nodes]]
        origins = [np.asarray([-1]), nodes]
        costs = [np.asarray([0.0]), self.distances[nodes + 1]]
        a, b = np.nonzero(np.triu(self.visible_edges, 1) & reachable[:, None] & reachable[None, :])
        if len(a):
            edge = self.vertices[b] - self.vertices[a]
            length2 = np.sum(edge * edge, axis=1)
            fraction = np.clip(np.sum((target - self.vertices[a]) * edge, axis=1) /
                               np.maximum(length2, 1e-20), 0, 1)
            points = rounded(self.vertices[a] + fraction[:, None] * edge)
            first = self.distances[a + 1] + np.linalg.norm(points - self.vertices[a], axis=1)
            second = self.distances[b + 1] + np.linalg.norm(points - self.vertices[b], axis=1)
            candidates.append(points)
            origins.append(np.where(first <= second, a, b))
            costs.append(np.minimum(first, second))
        candidates, origins, costs = np.concatenate(candidates), np.concatenate(origins), np.concatenate(costs)
        nearby = np.linalg.norm(candidates - target, axis=1) <= range_limit(self.cfg.visual_radius, construction=True)
        indices = np.flatnonzero(nearby)
        if not len(indices):
            return None, 0
        visible = visible32(target, candidates[indices], self.physical_solid_lo, self.physical_solid_hi)
        indices = indices[visible]
        indices = np.asarray([index for index in indices if point_valid(candidates[index], self)], dtype=int)
        if not len(indices):
            return None, 0
        index = indices[np.argmin(costs[indices])]
        node, point = int(origins[index]), candidates[index]
        path = self.vertices[self.paths[node]].copy() if node >= 0 else np.empty((0, 3))
        start = path[-1] if len(path) else self.route_start
        if np.linalg.norm(point - start) > 1e-7:
            # The projection is on an existing clear edge. Check the rounded
            # segment as well; do not return a nominally reachable witness.
            if not _visible_from(start, point[None], self.solid_lo, self.solid_hi)[0]:
                return None, len(indices)
            path = np.concatenate((path, point[None]))
        return path, len(indices)

    def report_routes(self, details=5):
        print("\n--- Roadmap attachment diagnostics ---")
        print(f"Protocol: {PROTOCOL}; numerical margin: {SAFETY_MARGIN_M:g}m")
        print(f"Planning components: {len(np.unique(self.components))}; "
              f"base-reachable nodes: {np.isfinite(self.distances[1:]).sum():,}/{len(self.vertices):,}")
        print(f"Endpoint misses before finder recovery: {self.route_counts['endpoint miss']:,}; "
              f"recovered by finder attachment: {self.route_counts['finder recovered']:,}; "
              f"unresolved: {self.route_counts['unresolved']:,}")
        print("Missing endpoint routes are not classified as physically impossible.")
        if self.route_counts['finder recovered']:
            print("Recovered paths may include an extra edge-finder waypoint outside the original vertex set.")
        cells = Counter((example['cell'], example['reason'], example['recovered']) for example in self.route_examples)
        for (cell, reason, recovered), count in sorted(cells.items()):
            print(f"  cell {cell}: {count:,}; {reason}; recovered={recovered}")
        selected = sorted(self.route_examples, key=lambda example: (example['recovered'], example['sample']))
        print(f"Detailed endpoint-miss examples: up to {details}")
        for example in selected[:details]:
            print(f"  Sample {example['sample']}; target={example['target'].tolist()}; "
                  f"cell={example['cell']}; recovered={example['recovered']}; {example['reason']}")
            print(f"    planning-visible nodes={example['visible']}; reachable-visible={example['reachable_visible']}; "
                  f"visible components={example['components']}; legal finder attachments={example['finders']}; "
                  f"nearest reachable node distance={example['nearest']:.6f}m")
        unresolved = [str(example['sample']) for example in self.route_examples if not example['recovered']]
        print("Unresolved zero-based indices: " + (", ".join(unresolved) or "none"))

    def iter_targets(self):
        """Yield roadmap/finder waypoints, or None for an unresolved target."""
        for sample, target in enumerate(self.targets):
            visible = _visible_from(target, self.vertices, self.solid_lo, self.solid_hi)
            costs = np.where(
                visible,
                self.distances[1:] + np.linalg.norm(self.vertices - target, axis=1),
                np.inf,
            )
            best = int(np.argmin(costs)) if len(costs) else -1
            via_length = costs[best] if best >= 0 else np.inf
            direct_visible = _visible_from(
                self.route_start, target[None], self.solid_lo, self.solid_hi
            )[0]
            direct_length = np.linalg.norm(target - self.route_start)
            if direct_visible and direct_length <= via_length:
                yield np.empty((0, 3)), target
            elif not np.isfinite(via_length):
                self.route_counts["endpoint miss"] += 1
                reachable = np.isfinite(self.distances[1:])
                path, finders = self.finder_attachment(target)
                recovered = path is not None
                reason = ("endpoint sees only disconnected graph components" if np.any(visible) else
                          "no planning-visible endpoint attachment")
                self.route_examples.append({
                    "sample": sample, "target": target.copy(),
                    "cell": tuple(np.floor(target / self.building.cell_size_m).astype(int).tolist()),
                    "reason": reason, "recovered": recovered, "visible": int(visible.sum()),
                    "reachable_visible": int(np.sum(visible & reachable)),
                    "components": np.unique(self.components[visible]).tolist(), "finders": finders,
                    "nearest": float(np.min(np.linalg.norm(self.vertices[reachable] - target, axis=1)))
                               if np.any(reachable) else float("inf"),
                })
                self.route_counts["finder recovered" if recovered else "unresolved"] += 1
                yield path, target
            else:
                yield self.vertices[self.paths[best]], target


def prepare_paths(args):
    level_path = resolve_level_file(args.level, LEVEL_DIR)
    env = yaml.safe_load(level_path.read_text(encoding="utf-8"))["env"]
    settings = (
        "drone_radius", "obstacle_planning_clearance_m", "target_wall_buffer_fraction",
        "roadmap_approach", "roadmap_node_density", "roadmap_merge_wall_end_nodes",
        "comm_radius_base", "comm_radius", "visual_radius",
    )
    cfg = EnvConfig(**{key: env[key] for key in settings if key in env})
    if min(cfg.comm_radius_base, cfg.comm_radius, cfg.visual_radius) <= 0:
        raise ValueError("Communication and visual ranges must be positive")
    map_path = resolve_map_file(args.map, MAP_DIR)
    building = load_building(map_path)
    base = np.asarray(building.base_position_m, dtype=np.float64)
    print(f"Map: {map_path.stem}; settings: {level_path.stem}; seed: {args.seed}", flush=True)
    print(f"Targets: {args.targets:,}; fixed base (m): {base.tolist()}", flush=True)
    print(f"Roadmap: {cfg.roadmap_approach}; density: {cfg.roadmap_node_density:g}; "
          f"merge wall-end nodes: {cfg.roadmap_merge_wall_end_nodes}", flush=True)
    print(f"Assignment protocol: {PROTOCOL}; safety margin: {SAFETY_MARGIN_M:g}m", flush=True)

    # Use existing free-volume spawn boxes, respecting noSpawn exclusions.
    lo, hi, volume = target_spawn_boxes(building, cfg)
    rng = np.random.default_rng(args.seed)
    box = rng.choice(len(lo), size=args.targets, p=volume.astype(float) / volume.sum(dtype=float))
    targets = rng.uniform(lo[box], hi[box])

    graph = building_roadmap(building, cfg, corner_bonus_m=0.0)
    vertices = graph.vertices.astype(np.float64)
    solid_lo = graph.solid_min - graph.clearance + 1e-4
    solid_hi = graph.solid_max + graph.clearance - 1e-4

    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                       [building.tile_thickness_m / 2 + cfg.drone_radius], dtype=float)
    upper = np.asarray(building.world_size_m, dtype=float) - lower
    # A station is not a drone. Start at the real spawn height, inset from
    # world bounds, and lift above any containing inflated slab deterministically.
    route_start = np.clip(base + [0, 0, cfg.drone_radius], lower + SAFETY_MARGIN_M,
                          upper - SAFETY_MARGIN_M)
    for _ in range(len(solid_lo) + 1):
        containing = np.all((route_start >= solid_lo) & (route_start <= solid_hi), axis=1)
        if not np.any(containing):
            break
        route_start[2] = float(solid_hi[containing, 2].max()) + SAFETY_MARGIN_M
    route_start = rounded(route_start)
    if np.any(route_start < lower) or np.any(route_start > upper):
        raise ValueError("No legal planning start above the authored station.")
    print(f"Legal drone planning start (m): {route_start.tolist()} "
          "(radio station coordinate remains unchanged)", flush=True)

    # One shortest-path tree serves every target. Only endpoint attachment is
    # repeated; visible_edges contains direct links, not all-pairs shortcuts.
    a, b = np.nonzero(graph.visible_edges)
    edge_lengths = np.linalg.norm(vertices[a] - vertices[b], axis=1)
    attached = np.flatnonzero(_visible_from(route_start, vertices, solid_lo, solid_hi))
    attachment_lengths = np.linalg.norm(vertices[attached] - route_start, axis=1)
    rows = np.concatenate((a + 1, np.zeros(len(attached), dtype=int), attached + 1))
    cols = np.concatenate((b + 1, attached + 1, np.zeros(len(attached), dtype=int)))
    weights = np.concatenate((edge_lengths, attachment_lengths, attachment_lengths))
    adjacency = coo_matrix((weights, (rows, cols)),
                           shape=(len(vertices) + 1, len(vertices) + 1)).tocsr()
    distances, predecessors = dijkstra(adjacency, directed=False, indices=0,
                                       return_predecessors=True)
    _, components = connected_components(coo_matrix(graph.visible_edges).tocsr(), directed=False)
    paths = []
    for index in range(len(vertices)):
        path = []
        node = index + 1
        if np.isfinite(distances[node]):
            while node != 0:
                path.append(node - 1)
                node = int(predecessors[node])
            path.reverse()
        paths.append(path)

    print(f"Roadmap ready: {len(vertices):,} nodes; inspecting targets...", flush=True)
    return PreparedPaths(cfg, base, route_start, targets, vertices, distances,
                         paths, solid_lo, solid_hi, graph.solid_min, graph.solid_max,
                         lower, upper, building, graph.visible_edges, components)

