"""On-demand, disk-cached reference routes for recorded replays."""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from swarmecho.core.config import EnvConfig, MAP_DIR
from swarmecho.env.buildings import compile_building, load_building
from swarmecho.env.roadmap_cpu import building_candidate_vertices


CACHE_VERSION = 1
SETTINGS = ("drone_radius", "obstacle_planning_clearance_m", "roadmap_approach",
            "roadmap_node_density", "roadmap_merge_wall_end_nodes", "roadmap_corner_bonus_m")


def cache_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".roadmap-path.json")


def source_stamp(manifest_path: Path) -> dict:
    """Invalidate a route when its recorded geometry or planner implementation changes."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = manifest_path.parent / manifest["data_file"]
    files = [manifest_path, data_path]
    if "building_snapshot" not in manifest:
        from swarmecho.core.compatibility import resolve_map_file
        files.append(resolve_map_file(MAP_DIR / f"{Path(manifest['map_name']).stem}.yaml", MAP_DIR))
    return {"version": CACHE_VERSION,
            "files": [[str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size]
                      for path in files]}


def cached_route(manifest_path: Path, stamp: dict) -> dict | None:
    path = cache_path(manifest_path)
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    route = saved.get("route") if isinstance(saved, dict) and saved.get("source") == stamp else None
    return route if isinstance(route, dict) and isinstance(route.get("path"), list) else None


def _visible(start: np.ndarray, ends: np.ndarray, lower: np.ndarray,
             upper: np.ndarray) -> np.ndarray:
    """Segment versus clearance-expanded boxes, matching roadmap_scan.visible."""
    if not len(lower):
        return np.ones(len(ends), dtype=bool)
    result = np.ones(len(ends), dtype=bool)
    for offset in range(0, len(ends), 128):
        direction = ends[offset:offset + 128, None, :] - start
        parallel = np.abs(direction) <= 1e-5
        safe = np.where(parallel, 1., direction)
        first, last = (lower - start) / safe, (upper - start) / safe
        near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
        far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
        outside = np.any(parallel & ((start < lower) | (start > upper)), axis=-1)
        hit = (far >= np.maximum(near, 1e-5)) & (near <= 1. - 1e-5) & ~outside
        result[offset:offset + 128] = ~np.any(hit, axis=-1)
    return result


def _shortest_route(adjacency: list[list[tuple[int, float]]]) -> list[int]:
    costs = [float("inf")] * len(adjacency)
    previous = [-1] * len(adjacency)
    costs[0] = 0.
    queue = [(0., 0)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > costs[node]:
            continue
        if node == 1:
            route = []
            while node != -1:
                route.append(node)
                node = previous[node]
            return route[::-1]
        for neighbor, weight in adjacency[node]:
            candidate = cost + weight
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return []


def compute_route(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(manifest_path.parent / manifest["data_file"], allow_pickle=False) as archive:
        names = ("base_position", "target_position", "target_present", "obstacle_min", "obstacle_max")
        arrays = {name: archive[name] for name in names if name in archive}
    snapshot = manifest.get("building_snapshot")
    if snapshot is not None:
        grid = snapshot["building_cell_grid"]
        cell = snapshot["cell_size_m"]
        building = compile_building({**snapshot, "width": grid["cols"] * cell,
                                     "height": grid["rows"] * cell,
                                     "depth": grid["layers"] * cell})
    else:
        building = load_building(MAP_DIR / f"{Path(manifest['map_name']).stem}.yaml")

    defaults = EnvConfig()
    settings = {name: manifest[name] for name in SETTINGS if name in manifest}
    cfg = replace(defaults, **settings)
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    corner_merge = building.wall_thickness_m + 2 * clearance
    base = np.asarray(arrays["base_position"][0], dtype=np.float32)
    target = np.asarray(arrays["target_position"][0], dtype=np.float32)
    if "target_present" in arrays and not bool(np.asarray(arrays["target_present"][0]).any()):
        raise ValueError("This replay has no target in its first frame")
    dynamic_lo = np.asarray(arrays["obstacle_min"][0], dtype=np.float32).reshape(-1, 3)
    dynamic_hi = np.asarray(arrays["obstacle_max"][0], dtype=np.float32).reshape(-1, 3)
    lower = np.concatenate((building.solid_min_m, dynamic_lo))
    upper = np.concatenate((building.solid_max_m, dynamic_hi))
    planning_lo = lower - clearance + 1e-4
    planning_hi = upper + clearance - 1e-4
    for label, point in (("Base", base), ("Target", target)):
        if np.any(np.all((point >= planning_lo) & (point <= planning_hi), axis=1)):
            raise ValueError(f"{label} intersects a clearance-expanded obstacle")

    vertices = building_candidate_vertices(building, cfg)
    if len(dynamic_lo):
        bounds_lo = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                               [building.tile_thickness_m / 2 + cfg.drone_radius], dtype=np.float32)
        bounds_hi = building.world_size_m - bounds_lo
        corners = np.asarray(list(product((0., 1.), repeat=3)), dtype=np.float32)
        expanded_lo, expanded_hi = dynamic_lo - clearance, dynamic_hi + clearance
        extra = np.clip((expanded_lo[:, None, :] + corners[None, :, :] *
                         (expanded_hi - expanded_lo)[:, None, :]).reshape(-1, 3),
                        bounds_lo, bounds_hi)
        vertices = np.concatenate((vertices, extra))
    vertices = np.unique(vertices, axis=0).reshape(-1, 3)
    if len(vertices):
        valid = [not np.any(np.all((v >= planning_lo) & (v <= planning_hi), axis=1))
                 and not np.array_equal(v, base) and not np.array_equal(v, target)
                 for v in vertices]
        vertices = vertices[valid]
    nodes = np.concatenate((base[None], target[None], vertices))
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for left in range(len(nodes)):
        for offset in range(left + 1, len(nodes), 128):
            ends = nodes[offset:offset + 128]
            for relative in np.flatnonzero(_visible(nodes[left], ends, planning_lo, planning_hi)):
                right = offset + int(relative)
                distance = float(np.linalg.norm(nodes[left] - nodes[right]))
                bonus = (cfg.roadmap_corner_bonus_m if
                         (left == 0 and right >= 2) or
                         (left >= 2 and distance > corner_merge + 1e-4) else 0.)
                adjacency[left].append((right, distance + bonus))
                adjacency[right].append((left, distance + bonus))
    route = _shortest_route(adjacency)
    if not route:
        raise ValueError("No route found in the replay's visibility graph")
    path = nodes[route]
    physical = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    weights = [dict(neighbors) for neighbors in adjacency]
    cost = sum(weights[a][b] for a, b in zip(route, route[1:]))
    layout_changes = (not np.array_equal(arrays["base_position"],
                                         np.broadcast_to(base, arrays["base_position"].shape))
                      or not np.array_equal(arrays["target_position"],
                                            np.broadcast_to(target, arrays["target_position"].shape))
                      or not np.array_equal(arrays["obstacle_min"],
                                            np.broadcast_to(arrays["obstacle_min"][0], arrays["obstacle_min"].shape))
                      or not np.array_equal(arrays["obstacle_max"],
                                            np.broadcast_to(arrays["obstacle_max"][0], arrays["obstacle_max"].shape)))
    return {"path": path.tolist(), "physical_length_m": physical, "roadmap_cost_m": cost,
            "waypoint_count": max(0, len(path) - 2),
            "settings_source": "recorded" if len(settings) == len(SETTINGS) else "defaults for missing settings",
            "geometry_source": "replay snapshot" if snapshot is not None else "current map",
            "layout": "first frame only" if layout_changes else "fixed"}


def compute_and_cache(manifest_name: str, stamp: dict) -> dict:
    # Keep the inspector's plotting and request threads ahead of this CPU-heavy job.
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    manifest_path = Path(manifest_name)
    route = compute_route(manifest_path)
    # A replay can be replaced while calculation is running. Never save a stale route.
    if source_stamp(manifest_path) != stamp:
        raise ValueError("Replay changed during path calculation; toggle the path again")
    path = cache_path(manifest_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"source": stamp, "route": route}), encoding="utf-8")
    temporary.replace(path)
    return route
