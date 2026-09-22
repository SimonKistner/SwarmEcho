"""Generate ranked, inspector-discoverable visibility paths for a map or level."""
from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from swarmecho.env.buildings import load_building
from swarmecho.env.roadmap_cpu import building_roadmap, authored_roadmap_vertices
from swarmecho.env.buildings import building_snapshot


def _shortest(adjacency, start, goal, banned, banned_nodes=frozenset()):
    queue, visited = [(0.0, start, ())], set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited or node in banned_nodes:
            continue
        visited.add(node)
        path = (*path, node)
        if node == goal:
            return list(path)
        for neighbor, weight in adjacency[node]:
            if tuple(sorted((node, neighbor))) not in banned:
                heapq.heappush(queue, (cost + weight, neighbor, path))
    return []


def ranked_paths(adjacency, count):
    """Yen's k shortest distinct loopless paths, in ascending length order."""
    first = _shortest(adjacency, 0, 1, set())
    if not first:
        return []
    routes, candidates, seen = [first], [], {tuple(first)}
    weights = [dict(neighbors) for neighbors in adjacency]
    for _ in range(count - 1):
        previous = routes[-1]
        for index in range(len(previous) - 1):
            root = previous[:index + 1]
            banned = {tuple(sorted((p[index], p[index + 1]))) for p in routes
                      if len(p) > index + 1 and p[:index + 1] == root}
            spur = _shortest(adjacency, root[-1], 1, banned, set(root[:-1]))
            if spur:
                route = tuple(root[:-1] + spur)
                if route not in seen:
                    seen.add(route)
                    cost = sum(weights[a][b] for a, b in zip(route, route[1:]))
                    heapq.heappush(candidates, (cost, route))
        if not candidates:
            break
        routes.append(list(heapq.heappop(candidates)[1]))
    return routes


def _visible(start, ends, lower, upper):
    """Vectorized segment/AABB test with the runtime's clearance-boundary epsilon."""
    if not len(lower):
        return np.ones(len(ends), dtype=bool)
    direction = ends[:, None, :] - start
    parallel = np.abs(direction) < 1e-9
    safe = np.where(parallel, 1.0, direction)
    a, b = (lower - start) / safe, (upper - start) / safe
    near = np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=-1)
    far = np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=-1)
    parallel_outside = np.any(parallel & ((start < lower) | (start > upper)), axis=-1)
    blocked = ~parallel_outside & (np.maximum(near, 0) <= np.minimum(far, 1))
    return ~np.any(blocked, axis=-1)


def generate(destination=None, *, level_name=None, map_name=None, target=None,
             start=None, path_count=5, layout_count=None, seed=9000, obstacles=False,
             merge_walls=None, corner_bonus_m=None, building=None, cfg=None,
             level_loader=None, obstacle_generator=None):
    if path_count < 1 or (layout_count is not None and layout_count < 1):
        raise ValueError("Path and layout counts must be positive.")
    explicit_building = building is not None
    name = Path(map_name).stem if map_name else "building"
    if building is not None:
        if cfg is None:
            raise ValueError("An explicit building requires cfg.")
    elif level_name or not map_name:
        if level_loader is None:
            from swarmecho.core.config import load_level
            level_loader = load_level
        level = level_loader(level_name or "M02_random_cuboid_obstacles")
        cfg, building, name = level.env, level.building, level.building_name
    else:
        from swarmecho.env.environment import EnvConfig
        cfg = EnvConfig()
    if merge_walls is not None:
        cfg = replace(cfg, roadmap_merge_walls=merge_walls)
    if corner_bonus_m is not None:
        cfg = replace(cfg, roadmap_corner_bonus_m=corner_bonus_m)
    if not np.isfinite(cfg.roadmap_corner_bonus_m) or cfg.roadmap_corner_bonus_m < 0:
        raise ValueError("Corner bonus must be finite and nonnegative.")
    if map_name and not explicit_building:
        from swarmecho.core.config import MAP_DIR
        source = Path(map_name)
        if not source.is_file():
            source = MAP_DIR / f"{source.stem}.yaml"
        building, name = load_building(source), source.stem
    world = np.asarray(building.world_size_m)
    base = np.asarray(building.base_position_m if start is None else start, dtype=float)
    target = np.asarray(target if target is not None else
                        [world[0] / 2, world[1] / 2, world[2] - building.cell_size_m / 2], dtype=float)
    for label, point in (("Start", base), ("Target", target)):
        if point.shape != (3,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= world):
            raise ValueError(f"{label} must be a finite XYZ point inside {world.tolist()} m.")
        if not building.interior_cells[tuple((point / building.cell_size_m).astype(int))]:
            raise ValueError(f"{label} is outside the interior volume.")
    layouts = []
    obstacle_count = cfg.num_obstacles if obstacles else 0
    count = layout_count if layout_count is not None else (5 if obstacle_count else 1)
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    corner_merge_distance = building.wall_thickness_m + 2 * clearance
    # Use the same bounds and authored candidates as the training roadmap.
    bounds_lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius,
                               building.wall_thickness_m / 2 + cfg.drone_radius,
                               building.tile_thickness_m / 2 + cfg.drone_radius], dtype=np.float32)
    bounds_upper = world - bounds_lower
    use_authored = cfg.roadmap_merge_walls or "roadmap_nodes_m" in json.loads(getattr(building, "extra_geometry_json", "{}"))
    if use_authored:
        authored_vertices = building_roadmap(building, cfg).vertices
    for index in range(count):
        if obstacle_count:
            import jax
            if obstacle_generator is None:
                from swarmecho.env.obstacles import generate_obstacles
                obstacle_generator = generate_obstacles
            lower, upper = obstacle_generator(
                jax.random.PRNGKey(seed + index), world, count=obstacle_count,
                size_min=cfg.obstacle_size_min_m, size_max=cfg.obstacle_size_max_m,
                z_min=cfg.obstacle_spawn_layer_min * building.cell_size_m,
                z_max=min(cfg.obstacle_spawn_layer_max * building.cell_size_m, world[2]),
                boundary_buffer=cfg.obstacle_boundary_buffer_m,
            )
        else:
            lower = upper = np.zeros((0, 3), dtype=np.float32)
        if use_authored:
            vertices = authored_vertices
            if obstacle_count:
                from swarmecho.env.obstacles import roadmap_vertices
                vertices = np.concatenate((authored_vertices, np.clip(
                    np.asarray(roadmap_vertices(lower, upper, clearance)), bounds_lower, bounds_upper)))
        lower = np.concatenate([building.solid_min_m, np.asarray(lower)], axis=0)
        upper = np.concatenate([building.solid_max_m, np.asarray(upper)], axis=0)
        planning_lower, planning_upper = lower - clearance + 1e-4, upper + clearance - 1e-4
        for label, point in (("Start", base), ("Target", target)):
            if np.any(np.all((point >= planning_lower) & (point <= planning_upper), axis=1)):
                raise ValueError(f"{label} intersects a clearance-expanded solid in layout {index}; choose another point or seed.")
        if not use_authored:
            from swarmecho.env.roadmap_cpu import ROADMAP_CORNERS
            vertices = (lower[:, None] - clearance + ROADMAP_CORNERS[None] * (upper - lower + 2 * clearance)[:, None]).reshape(-1, 3)
            vertices = vertices[np.all((vertices >= cfg.drone_radius) & (vertices <= world - cfg.drone_radius), axis=1)]
            if len(vertices):
                interior = building.interior_cells[tuple((vertices / building.cell_size_m).astype(int).T)]
                vertices = vertices[interior]
        vertices = np.unique(vertices, axis=0)
        if len(vertices):
            vertices = vertices[[not np.any(np.all((v >= planning_lower) & (v <= planning_upper), axis=1)) for v in vertices]]
        if len(vertices):
            vertices = vertices[np.any(vertices != base, axis=1) & np.any(vertices != target, axis=1)]
        nodes = np.concatenate([base[None], target[None], vertices])
        adjacency, edges = [[] for _ in nodes], []
        print(f"Layout {index + 1}/{count}: {len(lower)} solids, {len(vertices)} vertices", flush=True)
        for left in range(len(nodes)):
            for offset in range(left + 1, len(nodes), 128):
                ends = nodes[offset:offset + 128]
                for relative in np.flatnonzero(_visible(nodes[left], ends, planning_lower, planning_upper)):
                    right = offset + int(relative)
                    weight = float(np.linalg.norm(nodes[left] - nodes[right]))
                    if left == 0 and right >= 2:
                        weight += cfg.roadmap_corner_bonus_m
                    elif left >= 2 and weight > corner_merge_distance + 1e-4:
                        weight += cfg.roadmap_corner_bonus_m
                    adjacency[left].append((right, weight))
                    adjacency[right].append((left, weight))
                    if left >= 2 and right >= 2:
                        edges.append([left - 2, right - 2])
        routes = ranked_paths(adjacency, path_count)
        paths = [nodes[route].tolist() for route in routes]
        weights = [dict(neighbors) for neighbors in adjacency]
        lengths = [sum(weights[a][b] for a, b in zip(route, route[1:])) for route in routes]
        corner_counts = [0 if len(route) <= 2 else 1 + sum(
            np.linalg.norm(nodes[a] - nodes[b]) > corner_merge_distance + 1e-4
            for a, b in zip(route[1:-2], route[2:-1])) for route in routes]
        layouts.append({"obstacle_min": lower.tolist(), "obstacle_max": upper.tolist(),
                        "vertices": vertices.tolist(), "edges": edges, "paths": paths,
                        "path_lengths_m": lengths, "path_corner_counts": [int(count) for count in corner_counts],
                        "base": base.tolist(), "target": target.tolist()})
        print(f"  {len(paths)}/{path_count} paths; lengths (m): {[round(v, 3) for v in lengths]}", flush=True)
    point_label = "-".join(
        str(int(value)) if float(value).is_integer() else str(float(value))
        for value in target
    )
    output = Path(destination or f"outputs/testresults/{Path(name).stem}_[{point_label}].roadmap.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"format": "swarmecho-roadmap-test/v1",
        "manifest": {"world_size_m": world.tolist(), "map_name": name,
                     **({"building_snapshot": building_snapshot(building)} if hasattr(building, "target_exclusion") else {}),
                     "requested_paths": path_count, "clearance_m": clearance,
                     "comm_radius_m": cfg.comm_radius,
                     "roadmap_merge_walls": cfg.roadmap_merge_walls,
                     "roadmap_corner_bonus_m": cfg.roadmap_corner_bonus_m,
                     "corner_merge_distance_m": corner_merge_distance,
                     "path_metric": "Distance including corner allowance (equivalent m); lower is better",
                     "seed": seed, "level": level_name, "generated_obstacles": bool(obstacle_count)}, "layouts": layouts}, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", help="Level name or YAML path; uses its map and obstacle settings")
    parser.add_argument("--map", help="Map name or YAML path; optionally overrides --level's map")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Target in metres")
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Start in metres (default: base)")
    parser.add_argument("--paths", type=int, default=5, help="Maximum number of ranked paths")
    parser.add_argument("--corner-bonus-m", type=float, help="Override env.roadmap_corner_bonus_m (default: general/level config)")
    parser.add_argument("--obstacles", action="store_true", help="Enable generated obstacles using the level's settings (default: off)")
    parser.add_argument("--merge-walls", action=argparse.BooleanOptionalAction, default=None,
                        help="Override env.roadmap_merge_walls (otherwise uses the level/general config default)")
    parser.add_argument("--layouts", type=int, help="Layout count (default: 5 randomized, 1 static)")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--output", help="Output .roadmap.json path under outputs/testresults for discovery")
    args = parser.parse_args()
    try:
        print(generate(args.output, level_name=args.level, map_name=args.map, target=args.target,
                       start=args.start, path_count=args.paths, layout_count=args.layouts, seed=args.seed,
                       obstacles=args.obstacles, merge_walls=args.merge_walls, corner_bonus_m=args.corner_bonus_m))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
