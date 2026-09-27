"""Rasterize static map volume and export each point's farthest reachable point."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from swarmecho.analysis.roadmap_paths import generate, ranked_paths
from swarmecho.env.buildings import building_snapshot, load_building, target_spawn_boxes
from swarmecho.env.roadmap_cpu import building_roadmap


def visible(start, ends, lower, upper):
    """CPU equivalent of the training segment/AABB predicate, in small batches."""
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


def inside(points, lower, upper):
    return np.asarray([np.any(np.all((p >= lower) & (p <= upper), axis=-1)) for p in points])


def write_top_farthest(output, stem, map_name, building, cfg, graph, points, costs,
                       planning_lo, planning_hi, maximum, farthest, count, level_name):
    """Write distinct farthest-point pairs with one shortest route per pair."""
    ranked = []
    seen = set()
    for i in sorted(np.flatnonzero(np.isfinite(maximum)),
                    key=lambda index: (-float(maximum[index]), int(index))):
        j = int(farthest[i])
        key = tuple(sorted((int(i), j)))
        if j < 0 or key in seen:
            continue
        seen.add(key)  # The reverse direction is the same comparison pair.
        ranked.append((int(i), j, float(maximum[i])))
        if len(ranked) == count:
            break
    if not ranked:
        return None

    vertices = graph.vertices
    merge_distance = building.wall_thickness_m + 2 * graph.clearance
    edges = np.argwhere(np.triu(graph.visible_edges, 1))
    static = [[] for _ in range(len(vertices) + 2)]
    for left, right in edges:
        distance = float(np.linalg.norm(vertices[left] - vertices[right]))
        weight = distance + (cfg.roadmap_corner_bonus_m if distance > merge_distance + 1e-4 else 0.)
        static[int(left) + 2].append((int(right) + 2, weight))
        static[int(right) + 2].append((int(left) + 2, weight))

    pairs = []
    for rank, (target_index, base_index, raster_distance) in enumerate(ranked, 1):
        base, target = points[base_index], points[target_index]
        adjacency = [neighbors.copy() for neighbors in static]
        for vertex_index in np.flatnonzero(np.isfinite(costs[base_index])):
            weight = float(costs[base_index, vertex_index]) + cfg.roadmap_corner_bonus_m
            adjacency[0].append((int(vertex_index) + 2, weight))
            adjacency[int(vertex_index) + 2].append((0, weight))
        for vertex_index in np.flatnonzero(np.isfinite(costs[target_index])):
            weight = float(costs[target_index, vertex_index])
            adjacency[1].append((int(vertex_index) + 2, weight))
            adjacency[int(vertex_index) + 2].append((1, weight))
        if visible(base, target[None], planning_lo, planning_hi)[0]:
            weight = float(np.linalg.norm(target - base))
            adjacency[0].append((1, weight))
            adjacency[1].append((0, weight))
        route = ranked_paths(adjacency, 1)
        if not route:
            raise ValueError(f"No inspectable route for ranked raster pair {rank}.")
        path = np.concatenate((base[None], target[None], vertices))[route[0]]
        physical = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        corners = (0 if len(route[0]) <= 2 else 1 + sum(
            np.linalg.norm(path[k + 1] - path[k]) > merge_distance + 1e-4
            for k in range(1, len(path) - 2)))
        length = float(physical + corners * cfg.roadmap_corner_bonus_m)
        pairs.append({"rank": rank, "base": base.tolist(), "target": target.tolist(),
                      "path": path.tolist(), "path_length_m": length,
                      "path_corner_count": int(corners), "raster_distance_m": raster_distance})
        print(f"  #{rank}: {base.tolist()} -> {target.tolist()}: {length:.3f} m", flush=True)

    path = output / f"{stem}_top{len(pairs)}_longest_farthest.roadmap.json"
    path.write_text(json.dumps({"format": "swarmecho-roadmap-test/v1",
        "manifest": {"world_size_m": building.world_size_m.tolist(), "map_name": map_name,
                     "building_snapshot": building_snapshot(building),
                     "pair_ranking": "longest_farthest", "requested_pairs": count,
                     "clearance_m": graph.clearance, "comm_radius_m": cfg.comm_radius,
                     "roadmap_approach": cfg.roadmap_approach,
                     "roadmap_node_density": cfg.roadmap_node_density,
                     "roadmap_merge_wall_end_nodes": cfg.roadmap_merge_wall_end_nodes,
                     "roadmap_corner_bonus_m": cfg.roadmap_corner_bonus_m,
                     "corner_merge_distance_m": merge_distance,
                     "path_metric": "Shortest-route distance including corner allowance (equivalent m)",
                     "level": level_name, "generated_obstacles": False},
        "layouts": [{"obstacle_min": graph.solid_min.tolist(), "obstacle_max": graph.solid_max.tolist(),
                     "vertices": vertices.tolist(), "edges": edges.tolist(), "pairs": pairs,
                     "base": pairs[0]["base"], "target": pairs[0]["target"]}]}, indent=2), encoding="utf-8")
    return path


def scan(args, *, building=None, cfg=None):
    level = None
    if cfg is None:
        from swarmecho.core.config import load_level
        from swarmecho.env.environment import EnvConfig
        level = load_level(args.level) if args.level else None
        cfg = level.env if level else EnvConfig()
    if args.corner_bonus_m is not None:
        cfg = replace(cfg, roadmap_corner_bonus_m=args.corner_bonus_m)
    if getattr(args, "roadmap_approach", None) is not None:
        cfg = replace(cfg, roadmap_approach=args.roadmap_approach)
    if getattr(args, "roadmap_node_density", None) is not None:
        cfg = replace(cfg, roadmap_node_density=args.roadmap_node_density)
    if getattr(args, "merge_wall_end_nodes", None) is not None:
        cfg = replace(cfg, roadmap_merge_wall_end_nodes=args.merge_wall_end_nodes)
    if not np.isfinite(cfg.roadmap_corner_bonus_m) or cfg.roadmap_corner_bonus_m < 0:
        raise ValueError("Corner bonus must be finite and nonnegative.")
    if building is not None:
        name = Path(args.map).stem
    elif args.map:
        from swarmecho.core.config import MAP_DIR
        source = Path(args.map)
        source = source if source.is_file() else MAP_DIR / f"{source.stem}.yaml"
        building, name = load_building(source), source.stem
    elif level:
        building, name = level.building, level.building_name
    else:
        raise ValueError("Provide --level or --map.")
    spacing = args.spacing if args.spacing is not None else building.cell_size_m
    top_pairs = getattr(args, "top_pairs", 1)
    if not np.isfinite(spacing) or spacing <= 0 or args.paths < 1 or top_pairs < 1:
        raise ValueError("Spacing must be finite and positive; paths and top pairs must be positive.")
    if not np.isfinite(cfg.comm_radius) or cfg.comm_radius <= 0:
        raise ValueError("Drone communication range must be finite and positive.")
    world = np.asarray(building.world_size_m, dtype=np.float32)
    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius,
                        building.wall_thickness_m / 2 + cfg.drone_radius,
                        building.tile_thickness_m / 2 + cfg.drone_radius], dtype=np.float32)
    upper = world - lower
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    graph = building_roadmap(building, cfg, plan=True, corner_bonus_m=cfg.roadmap_corner_bonus_m)
    solid_lo, solid_hi = graph.solid_min, graph.solid_max
    planning_lo, planning_hi = solid_lo - clearance + 1e-4, solid_hi + clearance - 1e-4
    axes = []
    for extent in world:
        edges = np.arange(0., float(extent), spacing)
        axes.append((edges + np.minimum(edges + spacing, extent)) / 2)
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    cells = (points / building.cell_size_m).astype(int)
    valid = building.interior_cells[tuple(cells.T)] & np.all((points >= lower) & (points <= upper), axis=1)
    points = points[valid]
    points = points[~inside(points, planning_lo, planning_hi)]
    if len(points) < 2:
        raise ValueError("Fewer than two valid raster points; try finer spacing.")
    # Mark the actual sampler's target boxes and base/drone-offset clearance.
    target_lo, target_hi, _ = target_spawn_boxes(building, cfg)
    target_allowed = inside(points, np.asarray(target_lo), np.asarray(target_hi))
    target_allowed &= ~inside(points, solid_lo - cfg.obstacle_target_buffer_m,
                             solid_hi + cfg.obstacle_target_buffer_m)
    drone = points + np.asarray([0., 0., cfg.drone_radius], dtype=np.float32)
    base_allowed = np.all((drone >= lower) & (drone <= upper), axis=1)
    base_allowed &= ~inside(drone, solid_lo - cfg.drone_radius, solid_hi + cfg.drone_radius)
    vertices = graph.vertices
    print(f"Scanning {len(points):,} points at {spacing:g} m spacing; {len(vertices)} roadmap nodes. "
          f"Drone comm range: {cfg.comm_radius:g} m. Static authored geometry only.", flush=True)
    print(f"All distances include {cfg.roadmap_corner_bonus_m:g} m per waypoint group before ranking; "
          f"short links up to {building.wall_thickness_m + 2 * clearance:g} m share a group.", flush=True)
    costs = np.full((len(points), len(vertices)), np.inf, dtype=np.float32)
    for i, point in enumerate(points):
        clear = visible(point, vertices, planning_lo, planning_hi)
        costs[i, clear] = np.linalg.norm(vertices[clear] - point, axis=-1)
    graph_distances = np.where(graph.corner_distances < 1e6, graph.corner_distances, np.inf)
    attached = np.full_like(costs, np.inf)
    for node in range(len(vertices)):
        np.minimum(attached, costs[:, node, None] + cfg.roadmap_corner_bonus_m + graph_distances[node], out=attached)
    farthest = np.full(len(points), -1, dtype=int)
    farthest_base = farthest.copy()
    maximum = np.full(len(points), np.nan)
    maximum_base = maximum.copy()
    reachable_count = np.zeros(len(points), dtype=int)
    for i, point in enumerate(points):
        distances = np.full(len(points), np.inf, dtype=np.float32)
        for offset in range(0, len(points), 128):
            end = min(offset + 128, len(points))
            if len(vertices):
                distances[offset:end] = np.min(costs[offset:end] + attached[i], axis=1)
            clear = visible(point, points[offset:end], planning_lo, planning_hi)
            direct = np.where(clear, np.linalg.norm(points[offset:end] - point, axis=-1), np.inf)
            distances[offset:end] = np.minimum(distances[offset:end], direct)
        reachable = np.isfinite(distances) & (distances < 1e6)
        reachable[i] = False  # A point is not its own "other point".
        reachable_count[i] = np.count_nonzero(reachable)
        for mask, indices, values in ((reachable, farthest, maximum),
                                      (reachable & base_allowed, farthest_base, maximum_base)):
            if np.any(mask):
                index = int(np.argmax(np.where(mask, distances, -np.inf)))
                indices[i], values[i] = index, float(distances[index])
        if (i + 1) % max(1, len(points) // 10) == 0 or i + 1 == len(points):
            print(f"  {i + 1:,}/{len(points):,} points", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(name).stem}_raster_{spacing:g}m_corner_{cfg.roadmap_corner_bonus_m:g}m"
    csv_path = output / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_m", "y_m", "z_m", "target_allowed", "base_allowed", "reachable_other_points",
                         "farthest_x_m", "farthest_y_m", "farthest_z_m", "farthest_distance_m", "farthest_comm_ranges",
                         "farthest_base_x_m", "farthest_base_y_m", "farthest_base_z_m", "farthest_base_distance_m", "farthest_base_comm_ranges"])
        for i, point in enumerate(points):
            row = [*point, bool(target_allowed[i]), bool(base_allowed[i]), reachable_count[i]]
            for indices, values in ((farthest, maximum), (farthest_base, maximum_base)):
                row.extend([*points[indices[i]], values[i], values[i] / cfg.comm_radius] if indices[i] >= 0 else [""] * 5)
            writer.writerow(row)
    def report(label, eligible, indices, values, longest=False, save=True):
        if not np.any(eligible):
            print(f"{label}: no eligible points with a reachable partner.")
            return
        i = int(np.argmax(np.where(eligible, values, -np.inf)) if longest else
                np.argmin(np.where(eligible, values, np.inf)))
        j = int(indices[i])
        print(f"{label}: {points[i].tolist()} -> {points[j].tolist()}: "
              f"{values[i]:.3f} m = {values[i] / cfg.comm_radius:.4f} drone comm ranges", flush=True)
        if save:
            path = generate(output / f"{stem}_{label}.roadmap.json", level_name=args.level,
                            map_name=args.map, start=points[j], target=points[i], path_count=1,
                            corner_bonus_m=cfg.roadmap_corner_bonus_m, building=building, cfg=cfg)
            print(f"Inspectable roadmap: {path}")
    finite = np.isfinite(maximum)
    report("shortest_farthest", finite, farthest, maximum)
    report("longest_farthest", finite, farthest, maximum, longest=True, save=top_pairs == 1)
    if top_pairs > 1:
        path = write_top_farthest(output, stem, name, building, cfg, graph, points, costs,
                                  planning_lo, planning_hi, maximum, farthest, top_pairs, args.level)
        if path is not None:
            print(f"Inspectable ranked pairs: {path}", flush=True)
    report("minimum_target_to_base", target_allowed & np.isfinite(maximum_base), farthest_base, maximum_base, save=False)
    if np.any(finite):
        for label, value in zip(("Minimum", "Median", "Maximum"), np.percentile(maximum[finite], [0, 50, 100])):
            print(f"{label} farthest distance: {value:.3f} m = {value / cfg.comm_radius:.4f} comm ranges")
    print(f"Points with no reachable other raster point: {np.count_nonzero(~finite)}")
    print(f"Points unable to reach every other raster point: {np.count_nonzero(reachable_count < len(points) - 1)}")
    print(f"Allowed targets with no reachable base sample: {np.count_nonzero(target_allowed & ~np.isfinite(maximum_base))}")
    if level:
        threshold = level.training.minimum_geodesic_separation_multiplier * cfg.comm_radius
        count = np.count_nonzero(target_allowed & (~np.isfinite(maximum_base) | (maximum_base < threshold)))
        print(f"Allowed targets without a qualifying base sample at {threshold:g} m "
              f"({threshold / cfg.comm_radius:g} ranges): {count}/{np.count_nonzero(target_allowed)}")
    print(f"Point list: {csv_path}\nRaster approximation, not a continuous-space feasibility proof. Unreachable pairs are excluded from maxima.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", help="Level name/path; supplies drone communication range and roadmap settings")
    parser.add_argument("--map", help="Map name/path; optionally overrides the level map")
    parser.add_argument("--spacing", type=float, help="Raster spacing in metres (default: map cell size)")
    parser.add_argument("--paths", type=int, default=1, help="Legacy option; each of the two extreme pairs exports its best route only")
    parser.add_argument("--top-pairs", type=int, default=1,
                        help="Export this many distinct longest-farthest raster pairs in one inspectable artifact")
    parser.add_argument("--corner-bonus-m", type=float, help="Override env.roadmap_corner_bonus_m (default: general/level config)")
    parser.add_argument("--roadmap-approach", choices=("full", "minimal"), help="Override env.roadmap_approach")
    parser.add_argument("--roadmap-node-density", type=float, help="Override env.roadmap_node_density (full only)")
    parser.add_argument("--merge-wall-end-nodes", action=argparse.BooleanOptionalAction, default=None,
                        help="Override env.roadmap_merge_wall_end_nodes (full only)")
    parser.add_argument("--output-dir", default="outputs/testresults")
    args = parser.parse_args()
    try:
        scan(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
