"""Write an inspector-discoverable randomized obstacle roadmap TESTRESULT."""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import jax
import numpy as np

from swarmecho.core.config import load_level_3d
from swarmecho.env.obstacles3d import generate_obstacles, roadmap_vertices, segments_blocked


def _shortest(adjacency, start, goal, banned):
    queue = [(0.0, start, ())]
    best = {}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in best:
            continue
        best[node] = cost
        path = (*path, node)
        if node == goal:
            return list(path)
        for neighbour, weight in adjacency[node]:
            if tuple(sorted((node, neighbour))) not in banned:
                heapq.heappush(queue, (cost + weight, neighbour, path))
    return []


def generate(destination: str | Path = "outputs/testresults/obstacles.roadmap.json") -> Path:
    level = load_level_3d("M02_random_cuboid_obstacles_3D")
    cfg, building = level.env, level.building
    world = np.asarray(building.world_size_m)
    layouts = []
    for layout_index in range(5):
        lower, upper = generate_obstacles(
            jax.random.PRNGKey(9000 + layout_index), world,
            count=cfg.num_obstacles, size_min=cfg.obstacle_size_min_m,
            size_max=cfg.obstacle_size_max_m,
            z_min=cfg.obstacle_spawn_layer_min * building.cell_size_m,
            z_max=min(cfg.obstacle_spawn_layer_max * building.cell_size_m, world[2]),
            boundary_buffer=cfg.obstacle_boundary_buffer_m,
        )
        lower, upper = np.asarray(lower), np.asarray(upper)
        vertices = np.asarray(roadmap_vertices(
            lower, upper, cfg.drone_radius + cfg.obstacle_planning_clearance_m
        ))
        base = np.asarray(building.base_position_m)
        target = np.asarray([world[0] / 2, world[1] / 2, world[2] - building.cell_size_m / 2])
        nodes = np.concatenate([base[None], target[None], vertices])
        clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
        planning_lower = lower - clearance + 1e-4
        planning_upper = upper + clearance - 1e-4
        adjacency = [[] for _ in nodes]
        edges = []
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                if not bool(segments_blocked(
                    nodes[left], nodes[right], planning_lower, planning_upper
                )):
                    weight = float(np.linalg.norm(nodes[left] - nodes[right]))
                    adjacency[left].append((right, weight))
                    adjacency[right].append((left, weight))
                    if left >= 2 and right >= 2:
                        edges.append([left - 2, right - 2])
        paths, banned = [], set()
        for _ in range(5):
            route = _shortest(adjacency, 0, 1, banned)
            if not route:
                break
            paths.append(nodes[route].tolist())
            if len(route) > 2:
                banned.add(tuple(sorted((route[1], route[2]))))
            else:
                break
        layouts.append({
            "obstacle_min": lower.tolist(), "obstacle_max": upper.tolist(),
            "vertices": vertices.tolist(), "edges": edges, "paths": paths,
            "base": base.tolist(), "target": target.tolist(),
        })
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "format": "swarmecho-roadmap-test/v1",
        "manifest": {"world_size_m": world.tolist(), "map_name": level.building_name},
        "layouts": layouts,
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(generate())
