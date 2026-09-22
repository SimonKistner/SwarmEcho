"""Edit EVAL_FIXED_OBSTACLE_BOUNDS, run this file, then open the inspector."""

# M02 YAML (copy these exact rows under evaluation):
#   eval_fixed_obstacle_bounds:
#   - [2.0, 2.0, 11.0, 6.0, 7.0, 15.0]
#   - [12.0, 3.0, 15.0, 16.0, 8.0, 19.0]
#   - [6.0, 12.0, 19.0, 10.0, 17.0, 23.0]
EVAL_FIXED_OBSTACLE_BOUNDS = (
    (1.0, 5.0, 7.0, 14.0, 15.0, 10.0),
    (8.0, 3.0, 13.0, 16.0, 18.0, 15.0),
    (7.5, 7.5, 20.0, 12.5, 12.5, 26.0),
)

import heapq  # noqa: E402 - editable configuration intentionally comes first
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from swarmecho.core.config import load_level  # noqa: E402
from swarmecho.env.obstacles import roadmap_vertices, segments_blocked  # noqa: E402


def _shortest(adjacency, start, goal, banned):
    queue, visited = [(0.0, start, ())], set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = (*path, node)
        if node == goal:
            return list(path)
        for neighbour, weight in adjacency[node]:
            if tuple(sorted((node, neighbour))) not in banned:
                heapq.heappush(queue, (cost + weight, neighbour, path))
    return []


def generate(destination="outputs/testresults/static_eval_layout.roadmap.json") -> Path:
    level = load_level("M02_random_cuboid_obstacles")
    cfg, building = level.env, level.building
    bounds = np.asarray(EVAL_FIXED_OBSTACLE_BOUNDS, dtype=np.float32)
    if bounds.shape != (cfg.num_obstacles, 6):
        raise ValueError(f"Define exactly {cfg.num_obstacles} cuboids as min/max XYZ.")
    lower, upper = bounds[:, :3], bounds[:, 3:]
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    vertices = np.asarray(roadmap_vertices(lower, upper, clearance))
    world = np.asarray(building.world_size_m)
    base = np.asarray(building.base_position_m)
    target = np.asarray([world[0] / 2, world[1] / 2, world[2] - building.cell_size_m / 2])
    nodes = np.concatenate([base[None], target[None], vertices])
    adjacency, edges = [[] for _ in nodes], []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if not bool(segments_blocked(nodes[left], nodes[right], lower - clearance + 1e-4, upper + clearance - 1e-4)):
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
        if len(route) <= 2:
            break
        banned.add(tuple(sorted((route[1], route[2]))))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "format": "swarmecho-roadmap-test/v1",
        "manifest": {"world_size_m": world.tolist(), "map_name": "M02 static eval layout"},
        "layouts": [{
            "obstacle_min": lower.tolist(), "obstacle_max": upper.tolist(),
            "vertices": vertices.tolist(), "edges": edges, "paths": paths,
            "base": base.tolist(), "target": target.tolist(),
        }],
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(generate())
