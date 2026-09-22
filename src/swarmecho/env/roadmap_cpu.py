"""Host-only visibility roadmap compilation shared by runtime and inspection."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from time import perf_counter
import json
import heapq
import numpy as np
from swarmecho.core.terminal import init_message

ROADMAP_CORNERS = np.asarray(list(product((0., 1.), repeat=3)), dtype=np.float32)

@dataclass(frozen=True, eq=False)
class RoadmapArrays:
    solid_min: np.ndarray
    solid_max: np.ndarray
    vertices: np.ndarray
    distances: np.ndarray
    clearance: float
    visible_edges: np.ndarray | None = None
    corner_distances: np.ndarray | None = None


def shortest_path(matrix, directed=False, method="D"):
    """Use SciPy when installed; dependency-light inspection has a Dijkstra fallback."""
    try:
        from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
    except ImportError:
        adjacency = [[(int(j), float(matrix[i, j])) for j in np.flatnonzero(np.isfinite(matrix[i])) if j != i]
                     for i in range(len(matrix))]
        result = np.full_like(matrix, np.inf)
        for root in range(len(matrix)):
            result[root, root] = 0.
            queue = [(0., root)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance > result[root, node] + 1e-4:
                    continue
                for target, weight in adjacency[node]:
                    candidate = distance + weight
                    if candidate < result[root, target]:
                        result[root, target] = candidate
                        heapq.heappush(queue, (float(result[root, target]), target))
        return result
    return scipy_shortest_path(matrix, directed=directed, method=method)

def authored_roadmap_vertices(obstacle_min, obstacle_max, lower, upper, clearance, edge_spacing):
    """Merge exact rectangular unions, then sample their clearance-offset edges.

    Only roadmap candidates change; collision geometry remains untouched.
    This host-side operation runs once per authored layout, outside JAX resets.
    """
    lo = np.asarray(obstacle_min, dtype=np.float32).reshape(-1, 3)
    hi = np.asarray(obstacle_max, dtype=np.float32).reshape(-1, 3)
    boxes = [(a.copy(), b.copy()) for a, b in zip(lo, hi)]
    # Equal cross sections and overlapping/touching intervals guarantee that
    # merging fills no doorway or other empty space. Repeat across all axes.
    changed = True
    while changed:
        before = len(boxes)
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            groups = {}
            for a, b in boxes:
                key = tuple(a[others]) + tuple(b[others])
                groups.setdefault(key, []).append((a, b))
            boxes = []
            for group in groups.values():
                group.sort(key=lambda box: float(box[0][axis]))
                a, b = group[0]
                for next_a, next_b in group[1:]:
                    if next_a[axis] <= b[axis]:
                        b[axis] = max(b[axis], next_b[axis])
                    else:
                        boxes.append((a, b))
                        a, b = next_a, next_b
                boxes.append((a, b))
        changed = len(boxes) < before
    samples = []
    for a, b in boxes:
        a, b = np.clip(a - clearance, lower, upper), np.clip(b + clearance, lower, upper)
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            count = max(1, int(np.ceil((b[axis] - a[axis]) / edge_spacing)))
            for sides in product((0, 1), repeat=2):
                edge = np.broadcast_to(a, (count + 1, 3)).copy()
                edge[:, axis] = np.linspace(a[axis], b[axis], count + 1)
                for other, side in zip(others, sides):
                    edge[:, other] = b[other] if side else a[other]
                samples.append(edge)
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    vertices = np.unique(np.concatenate(samples), axis=0)
    expanded_lo, expanded_hi = lo - clearance + 1e-4, hi + clearance - 1e-4
    exposed = [not np.any(np.all((v >= expanded_lo) & (v <= expanded_hi), axis=-1)) for v in vertices]
    return vertices[np.asarray(exposed, dtype=bool)]


def shared_roadmap(obstacle_min, obstacle_max, lower, upper, clearance, *, plan,
                   merge_walls=False, edge_spacing=5.0, corner_bonus_m=0.0, corner_merge_distance_m=0.0,
                   candidate_vertices=None):
    arrays = [np.asarray(a, dtype=np.float32) for a in (obstacle_min, obstacle_max, lower, upper)]
    return _shared_roadmap(*(a.tobytes() for a in arrays), float(clearance), bool(plan),
                           bool(merge_walls), float(edge_spacing), float(corner_bonus_m), float(corner_merge_distance_m),
                           None if candidate_vertices is None else np.asarray(candidate_vertices, dtype=np.float32).tobytes())


def building_roadmap(building, cfg, *, plan=True, corner_bonus_m=0.0):
    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                       [building.tile_thickness_m / 2 + cfg.drone_radius], dtype=np.float32)
    return shared_roadmap(building.solid_min_m, building.solid_max_m, lower, building.world_size_m - lower,
        cfg.drone_radius + cfg.obstacle_planning_clearance_m, plan=plan,
        merge_walls=cfg.roadmap_merge_walls, edge_spacing=building.cell_size_m,
        corner_bonus_m=corner_bonus_m,
        corner_merge_distance_m=building.wall_thickness_m + 2 * (cfg.drone_radius + cfg.obstacle_planning_clearance_m),
        candidate_vertices=json.loads(getattr(building, "extra_geometry_json", "{}")).get("roadmap_nodes_m"))


@lru_cache(maxsize=8)
def _shared_roadmap(min_bytes, max_bytes, lower_bytes, upper_bytes, clearance, plan, merge_walls, edge_spacing,
                    corner_bonus_m, corner_merge_distance_m, candidate_bytes=None):
    """Build static all-pairs distances on the host, once per geometry/configuration."""
    lo = np.frombuffer(min_bytes, dtype=np.float32).reshape(-1, 3)
    hi = np.frombuffer(max_bytes, dtype=np.float32).reshape(-1, 3)
    lower = np.frombuffer(lower_bytes, dtype=np.float32)
    upper = np.frombuffer(upper_bytes, dtype=np.float32)
    vertices = np.empty((0, 3), dtype=np.float32)
    distances = np.empty((0, 0), dtype=np.float32)
    visible_edges = np.empty((0, 0), dtype=bool)
    corner_distances = distances
    if plan and len(lo):
        started = perf_counter()
        init_message(f"Building shared geodesic roadmap: {len(lo)} authored solids")
        expanded_lo, expanded_hi = lo - clearance + 1e-4, hi + clearance - 1e-4
        if candidate_bytes is not None:
            vertices = np.unique(np.frombuffer(candidate_bytes, dtype=np.float32).reshape(-1, 3), axis=0)
            valid = [np.all((v >= lower) & (v <= upper)) and
                     not np.any(np.all((v >= expanded_lo) & (v <= expanded_hi), axis=-1)) for v in vertices]
            vertices = vertices[np.asarray(valid, dtype=bool)]
        elif merge_walls:
            vertices = authored_roadmap_vertices(lo, hi, lower, upper, clearance, edge_spacing)
        else:
            vertices = (lo[:, None] - clearance + ROADMAP_CORNERS[None] * (hi - lo + 2 * clearance)[:, None]).reshape(-1, 3)
            # Retain routes alongside floor-to-ceiling walls.
            vertices = np.unique(np.clip(vertices, lower, upper), axis=0)
            valid = [not np.any(np.all((v >= expanded_lo) & (v <= expanded_hi), axis=-1)) for v in vertices]
            vertices = vertices[np.asarray(valid, dtype=bool)]
        distances = np.full((len(vertices), len(vertices)), np.inf, dtype=np.float32)
        for index, start in enumerate(vertices):
            # Bound temporary segment/solid arrays instead of allocating V*V*S.
            for offset in range(index + 1, len(vertices), 128):
                ends = vertices[offset:offset + 128]
                direction = ends[:, None] - start
                parallel = np.abs(direction) <= 1e-5
                safe = np.where(parallel, 1.0, direction)
                first, last = (expanded_lo - start) / safe, (expanded_hi - start) / safe
                near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
                far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
                outside = np.any(parallel & ((start < expanded_lo) | (start > expanded_hi)), axis=-1)
                blocked = np.any((far >= np.maximum(near, 1e-5)) & (near <= 1 - 1e-5) & ~outside, axis=-1)
                weights = np.where(blocked, np.inf, np.linalg.norm(ends - start, axis=-1))
                distances[index, offset:offset + len(ends)] = weights
                distances[offset:offset + len(ends), index] = weights
        np.fill_diagonal(distances, 0)
        visible_edges = np.isfinite(distances)
        np.fill_diagonal(visible_edges, False)
        # Entering the first waypoint group is charged at query time. Short
        # internal links stay in that group; longer links enter another one.
        corner_distances = distances + np.where(
            visible_edges & (distances > corner_merge_distance_m + 1e-4), corner_bonus_m, 0.)
        if len(vertices):
            distances = shortest_path(distances, directed=False, method="D").astype(np.float32)
            distances = np.minimum(distances, 1e6)
            corner_distances = (np.minimum(shortest_path(corner_distances, directed=False, method="D"), 1e6)
                                .astype(np.float32)) if corner_bonus_m else distances
        init_message(f"Shared geodesic roadmap ready: {len(vertices)} vertices in {perf_counter() - started:.1f}s")
    for array in (vertices, distances, visible_edges, corner_distances):
        array.setflags(write=False)
    return RoadmapArrays(lo, hi, vertices, distances, clearance, visible_edges, corner_distances)


