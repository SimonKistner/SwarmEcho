"""Fixed-shape 3D cuboid geometry and compact visibility roadmaps."""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from functools import lru_cache
from swarmecho.core.terminal import init_message
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


ROADMAP_CORNERS = np.asarray(list(product((0.0, 1.0), repeat=3)), dtype=np.float32)
INF = jnp.float32(1.0e6)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class SharedRoadmap:
    """Immutable geometry for one map; no leaves to batch or carry through scans.

    Layout-specific generated geometry and roadmaps remain ordinary state arrays.
    The cache key includes geometry and bounds, never just a map name.
    """

    solid_min: np.ndarray
    solid_max: np.ndarray
    vertices: np.ndarray
    distances: np.ndarray
    clearance: float
    visible_edges: np.ndarray | None = None
    corner_distances: np.ndarray | None = None

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return auxiliary


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
                   merge_walls=False, edge_spacing=5.0, corner_bonus_m=0.0, corner_merge_distance_m=0.0):
    arrays = [np.asarray(a, dtype=np.float32) for a in (obstacle_min, obstacle_max, lower, upper)]
    return _shared_roadmap(*(a.tobytes() for a in arrays), float(clearance), bool(plan),
                           bool(merge_walls), float(edge_spacing), float(corner_bonus_m), float(corner_merge_distance_m))


@lru_cache(maxsize=8)
def _shared_roadmap(min_bytes, max_bytes, lower_bytes, upper_bytes, clearance, plan, merge_walls, edge_spacing,
                    corner_bonus_m, corner_merge_distance_m):
    """Build static all-pairs distances on the host, once per geometry/configuration."""
    from scipy.sparse.csgraph import shortest_path

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
        if merge_walls:
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
    return SharedRoadmap(lo, hi, vertices, distances, clearance, visible_edges, corner_distances)


def points_inside_aabbs(points, obstacle_min, obstacle_max, buffer=0.0):
    """Return whether each point is inside any buffered axis-aligned box."""
    points = jnp.asarray(points)
    inside = (points[..., None, :] >= obstacle_min - buffer) & (
        points[..., None, :] <= obstacle_max + buffer
    )
    return jnp.any(jnp.all(inside, axis=-1), axis=-1)


def segments_blocked(start, end, obstacle_min, obstacle_max, epsilon=1e-5):
    """Exact slab test, with bounded wall reductions inside batched kernels.

    A full authored map fused into one reduction causes pathological GPU
    register spilling/compilation. Keep at most eight solids in each loop
    body, including when this function is nested inside vmap and scan.
    """
    start, end = jnp.asarray(start), jnp.asarray(end)
    obstacle_min, obstacle_max = jnp.asarray(obstacle_min), jnp.asarray(obstacle_max)
    count = obstacle_min.shape[0]
    if count <= 8:
        return _segments_blocked_block(start, end, obstacle_min, obstacle_max, epsilon)

    def check_chunk(index, blocked):
        # The last slice overlaps the preceding slice instead of introducing
        # fictitious padded solids. OR is unchanged by checking a wall twice.
        offset = jnp.minimum(index * 8, count - 8)
        lower = jax.lax.dynamic_slice_in_dim(obstacle_min, offset, 8, axis=0)
        upper = jax.lax.dynamic_slice_in_dim(obstacle_max, offset, 8, axis=0)
        return blocked | _segments_blocked_block(start, end, lower, upper, epsilon)

    shape = jnp.broadcast_shapes(start.shape[:-1], end.shape[:-1])
    return jax.lax.fori_loop(
        0, (count + 7) // 8, check_chunk, jnp.zeros(shape, dtype=jnp.bool_)
    )


def _segments_blocked_block(start, end, obstacle_min, obstacle_max, epsilon):
    """Slab arithmetic for one small group of solids."""
    start, end = jnp.asarray(start), jnp.asarray(end)
    direction = end - start
    safe = jnp.where(jnp.abs(direction) > epsilon, direction, 1.0)
    first = (obstacle_min - start[..., None, :]) / safe[..., None, :]
    last = (obstacle_max - start[..., None, :]) / safe[..., None, :]
    parallel = jnp.abs(direction)[..., None, :] <= epsilon
    t_near = jnp.max(jnp.where(parallel, -jnp.inf, jnp.minimum(first, last)), axis=-1)
    t_far = jnp.min(jnp.where(parallel, jnp.inf, jnp.maximum(first, last)), axis=-1)
    outside_parallel = parallel & (
        (start[..., None, :] < obstacle_min)
        | (start[..., None, :] > obstacle_max)
    )
    hit = (t_far >= jnp.maximum(t_near, epsilon)) & (t_near <= 1.0 - epsilon)
    return jnp.any(hit & ~jnp.any(outside_parallel, axis=-1), axis=-1)


def generate_obstacles(
    key,
    world_size,
    *,
    count: int,
    size_min: float,
    size_max: float,
    z_min: float,
    z_max: float,
    boundary_buffer: float,
):
    """Generate a deterministic fixed-shape AABB layout from one environment key."""
    if count == 0:
        empty = jnp.zeros((0, 3), dtype=jnp.float32)
        return empty, empty
    mins = jnp.zeros((count, 3), dtype=jnp.float32)
    maxs = jnp.zeros((count, 3), dtype=jnp.float32)
    for index in range(count):
        def draw(draw_key):
            size_key, centre_key = jax.random.split(draw_key)
            size = jax.random.uniform(size_key, (3,), minval=size_min, maxval=size_max)
            low = jnp.asarray([boundary_buffer, boundary_buffer, z_min]) + size / 2
            high = jnp.asarray([world_size[0] - boundary_buffer, world_size[1] - boundary_buffer, z_max]) - size / 2
            centre = jax.random.uniform(centre_key, (3,), minval=low, maxval=jnp.maximum(low, high))
            return centre - size / 2, centre + size / 2

        key, first_key = jax.random.split(key)
        first_min, first_max = draw(first_key)

        def overlaps(candidate_min, candidate_max):
            separated = (candidate_max <= mins - boundary_buffer) | (
                candidate_min >= maxs + boundary_buffer
            )
            return jnp.any(jnp.all(~separated, axis=-1) & (jnp.arange(count) < index))

        def condition(carry):
            attempt, _, candidate_min, candidate_max = carry
            return (attempt < 64) & overlaps(candidate_min, candidate_max)

        def retry(carry):
            attempt, retry_key, _, _ = carry
            retry_key, draw_key = jax.random.split(retry_key)
            candidate_min, candidate_max = draw(draw_key)
            return attempt + 1, retry_key, candidate_min, candidate_max

        _, key, candidate_min, candidate_max = jax.lax.while_loop(
            condition, retry, (0, key, first_min, first_max)
        )
        fallback_size = jnp.full(3, size_min)
        fallback_centre = jnp.asarray([
            world_size[0] * (index + 1) / (count + 1),
            world_size[1] * (0.35 + 0.3 * (index % 2)),
            (z_min + z_max) / 2,
        ])
        still_overlaps = overlaps(candidate_min, candidate_max)
        candidate_min = jnp.where(still_overlaps, fallback_centre - fallback_size / 2, candidate_min)
        candidate_max = jnp.where(still_overlaps, fallback_centre + fallback_size / 2, candidate_max)
        mins = mins.at[index].set(candidate_min)
        maxs = maxs.at[index].set(candidate_max)
    return mins, maxs


def roadmap_vertices(obstacle_min, obstacle_max, clearance: float):
    """Return the eight clearance-expanded corners of every cuboid."""
    corners = jnp.asarray(ROADMAP_CORNERS)
    lower = obstacle_min - clearance
    upper = obstacle_max + clearance
    return (lower[:, None, :] + corners[None, :, :] * (upper - lower)[:, None, :]).reshape(-1, 3)


def roadmap_distances(vertices, obstacle_min, obstacle_max, clearance: float,
                      corner_bonus_m=0.0, corner_merge_distance_m=0.0):
    """Build a visibility graph and its fixed-size Floyd-Warshall distance matrix."""
    # Slightly shrink the forbidden volume so edges lying on the clearance
    # boundary remain legal while paths through its interior remain blocked.
    expanded_min = obstacle_min - clearance + 1e-4
    expanded_max = obstacle_max + clearance - 1e-4
    if vertices.shape[0] == 0:
        return jnp.zeros((0, 0), dtype=jnp.float32)

    def row(index, distance):
        visible = ~segments_blocked(vertices[index], vertices, expanded_min, expanded_max)
        length = jnp.linalg.norm(vertices[index] - vertices, axis=-1)
        weighted = length + jnp.where(length > corner_merge_distance_m + 1e-4, corner_bonus_m, 0.0)
        return distance.at[index].set(jnp.where(visible, weighted, INF))

    distance = jax.lax.fori_loop(0, vertices.shape[0], row,
                               jnp.full((vertices.shape[0], vertices.shape[0]), INF))
    distance = distance.at[jnp.arange(vertices.shape[0]), jnp.arange(vertices.shape[0])].set(0.0)
    def relax(pivot, distance):
        return jnp.minimum(distance, distance[:, pivot, None] + distance[pivot, None, :])

    return jax.lax.fori_loop(0, vertices.shape[0], relax, distance)


def _roadmap_attachment(points, vertices, obstacle_min, obstacle_max):
    cost = jnp.linalg.norm(points[:, None, :] - vertices[None, :, :], axis=-1)
    return jnp.where(
        segments_blocked(points[:, None, :], vertices[None, :, :], obstacle_min, obstacle_max),
        INF, cost,
    )


def _route_costs(start_cost, roadmap):
    """Exact min-plus product in small blocks instead of one loop per vertex."""
    count = roadmap.shape[0]
    if count == 0:
        return start_cost
    block = min(8, count)

    def attach(index, cost):
        # Overlap the final block; repeated minima leave the result unchanged.
        offset = jnp.minimum(index * block, count - block)
        source = jax.lax.dynamic_slice_in_dim(start_cost, offset, block, axis=1)
        rows = jax.lax.dynamic_slice_in_dim(roadmap, offset, block, axis=0)
        candidate = jnp.min(source[:, :, None] + rows[None, :, :], axis=1)
        return jnp.minimum(cost, candidate)

    return jax.lax.fori_loop(0, (count + block - 1) // block, attach,
                            jnp.full_like(start_cost, INF))


def pairwise_free_space_distance(points, vertices, roadmap, obstacle_min, obstacle_max):
    """Query one node set, sharing all endpoint attachments across its pairs."""
    points = jnp.atleast_2d(points)
    direct = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    direct = jnp.where(segments_blocked(points[:, None, :], points[None, :, :],
                                       obstacle_min, obstacle_max), INF, direct)
    if vertices.shape[0] == 0:
        return direct
    cost = _roadmap_attachment(points, vertices, obstacle_min, obstacle_max)
    attached = _route_costs(cost, roadmap)
    via = jnp.min(attached[:, :, None] + cost.T[None, :, :], axis=1)
    return jnp.minimum(direct, via)


def free_space_distance(start, end, vertices, roadmap, obstacle_min, obstacle_max, corner_bonus_m=0.0):
    """Obstacle-aware distance between all pairs in ``start`` and ``end``."""
    start = jnp.atleast_2d(start)
    end = jnp.atleast_2d(end)
    direct = jnp.linalg.norm(start[:, None, :] - end[None, :, :], axis=-1)
    direct = jnp.where(
        segments_blocked(start[:, None, :], end[None, :, :], obstacle_min, obstacle_max),
        INF,
        direct,
    )
    if vertices.shape[0] == 0:
        return direct
    start_cost = _roadmap_attachment(start, vertices, obstacle_min, obstacle_max) + corner_bonus_m
    end_cost = _roadmap_attachment(end, vertices, obstacle_min, obstacle_max)
    attached = _route_costs(start_cost, roadmap)
    via = jnp.min(attached[:, :, None] + end_cost.T[None, :, :], axis=1)
    return jnp.minimum(direct, via)
