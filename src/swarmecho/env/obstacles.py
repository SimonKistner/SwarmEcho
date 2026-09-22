"""Fixed-shape cuboid geometry and compact visibility roadmaps."""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from functools import lru_cache
from swarmecho.core.terminal import init_message
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import json


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


# Keep the existing public JAX-facing API and static-pytree contract.
from swarmecho.env.roadmap_cpu import authored_roadmap_vertices
from swarmecho.env.roadmap_cpu import shared_roadmap as _cpu_shared_roadmap
from swarmecho.env.roadmap_cpu import building_roadmap as _cpu_building_roadmap

@lru_cache(maxsize=8)
def _as_shared_roadmap(graph):
    return SharedRoadmap(graph.solid_min, graph.solid_max, graph.vertices, graph.distances,
                         graph.clearance, graph.visible_edges, graph.corner_distances)

def shared_roadmap(*args, **kwargs):
    return _as_shared_roadmap(_cpu_shared_roadmap(*args, **kwargs))

def building_roadmap(*args, **kwargs):
    return _as_shared_roadmap(_cpu_building_roadmap(*args, **kwargs))

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
