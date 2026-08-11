"""Fixed-shape 3D cuboid geometry and compact visibility roadmaps."""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


ROADMAP_CORNERS = np.asarray(list(product((0.0, 1.0), repeat=3)), dtype=np.float32)
INF = jnp.float32(1.0e6)


def points_inside_aabbs(points, obstacle_min, obstacle_max, buffer=0.0):
    """Return whether each point is inside any buffered axis-aligned box."""
    points = jnp.asarray(points)
    inside = (points[..., None, :] >= obstacle_min - buffer) & (
        points[..., None, :] <= obstacle_max + buffer
    )
    return jnp.any(jnp.all(inside, axis=-1), axis=-1)


def segments_blocked(start, end, obstacle_min, obstacle_max, epsilon=1e-5):
    """Vectorized slab test for segments whose leading dimensions may differ."""
    start, end = jnp.asarray(start), jnp.asarray(end)
    direction = end - start
    safe = jnp.where(jnp.abs(direction) > epsilon, direction, 1.0)
    first = (obstacle_min - start[..., None, :]) / safe[..., None, :]
    last = (obstacle_max - start[..., None, :]) / safe[..., None, :]
    t_near = jnp.max(jnp.minimum(first, last), axis=-1)
    t_far = jnp.min(jnp.maximum(first, last), axis=-1)
    parallel = jnp.abs(direction)[..., None, :] <= epsilon
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


def roadmap_distances(vertices, obstacle_min, obstacle_max, clearance: float):
    """Build a visibility graph and its fixed-size Floyd-Warshall distance matrix."""
    # Slightly shrink the forbidden volume so edges lying on the clearance
    # boundary remain legal while paths through its interior remain blocked.
    expanded_min = obstacle_min - clearance + 1e-4
    expanded_max = obstacle_max + clearance - 1e-4
    start = vertices[:, None, :]
    end = vertices[None, :, :]
    visible = ~segments_blocked(start, end, expanded_min, expanded_max)
    length = jnp.linalg.norm(start - end, axis=-1)
    distance = jnp.where(visible, length, INF)
    distance = distance.at[jnp.arange(vertices.shape[0]), jnp.arange(vertices.shape[0])].set(0.0)
    for pivot in range(vertices.shape[0]):
        distance = jnp.minimum(distance, distance[:, pivot, None] + distance[pivot, None, :])
    return distance


def free_space_distance(start, end, vertices, roadmap, obstacle_min, obstacle_max):
    """Obstacle-aware distance between all pairs in ``start`` and ``end``."""
    start = jnp.atleast_2d(start)
    end = jnp.atleast_2d(end)
    direct = jnp.linalg.norm(start[:, None, :] - end[None, :, :], axis=-1)
    direct = jnp.where(
        segments_blocked(start[:, None, :], end[None, :, :], obstacle_min, obstacle_max),
        INF,
        direct,
    )
    start_cost = jnp.linalg.norm(start[:, None, :] - vertices[None, :, :], axis=-1)
    start_cost = jnp.where(
        segments_blocked(start[:, None, :], vertices[None, :, :], obstacle_min, obstacle_max),
        INF,
        start_cost,
    )
    end_cost = jnp.linalg.norm(end[:, None, :] - vertices[None, :, :], axis=-1)
    end_cost = jnp.where(
        segments_blocked(end[:, None, :], vertices[None, :, :], obstacle_min, obstacle_max),
        INF,
        end_cost,
    )
    via = jnp.min(
        start_cost[:, :, None, None]
        + roadmap[None, :, :, None]
        + end_cost.T[None, None, :, :],
        axis=(1, 2),
    )
    return jnp.minimum(direct, via)
