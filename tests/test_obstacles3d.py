from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.env.obstacles3d import (
    free_space_distance,
    generate_obstacles,
    points_inside_aabbs,
    roadmap_distances,
    roadmap_vertices,
    segments_blocked,
)


def test_seeded_layout_is_valid_deterministic_and_non_overlapping():
    args = dict(
        count=3, size_min=2.0, size_max=4.0, z_min=10.0, z_max=25.0,
        boundary_buffer=0.5,
    )
    first = generate_obstacles(jax.random.PRNGKey(7), jnp.array([20.0, 20.0, 30.0]), **args)
    second = generate_obstacles(jax.random.PRNGKey(7), jnp.array([20.0, 20.0, 30.0]), **args)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert np.all(np.asarray(first[0])[:, 2] >= 10.0)
    assert np.all(np.asarray(first[1])[:, 2] <= 25.0)
    for left in range(3):
        for right in range(left + 1, 3):
            assert np.any(np.asarray(first[1][left]) <= np.asarray(first[0][right])) or np.any(
                np.asarray(first[1][right]) <= np.asarray(first[0][left])
            )


def test_one_aabb_consistently_blocks_points_segments_and_roadmap_direct_path():
    lower = jnp.array([[4.0, 4.0, 4.0]])
    upper = jnp.array([[6.0, 6.0, 6.0]])
    assert bool(points_inside_aabbs(jnp.array([5.0, 5.0, 5.0]), lower, upper))
    assert bool(segments_blocked(jnp.array([2.0, 5.0, 5.0]), jnp.array([8.0, 5.0, 5.0]), lower, upper))
    assert not bool(segments_blocked(jnp.array([2.0, 2.0, 2.0]), jnp.array([8.0, 2.0, 2.0]), lower, upper))
    vertices = roadmap_vertices(lower, upper, 0.25)
    roadmap = roadmap_distances(vertices, lower, upper, 0.25)
    distance = free_space_distance(
        jnp.array([2.0, 5.0, 5.0]), jnp.array([8.0, 5.0, 5.0]),
        vertices, roadmap, lower, upper,
    )[0, 0]
    assert float(distance) > 6.0


def test_symmetric_routes_have_equal_cost_and_jit_fixed_shapes():
    lower = jnp.array([[4.0, 4.0, 4.0]])
    upper = jnp.array([[6.0, 6.0, 6.0]])
    vertices = roadmap_vertices(lower, upper, 0.25)
    roadmap = roadmap_distances(vertices, lower, upper, 0.25)
    distance = jax.jit(free_space_distance)(
        jnp.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]]),
        jnp.array([[5.0, 2.0, 5.0], [5.0, 8.0, 5.0]]),
        vertices, roadmap, lower, upper,
    )
    assert distance.shape == (2, 2)
    np.testing.assert_allclose(distance[0, 0], distance[0, 1], rtol=1e-5)

