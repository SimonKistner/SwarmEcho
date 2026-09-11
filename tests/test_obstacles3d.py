from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.env.obstacles3d import (
    free_space_distance,
    generate_obstacles,
    points_inside_aabbs,
    roadmap_distances,
    roadmap_vertices,
    segments_blocked,
    shared_roadmap,
    _segments_blocked_block,
    _route_costs,
    pairwise_free_space_distance,
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


def test_empty_roadmap_returns_euclidean_distance():
    empty = jnp.zeros((0, 3), dtype=jnp.float32)
    result = jax.jit(free_space_distance)(
        jnp.array([0., 0., 0.]), jnp.array([3., 4., 0.]),
        empty, jnp.zeros((0, 0)), empty, empty,
    )
    np.testing.assert_allclose(result, [[5.]])


def test_shared_roadmap_is_unbatched_cached_and_matches_runtime_solver():
    lo = np.array([[9., 5., 0.]], dtype=np.float32)
    hi = np.array([[11., 15., 20.]], dtype=np.float32)
    bounds = (np.full(3, .4), np.full(3, 19.6))
    graph = shared_roadmap(lo, hi, *bounds, .35, plan=True)
    assert shared_roadmap(lo.copy(), hi.copy(), *bounds, .35, plan=True) is graph
    assert jax.tree_util.tree_leaves(graph) == []
    assert shared_roadmap(lo, hi, *bounds, .5, plan=True) is not graph
    changed = hi.copy()
    changed[0, 1] = 16.
    assert shared_roadmap(lo, changed, *bounds, .35, plan=True) is not graph
    assert not graph.distances.flags.writeable
    np.testing.assert_allclose(
        graph.distances,
        jax.jit(roadmap_distances)(jnp.asarray(graph.vertices), jnp.asarray(lo), jnp.asarray(hi), .35),
        atol=1e-4,
    )
    distance = free_space_distance(
        jnp.array([5., 10., 10.]), jnp.array([15., 10., 10.]),
        jnp.asarray(graph.vertices), jnp.asarray(graph.distances),
        jnp.asarray(lo - .35 + 1e-4), jnp.asarray(hi + .35 - 1e-4),
    )
    assert 10. < float(distance[0, 0]) < 1e6


def test_parallel_axis_inside_thin_slab_does_not_limit_segment_parameter():
    # The x crossing occurs at t=.5; the narrow parallel y slab must not
    # incorrectly restrict the valid parameter interval to [-.1, .1].
    assert bool(segments_blocked(
        jnp.array([0., 5., 5.]), jnp.array([10., 5., 5.]),
        jnp.array([[4., 4.9, 4.]]), jnp.array([[6., 5.1, 6.]]),
    ))


def test_bounded_route_query_matches_full_pairwise_reduction():
    lo, hi = jnp.array([[4., 4., 4.]]), jnp.array([[6., 6., 6.]])
    vertices = roadmap_vertices(lo, hi, .25)
    graph = roadmap_distances(vertices, lo, hi, .25)
    starts = jnp.array([[2., 5., 5.], [5., 2., 5.]])
    ends = jnp.array([[8., 5., 5.], [5., 8., 5.], [5., 5., 8.]])

    def links(left, right):
        distance = jnp.linalg.norm(left[:, None] - right[None], axis=-1)
        return jnp.where(segments_blocked(left[:, None], right[None], lo, hi), 1e6, distance)

    full = links(starts, vertices)[:, :, None, None] + graph[None, :, :, None]
    full = full + links(ends, vertices).T[None, None, :, :]
    expected = jnp.minimum(links(starts, ends), jnp.min(full, axis=(1, 2)))
    actual = jax.jit(free_space_distance)(starts, ends, vertices, graph, lo, hi)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("count", [0, 1, 8, 9, 16, 17, 141])
def test_chunked_intersections_match_full_reduction_under_vmap(count):
    rng = np.random.default_rng(31)
    lower = jnp.asarray(rng.uniform(-3., 3., (count, 3)), dtype=jnp.float32)
    upper = lower + jnp.asarray(rng.uniform(.1, 1., (count, 3)), dtype=jnp.float32)
    points = jnp.asarray(rng.uniform(-4., 4., (3, 4, 3)), dtype=jnp.float32)
    # Includes zero-length self-segments and nested agent-pair broadcasting.
    def pairs(fn, lane, lo, hi):
        return fn(lane[:, None, :], lane[None, :, :], lo, hi, 1e-5)

    expected = jax.vmap(lambda lane: pairs(_segments_blocked_block, lane, lower, upper))(points)
    actual = jax.jit(jax.vmap(lambda lane: pairs(segments_blocked, lane, lower, upper)))(points)
    np.testing.assert_array_equal(actual, expected)
    # Also exercise per-environment geometry rather than only shared constants.
    lowers = jnp.stack([lower, lower + .2, lower - .3])
    uppers = jnp.stack([upper, upper + .2, upper - .3])
    actual = jax.jit(jax.vmap(lambda lane, lo, hi: pairs(segments_blocked, lane, lo, hi)))(points, lowers, uppers)
    expected = jax.vmap(lambda lane, lo, hi: pairs(_segments_blocked_block, lane, lo, hi))(points, lowers, uppers)
    np.testing.assert_array_equal(actual, expected)


def test_chunked_intersections_include_last_partial_chunk():
    lower = jnp.full((141, 3), 100.).at[-1].set(jnp.array([4., -1., -1.]))
    upper = (lower + 1.).at[-1].set(jnp.array([6., 1., 1.]))
    result = jax.jit(segments_blocked)(
        jnp.array([[0., 0., 0.], [0., 2., 0.]]),
        jnp.array([[10., 0., 0.], [10., 2., 0.]]), lower, upper,
    )
    np.testing.assert_array_equal(result, [True, False])


@pytest.mark.parametrize("count", [0, 1, 8, 9, 17, 972])
def test_blocked_min_plus_matches_original_pivot_reduction(count):
    rng = np.random.default_rng(837)
    start = rng.uniform(0., 20., (3, count)).astype(np.float32)
    graph = rng.uniform(0., 100., (count, count)).astype(np.float32)
    if count:
        start[0] = 1e6  # Unreachable endpoint, including the finite-INF cap.
        graph[:, -1] = 1e6
    expected = np.full_like(start, 1e6)
    for pivot in range(count):
        expected = np.minimum(expected, start[:, pivot, None] + graph[pivot, None, :])
    actual = jax.jit(_route_costs)(jnp.asarray(start), jnp.asarray(graph))
    np.testing.assert_array_equal(actual, expected)


def test_shared_pairwise_query_matches_separate_endpoint_queries():
    lo, hi = jnp.array([[4., 4., 0.]]), jnp.array([[6., 6., 10.]])
    vertices = roadmap_vertices(lo, hi, .25)
    graph = roadmap_distances(vertices, lo, hi, .25)
    points = jnp.array([[2., 5., 5.], [8., 5., 5.], [2., 2., 5.], [8., 8., 5.]])
    actual = jax.jit(pairwise_free_space_distance)(points, vertices, graph, lo, hi)
    expected = free_space_distance(points, points, vertices, graph, lo, hi)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

