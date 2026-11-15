"""Building maps, visibility roadmaps, and obstacle geometry checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from swarmecho.analysis import roadmap_paths as generator
from swarmecho.core.config import EnvConfig, load_level_cli, RandomBuildingConfig
from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
    _compile_one,
    _map_digest,
    required_settings,
)
from swarmecho.env.buildings import (
    BuildingValidationError,
    compile_building,
    load_building,
    make_cuboid_building,
    target_spawn_boxes,
)
from swarmecho.env.obstacles import (
    _route_costs,
    _segments_blocked_block,
    free_space_distance,
    generate_obstacles,
    pairwise_free_space_distance,
    points_inside_aabbs,
    roadmap_distances,
    roadmap_vertices,
    segments_blocked,
    shared_roadmap,
)
from swarmecho.env.random_buildings import (
    advanced_target_exclusions,
    generate_building,
    generate_maps,
    random_building_grid,
)


# Authored buildings

BUILDING_PATH = Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")


def _data():
    return yaml.safe_load(BUILDING_PATH.read_text(encoding="utf-8"))


def test_baseline_building_compiles_to_static_arrays():
    source = _data()
    assert source["name"].startswith("M00_")
    assert source["building_cell_grid"] == {"cols": 4, "rows": 4, "layers": 4}
    building = load_building(BUILDING_PATH)

    assert building.tiles.shape == (4, 4, 5)
    assert building.x_walls.shape == (5, 4, 4)
    assert building.y_walls.shape == (4, 5, 4)
    assert building.target_exclusion.shape == (4, 4, 4)
    np.testing.assert_allclose(building.base_position_m, [10.0, 10.0, 0.125])
    assert building.tile_thickness_m == 0.25
    assert building.wall_thickness_m == 0.25
    assert building.max_base_to_top_corner_m == pytest.approx(24.393, abs=0.001)


@pytest.mark.parametrize(
    ("collection", "coordinate", "message"),
    [
        ("tiles", [0, 0, 0], "bottom floor and top roof"),
        ("x_walls", [0, 0, 0], "outer X walls"),
        ("y_walls", [0, 0, 0], "outer Y walls"),
    ],
)
def test_missing_shell_geometry_is_rejected(collection, coordinate, message):
    data = deepcopy(_data())
    data["geometry"][collection].remove(coordinate)

    with pytest.raises(BuildingValidationError, match=message):
        compile_building(data)


def test_intermediate_tile_holes_are_allowed():
    building = compile_building(_data())

    assert not building.tiles[:, :, 1:-1].any()


def test_out_of_bounds_target_exclusion_is_rejected():
    data = deepcopy(_data())
    data["target_exclusion_cells"].append([4, 0, 0])

    with pytest.raises(BuildingValidationError, match="outside valid bounds"):
        compile_building(data)


def test_base_is_an_explicit_coordinate_not_a_cell():
    data = deepcopy(_data())
    np.testing.assert_allclose(compile_building(data).base_position_m, [10.0, 10.0, 0.125])

    data["base_position_m"] = [10.0, 10.0, 20.1]
    with pytest.raises(BuildingValidationError, match="inside the building bounds"):
        compile_building(data)


def test_generated_cuboid_has_closed_shell_and_central_base():
    building = make_cuboid_building((6, 4, 3))
    assert building.tiles[:, :, 0].all()
    assert building.tiles[:, :, -1].all()
    assert building.x_walls[0].all() and building.x_walls[-1].all()
    assert building.y_walls[:, 0].all() and building.y_walls[:, -1].all()
    np.testing.assert_allclose(building.base_position_m, [15.0, 10.0, 0.125])
    assert building.target_exclusion.sum() == 1


# Obstacle geometry

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


# Generated map exclusions

G03_ENV = EnvConfig(comm_radius_base=15., visual_radius=5.)
SETTINGS = RandomBuildingConfig(enabled=True, length_m=15., width_m=15., stories=3,
                               rooms_per_story_min=2, rooms_per_story_max=4, staircase_max=2)


def representative_fixture(monkeypatch, *, wall=False, radius=20.):
    # All 12 cells have a legal representative. The wall blocks the diagonal
    # (0,0)->(1,1), while (0,1)->(1,1) is clear around its far end.
    cells = np.array([(x, y, 0) for x in range(4) for y in range(3)])
    centres = (cells + .5) * 5.
    lo, hi = centres - 1., centres + 1.
    monkeypatch.setattr("swarmecho.env.random_buildings.target_spawn_boxes",
                        lambda *args, **kwargs: (lo, hi, np.full(len(cells), 8.)))
    building = SimpleNamespace(base_position_m=np.array([2.5, 2.5, 2.5]), cell_size_m=5.,
        solid_min_m=np.array([[4.9, 0., 0.]]) if wall else np.empty((0, 3)),
        solid_max_m=np.array([[5.1, 5.1, 5.]]) if wall else np.empty((0, 3)))
    cfg = replace(G03_ENV, comm_radius_base=radius-1., visual_radius=1.)
    return advanced_target_exclusions(building, cfg)


def test_all_visible_cells_are_excluded_instead_of_only_three(monkeypatch):
    cells, details = representative_fixture(monkeypatch, radius=10.1)
    assert {tuple(cell) for cell in cells} == {
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0),
    }
    assert len(details["direct_los_cells"]) == 6
    assert details["around_corner_cell"] is None


def test_diagonal_wall_blocks_los_and_only_one_hidden_corner_is_added(monkeypatch):
    cells, details = representative_fixture(monkeypatch, wall=True, radius=15.1)
    direct = {tuple(cell) for cell in details["direct_los_cells"]}
    assert (1, 1, 0) not in direct  # diagonal clips the wall
    assert (0, 1, 0) in direct
    assert details["around_corner_cell"] == [1, 1, 0]
    assert details["around_corner_via"] == [0, 1, 0]
    assert len(cells) == len(direct) + 1
    assert [1, 0, 0] not in cells  # second hidden cell is not also excluded


def test_fully_excluded_layout_fails_instead_of_silent_fallback(monkeypatch):
    with pytest.raises(ValueError, match="excludes all legal target volume"):
        representative_fixture(monkeypatch)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_toggle_requires_a_boolean(value):
    with pytest.raises(ValueError, match="advanced_exclustion_base_LOS must be boolean"):
        random_building_grid(replace(SETTINGS, advanced_exclustion_base_LOS=value))


@pytest.mark.parametrize("seed", [0, 1, 2, 456789])
def test_generation_preserves_geometry_and_default_exclusions(seed):
    ordinary, base_building = generate_building(SETTINGS, seed, env_cfg=G03_ENV)
    enabled = replace(SETTINGS, advanced_exclustion_base_LOS=True)
    advanced, building = generate_building(enabled, seed, env_cfg=G03_ENV)
    repeated, _ = generate_building(enabled, seed, env_cfg=G03_ENV)
    assert base_building.target_exclusion.sum() == 1
    assert advanced == repeated
    assert advanced["geometry"] == ordinary["geometry"]
    assert advanced["base_position_m"] == ordinary["base_position_m"]
    np.testing.assert_array_equal(building.solid_min_m, base_building.solid_min_m)
    np.testing.assert_array_equal(building.solid_max_m, base_building.solid_max_m)
    assert building.target_exclusion.sum() > 1
    assert target_spawn_boxes(building, G03_ENV)[2].sum() > 0
    assert advanced["generation"]["target_exclusion_parameters"]["comm_radius_base"] == 15.


def test_serialized_generated_maps_preserve_exclusions_and_environment(tmp_path):
    enabled = replace(SETTINGS, advanced_exclustion_base_LOS=True)
    buildings, manifest = generate_maps(enabled, 456789, 2, directory=tmp_path, env_cfg=G03_ENV)
    for record, building in zip(manifest["maps"], buildings):
        path = tmp_path / f"{record['id']}.yaml"
        np.testing.assert_array_equal(load_building(path).target_exclusion, building.target_exclusion)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["generation"]["target_exclusion_parameters"]["visual_radius"] == 5.
    assert manifest["target_exclusion_parameters"]["comm_radius_base"] == 15.


@pytest.mark.parametrize("level_name", ["G03_legacy_random_buildings", "G03_CLU_random_buildings"])
def test_cli_and_bank_identity_include_toggle_and_active_ranges(level_name):
    level = load_level_cli([f"level={level_name}", "random_buildings.advanced_exclustion_base_LOS=false"])
    assert not level.random_buildings.advanced_exclustion_base_LOS
    advanced = load_level_cli([f"level={level_name}"])
    assert advanced.random_buildings.advanced_exclustion_base_LOS
    assert advanced.training.map_pool_path is None
    assert advanced.building.target_exclusion.sum() > 1
    plain_settings, active_settings = required_settings(level), required_settings(advanced)
    assert plain_settings != active_settings
    assert active_settings["random_buildings"]["advanced_exclustion_base_LOS"] is True
    for field in ("comm_radius_base", "visual_radius"):
        changed_plain = replace(level, env=replace(level.env, **{field: 12.}))
        changed_active = replace(advanced, env=replace(advanced.env, **{field: 12.}))
        assert required_settings(changed_plain) == plain_settings
        assert required_settings(changed_active) != active_settings


def test_bank_worker_saves_only_nonexcluded_target_boxes(tmp_path):
    enabled = replace(SETTINGS, advanced_exclustion_base_LOS=True)
    cfg = replace(G03_ENV, comm_radius_base=9., visual_radius=3., target_wall_buffer_fraction=.15)
    data, _ = generate_building(enabled, 456789, name="map_00000", env_cfg=cfg)
    (tmp_path / "maps").mkdir()
    (tmp_path / "records").mkdir()
    _compile_one(0, 456789, _map_digest(data), enabled, cfg, False, 0., tmp_path)
    building = load_building(tmp_path / "maps/map_00000.yaml")
    with np.load(tmp_path / "records/map_00000.npz") as record:
        centres = (record["target_lower"] + record["target_upper"]) * .5
        cells = np.floor(centres / building.cell_size_m).astype(int)
        assert not building.target_exclusion[tuple(cells.T)].any()
        assert record["target_volumes"].sum() > 0


def test_generated_validation_reloads_matching_policy_and_rejects_changed_range(tmp_path):
    from swarmecho.training.validation import prepare_generated_validation_maps

    level = load_level_cli(["level=G03_legacy_random_buildings", "validation.num_maps=2",
                           "inspection_evaluation.inspection_maps=2",
                           "random_buildings.advanced_exclustion_base_LOS=true",
                           "env.comm_radius_base=9", "env.visual_radius=3"])
    _, training = generate_building(level.random_buildings, 456789, env_cfg=level.env)
    paths = prepare_generated_validation_maps(level, tmp_path, lambda message: None, [training])
    assert len(paths) == 2
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["generation"]["target_exclusion_parameters"]["comm_radius_base"] == 9.
        assert load_building(path).target_exclusion.sum() > 1
    assert prepare_generated_validation_maps(level, tmp_path, lambda message: None) == paths
    changed = replace(level, env=replace(level.env, comm_radius_base=10.))
    with pytest.raises(ValueError, match="validation pool settings changed"):
        prepare_generated_validation_maps(changed, tmp_path, lambda message: None)


# Ranked visibility routes

def test_ranked_paths_are_unique_loopless_and_sorted():
    adjacency = [[] for _ in range(5)]
    for a, b, cost in [(0, 2, 1), (2, 1, 1), (0, 3, 1), (3, 1, 2),
                       (0, 4, 2), (4, 1, 2), (2, 3, .5)]:
        adjacency[a].append((b, cost))
        adjacency[b].append((a, cost))
    routes = generator.ranked_paths(adjacency, 10)
    weights = [dict(edges) for edges in adjacency]
    lengths = [sum(weights[a][b] for a, b in zip(p, p[1:])) for p in routes]
    assert lengths == sorted(lengths)
    assert lengths[0] == 2
    assert len({tuple(p) for p in routes}) == len(routes)
    assert all(len(set(p)) == len(p) for p in routes)
    assert generator.ranked_paths([[], []], 5) == []


def test_visibility_blocks_crossings_but_allows_parallel_clear_paths():
    visible = generator._visible(
        np.array([0., 0., 0.]), np.array([[3., 0., 0.], [0., 3., 0.]]),
        np.array([[1., -.5, -.5]]), np.array([[2., .5, .5]]),
    )
    assert visible.tolist() == [False, True]


def test_obstacles_default_off_even_for_an_obstacle_level(tmp_path):
    from types import SimpleNamespace
    from swarmecho.env.environment import EnvConfig

    building = SimpleNamespace(
        world_size_m=np.array([10., 10., 10.]), base_position_m=np.array([1., 1., 1.]),
        cell_size_m=5., interior_cells=np.ones((2, 2, 2), dtype=bool),
        wall_thickness_m=.25, tile_thickness_m=.25,
        solid_min_m=np.zeros((0, 3)), solid_max_m=np.zeros((0, 3)),
    )
    def unexpected(*args, **kwargs):
        raise AssertionError("Random obstacles must be opt-in")
    output = generator.generate(
        tmp_path / "test.roadmap.json", building=building, cfg=EnvConfig(num_obstacles=3),
        target=[8., 8., 8.], obstacle_generator=unexpected,
    )
    data = generator.json.loads(output.read_text())
    assert not data["manifest"]["generated_obstacles"]
    assert len(data["layouts"]) == 1
    assert data["layouts"][0]["obstacle_min"] == []
    assert len(data["layouts"][0]["paths"]) == 1
