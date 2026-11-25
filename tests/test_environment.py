"""Environment dynamics, observations, and reset checks."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import AdaptiveTargetSpawnConfig, EnvConfig, RewardConfig
from swarmecho.env.buildings import load_building, target_spawn_boxes
from swarmecho.env.environment import (
    compute_rewards,
    coverage_grid_geometry,
    EnvState,
    final_chain_length,
    make_autoreset_fns,
    make_env_fns,
    maximum_chain_distance,
    maximum_five_drone_chain_distance,
    observation_dim,
    obstacle_chain_diagnostics,
    spherical_directions,
)
from swarmecho.training.adaptive_spawn import AdaptiveTargetSpawnController


# Environment and sensors

BUILDING = load_building(
    Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml")
)


def _partial_wall_building():
    """Keep the authored wall grid and its solid prism in agreement."""
    x_walls = BUILDING.x_walls.copy()
    x_walls[2, 1:3, :] = True
    half_width = BUILDING.wall_thickness_m / 2
    return replace(
        BUILDING,
        x_walls=x_walls,
        solid_min_m=np.array([[10. - half_width, 5., 0.]], dtype=np.float32),
        solid_max_m=np.array([[10. + half_width, 15., 20.]], dtype=np.float32),
        wall_solid_range=(0, 1),
    )


def test_authored_geodesic_roadmap_is_shared_through_batched_autoreset():
    building = _partial_wall_building()
    cfg = replace(EnvConfig(), num_agents=1, visual_radius=1.,
                  comm_radius=1., comm_radius_base=1., max_steps=1)
    reward_cfg = RewardConfig()
    reset, step, _, _ = make_autoreset_fns(building, cfg, reward_cfg)
    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(21), 2))
    assert states.obstacle_min.shape == (2, 0, 3)
    assert states.roadmap_distances.shape == (2, 0, 0)
    assert states.shared_geometry.distances.shape[0] > 0
    state = jax.tree_util.tree_map(lambda value: value[0], states)._replace(
        base_pos=jnp.array([5., 10., 10.]), target_pos=jnp.array([15., 10., 10.]),
        is_conn_base=jnp.array([False]), is_conn_target=jnp.array([False]),
        base_target_connected=jnp.bool_(False),
    )
    gap = jax.jit(obstacle_chain_diagnostics)(state)[0]
    assert 10. < float(gap) < 1e6
    next_states, _, done, _ = jax.jit(step.batched)(states, jnp.zeros((2, 1, 3)))
    assert np.all(done)
    assert next_states.shared_geometry is states.shared_geometry
    assert next_states.roadmap_distances.shape == (2, 0, 0)


def test_dynamic_roadmap_combines_authored_and_persisted_layout_geometry():
    building = _partial_wall_building()
    cfg = replace(EnvConfig(), num_agents=1)
    reset, _, _, _ = make_env_fns(building, cfg)
    lo, hi = jnp.array([[2., 2., 2.]]), jnp.array([[3., 3., 3.]])
    state = reset(jax.random.PRNGKey(4), target_pos=jnp.array([15., 10., 10.]),
                  obstacle_min=lo, obstacle_max=hi)
    # Authored edges are sampled; the generated cuboid still adds eight corners.
    assert state.roadmap_vertices.shape[1] == 3
    assert state.roadmap_vertices.shape[0] > 8
    assert state.solid_min.shape == (2, 3)
    np.testing.assert_allclose(state.solid_min[0], building.solid_min_m[0])
    assert state.shared_geometry.distances.size > 0
    np.testing.assert_array_equal(state.planning_distances, state.roadmap_distances)
    query = state._replace(
        base_pos=jnp.array([5., 10., 10.]),
        is_conn_base=jnp.array([False]), is_conn_target=jnp.array([False]),
        base_target_connected=jnp.bool_(False),
    )
    assert 10. < float(obstacle_chain_diagnostics(query)[0]) < 1e6
    # Legacy generated-only graph arrays cannot bypass the authored walls.
    legacy = reset(jax.random.PRNGKey(5), obstacle_min=lo, obstacle_max=hi,
                   stored_vertices=jnp.zeros((8, 3)), stored_distances=jnp.zeros((8, 8)))
    np.testing.assert_allclose(legacy.roadmap_distances, state.roadmap_distances)
    restored = reset(jax.random.PRNGKey(5), obstacle_min=lo, obstacle_max=hi,
                     stored_vertices=state.roadmap_vertices,
                     stored_distances=state.roadmap_distances)
    np.testing.assert_allclose(restored.roadmap_distances, state.roadmap_distances)


def test_combined_chain_query_preserves_original_frontier_selection():
    from swarmecho.env.obstacles import free_space_distance

    cfg = replace(EnvConfig(), num_agents=3)
    reset, _, _, _ = make_env_fns(BUILDING, cfg)
    state = reset(jax.random.PRNGKey(51), target_pos=jnp.array([15., 10., 5.]),
                  obstacle_min=jnp.array([[9., 5., 0.]]),
                  obstacle_max=jnp.array([[11., 15., 10.]]))
    state = state._replace(base_pos=jnp.array([5., 10., 5.]),
                           pos=jnp.array([[6., 10., 5.], [12., 16., 5.], [14., 10., 5.]]),
                           is_conn_base=jnp.array([True, False, False]),
                           is_conn_target=jnp.array([False, True, True]),
                           target_known=jnp.array([False, True, True]),
                           base_target_connected=jnp.bool_(False))
    def distance(a, b):
        return free_space_distance(a, b, state.planning_vertices, state.planning_distances,
                                   state.planning_min, state.planning_max)
    base_candidates = jnp.concatenate((state.base_pos[None], state.pos))
    target_candidates = jnp.concatenate((state.target_pos[None], state.pos))
    gaps = distance(base_candidates, target_candidates)
    mission = distance(state.base_pos, state.target_pos)[0, 0]
    excess = jnp.maximum(distance(state.base_pos, base_candidates)[0, :, None] + gaps
                         + distance(target_candidates, state.target_pos)[:, 0][None, :] - mission, 0.)
    valid_b = jnp.concatenate((jnp.array([True]), state.is_conn_base & state.active))
    valid_t = jnp.concatenate((jnp.array([True]), state.is_conn_target & state.active & state.target_known))
    score = jnp.where(valid_b[:, None] & valid_t[None, :], gaps + jnp.minimum(excess, 1e3)*1e-6, 1e6)
    flat = jnp.argmin(score)
    b, t = flat // score.shape[1], flat % score.shape[1]
    expected = (gaps[b,t], jnp.clip(100*(1-gaps[b,t]/jnp.maximum(mission,1e-6)),0,100),
                excess[b,t], jnp.array([b-1,t-1]))
    actual = jax.jit(obstacle_chain_diagnostics)(state)
    for got, want in zip(actual[:3], expected[:3]):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[3], expected[3])


def test_euclidean_factory_does_not_build_authored_or_dynamic_roadmaps():
    building = replace(
        BUILDING,
        solid_min_m=np.array([[9., 5., 0.]], dtype=np.float32),
        solid_max_m=np.array([[11., 15., 20.]], dtype=np.float32),
    )
    cfg = replace(EnvConfig(), num_agents=1)
    reset, _, _, _ = make_env_fns(building, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(4), target_pos=jnp.array([15., 10., 10.]),
                  obstacle_min=jnp.array([[2., 2., 2.]]),
                  obstacle_max=jnp.array([[3., 3., 3.]]))
    assert state.roadmap_distances.shape == (0, 0)
    assert state.shared_geometry.distances.shape == (0, 0)
    assert state.solid_min.shape == (2, 3)


def _functions(**overrides):
    cfg = replace(EnvConfig(), **overrides)
    return cfg, make_env_fns(BUILDING, cfg)


def test_target_knowledge_relays_one_hop_per_communication_tick_and_reaches_base():
    cfg = replace(
        EnvConfig(), num_agents=3, spawn_delay=0, comm_radius=1.5,
        comm_radius_base=1.5, visual_radius=1.0, max_steps=20,
        hold_chain_for=20,
    )
    reset, step, _, _ = make_env_fns(
        BUILDING, cfg,
    )
    state = reset(jax.random.PRNGKey(91), target_pos=jnp.array([17., 17., 12.]))
    state = state._replace(
        pos=jnp.array([[10., 10., 2.], [11., 10., 2.], [12.4, 10., 2.]]),
        vel=jnp.zeros((3, 3)), base_pos=jnp.array([2., 2., 2.]),
        active=jnp.ones(3, dtype=jnp.bool_),
        target_known=jnp.array([True, False, False]),
        base_target_known=jnp.bool_(False),
    )
    action = jnp.zeros((3, 3))
    state = step(state, action)  # Tick 0: A informs B, but B cannot relay yet.
    np.testing.assert_array_equal(state.target_known, [True, True, False])
    assert not state.base_target_known
    state = step(state, action)  # B informs C without a target sighting.
    np.testing.assert_array_equal(state.target_known, [True, True, True])
    assert not state.base_target_connected

    # B can report the remembered target after moving out of the original group.
    state = state._replace(
        pos=state.pos.at[1].set(jnp.array([3., 2., 2.])),
        vel=jnp.zeros((3, 3)),
    )
    state = step(state, action)
    assert state.base_target_known
    assert not state.success
    assert not state.base_target_connected


def test_informed_base_broadcasts_remembered_target_on_a_later_tick():
    cfg = replace(
        EnvConfig(), num_agents=2, spawn_delay=0, comm_radius=1.5,
        comm_radius_base=1.5, visual_radius=1.0, max_steps=20,
        hold_chain_for=20,
    )
    reset, step, _, _ = make_env_fns(
        BUILDING, cfg,
    )
    state = reset(jax.random.PRNGKey(92), target_pos=jnp.array([17., 17., 12.]))
    state = state._replace(
        pos=jnp.array([[3., 2., 2.], [5., 2., 2.]]),
        vel=jnp.zeros((2, 3)), base_pos=jnp.array([2., 2., 2.]),
        active=jnp.ones(2, dtype=jnp.bool_),
        target_known=jnp.array([False, False]),
        base_target_known=jnp.bool_(True),
    )
    action = jnp.zeros((2, 3))
    state = step(state, action)  # Base informs only the drone in range.
    np.testing.assert_array_equal(state.target_known, [True, False])
    state = step(step(state, action), action)
    np.testing.assert_array_equal(state.target_known, [True, False])
    state = state._replace(pos=state.pos.at[1].set(jnp.array([4., 2., 2.])))
    state = step(state, action)  # Next tick: the newly informed drone can relay.
    np.testing.assert_array_equal(state.target_known, [True, True])


def test_target_knowledge_respects_line_of_sight():
    cfg = replace(
        EnvConfig(), num_agents=2, spawn_delay=0,
        comm_radius=3.0, comm_radius_base=1.0, visual_radius=1.0,

    )
    reset, step, _, _ = make_env_fns(BUILDING, cfg)
    state = reset(
        jax.random.PRNGKey(93), target_pos=jnp.array([17., 17., 12.]),
        obstacle_min=jnp.array([[10.45, 9., 1.]]),
        obstacle_max=jnp.array([[10.55, 11., 3.]]),
    )._replace(
        pos=jnp.array([[10., 10., 2.], [11., 10., 2.]]),
        vel=jnp.zeros((2, 3)), base_pos=jnp.array([2., 2., 2.]),
        active=jnp.ones(2, dtype=jnp.bool_),
        target_known=jnp.array([True, False]),
    )
    next_state = step(state, jnp.zeros((2, 3)))
    np.testing.assert_array_equal(next_state.target_known, [True, False])


def test_octant_radar_and_configured_distance_contracts():
    directions = spherical_directions(8)
    assert directions.shape == (8, 3)
    assert len({tuple(np.sign(row)) for row in directions}) == 8
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    cfg = EnvConfig()
    assert maximum_five_drone_chain_distance(cfg) == 30.0
    assert maximum_chain_distance(replace(cfg, num_agents=6)) == 35.0


def test_peer_radar_hides_drones_behind_walls():
    """Problem 9: all three peer channels must respect line of sight."""
    cfg = replace(EnvConfig(), num_agents=2, spawn_delay=0, wall_sensing_rays=8)
    reset, _, observations, _ = make_env_fns(
        _partial_wall_building(), cfg, plan_geodesic=False,
    )
    state = reset(jax.random.PRNGKey(95))._replace(
        pos=jnp.array([[9., 4., 2.], [11., 4., 2.]]),
        active=jnp.ones(2, dtype=jnp.bool_),
        # Peers may belong to either component via other relay paths.
        is_conn_base=jnp.ones(2, dtype=jnp.bool_),
        is_conn_target=jnp.ones(2, dtype=jnp.bool_),
    )
    observe = jax.jit(observations)

    def peer_channels(value):
        # Six self features, then bins of [wall, peer, target-peer, base-peer].
        return np.asarray(observe(value))[:, 6:].reshape(2, cfg.drone_radar_bins, 4)[:, :, 1:]

    # Both drones see each other through the opening below the wall (y < 5).
    np.testing.assert_allclose(
        peer_channels(state).max(axis=1), 1.0 - 2.0 / cfg.comm_radius,
        rtol=1e-6,
    )
    # Same separation, but the wall at x=10 now blocks both directions.
    blocked = state._replace(pos=jnp.array([[9., 10., 2.], [11., 10., 2.]]))
    np.testing.assert_array_equal(
        peer_channels(blocked), 0.0,
        err_msg="Peer radar leaks through the wall in problem 9",
    )


def _counting_radar_counts(observation, cfg):
    """Decode counting-mode fixtures with the six mandatory self values."""
    values = np.asarray(observation)
    count_start = 6 + cfg.wall_sensing_rays
    counts = values[:, count_start:].reshape(cfg.num_agents, cfg.drone_radar_bins,
                                            max(1, cfg.distance_bands), 3)
    return counts


def test_counting_radar_places_and_caps_visible_neighbors_in_exact_cells():
    """Check crowd/category counts, inactive/self filtering, and the range edge."""
    cfg = replace(EnvConfig(), num_agents=10, spawn_delay=0, comm_radius=15.,
                  wall_sensing_rays=16, drone_radar_bins=8,
                  radar_mode="band_sep_counting", distance_bands=4, radar_count_cap=3)
    reset, _, observations, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    # All neighbors are on +X. In the deterministic eight-octant layout,
    # ties resolve to octant 4 (+X, -Y, -Z). Exact axis distances avoid
    # rounding ambiguity at the 7.5 m and 15 m boundaries.
    distances = jnp.array([0., 1., 2., 2.5, 4., 7.5, 15., 15.25, 1.5, 3.])
    positions = jnp.stack([2. + distances, jnp.full(10, 2.), jnp.full(10, 2.)], axis=-1)
    state = reset(jax.random.PRNGKey(96))._replace(
        pos=positions,
        active=jnp.array([True, True, True, True, True, True, True, True, False, True]),
        is_conn_target=jnp.array([True, True, False, True, False, True, False, True, True, False]),
        is_conn_base=jnp.array([True, False, True, True, True, False, True, True, True, False]),
    )
    observed = jax.jit(observations)(state)
    assert observed.shape == (10, 118)
    counts = _counting_radar_counts(observed, cfg)
    expected_counts = np.zeros((8, 4, 3), dtype=np.float32)
    # Four visible drones saturate the inner all-drone count. Each connected
    # category still has two; self/inactive leaks would change those values.
    expected_counts[4] = np.array([[3, 2, 2], [1, 0, 1], [1, 1, 0], [1, 0, 1]]) / 3.
    np.testing.assert_allclose(counts[0], expected_counts, rtol=1e-6, atol=0.)
    np.testing.assert_array_equal(np.asarray(observed)[8], 0.)


def test_counting_radar_moves_a_neighbor_inward_across_each_band_boundary():
    cfg = replace(EnvConfig(), num_agents=2, spawn_delay=0, comm_radius=15.,
                  wall_sensing_rays=16, drone_radar_bins=8,
                  radar_mode="band_sep_counting", distance_bands=4, radar_count_cap=3)
    reset, _, observations, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(97))._replace(
        active=jnp.ones(2, dtype=jnp.bool_),
        is_conn_target=jnp.zeros(2, dtype=jnp.bool_),
        is_conn_base=jnp.zeros(2, dtype=jnp.bool_),
    )
    observe = jax.jit(observations)
    # Borders belong to the outer band. Just below each border the drone
    # moves inward and leaves its old cell empty.
    for distance, expected_band in [(14., 3), (11.25, 3), (11.249, 2),
                                    (7.5, 2), (7.499, 1), (3.75, 1),
                                    (3.749, 0), (1., 0)]:
        moved = state._replace(pos=jnp.array([[2., 2., 2.], [2. + distance, 2., 2.]]))
        counts = _counting_radar_counts(observe(moved), cfg)
        expected = np.zeros((8, 4, 3), dtype=np.float32)
        expected[4, expected_band, 0] = 1. / 3.
        np.testing.assert_allclose(counts[0], expected, rtol=1e-6, atol=0.,
                                   err_msg=f"Incorrect cell at distance {distance} m")


@pytest.mark.parametrize("distance_bands", [0, 1])
def test_counting_radar_single_band_counts_without_subdivision(distance_bands):
    cfg = replace(EnvConfig(), num_agents=3, spawn_delay=0, comm_radius=15.,
                  wall_sensing_rays=16, drone_radar_bins=8,
                  radar_mode="band_sep_counting", distance_bands=distance_bands,
                  radar_count_cap=2)
    reset, _, observations, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(98))._replace(
        pos=jnp.array([[2., 2., 2.], [3., 2., 2.], [16., 2., 2.]]),
        active=jnp.ones(3, dtype=jnp.bool_),
        is_conn_target=jnp.array([False, True, False]),
        is_conn_base=jnp.array([False, False, True]),
    )
    observed = jax.jit(observations)(state)
    assert observed.shape == (3, 46)
    counts = _counting_radar_counts(observed, cfg)
    expected = np.zeros((8, 1, 3), dtype=np.float32)
    expected[4, 0] = [1., .5, .5]
    np.testing.assert_array_equal(counts[0], expected)


@pytest.mark.parametrize("geometry", ["authored_wall", "generated_obstacle"])
def test_counting_radar_respects_line_of_sight_in_all_categories_and_bands(geometry):
    cfg = replace(EnvConfig(), num_agents=3, spawn_delay=0, comm_radius=8.,
                  wall_sensing_rays=16, drone_radar_bins=8,
                  radar_mode="band_sep_counting", distance_bands=4, radar_count_cap=3)
    building = _partial_wall_building() if geometry == "authored_wall" else BUILDING
    reset, _, observations, _ = make_env_fns(building, cfg, plan_geodesic=False)
    reset_kwargs = {}
    if geometry == "generated_obstacle":
        half_width = BUILDING.wall_thickness_m / 2
        reset_kwargs = {
            "obstacle_min": jnp.array([[10. - half_width, 5., 0.]]),
            "obstacle_max": jnp.array([[10. + half_width, 15., 20.]]),
        }
    state = reset(jax.random.PRNGKey(99), target_pos=jnp.array([17., 17., 12.]),
                  **reset_kwargs)._replace(
        pos=jnp.array([[9.5, 4., 2.], [8.5, 4., 2.], [10.5, 4., 2.]]),
        active=jnp.ones(3, dtype=jnp.bool_),
        # Connectivity through other paths must not bypass local occlusion.
        is_conn_target=jnp.ones(3, dtype=jnp.bool_),
        is_conn_base=jnp.ones(3, dtype=jnp.bool_),
    )
    observe = jax.jit(observations)
    # Exercise occlusion in each of the four bands. The receiver stays on
    # the left of x=10; one peer is on either side at the same distance.
    for distance, band in [(1., 0), (3., 1), (5., 2), (7., 3)]:
        open_state = state._replace(pos=jnp.array(
            [[9.5, 4., 2.], [9.5 - distance, 4., 2.], [9.5 + distance, 4., 2.]]))
        # At y=4 the wall starts above the line joining the drones.
        open_counts = _counting_radar_counts(observe(open_state), cfg)
        expected_counts = np.zeros((8, 4, 3), dtype=np.float32)
        expected_counts[[0, 4], band, :] = 1. / 3.
        np.testing.assert_allclose(open_counts[0], expected_counts, rtol=1e-6, atol=0.)
        # At y=10 only the +X neighbor is hidden; the -X peer stays visible.
        # Thus a broken implementation that blanks all counts also fails.
        blocked = open_state._replace(pos=open_state.pos.at[:, 1].set(10.))
        blocked_counts = _counting_radar_counts(observe(blocked), cfg)
        expected_counts[4] = 0.
        np.testing.assert_allclose(blocked_counts[0], expected_counts, rtol=1e-6, atol=0.,
                                   err_msg=f"{geometry}: count leak in band {band}")


@pytest.mark.parametrize("bins", [8, 16, 32])
def test_reset_observation_and_step_jit_contract(bins):
    cfg, (reset, step, observations, metrics) = _functions(wall_sensing_rays=bins, drone_radar_bins=bins)
    state = jax.jit(reset)(jax.random.PRNGKey(0))
    obs = jax.jit(observations)(state)
    next_state = jax.jit(step)(state, jnp.zeros((cfg.num_agents, 3)))

    assert state.pos.shape == (cfg.num_agents, 3)
    assert obs.shape == (cfg.num_agents, 6 + bins * 4)
    assert next_state.coverage.shape == BUILDING.target_exclusion.shape
    assert jnp.isfinite(obs).all()
    assert metrics(next_state)["coverage_fraction"] > 0


def test_default_targets_sample_continuously_in_buffered_valid_layers():
    cfg, (reset, _, _, _) = _functions()
    targets = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(41), 128)).target_pos
    targets = np.asarray(targets)

    # Both complete lower layers are excluded, while the 0.5 m buffer plus
    # half of the 0.25 m wall thickness protects every outer wall surface.
    assert np.all(targets[:, 2] >= 10.0)
    assert np.all(targets >= np.asarray([0.625, 0.625, 0.625]))
    assert np.all(targets <= np.asarray([19.375, 19.375, 19.375]))
    assert np.unique(targets, axis=0).shape[0] == len(targets)
    assert np.any(np.mod(targets, BUILDING.cell_size_m) != 2.5)


def test_target_sampling_is_independent_of_communication_ranges():
    keys = jax.random.split(jax.random.PRNGKey(41), 32)
    _, (reset, _, _, _) = _functions()
    _, (wide_reset, _, _, _) = _functions(
        comm_radius_base=500., comm_radius=500.
    )
    targets = jax.vmap(reset)(keys).target_pos
    wide_targets = jax.vmap(wide_reset)(keys).target_pos
    np.testing.assert_array_equal(targets, wide_targets)
    assert np.unique(np.asarray(wide_targets), axis=0).shape[0] == len(keys)


def test_vmapped_step_and_high_speed_boundary_collision():
    cfg, (reset, step, _, _) = _functions(max_force=10_000.0, max_speed=1_000.0)
    reset_batch = jax.jit(jax.vmap(reset))
    step_batch = jax.jit(jax.vmap(step))
    states = reset_batch(jax.random.split(jax.random.PRNGKey(1), 4))
    actions = jnp.full((4, cfg.num_agents, 3), 1.0)

    for _ in range(10):
        states = step_batch(states, actions)

    lower = jnp.asarray(
        [
            BUILDING.wall_thickness_m / 2 + cfg.drone_radius,
            BUILDING.wall_thickness_m / 2 + cfg.drone_radius,
            BUILDING.tile_thickness_m / 2 + cfg.drone_radius,
        ]
    )
    upper = jnp.asarray(BUILDING.world_size_m) - lower
    assert jnp.all(states.pos >= lower)
    assert jnp.all(states.pos <= upper)


def test_scripted_five_drone_chain_uses_visual_final_hop():
    cfg, (_, step, _, _) = _functions(
        comm_radius_base=4.0,
        comm_radius=3.0,
        visual_radius=2.0,
        spawn_delay=0,
    )
    base = jnp.asarray([1.0, 1.0, 1.0])
    # 3.5 base hop, four 2.5 communication hops, then a 1.5 visual hop.
    x = jnp.asarray([4.5, 7.0, 9.5, 12.0, 14.5])
    positions = jnp.stack([x, jnp.ones(5), jnp.ones(5)], axis=-1)
    target = jnp.asarray([16.0, 1.0, 1.0])
    state = EnvState(
        pos=positions,
        vel=jnp.zeros((5, 3)),
        base_pos=base,
        target_pos=target,
        active=jnp.ones(5, dtype=jnp.bool_),
        coverage=jnp.zeros(BUILDING.target_exclusion.shape, dtype=jnp.bool_),
        step=jnp.int32(0),
        key=jax.random.PRNGKey(0),
        directly_sees_target=jnp.zeros(5, dtype=jnp.bool_),
        is_conn_base=jnp.zeros(5, dtype=jnp.bool_),
        is_conn_target=jnp.zeros(5, dtype=jnp.bool_),
        target_known=jnp.zeros(5, dtype=jnp.bool_),
        success=jnp.bool_(False),
        base_target_connected=jnp.bool_(False),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(5, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(5),
    )
    connected = state
    for _ in range(cfg.hold_chain_for):
        connected = step(connected, jnp.zeros((5, 3)))
    assert connected.success and connected.done

    broken = state._replace(pos=positions.at[2, 0].set(10.1))
    assert not step(broken, jnp.zeros((5, 3))).success


def test_coverage_changes_across_height_layers():
    cfg, (_, step, _, _) = _functions()
    state = EnvState(
        pos=jnp.tile(jnp.asarray([[2.5, 2.5, 2.5]]), (cfg.num_agents, 1)),
        vel=jnp.zeros((cfg.num_agents, 3)),
        base_pos=jnp.asarray([2.5, 2.5, 2.5]),
        target_pos=jnp.asarray([17.5, 17.5, 17.5]),
        active=jnp.ones(cfg.num_agents, dtype=jnp.bool_),
        coverage=jnp.zeros(BUILDING.target_exclusion.shape, dtype=jnp.bool_),
        step=jnp.int32(0),
        key=jax.random.PRNGKey(0),
        directly_sees_target=jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
        is_conn_base=jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
        is_conn_target=jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
        target_known=jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
        success=jnp.bool_(False),
        base_target_connected=jnp.bool_(False),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(cfg.num_agents),
    )
    low = step(state, jnp.zeros((cfg.num_agents, 3)))
    high = step(low._replace(pos=low.pos.at[0, 2].set(12.5)), jnp.zeros((cfg.num_agents, 3)))
    assert high.coverage[0, 0, 0]
    assert high.coverage[0, 0, 2]


def test_coverage_voxel_size_is_independent_of_building_cell_size():
    # Coverage probes are enabled here only to exercise the legacy debug option.
    cfg, (reset, step, observations, _) = _functions(
        coverage_voxel_size=2.5,
    )
    state = reset(jax.random.PRNGKey(11))
    next_state = step(state, jnp.zeros((cfg.num_agents, 3)))

    voxel_size, shape = coverage_grid_geometry(BUILDING, cfg)
    assert voxel_size == 2.5
    assert shape == (8, 8, 8)
    assert state.coverage.shape == shape
    assert next_state.coverage.shape == shape
    assert observations(state).shape == (cfg.num_agents, observation_dim(cfg))


def test_coverage_voxel_size_must_tile_the_building():
    with pytest.raises(ValueError, match="evenly divide"):
        make_env_fns(BUILDING, EnvConfig(coverage_voxel_size=3.0))


def test_reward_terms_preserve_local_credit_and_shared_events():
    cfg, (reset, step, _, _) = _functions()
    previous = reset(jax.random.PRNGKey(7))
    current = step(previous, jnp.ones((cfg.num_agents, 3)))
    reward, terms = compute_rewards(previous, current, RewardConfig(), cfg)
    assert reward.shape == (cfg.num_agents,)
    assert terms["exploration_reward"].shape == (cfg.num_agents,)
    assert terms["collision_reward"].shape == (cfg.num_agents,)
    assert terms["time_presure_reward"].shape == (cfg.num_agents,)
    assert terms["gap_reward"].shape == (cfg.num_agents,)
    assert jnp.isfinite(reward).all()
    assert jnp.sum(current.coverage_credit) == pytest.approx(
        jnp.sum(current.coverage & ~previous.coverage), abs=1e-5
    )


def test_target_delivery_and_success_are_distinct_one_shot_rewards():
    """must retain 2D's delivery milestone and resist finder farming."""
    reward_cfg = RewardConfig(
        base_reporter_reward=50.0,
        target_delivery_reward=100.0,
        success_reward=500.0,
    )
    base = jnp.asarray([1.0, 1.0, 1.0])
    target = jnp.asarray([11.0, 1.0, 1.0])
    env_cfg = EnvConfig(num_agents=2, comm_radius_base=6.0)
    previous = EnvState(
        pos=jnp.asarray([[4.0, 1.0, 1.0], [8.0, 1.0, 1.0]]),
        vel=jnp.zeros((2, 3)),
        base_pos=base,
        target_pos=target,
        active=jnp.ones(2, dtype=jnp.bool_),
        coverage=jnp.zeros(BUILDING.target_exclusion.shape, dtype=jnp.bool_),
        step=jnp.int32(0),
        key=jax.random.PRNGKey(0),
        directly_sees_target=jnp.asarray([False, False]),
        is_conn_base=jnp.asarray([True, True]),
        is_conn_target=jnp.asarray([False, False]),
        target_known=jnp.asarray([True, False]),
        success=jnp.bool_(False),
        base_target_connected=jnp.bool_(False),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(2, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(2),
        base_target_known=jnp.bool_(False),
    )
    delivered = previous._replace(
        is_conn_target=jnp.asarray([True, True]),
        target_known=jnp.asarray([True, True]),
        base_target_connected=jnp.bool_(True),
        base_target_known=jnp.bool_(True),
    )
    _, delivered_terms = compute_rewards(previous, delivered, reward_cfg, env_cfg)
    np.testing.assert_allclose(delivered_terms["target_delivery_reward"], [100.0, 100.0])
    np.testing.assert_allclose(delivered_terms["base_reporter_reward"], [50.0, 0.0])
    np.testing.assert_allclose(delivered_terms["success_reward"], [0.0, 0.0])

    # A later visual re-entry, and a reconnection after the base already
    # knows the target, cannot earn delivery or finder reward again.
    reentry = delivered._replace(
        directly_sees_target=jnp.asarray([False, True]),
        base_target_connected=jnp.bool_(False),
    )
    _, reentry_terms = compute_rewards(delivered, reentry, reward_cfg, env_cfg)
    np.testing.assert_allclose(reentry_terms["base_reporter_reward"], [0.0, 0.0])
    reconnected = reentry._replace(
        directly_sees_target=jnp.asarray([False, False]),
        base_target_connected=jnp.bool_(True),
    )
    _, reconnected_terms = compute_rewards(reentry, reconnected, reward_cfg, env_cfg)
    np.testing.assert_allclose(reconnected_terms["target_delivery_reward"], [0.0, 0.0])
    np.testing.assert_allclose(reconnected_terms["base_reporter_reward"], [0.0, 0.0])

    succeeded = delivered._replace(success=jnp.bool_(True))
    _, success_terms = compute_rewards(delivered, succeeded, reward_cfg, env_cfg)
    np.testing.assert_allclose(success_terms["success_reward"], [500.0, 500.0])
    np.testing.assert_allclose(success_terms["target_delivery_reward"], [0.0, 0.0])


def test_post_delivery_gap_credit_is_limited_to_relay_route():
    """A completed route keeps paying its relay drones during the hold."""
    env_cfg = EnvConfig(
        num_agents=4,
        comm_radius_base=4.0,
        comm_radius=4.0,
        visual_radius=4.0,
    )
    reward_cfg = RewardConfig(time_presure_reward=-1.25, closed_gap_reward_per_meter=0.125)
    state = EnvState(
        pos=jnp.asarray(
            [[4.5, 1.0, 1.0], [8.5, 1.0, 1.0], [12.5, 1.0, 1.0], [1.0, 4.5, 1.0]]
        ),
        vel=jnp.zeros((4, 3)),
        base_pos=jnp.asarray([1.0, 1.0, 1.0]),
        target_pos=jnp.asarray([15.0, 1.0, 1.0]),
        active=jnp.ones(4, dtype=jnp.bool_),
        coverage=jnp.zeros(BUILDING.target_exclusion.shape, dtype=jnp.bool_),
        step=jnp.int32(1),
        key=jax.random.PRNGKey(0),
        directly_sees_target=jnp.asarray([False, False, True, False]),
        is_conn_base=jnp.asarray([True, True, True, True]),
        is_conn_target=jnp.asarray([True, True, True, False]),
        target_known=jnp.asarray([True, True, True, False]),
        success=jnp.bool_(False),
        base_target_connected=jnp.bool_(True),
        chain_held_steps=jnp.int32(1),
        done=jnp.bool_(False),
        collided=jnp.zeros(4, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(4),
        base_target_known=jnp.bool_(True),
    )
    _, terms = compute_rewards(state, state, reward_cfg, env_cfg)
    np.testing.assert_allclose(terms["time_presure_reward"], [-1.25] * 4)
    np.testing.assert_allclose(terms["gap_reward"], [1.75] * 3 + [0.0])


def test_partial_chain_gap_credit_uses_geodesic_leaders():
    """Training reward must trace to the same leaders used to measure its gap."""
    env_cfg = EnvConfig(num_agents=5, comm_radius_base=5., comm_radius=5., visual_radius=5.)
    state = EnvState(
        pos=jnp.array([[4., 0., 0.], [4., 4., 0.], [16., 0., 0.],
                       [16., 4., 0.], [0., -4., 0.]]),
        vel=jnp.zeros((5, 3)),
        base_pos=jnp.array([0., 0., 0.]),
        target_pos=jnp.array([20., 0., 0.]),
        active=jnp.ones(5, dtype=bool),
        coverage=jnp.zeros(BUILDING.target_exclusion.shape, dtype=bool),
        step=jnp.int32(1),
        key=jax.random.PRNGKey(0),
        directly_sees_target=jnp.array([False, False, True, False, False]),
        is_conn_base=jnp.array([True, True, False, False, True]),
        is_conn_target=jnp.array([False, False, True, True, False]),
        target_known=jnp.ones(5, dtype=bool),
        success=jnp.bool_(False),
        base_target_connected=jnp.bool_(False),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(5, dtype=bool),
        coverage_credit=jnp.zeros(5),
        base_target_known=jnp.bool_(True),
    )
    # A corridor detour makes D1/D3 the roadmap leaders, while straight-line
    # distance to the opposite endpoint makes D0/D2 the Euclidean tips.
    route_positions = jnp.array([0., 28., 4., 8., 24., 20., -4.])
    roadmap_distances = jnp.abs(route_positions[:, None] - route_positions[None, :])
    gap, _, _, leaders = obstacle_chain_diagnostics(state, roadmap_distances)
    np.testing.assert_array_equal(leaders, [1, 3])
    np.testing.assert_allclose(gap, 12.)

    obstacle_cfg = RewardConfig(time_presure_reward=-1., closed_gap_reward_per_meter=0.125)
    _, obstacle_terms = compute_rewards(
        state, state, obstacle_cfg, env_cfg, roadmap_distances,
    )
    np.testing.assert_allclose(obstacle_terms["time_presure_reward"], [-1.] * 5)
    np.testing.assert_allclose(obstacle_terms["gap_reward"],
                               [2., 2., 2., 2., 0.], rtol=1e-6)
    _, longer_terms = compute_rewards(
        state, state, obstacle_cfg, env_cfg, roadmap_distances * 2.,
    )
    np.testing.assert_allclose(longer_terms["gap_reward"],
                               [4., 4., 4., 4., 0.], rtol=1e-6)
    _, unknown_terms = compute_rewards(
        state._replace(base_target_known=jnp.bool_(False)),
        state._replace(base_target_known=jnp.bool_(False)),
        obstacle_cfg, env_cfg, roadmap_distances,
    )
    np.testing.assert_array_equal(unknown_terms["gap_reward"], [0.] * 5)
    np.testing.assert_allclose(unknown_terms["time_presure_reward"], [-1.] * 5)

    # With no target-connected drone, the target endpoint is leader -1.
    # Only the route from the base to D1 earns the partial-chain bonus.
    no_target_front = state._replace(
        directly_sees_target=jnp.zeros(5, dtype=bool),
        is_conn_target=jnp.zeros(5, dtype=bool),
    )
    _, _, _, leaders = obstacle_chain_diagnostics(no_target_front, roadmap_distances)
    np.testing.assert_array_equal(leaders, [1, -1])
    _, endpoint_terms = compute_rewards(
        no_target_front, no_target_front, obstacle_cfg, env_cfg, roadmap_distances,
    )
    np.testing.assert_allclose(endpoint_terms["gap_reward"],
                               [1., 1., 0., 0., 0.], rtol=1e-6)



def test_time_limit_autoreset_preserves_terminal_info_and_changes_target():
    cfg = replace(EnvConfig(), max_steps=1)
    reset, autoreset_step, _, _ = make_autoreset_fns(BUILDING, cfg)
    state = reset(jax.random.PRNGKey(99))
    next_state, reward, done, info = autoreset_step(
        state,
        jnp.zeros((cfg.num_agents, 3)),
    )
    assert done
    assert info["done"]
    assert info["terminal_target_pos"].shape == (3,)
    assert info["global_target_found"].shape == ()
    assert info["chain_gap_dist"].shape == ()
    assert info["chain_progress_pct"].shape == ()
    assert info["time_presure_reward"].shape == (cfg.num_agents,)
    assert info["gap_reward"].shape == (cfg.num_agents,)
    assert info["global_coverage"].shape == ()
    assert reward.shape == (cfg.num_agents,)
    assert next_state.step == 0
    assert not next_state.done


def test_batched_autoreset_matches_reset_only_for_terminal_lanes():
    cfg = replace(EnvConfig(), num_agents=1, max_steps=10,
                  hold_chain_for=1)
    reward_cfg = RewardConfig()
    reset, plain_step, _, _ = make_env_fns(BUILDING, cfg)
    _, autoreset_step, _, _ = make_autoreset_fns(BUILDING, cfg, reward_cfg)
    states = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(823), 4))
    # Ongoing, time limit, ongoing stationary, and successful chain.
    states = states._replace(
        step=jnp.array([0, 9, 0, 0], dtype=jnp.int32),
        stationary_steps=jnp.array([0, 0, 4, 0], dtype=jnp.int32),
        target_pos=states.target_pos.at[3].set(states.pos[3, 0] + jnp.array([1., 0., 0.])),
    )
    actions = jnp.zeros((4, 1, 3))
    actual, rewards, done, info = jax.jit(autoreset_step.batched)(states, actions)
    np.testing.assert_array_equal(done, [False, True, False, True])
    for lane in range(4):
        previous = jax.tree_util.tree_map(lambda a: a[lane], states)
        terminal = plain_step(previous, actions[lane])
        expected = terminal
        if bool(terminal.done):
            expected = reset(terminal.key, obstacle_min=terminal.obstacle_min,
                             obstacle_max=terminal.obstacle_max,
                             stored_vertices=terminal.roadmap_vertices,
                             stored_distances=terminal.roadmap_distances)
        current = jax.tree_util.tree_map(lambda a: a[lane], actual)
        for got, want in zip(jax.tree_util.tree_leaves(current), jax.tree_util.tree_leaves(expected)):
            if np.issubdtype(np.asarray(got).dtype, np.floating):
                np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
            else:
                np.testing.assert_array_equal(got, want)
        expected_reward, _ = compute_rewards(previous, terminal, reward_cfg, cfg)
        np.testing.assert_allclose(rewards[lane], expected_reward, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(info["terminal_target_pos"][lane], terminal.target_pos)
        np.testing.assert_array_equal(current.obstacle_min, previous.obstacle_min)


@pytest.mark.parametrize("terminal_count", [0, 1, 17, 18, 35])
def test_sparse_and_dense_reset_paths_match_reference_over_multiple_steps(terminal_count):
    cfg = replace(EnvConfig(), num_agents=1, max_steps=10,
                  hold_chain_for=50)
    reward_cfg = RewardConfig()
    reset, step, _, _ = make_autoreset_fns(BUILDING, cfg, reward_cfg)
    states = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(436), 35))
    # Put terminal lanes at the end to exercise sparse gather/scatter indices.
    states = states._replace(step=jnp.where(jnp.arange(35) >= 35-terminal_count, 9, 0).astype(jnp.int32))
    actions = jnp.zeros((35, 1, 3))
    reference_step = jax.jit(jax.vmap(step))
    optimized_step = jax.jit(step.batched)
    reference = optimized = states
    for iteration in range(7):  # Cross multiple episode/key boundaries, not just one reset.
        reference_out = reference_step(reference, actions)
        optimized_out = optimized_step(optimized, actions)
        names = ("state", "reward", "done", "info")
        actual_leaves, actual_tree = jax.tree_util.tree_flatten_with_path(dict(zip(names, optimized_out)))
        expected_leaves, expected_tree = jax.tree_util.tree_flatten_with_path(dict(zip(names, reference_out)))
        assert actual_tree == expected_tree
        for (path, got), (_, want) in zip(actual_leaves, expected_leaves):
            field = jax.tree_util.keystr(path)
            message = f"transition {iteration}, {field}"
            if np.issubdtype(np.asarray(got).dtype, np.floating):
                # 100 * (1 - gap / mission) amplifies float32 rounding near
                # zero progress across separately compiled execution paths.
                # Allow two float32 epsilons on the percentage's 100-unit
                # scale, only for this diagnostic; keep state/rewards strict.
                atol = (2 * np.finfo(np.float32).eps * 100
                        if field == "['info']['chain_progress_pct']" else 1e-6)
                if field == "['info']['chain_route_excess']":
                    # Route length minus mission length cancels metre-scale
                    # operands. Relative error in the tiny remainder is not
                    # meaningful; permit 4 micrometres of absolute rounding.
                    # Frontier indices (which use excess to break ties) must
                    # still match exactly, as must all other integer fields.
                    atol = 4e-6
                np.testing.assert_allclose(got, want, rtol=1e-6, atol=atol, err_msg=message)
            else:
                np.testing.assert_array_equal(got, want, err_msg=message)
        reference, optimized = reference_out[0], optimized_out[0]


def test_chain_efficiency_rewards_one_path_and_gap_credit_splits_ties():
    from swarmecho.env.environment import simple_chain_paths, chain_path_efficiencies

    cfg = replace(EnvConfig(), num_agents=4, comm_radius=2.3,
                  comm_radius_base=2.1, visual_radius=2.3, observe_chain_contributor=True)
    reset, _, _, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(12))._replace(
        base_pos=jnp.array([1., 1., 2.]), target_pos=jnp.array([7., 1., 2.]),
        pos=jnp.array([[3., 1., 2.], [5., 2., 2.], [5., 0., 2.], [3., 3.3, 2.]]),
        active=jnp.ones(4, dtype=bool), target_known=jnp.ones(4, dtype=bool),
        is_conn_base=jnp.ones(4, dtype=bool), is_conn_target=jnp.ones(4, dtype=bool),
        directly_sees_target=jnp.array([False, True, True, False]),
        base_target_connected=jnp.bool_(True), base_target_known=jnp.bool_(True),
    )
    # D0 links the base to alternate relays D1/D2. D3 is only a dead-end
    # neighbour of D0; walking out to it and back must never count as a path.
    paths = simple_chain_paths(state, cfg)
    assert int(jnp.sum(paths.counts)) == 4
    reference = jnp.zeros((6, 6)).at[0, 1].set(6.)
    expected = 6. / (2. + 2. * np.sqrt(5.))
    np.testing.assert_allclose(jnp.max(chain_path_efficiencies(paths, 6.)), expected)
    reward_cfg = RewardConfig(time_presure_reward=-1.25, closed_gap_reward_per_meter=.125,
                              chain_efficiency_reward=.125)
    reward, terms = compute_rewards(state, state, reward_cfg, cfg, reference)
    np.testing.assert_allclose(terms["time_presure_reward"], [-1.25] * 4)
    np.testing.assert_allclose(terms["gap_reward"], [.75, .375, .375, 0.])
    credit = np.asarray(terms["chain_efficiency_reward"])
    np.testing.assert_allclose(credit[[0, 3]], [expected * .125, 0.])
    np.testing.assert_allclose(np.sort(credit[1:3]), [0., expected * .125])
    np.testing.assert_allclose(reward, terms["time_presure_reward"] + terms["gap_reward"] + credit)
    _, disconnected = compute_rewards(
        state, state._replace(base_target_connected=jnp.bool_(False)), reward_cfg, cfg, reference,
    )
    np.testing.assert_array_equal(disconnected["chain_efficiency_reward"], np.zeros(4))
    _, _, observe, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    np.testing.assert_array_equal(observe(state)[:, 6], [1., 1., 0., 0.])
    straight = state._replace(pos=state.pos.at[1].set(jnp.array([5., 1., 2.])))
    np.testing.assert_allclose(jnp.max(chain_path_efficiencies(simple_chain_paths(straight, cfg), 6.)), 1.)


def test_autoreset_step_uses_the_configured_reward_terms():
    cfg = replace(EnvConfig(), max_steps=1)
    zero_reward = RewardConfig(
        exploration_reward=0.0,
        collision_reward=0.0,
        base_reporter_reward=0.0,
        time_presure_reward=0.0,
        closed_gap_reward_per_meter=0.0,
        target_delivery_reward=0.0,
        success_reward=0.0,
    )
    reset, autoreset_step, _, _ = make_autoreset_fns(
        BUILDING, cfg, zero_reward
    )
    _, reward, _, _ = autoreset_step(
        reset(jax.random.PRNGKey(123)),
        jnp.zeros((cfg.num_agents, 3)),
    )
    np.testing.assert_allclose(reward, 0.0)


def test_chain_diagnostics_are_zero_gap_and_full_progress_for_a_complete_chain():
    state = EnvState(
        pos=jnp.asarray([[4.0, 1.0, 1.0], [8.0, 1.0, 1.0]]),
        vel=jnp.zeros((2, 3)),
        base_pos=jnp.asarray([1.0, 1.0, 1.0]),
        target_pos=jnp.asarray([11.0, 1.0, 1.0]),
        active=jnp.ones(2, dtype=jnp.bool_),
        coverage=jnp.zeros(BUILDING.target_exclusion.shape, dtype=jnp.bool_),
        step=jnp.int32(0),
        key=jax.random.PRNGKey(0),
        directly_sees_target=jnp.zeros(2, dtype=jnp.bool_),
        is_conn_base=jnp.asarray([True, True]),
        is_conn_target=jnp.asarray([True, True]),
        target_known=jnp.asarray([True, True]),
        success=jnp.bool_(False),
        base_target_connected=jnp.bool_(True),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(2, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(2),
    )
    gap, progress, _, _ = obstacle_chain_diagnostics(state)
    assert gap == 0.0
    assert progress == 100.0
    state = state._replace(directly_sees_target=jnp.asarray([False, True]))
    assert final_chain_length(state, EnvConfig(num_agents=2)) == pytest.approx(10.0)


# Adaptive target sampling

ADAPTIVE_SPAWN_BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/custom_maps/g01_static_map.yaml"))
ENV = replace(EnvConfig(), num_agents=1, success_condition="chain_held", max_steps=3)


def controller(config=None):
    return AdaptiveTargetSpawnController(ADAPTIVE_SPAWN_BUILDING, ENV, config or AdaptiveTargetSpawnConfig(enabled=True))


def cell_points(curriculum):
    lower, upper, _ = target_spawn_boxes(ADAPTIVE_SPAWN_BUILDING, ENV)
    centres = (lower + upper) / 2
    return np.asarray([centres[np.flatnonzero(curriculum.box_cell_indices == i)[0]]
                       for i in range(len(curriculum.cells))])


def assert_same_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(a, b)


def test_volume_baseline_floor_redistribution_and_two_smoothing_stages():
    c = controller()
    points = cell_points(c)
    _, _, volumes = target_spawn_boxes(ADAPTIVE_SPAWN_BUILDING, ENV)
    assert len(c.cells) == 23
    np.testing.assert_allclose(c.box_weights, volumes / volumes.sum(), rtol=2e-7)

    # A half-success cell loses probability, while untouched cells stay at zero success.
    c.observe(np.repeat(points[:1], 10, axis=0), [1] * 5 + [0] * 5)
    assert c.rates[0] == .5 and c.rates[1] == 0
    c.observe(np.empty((0, 3)), [])
    assert c.rates[0] == .5
    c.advance(1, 100)
    assert c.probabilities[0] < c.baseline[0]

    c = controller()
    c.observe(np.repeat(points, 500, axis=0), np.repeat([0] + [1] * 22, 500))
    expected = .2 * c.baseline.copy()
    expected[0] += .8
    start = c.probabilities.copy()
    for update in range(1, 4):
        c.advance(update, update * 100)
    np.testing.assert_allclose(c.probabilities, (start + expected) / 2, atol=1e-12)

    # Replace just the difficult cell's last 500 outcomes: other windows are unchanged.
    c.observe(np.repeat(points[:1], 500, axis=0), [0] * 250 + [1] * 250)
    expected = .6 * c.baseline.copy()
    expected[0] += .4
    start = c.probabilities.copy()
    for update in range(4, 7):
        row, _ = c.advance(update, update * 100)
        assert np.all(np.asarray(row["probabilities"]) >= .2 * c.baseline - 1e-12)
    np.testing.assert_allclose(c.probabilities, (start + expected) / 2, atol=1e-12)
    np.testing.assert_allclose(np.bincount(c.box_cell_indices, weights=c.box_weights),
                               c.probabilities, rtol=2e-7)

    c.observe(np.repeat(points[:1], 500, axis=0), [1] * 500)
    start = c.probabilities.copy()
    for update in range(7, 10):
        c.advance(update, update * 100)
    np.testing.assert_allclose(c.probabilities, (start + c.baseline) / 2, atol=1e-12)


def test_dynamic_weights_reach_jitted_resets_without_retracing():
    c = controller(replace(AdaptiveTargetSpawnConfig(enabled=True), min_probability_fraction=0.,
                           probability_smoothing_half_life_updates=1e-6))
    reset, _, _, _ = make_env_fns(ADAPTIVE_SPAWN_BUILDING, ENV, plan_geodesic=False)
    traces = []

    def draw(keys, weights):
        traces.append(True)
        return jax.vmap(lambda key: reset(key, target_spawn_weights=weights).target_pos)(keys)

    draw_jit = jax.jit(draw)
    keys = jax.random.split(jax.random.PRNGKey(117), 64)
    for update, selected in enumerate((0, len(c.cells) - 1), start=1):
        # Two limiting distributions from the real controller, using the same compiled sampler.
        c.rates[:] = 1.
        c.rates[selected] = 0.
        c.advance(update, update * 100)
        targets = draw_jit(keys, jnp.asarray(c.box_weights))
        cells = np.floor(np.asarray(targets) / c.cell_size).astype(int)
        np.testing.assert_array_equal(cells, np.repeat(c.cells[selected:selected + 1], 64, axis=0))
    assert len(traces) == 1


def test_disabled_reset_keeps_the_legacy_keyed_volume_draw():
    reset, _, _, _ = make_env_fns(ADAPTIVE_SPAWN_BUILDING, ENV, plan_geodesic=False)
    lower, upper, volumes = map(jnp.asarray, target_spawn_boxes(ADAPTIVE_SPAWN_BUILDING, ENV))

    def legacy_first_draw(key):
        # Freeze the pre-feature PRNG contract independently of reset/autoreset wiring.
        target_key = jax.random.split(key, 3)[1]
        first_key = jax.random.split(target_key)[1]
        box_key, point_key = jax.random.split(first_key)
        box = jax.random.categorical(box_key, jnp.log(volumes))
        return jax.random.uniform(point_key, (3,), minval=lower[box], maxval=upper[box])

    keys = jax.random.split(jax.random.PRNGKey(883), 64)
    default = jax.jit(jax.vmap(reset))(keys)
    explicit_none = jax.jit(jax.vmap(lambda key: reset(key, target_spawn_weights=None)))(keys)
    assert_same_tree(default, explicit_none)
    np.testing.assert_array_equal(default.target_pos, jax.jit(jax.vmap(legacy_first_draw))(keys))


@pytest.mark.parametrize("sparse", [False, True])
def test_weights_reach_dense_and_sparse_autoresets_and_disabled_results_match(sparse):
    c = controller()
    reset, step, _, _ = make_autoreset_fns(ADAPTIVE_SPAWN_BUILDING, ENV, RewardConfig())
    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(227), 4))
    # A 50-step chain hold cannot finish here: only selected lanes reach the time limit.
    steps = jnp.asarray([2, 0, 0, 0] if sparse else [2, 2, 2, 2], dtype=jnp.int32)
    states = states._replace(step=steps)
    actions = jnp.zeros((4, ENV.num_agents, 3))
    default = jax.jit(step.batched)(states, actions)
    explicit_none = jax.jit(lambda s, a: step.batched(s, a, None))(states, actions)
    scalar_reference = jax.jit(jax.vmap(step))(states, actions)
    assert_same_tree(default, explicit_none)
    assert_same_tree(default, scalar_reference)

    selected = len(c.cells) - 1
    weights = np.where(c.box_cell_indices == selected,
                       c.box_volumes / c.cell_volumes[selected], 0.).astype(np.float32)
    batched = jax.jit(step.batched)
    updated, _, done, _ = batched(states, actions, jnp.asarray(weights))
    finished = np.asarray(done, dtype=bool)
    assert finished.sum() == (1 if sparse else 4)
    cells = np.floor(np.asarray(updated.target_pos)[finished] / c.cell_size).astype(int)
    np.testing.assert_array_equal(cells, np.repeat(c.cells[selected:selected + 1], finished.sum(), axis=0))
    np.testing.assert_array_equal(np.asarray(updated.target_pos)[~finished], np.asarray(states.target_pos)[~finished])


def test_controller_checkpoint_restores_windows_and_next_update(tmp_path):
    original = controller()
    original.observe(cell_points(original)[[0, 1]], [1, 0])
    original.advance(1, 100)
    original.save(tmp_path)
    restored = controller()
    assert restored.restore(tmp_path)
    assert original.advance(2, 200) == restored.advance(2, 200)
    np.testing.assert_array_equal(original.box_weights, restored.box_weights)
