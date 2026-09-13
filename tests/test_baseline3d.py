from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.env.baseline3d import (
    Baseline3DConfig,
    Baseline3DRewardConfig,
    Baseline3DState,
    chain_diagnostics_3d,
    coverage_grid_geometry,
    final_chain_length_3d,
    make_autoreset_3d_fns,
    make_baseline_3d_fns,
    maximum_chain_distance,
    maximum_five_drone_chain_distance,
    observation_dim_3d,
    obstacle_chain_diagnostics_3d,
    rewards_3d,
    spherical_directions,
)
from swarmecho.env.buildings import load_building


BUILDING = load_building(
    Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")
)


def test_authored_geodesic_roadmap_is_shared_through_batched_autoreset():
    building = replace(
        BUILDING,
        solid_min_m=np.array([[9., 5., 0.]], dtype=np.float32),
        solid_max_m=np.array([[11., 15., 20.]], dtype=np.float32),
    )
    cfg = replace(Baseline3DConfig(), num_agents=1, visual_radius=1.,
                  comm_radius=1., comm_radius_base=1., max_steps=1)
    reward_cfg = Baseline3DRewardConfig(chain_reward_system="obstacle_geodesic")
    reset, step, _, _ = make_autoreset_3d_fns(building, cfg, reward_cfg)
    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(21), 2))
    assert states.obstacle_min.shape == (2, 0, 3)
    assert states.roadmap_distances.shape == (2, 0, 0)
    assert states.shared_geometry.distances.shape[0] > 0
    state = jax.tree_util.tree_map(lambda value: value[0], states)._replace(
        base_pos=jnp.array([5., 10., 10.]), target_pos=jnp.array([15., 10., 10.]),
        is_conn_base=jnp.array([False]), is_conn_target=jnp.array([False]),
        fully_connected=jnp.bool_(False),
    )
    gap = jax.jit(obstacle_chain_diagnostics_3d)(state)[0]
    assert 10. < float(gap) < 1e6
    next_states, _, done, _ = jax.jit(step.batched)(states, jnp.zeros((2, 1, 3)))
    assert np.all(done)
    assert next_states.shared_geometry is states.shared_geometry
    assert next_states.roadmap_distances.shape == (2, 0, 0)


def test_dynamic_roadmap_combines_authored_and_persisted_layout_geometry():
    building = replace(
        BUILDING,
        solid_min_m=np.array([[9., 5., 0.]], dtype=np.float32),
        solid_max_m=np.array([[11., 15., 20.]], dtype=np.float32),
    )
    cfg = replace(Baseline3DConfig(), num_agents=1, num_obstacles=1)
    reset, _, _, _ = make_baseline_3d_fns(building, cfg)
    lo, hi = jnp.array([[2., 2., 2.]]), jnp.array([[3., 3., 3.]])
    state = reset(jax.random.PRNGKey(4), target_pos=jnp.array([15., 10., 10.]),
                  obstacle_min=lo, obstacle_max=hi)
    # Authored edges are sampled; the generated cuboid still adds eight corners.
    assert state.roadmap_vertices.shape[1] == 3
    assert state.roadmap_vertices.shape[0] > 8
    assert state.solid_min.shape == (2, 3)
    np.testing.assert_allclose(state.solid_min[0], building.solid_min_m[0])
    assert state.shared_geometry.distances.shape == (0, 0)
    query = state._replace(
        base_pos=jnp.array([5., 10., 10.]),
        is_conn_base=jnp.array([False]), is_conn_target=jnp.array([False]),
        fully_connected=jnp.bool_(False),
    )
    assert 10. < float(obstacle_chain_diagnostics_3d(query)[0]) < 1e6
    # Legacy generated-only graph arrays cannot bypass the authored walls.
    legacy = reset(jax.random.PRNGKey(5), obstacle_min=lo, obstacle_max=hi,
                   stored_vertices=jnp.zeros((8, 3)), stored_distances=jnp.zeros((8, 8)))
    np.testing.assert_allclose(legacy.roadmap_distances, state.roadmap_distances)
    restored = reset(jax.random.PRNGKey(5), obstacle_min=lo, obstacle_max=hi,
                     stored_vertices=state.roadmap_vertices,
                     stored_distances=state.roadmap_distances)
    np.testing.assert_allclose(restored.roadmap_distances, state.roadmap_distances)


def test_combined_chain_query_preserves_original_frontier_selection():
    from swarmecho.env.obstacles3d import free_space_distance

    cfg = replace(Baseline3DConfig(), num_agents=3, num_obstacles=1,
                  obstacle_spawn_layer_min=1, obstacle_spawn_layer_max=2)
    reset, _, _, _ = make_baseline_3d_fns(BUILDING, cfg)
    state = reset(jax.random.PRNGKey(51), target_pos=jnp.array([15., 10., 5.]),
                  obstacle_min=jnp.array([[9., 5., 0.]]),
                  obstacle_max=jnp.array([[11., 15., 10.]]))
    state = state._replace(base_pos=jnp.array([5., 10., 5.]),
                           pos=jnp.array([[6., 10., 5.], [12., 16., 5.], [14., 10., 5.]]),
                           is_conn_base=jnp.array([True, False, False]),
                           is_conn_target=jnp.array([False, True, True]),
                           target_known=jnp.array([False, True, True]),
                           fully_connected=jnp.bool_(False))
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
    actual = jax.jit(obstacle_chain_diagnostics_3d)(state)
    for got, want in zip(actual[:3], expected[:3]):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[3], expected[3])


def test_euclidean_factory_does_not_build_authored_or_dynamic_roadmaps():
    building = replace(
        BUILDING,
        solid_min_m=np.array([[9., 5., 0.]], dtype=np.float32),
        solid_max_m=np.array([[11., 15., 20.]], dtype=np.float32),
    )
    cfg = replace(Baseline3DConfig(), num_agents=1, num_obstacles=1)
    reset, _, _, _ = make_baseline_3d_fns(building, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(4), target_pos=jnp.array([15., 10., 10.]),
                  obstacle_min=jnp.array([[2., 2., 2.]]),
                  obstacle_max=jnp.array([[3., 3., 3.]]))
    assert state.roadmap_distances.shape == (0, 0)
    assert state.shared_geometry.distances.shape == (0, 0)
    assert state.solid_min.shape == (2, 3)


def _functions(**overrides):
    cfg = replace(Baseline3DConfig(), **overrides)
    return cfg, make_baseline_3d_fns(BUILDING, cfg)


def test_octant_radar_and_configured_distance_contracts():
    directions = spherical_directions(8)
    assert directions.shape == (8, 3)
    assert len({tuple(np.sign(row)) for row in directions}) == 8
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    cfg = Baseline3DConfig()
    assert maximum_five_drone_chain_distance(cfg) == 30.0
    assert maximum_chain_distance(replace(cfg, num_agents=6)) == 35.0


def test_optional_observation_features_match_the_maintained_config_switches():
    cfg, (reset, _, observations, _) = _functions(
        observe_base_vector=True,
        observe_target_vector=True,
        observe_coverage_probe=True,
    )
    obs = observations(reset(jax.random.PRNGKey(7)))

    assert obs.shape == (cfg.num_agents, observation_dim_3d(cfg))


@pytest.mark.parametrize("bins", [8, 16, 32])
def test_reset_observation_and_step_jit_contract(bins):
    cfg, (reset, step, observations, metrics) = _functions(radar_bins=bins)
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
        comm_radius_base=500., comm_radius=500., target_spawn_buffer=500.
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
    )
    base = jnp.asarray([1.0, 1.0, 1.0])
    # 3.5 base hop, four 2.5 communication hops, then a 1.5 visual hop.
    x = jnp.asarray([4.5, 7.0, 9.5, 12.0, 14.5])
    positions = jnp.stack([x, jnp.ones(5), jnp.ones(5)], axis=-1)
    target = jnp.asarray([16.0, 1.0, 1.0])
    state = Baseline3DState(
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
        fully_connected=jnp.bool_(False),
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
    state = Baseline3DState(
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
        fully_connected=jnp.bool_(False),
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
    cfg, (reset, step, observations, _) = _functions(
        coverage_voxel_size=2.5,
        observe_coverage_probe=True,
    )
    state = reset(jax.random.PRNGKey(11))
    next_state = step(state, jnp.zeros((cfg.num_agents, 3)))

    voxel_size, shape = coverage_grid_geometry(BUILDING, cfg)
    assert voxel_size == 2.5
    assert shape == (8, 8, 8)
    assert state.coverage.shape == shape
    assert next_state.coverage.shape == shape
    assert observations(state).shape == (cfg.num_agents, observation_dim_3d(cfg))


def test_coverage_voxel_size_must_tile_the_building():
    with pytest.raises(ValueError, match="evenly divide"):
        make_baseline_3d_fns(BUILDING, Baseline3DConfig(coverage_voxel_size=3.0))


def test_reward_terms_preserve_local_credit_and_shared_events():
    cfg, (reset, step, _, _) = _functions()
    previous = reset(jax.random.PRNGKey(7))
    current = step(previous, jnp.ones((cfg.num_agents, 3)))
    reward, terms = rewards_3d(previous, current, Baseline3DRewardConfig(), cfg)
    assert reward.shape == (cfg.num_agents,)
    assert terms["coverage"].shape == (cfg.num_agents,)
    assert terms["collision"].shape == (cfg.num_agents,)
    assert terms["chain_gap"].shape == (cfg.num_agents,)
    assert jnp.isfinite(reward).all()
    assert jnp.sum(current.coverage_credit) == pytest.approx(
        jnp.sum(current.coverage & ~previous.coverage), abs=1e-5
    )


def test_target_delivery_and_success_are_distinct_one_shot_rewards():
    """3D must retain 2D's delivery milestone and resist finder farming."""
    reward_cfg = Baseline3DRewardConfig(
        target_found_requires_delivery=True,
        finder_bonus=50.0,
        target_found_bonus=100.0,
        success_bonus=500.0,
    )
    base = jnp.asarray([1.0, 1.0, 1.0])
    target = jnp.asarray([11.0, 1.0, 1.0])
    env_cfg = Baseline3DConfig(num_agents=2, comm_radius_base=6.0)
    previous = Baseline3DState(
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
        fully_connected=jnp.bool_(False),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(2, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(2),
        base_target_known=jnp.bool_(False),
    )
    delivered = previous._replace(
        is_conn_target=jnp.asarray([True, True]),
        target_known=jnp.asarray([True, True]),
        fully_connected=jnp.bool_(True),
        base_target_known=jnp.bool_(True),
    )
    _, delivered_terms = rewards_3d(previous, delivered, reward_cfg, env_cfg)
    np.testing.assert_allclose(delivered_terms["target_found"], [50.0, 50.0])
    np.testing.assert_allclose(delivered_terms["finder"], [50.0, 0.0])
    np.testing.assert_allclose(delivered_terms["success"], [0.0, 0.0])

    # A later visual re-entry, and a reconnection after the base already
    # knows the target, cannot earn delivery or finder reward again.
    reentry = delivered._replace(
        directly_sees_target=jnp.asarray([False, True]),
        fully_connected=jnp.bool_(False),
    )
    _, reentry_terms = rewards_3d(delivered, reentry, reward_cfg, env_cfg)
    np.testing.assert_allclose(reentry_terms["finder"], [0.0, 0.0])
    reconnected = reentry._replace(
        directly_sees_target=jnp.asarray([False, False]),
        fully_connected=jnp.bool_(True),
    )
    _, reconnected_terms = rewards_3d(reentry, reconnected, reward_cfg, env_cfg)
    np.testing.assert_allclose(reconnected_terms["target_found"], [0.0, 0.0])
    np.testing.assert_allclose(reconnected_terms["finder"], [0.0, 0.0])

    succeeded = delivered._replace(success=jnp.bool_(True))
    _, success_terms = rewards_3d(delivered, succeeded, reward_cfg, env_cfg)
    np.testing.assert_allclose(success_terms["success"], [250.0, 250.0])
    np.testing.assert_allclose(success_terms["target_found"], [0.0, 0.0])


def test_post_delivery_gap_credit_is_limited_to_relay_route():
    """Match 2D: non-route drones retain the maximum gap penalty."""
    env_cfg = Baseline3DConfig(
        num_agents=4,
        comm_radius_base=4.0,
        comm_radius=4.0,
        visual_radius=4.0,
    )
    reward_cfg = Baseline3DRewardConfig(max_gap_penalty=5.0)
    state = Baseline3DState(
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
        fully_connected=jnp.bool_(True),
        chain_held_steps=jnp.int32(1),
        done=jnp.bool_(False),
        collided=jnp.zeros(4, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(4),
        base_target_known=jnp.bool_(True),
    )
    _, terms = rewards_3d(state, state, reward_cfg, env_cfg)
    np.testing.assert_allclose(terms["chain_gap"][:3], 0.0)
    np.testing.assert_allclose(terms["chain_gap"][3], -1.25)


def test_time_limit_autoreset_preserves_terminal_info_and_changes_target():
    cfg = replace(Baseline3DConfig(), max_steps=1)
    reset, autoreset_step, _, _ = make_autoreset_3d_fns(BUILDING, cfg)
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
    assert info["global_coverage"].shape == ()
    assert reward.shape == (cfg.num_agents,)
    assert next_state.step == 0
    assert not next_state.done


def test_idle_termination_applies_configured_penalty():
    cfg = replace(
        Baseline3DConfig(),
        num_agents=1,
        max_steps=10,
        no_movement_termination_steps=1,
    )
    reward_cfg = Baseline3DRewardConfig(
        exploration_bonus=0.0,
        collision_penalty=0.0,
        finder_bonus=0.0,
        max_gap_penalty=0.0,
        target_found_bonus=0.0,
        success_bonus=0.0,
        no_movement_termination_penalty=-300.0,
    )
    reset, autoreset_step, _, _ = make_autoreset_3d_fns(BUILDING, cfg, reward_cfg)
    _, reward, done, info = autoreset_step(
        reset(jax.random.PRNGKey(101)),
        jnp.zeros((cfg.num_agents, 3)),
    )

    assert done
    assert info["idle_terminated"]
    np.testing.assert_allclose(reward, [-300.0])
    np.testing.assert_allclose(info["no_movement_termination"], [-300.0])


@pytest.mark.parametrize("obstacles", [0, 1])
def test_batched_autoreset_matches_reset_only_for_terminal_lanes(obstacles):
    cfg = replace(Baseline3DConfig(), num_agents=1, max_steps=10,
                  no_movement_termination_steps=5, hold_chain_for=1,
                  num_obstacles=obstacles, obstacle_spawn_layer_min=1,
                  obstacle_spawn_layer_max=2)
    reward_cfg = Baseline3DRewardConfig()
    reset, plain_step, _, _ = make_baseline_3d_fns(BUILDING, cfg, plan_geodesic=False)
    _, autoreset_step, _, _ = make_autoreset_3d_fns(BUILDING, cfg, reward_cfg)
    states = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(823), 4))
    # Ongoing, time limit, idle termination, successful chain respectively.
    states = states._replace(
        step=jnp.array([0, 9, 0, 0], dtype=jnp.int32),
        stationary_steps=jnp.array([0, 0, 4, 0], dtype=jnp.int32),
        target_pos=states.target_pos.at[3].set(states.pos[3, 0] + jnp.array([1., 0., 0.])),
    )
    actions = jnp.zeros((4, 1, 3))
    actual, rewards, done, info = jax.jit(autoreset_step.batched)(states, actions)
    np.testing.assert_array_equal(done, [False, True, True, True])
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
        expected_reward, _ = rewards_3d(previous, terminal, reward_cfg, cfg)
        np.testing.assert_allclose(rewards[lane], expected_reward, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(info["terminal_target_pos"][lane], terminal.target_pos)
        np.testing.assert_array_equal(current.obstacle_min, previous.obstacle_min)


@pytest.mark.parametrize("terminal_count", [0, 1, 17, 18, 35])
@pytest.mark.parametrize("geodesic", [False, True])
def test_sparse_and_dense_reset_paths_match_reference_over_multiple_steps(terminal_count, geodesic):
    cfg = replace(Baseline3DConfig(), num_agents=1, max_steps=10,
                  no_movement_termination_steps=5, hold_chain_for=50,
                  num_obstacles=1, obstacle_spawn_layer_min=1, obstacle_spawn_layer_max=2)
    reward_cfg = Baseline3DRewardConfig(chain_reward_system="obstacle_geodesic" if geodesic else "euclidean")
    reset, step, _, _ = make_autoreset_3d_fns(BUILDING, cfg, reward_cfg)
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


def test_redundancy_and_efficiency_reward_modes():
    from swarmecho.env.baseline3d import simple_chain_paths_3d, chain_path_efficiencies_3d

    cfg = replace(Baseline3DConfig(), num_agents=4, comm_radius=2.3,
                  comm_radius_base=2.1, visual_radius=2.3, observe_chain_contributor=True)
    reset, _, _, _ = make_baseline_3d_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(12))._replace(
        base_pos=jnp.array([1., 1., 2.]), target_pos=jnp.array([7., 1., 2.]),
        pos=jnp.array([[3., 1., 2.], [5., 2., 2.], [5., 0., 2.], [3., 3.3, 2.]]),
        active=jnp.ones(4, dtype=bool), target_known=jnp.ones(4, dtype=bool),
        is_conn_base=jnp.ones(4, dtype=bool), is_conn_target=jnp.ones(4, dtype=bool),
        directly_sees_target=jnp.array([False, True, True, False]),
        fully_connected=jnp.bool_(True), base_target_known=jnp.bool_(True),
    )
    # D0 links the base to alternate relays D1/D2. D3 is only a dead-end
    # neighbour of D0; walking out to it and back must never count as a path.
    paths = simple_chain_paths_3d(state, cfg)
    assert int(jnp.sum(paths.counts)) == 4
    reference = jnp.zeros((6, 6)).at[0, 1].set(6.)
    expected = 6. / (2. + 2. * np.sqrt(5.))
    np.testing.assert_allclose(jnp.max(chain_path_efficiencies_3d(paths, 6.)), expected)
    for redundancy in (False, True):
        for efficiency in (False, True):
            reward_cfg = Baseline3DRewardConfig(allow_redundancy_reward=redundancy,
                                               enable_chain_efficiency_reward=efficiency)
            reward, terms = rewards_3d(state, state, reward_cfg, cfg, reference)
            np.testing.assert_allclose(terms["chain_gap"], [0., 0., 0. if redundancy else -1.25, -1.25])
            credit = np.array([expected, expected, expected if redundancy else 0., 0.]) * .5 / 4
            if efficiency:
                np.testing.assert_allclose(terms["efficiency"], credit, rtol=1e-6)
                _, disconnected = rewards_3d(state, state._replace(fully_connected=jnp.bool_(False)), reward_cfg, cfg, reference)
                np.testing.assert_array_equal(disconnected["efficiency"], np.zeros(4))
            else:
                assert "efficiency" not in terms
                credit = np.zeros(4)
            np.testing.assert_allclose(reward, terms["chain_gap"] + credit, rtol=1e-6)
        _, _, observe, _ = make_baseline_3d_fns(BUILDING, cfg, plan_geodesic=False,
                                              allow_redundancy_reward=redundancy)
        np.testing.assert_array_equal(observe(state)[:, 6], [1., 1., float(redundancy), 0.])
    straight = state._replace(pos=state.pos.at[1].set(jnp.array([5., 1., 2.])))
    np.testing.assert_allclose(jnp.max(chain_path_efficiencies_3d(simple_chain_paths_3d(straight, cfg), 6.)), 1.)


def test_autoreset_step_uses_the_configured_reward_terms():
    cfg = replace(Baseline3DConfig(), max_steps=1)
    zero_reward = Baseline3DRewardConfig(
        exploration_bonus=0.0,
        collision_penalty=0.0,
        finder_bonus=0.0,
        max_gap_penalty=0.0,
        target_found_bonus=0.0,
        success_bonus=0.0,
    )
    reset, autoreset_step, _, _ = make_autoreset_3d_fns(
        BUILDING, cfg, zero_reward
    )
    _, reward, _, _ = autoreset_step(
        reset(jax.random.PRNGKey(123)),
        jnp.zeros((cfg.num_agents, 3)),
    )
    np.testing.assert_allclose(reward, 0.0)


def test_episode_terminates_after_configured_consecutive_stationary_steps():
    cfg, (reset, step, _, metrics) = _functions(
        max_steps=10,
        no_movement_termination_steps=3,
    )
    state = reset(jax.random.PRNGKey(17))
    zero_action = jnp.zeros((cfg.num_agents, 3))

    for expected_steps in (1, 2):
        state = step(state, zero_action)
        assert state.stationary_steps == expected_steps
        assert not state.idle_terminated
        assert not state.done

    state = step(state, zero_action)
    assert state.stationary_steps == 3
    assert state.idle_terminated
    assert state.done
    assert metrics(state)["idle_terminated"]


def test_any_agent_motion_resets_stationary_step_counter():
    cfg, (reset, step, _, _) = _functions(no_movement_termination_steps=3)
    state = step(reset(jax.random.PRNGKey(19)), jnp.zeros((cfg.num_agents, 3)))
    state = step(state, jnp.ones((cfg.num_agents, 3)))

    assert state.stationary_steps == 0
    assert not state.idle_terminated


def test_chain_diagnostics_are_zero_gap_and_full_progress_for_a_complete_chain():
    state = Baseline3DState(
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
        fully_connected=jnp.bool_(True),
        chain_held_steps=jnp.int32(0),
        done=jnp.bool_(False),
        collided=jnp.zeros(2, dtype=jnp.bool_),
        coverage_credit=jnp.zeros(2),
    )
    gap, progress = chain_diagnostics_3d(state)
    assert gap == 0.0
    assert progress == 100.0
    state = state._replace(directly_sees_target=jnp.asarray([False, True]))
    assert final_chain_length_3d(state, Baseline3DConfig(num_agents=2)) == pytest.approx(10.0)
