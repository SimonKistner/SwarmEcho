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
    make_autoreset_3d_fns,
    make_baseline_3d_fns,
    maximum_chain_distance,
    maximum_five_drone_chain_distance,
    minimum_target_distance,
    observation_dim_3d,
    rewards_3d,
    spherical_directions,
)
from swarmecho.env.buildings import load_building


BUILDING = load_building(
    Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")
)


def _functions(**overrides):
    cfg = replace(Baseline3DConfig(), **overrides)
    return cfg, make_baseline_3d_fns(BUILDING, cfg)


def test_octant_radar_and_configured_distance_contracts():
    directions = spherical_directions(8)
    assert directions.shape == (8, 3)
    assert len({tuple(np.sign(row)) for row in directions}) == 8
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    cfg = Baseline3DConfig()
    assert minimum_target_distance(cfg) == 9.0
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
    assert jnp.linalg.norm(state.target_pos - state.base_pos) > minimum_target_distance(cfg)


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
