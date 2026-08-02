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


def test_reward_terms_preserve_local_credit_and_shared_events():
    cfg, (reset, step, _, _) = _functions()
    previous = reset(jax.random.PRNGKey(7))
    current = step(previous, jnp.ones((cfg.num_agents, 3)))
    reward, terms = rewards_3d(previous, current, Baseline3DRewardConfig())
    assert reward.shape == (cfg.num_agents,)
    assert terms["coverage"].shape == (cfg.num_agents,)
    assert terms["collision"].shape == (cfg.num_agents,)
    assert terms["chain_gap"].shape == (cfg.num_agents,)
    assert jnp.isfinite(reward).all()
    assert jnp.sum(current.coverage_credit) == pytest.approx(
        jnp.sum(current.coverage & ~previous.coverage), abs=1e-5
    )


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
    assert reward.shape == (cfg.num_agents,)
    assert next_state.step == 0
    assert not next_state.done
