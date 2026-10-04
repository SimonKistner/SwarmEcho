"""Behavioral checks for gap credit earned before target delivery."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import load_level_cli
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import (
    EnvConfig,
    RewardConfig,
    compute_rewards,
    make_env_fns,
)


BUILDING = load_building(
    Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")
)


def test_delivery_gap_vesting_config_requires_change_reward_and_positive_steps():
    with pytest.raises(ValueError, match="requires gap-change reward"):
        load_level_cli([
            "level=M00_no_maze_open_cuboid",
            "reward.delivery_gap_vesting_enabled=true",
        ])
    with pytest.raises(ValueError, match="positive integer"):
        load_level_cli([
            "level=M00_no_maze_open_cuboid",
            "reward.delivery_gap_vesting_steps=0",
        ])


def _scenario(*, hold_chain_for=50, second_agent_x=9.0):
    env_cfg = EnvConfig(
        num_agents=2, spawn_delay=0, max_steps=100,
        comm_radius_base=2.5, comm_radius=3.0,
        visual_radius=5.0 if second_agent_x == 7.0 else 4.0,
        hold_chain_for=hold_chain_for,
    )
    reward_cfg = RewardConfig(
        chain_reward_system="euclidean", gap_reward_uses_change=True,
        gap_change_meter_bonus=3.0, delivery_gap_vesting_enabled=True,
        delivery_gap_vesting_steps=10, exploration_bonus=0.0,
        finder_bonus=0.0, target_found_bonus=0.0, success_bonus=0.0,
        time_penalty_per_step=0.0, collision_penalty=0.0,
    )
    reset, step, _, _ = make_env_fns(
        BUILDING, env_cfg, plan_geodesic=False,
        delivery_gap_vesting_enabled=True,
    )
    state = reset(jax.random.PRNGKey(12))._replace(
        pos=jnp.array([[4., 2., 2.], [second_agent_x, 2., 2.]]),
        vel=jnp.zeros((2, 3)),
        base_pos=jnp.array([2., 2., 2.]),
        target_pos=jnp.array([12., 2., 2.]),
        active=jnp.ones(2, dtype=jnp.bool_),
        step=jnp.int32(0),
        target_known=jnp.ones(2, dtype=jnp.bool_),
        base_target_known=jnp.bool_(False),
        chain_held_steps=jnp.int32(0),
        best_chain_held_steps=jnp.int32(0),
        success=jnp.bool_(False),
        done=jnp.bool_(False),
    )
    return env_cfg, reward_cfg, state, step


def _gap_reward(before, after, reward_cfg, env_cfg):
    _, terms = compute_rewards(before, after, reward_cfg, env_cfg)
    return np.asarray(terms["gap_reduction"])


def test_pre_delivery_gap_credit_vests_to_the_same_total_as_later_construction():
    env_cfg, reward_cfg, state, step = _scenario()
    step = jax.jit(step)
    gap_reward = jax.jit(
        lambda before, after: compute_rewards(before, after, reward_cfg, env_cfg)[1]["gap_reduction"]
    )
    delivered = step(state, jnp.zeros((2, 3)))
    assert delivered.base_target_known
    assert int(delivered.delivery_gap_step) == 1
    np.testing.assert_allclose(delivered.delivery_gap_credit, [5., 5.])
    np.testing.assert_allclose(gap_reward(state, delivered), [0., 0.])

    paid = np.zeros(2)
    current = delivered
    for _ in range(reward_cfg.delivery_gap_vesting_steps):
        following = step(current, jnp.zeros((2, 3)))
        tranche = np.asarray(gap_reward(current, following))
        np.testing.assert_allclose(tranche, [1.5, 1.5], atol=1e-5)
        paid += tranche
        current = following
    np.testing.assert_allclose(paid, [15., 15.], atol=1e-5)
    following = step(current, jnp.zeros((2, 3)))
    np.testing.assert_allclose(gap_reward(current, following), [0., 0.])

    # Establish the identical credited chain after delivery from zero progress.
    empty = delivered._replace(
        pos=jnp.array([[6., 6., 2.], [6., 8., 2.]]),
        directly_sees_target=jnp.zeros(2, dtype=jnp.bool_),
        is_conn_base=jnp.zeros(2, dtype=jnp.bool_),
        is_conn_target=jnp.zeros(2, dtype=jnp.bool_),
        delivery_gap_credit=jnp.zeros(2),
    )
    built = delivered._replace(
        step=jnp.int32(2), delivery_gap_credit=jnp.zeros(2),
    )
    np.testing.assert_allclose(gap_reward(empty, built), paid)


def test_unpaid_credit_absorbs_loss_and_new_progress_pays_immediately():
    env_cfg, reward_cfg, state, step = _scenario()
    delivered = step(state, jnp.zeros((2, 3)))
    first = step(delivered, jnp.zeros((2, 3)))
    np.testing.assert_allclose(_gap_reward(delivered, first, reward_cfg, env_cfg), [1.5, 1.5])

    # Move the target-side relay one metre toward the base: this new work pays
    # immediately in addition to the second instalment of the earlier credit.
    improved = first._replace(
        step=jnp.int32(3),
        pos=first.pos.at[1, 0].set(8.0),
    )
    np.testing.assert_allclose(_gap_reward(first, improved, reward_cfg, env_cfg), [4.5, 4.5])

    # Losing both credited paths after the first instalment reclaims only that
    # instalment, not the full 15 that was still mostly pending.
    broken = first._replace(
        step=jnp.int32(3),
        pos=jnp.array([[6., 6., 2.], [6., 8., 2.]]),
        directly_sees_target=jnp.zeros(2, dtype=jnp.bool_),
        is_conn_base=jnp.zeros(2, dtype=jnp.bool_),
        is_conn_target=jnp.zeros(2, dtype=jnp.bool_),
        fully_connected=jnp.bool_(False),
    )
    np.testing.assert_allclose(_gap_reward(first, broken, reward_cfg, env_cfg), [-1.5, -1.5])
    np.testing.assert_allclose(_gap_reward(broken, broken, reward_cfg, env_cfg), [0., 0.])

    # The optional feature leaves the prior zero-on-activation behavior intact.
    original = replace(reward_cfg, delivery_gap_vesting_enabled=False)
    np.testing.assert_allclose(_gap_reward(state, delivered, original, env_cfg), [0., 0.])
    np.testing.assert_allclose(_gap_reward(delivered, first, original, env_cfg), [0., 0.])
    np.testing.assert_allclose(_gap_reward(first, broken, original, env_cfg), [-15., -15.])


def test_terminal_transition_releases_pending_credit_for_success_or_timeout():
    env_cfg, reward_cfg, state, step = _scenario(hold_chain_for=2, second_agent_x=7.0)
    delivered = step(state, jnp.zeros((2, 3)))
    assert delivered.fully_connected and not delivered.success
    np.testing.assert_allclose(delivered.delivery_gap_credit, [10., 10.])
    np.testing.assert_allclose(_gap_reward(state, delivered, reward_cfg, env_cfg), [0., 0.])

    finished = step(delivered, jnp.zeros((2, 3)))
    assert finished.success and finished.done
    np.testing.assert_allclose(_gap_reward(delivered, finished, reward_cfg, env_cfg), [30., 30.])

    partial_cfg, partial_reward, partial_state, partial_step = _scenario()
    partial_delivery = partial_step(partial_state, jnp.zeros((2, 3)))
    timed_out = partial_delivery._replace(step=jnp.int32(2), done=jnp.bool_(True))
    np.testing.assert_allclose(
        _gap_reward(partial_delivery, timed_out, partial_reward, partial_cfg),
        [15., 15.],
    )
