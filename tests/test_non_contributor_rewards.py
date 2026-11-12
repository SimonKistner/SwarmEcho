"""Surplus-agent movement, safe exit and disabled-path regression checks."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from swarmecho.core.config import AdaptSizeConfig, EnvConfig, RewardConfig, load_level, validate_reward_config
from swarmecho.env.adapt_size import initial_state
from swarmecho.env.environment import (
    EnvState, _return_progress_metres, chain_distance_matrix, compute_rewards,
    make_autoreset_fns, success_contributors,
)
from swarmecho.env.buildings import load_building
from swarmecho.env.obstacles import roadmap_distances


def scenario(y=5.):
    env = EnvConfig(num_agents=4, comm_radius=3.1, comm_radius_base=3.1,
                    visual_radius=1.1, observe_chain_contributor=True,
                    adapt_size=AdaptSizeConfig(enabled=True, initial_agents=4,
                                              decommission_radius_m=1.))
    active = jnp.ones(4, dtype=bool)
    zero = jnp.zeros(4, dtype=bool)
    state = EnvState(
        pos=jnp.array([[3., 0., 0.], [6., 0., 0.], [9., 0., 0.], [0., y, 1.]]),
        vel=jnp.zeros((4, 3)), base_pos=jnp.zeros(3), target_pos=jnp.array([10., 0., 0.]),
        active=active, coverage=jnp.zeros((1, 1, 1), dtype=bool), step=jnp.int32(10),
        key=jax.random.PRNGKey(0), directly_sees_target=jnp.array([0, 0, 1, 0], dtype=bool),
        is_conn_base=jnp.array([1, 1, 1, 0], dtype=bool),
        is_conn_target=jnp.array([1, 1, 1, 0], dtype=bool), target_known=active,
        base_target_known=jnp.bool_(True), success=jnp.bool_(False),
        fully_connected=jnp.bool_(True), chain_held_steps=jnp.int32(10), done=jnp.bool_(False),
        collided=zero, coverage_credit=jnp.zeros(4),
        adaptive=initial_state(4, 4, jnp.array([0., 0., 1.]), max_steps=32),
    )
    return env, state


def reward_config(**overrides):
    return replace(RewardConfig(
        exploration_reward=0., finder_reward=0., target_found_reward=0., success_reward=0.,
        collision_reward=0., time_presure_reward=0., gap_progress_reward_per_meter=0.,
        reinforcement_reward=0., rejected_call_reward=0., all_decommissioned_penalty=0.,
        useless_agent_at_success_reward=0., useless_agent_at_success_all_agents_reward=0.,
        non_contributor_return_reward_per_meter=3.,
        non_contributor_departure_reward_per_meter=-3.,
        non_contributor_decommission_reward=1500.,
    ), **overrides)


def move(state, y):
    return state._replace(pos=state.pos.at[3, 1].set(y), step=state.step + 1)


@pytest.mark.parametrize("before,after,inward,outward", [
    (6., 5., 3., 0.), (5., 6., 0., -3.), (5., 5., 0., 0.),
    (1., 1.5, 0., -1.5), (1.5, 1., 1.5, 0.), (2., 1., 3., 0.),
    (2.5, 1.5, 3., 0.), (1.5, 2.5, 0., -3.),
    (2., 2.5, 0., -1.5), (2.5, 2., 1.5, 0.),
    (0., 0., 0., 0.), (.5, 0., 1.5, 0.), (0., .5, 0., -1.5),
    (-.5, 0., 1.5, 0.), (0., -.5, 0., -1.5),
    (.0001, 0., .0003, 0.), (0., .0001, 0., -.0003),
])
def test_progress_to_zone_centre_including_inside_zone_and_exact_centre(before, after, inward, outward):
    env, state = scenario(before)
    np.testing.assert_array_equal(success_contributors(state, env), [1, 1, 1, 0])
    total, terms = compute_rewards(state, move(state, after), reward_config(), env)
    np.testing.assert_allclose(terms["non_contributor_return_reward"], [0, 0, 0, inward], atol=1e-5)
    np.testing.assert_allclose(terms["non_contributor_departure_reward"], [0, 0, 0, outward], atol=1e-5)
    np.testing.assert_allclose(total, [0, 0, 0, inward + outward], atol=1e-5)


def test_longer_return_pays_more_and_stationary_hold_does_not_pay():
    totals = []
    for start in (6., 12.):
        env, state = scenario(start)
        total = 0.
        for y in (4., 2., 0., 0.):
            following = move(state, y)
            total += float(compute_rewards(state, following, reward_config(), env)[0][3])
            state = following
        totals.append(total)
    np.testing.assert_allclose(totals, [18., 36.])


def test_moving_toward_base_but_away_from_zone_centre_is_penalized():
    env, before = scenario(0.)
    after = before._replace(pos=before.pos.at[3, 2].set(.5))
    assert float(jnp.linalg.norm(after.pos[3] - after.base_pos)) < float(jnp.linalg.norm(before.pos[3] - before.base_pos))
    _, terms = compute_rewards(before, after, reward_config(), env, chain_distance_matrix(after))
    np.testing.assert_array_equal(terms["non_contributor_return_reward"], 0.)
    np.testing.assert_array_equal(terms["non_contributor_departure_reward"], [0, 0, 0, -1.5])


@pytest.mark.parametrize("change", ["chain_not_built", "chain_broken", "already_done", "inactive",
                                        "replacement", "invalid_zone", "becomes_contributor"])
def test_only_surviving_surplus_lifetimes_receive_movement(change):
    env, before = scenario(6.)
    after = move(before, 5.)
    if change == "chain_not_built":
        before = before._replace(fully_connected=jnp.bool_(False))
    elif change == "chain_broken":
        after = after._replace(fully_connected=jnp.bool_(False))
    elif change == "already_done":
        before = before._replace(done=jnp.bool_(True))
    elif change == "inactive":
        before = before._replace(active=before.active.at[3].set(False))
    elif change == "replacement":
        after = after._replace(adaptive=after.adaptive._replace(spawned=after.active.at[:3].set(False)))
    elif change == "invalid_zone":
        after = after._replace(adaptive=after.adaptive._replace(zone_valid=jnp.bool_(False)))
    else:
        after = after._replace(pos=after.pos.at[2].set([20., 0., 0.]).at[3].set([9., 0., 0.]),
                               directly_sees_target=jnp.array([0, 0, 0, 1], dtype=bool))
    _, terms = compute_rewards(before, after, reward_config(), env)
    np.testing.assert_array_equal(terms["non_contributor_return_reward"], 0.)
    np.testing.assert_array_equal(terms["non_contributor_departure_reward"], 0.)


def removed_state(state, slot=3, *, replacement=False):
    removed = jnp.zeros(4, dtype=bool).at[slot].set(True)
    return state._replace(
        pos=state.pos.at[slot].set(state.base_pos),
        active=state.active.at[slot].set(replacement), step=state.step + 1,
        adaptive=state.adaptive._replace(
            decommissioned=removed, decommissions=jnp.int32(1), spawned=removed & replacement,
            ids=state.adaptive.ids.at[slot].set(4 if replacement else -1),
        ),
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_qualified_decommission_is_paid_once_to_departing_lifetime(replacement):
    env, before = scenario(1.)
    after = removed_state(before, replacement=replacement)
    total, terms = jax.jit(lambda a, b: compute_rewards(a, b, reward_config(), env))(before, after)
    np.testing.assert_array_equal(terms["non_contributor_decommission_reward"], [0, 0, 0, 1500])
    np.testing.assert_array_equal(total, [0, 0, 0, 1500])
    np.testing.assert_array_equal(compute_rewards(after, after, reward_config(), env)[0], 0.)


@pytest.mark.parametrize("change", ["contributor", "broken", "no_previous_chain", "not_removed", "stale_event"])
def test_decommission_iff_chain_intact_and_prior_non_contributor_is_removed(change):
    env, before = scenario(1.)
    after = removed_state(before, slot=2 if change == "contributor" else 3)
    if change == "broken":
        after = after._replace(fully_connected=jnp.bool_(False))
    elif change == "no_previous_chain":
        before = before._replace(fully_connected=jnp.bool_(False))
    elif change == "not_removed":
        after = after._replace(adaptive=after.adaptive._replace(decommissioned=jnp.zeros(4, dtype=bool)))
    elif change == "stale_event":
        before = before._replace(adaptive=before.adaptive._replace(decommissions=jnp.int32(1)))
    _, terms = compute_rewards(before, after, reward_config(), env)
    np.testing.assert_array_equal(terms["non_contributor_decommission_reward"], 0.)


def test_each_reward_can_be_enabled_without_the_others():
    env, before = scenario(5.)
    disabled = reward_config(non_contributor_return_reward_per_meter=0.,
                             non_contributor_departure_reward_per_meter=0.,
                             non_contributor_decommission_reward=0.)
    for field, value, after, expected in (
        ("non_contributor_return_reward_per_meter", 3., move(before, 4.), 3.),
        ("non_contributor_departure_reward_per_meter", -3., move(before, 6.), -3.),
        ("non_contributor_decommission_reward", 1500., removed_state(before), 1500.),
    ):
        total, _ = compute_rewards(before, after, replace(disabled, **{field: value}), env)
        np.testing.assert_array_equal(total, [0, 0, 0, expected])


def test_all_zero_skips_new_logic_and_preserves_exact_existing_terms(monkeypatch):
    from swarmecho.env import environment
    env, before = scenario()
    after = move(before, 4.)
    baseline = RewardConfig()
    expected, expected_terms = compute_rewards(before, after, baseline, env)

    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled rewards must not compute new eligibility or paths")

    monkeypatch.setattr(environment, "non_contributor_rewards", unexpected)
    explicit_zero = replace(baseline, non_contributor_return_reward_per_meter=0.,
                            non_contributor_departure_reward_per_meter=0.,
                            non_contributor_decommission_reward=0.)
    actual, terms = compute_rewards(before, after, explicit_zero, env)
    np.testing.assert_array_equal(actual, expected)
    assert terms.keys() == expected_terms.keys()
    assert not any(name.startswith("non_contributor_") for name in terms)
    for name in terms:
        np.testing.assert_array_equal(terms[name], expected_terms[name])


def test_geodesic_progress_penalizes_wrong_way_even_when_euclidean_distance_decreases():
    env, before = scenario()
    lo, hi = jnp.array([[1., -2., -100.]]), jnp.array([[3., 2., 100.]])
    vertices = jnp.array([[.9999, -2.0001, 0.], [3.0001, -2.0001, 0.],
                          [.9999, 2.0001, 0.], [3.0001, 2.0001, 0.]])
    before = before._replace(pos=before.pos.at[3].set([4., 3., 0.]), obstacle_min=lo, obstacle_max=hi,
                             roadmap_vertices=vertices, roadmap_distances=roadmap_distances(vertices, lo, hi, 0.))
    after = before._replace(pos=before.pos.at[3].set([4., 0., 0.]))
    centre = before.adaptive.zone_center
    assert float(jnp.linalg.norm(after.pos[3] - centre)) < float(jnp.linalg.norm(before.pos[3] - centre))
    progress = _return_progress_metres(before, after)
    assert float(progress[3]) < -.5


def test_jit_vmap_handles_mixed_inward_outward_and_near_centre_movements():
    env, state = scenario()
    before = jax.tree.map(lambda x: jnp.stack([x, x, x]), state)
    before = before._replace(pos=before.pos.at[:, 3, 1].set(jnp.array([6., 5., 1.])))
    after = before._replace(pos=before.pos.at[:, 3, 1].set(jnp.array([5., 6., 1.5])))
    total, _ = jax.jit(jax.vmap(lambda a, b: compute_rewards(a, b, reward_config(), env)))(before, after)
    np.testing.assert_allclose(total[:, 3], [3., -3., -1.5])


@pytest.mark.parametrize("name", ["G03_random_buildings", "G03_CLU_random_buildings"])
def test_both_g03_levels_explicitly_configure_independent_defaults(name):
    data = yaml.safe_load(Path(f"src/swarmecho/curriculum_config/levels/G/{name}.yaml").read_text())
    cfg = data["reward"]
    assert data["env"]["observe_chain_contributor"] is True
    assert cfg["non_contributor_return_reward_per_meter"] == 3.
    assert cfg["non_contributor_departure_reward_per_meter"] == -3.
    assert cfg["non_contributor_decommission_reward"] == cfg["success_reward"] / 2 == 1500.
    assert cfg["decommission_reward"] == 0.
    level = load_level(name)
    assert level.reward.non_contributor_return_reward_per_meter == 3.
    assert level.reward.non_contributor_departure_reward_per_meter == -3.
    assert level.reward.non_contributor_decommission_reward == 1500.


@pytest.mark.parametrize("field", ["non_contributor_return_reward_per_meter",
                                    "non_contributor_departure_reward_per_meter", "non_contributor_decommission_reward"])
def test_coefficients_are_signed_finite_and_zero_disabled(field):
    assert getattr(RewardConfig(), field) == 0.
    for value in (0., -3., 3.):
        validate_reward_config(replace(RewardConfig(), **{field: value}))
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match=field):
            validate_reward_config(replace(RewardConfig(), **{field: value}))


def test_removal_payment_survives_real_protocol_and_batched_autoreset():
    building = load_building(Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"))
    env = EnvConfig(num_agents=2, spawn_delay=0, hold_chain_for=2, comm_radius_base=10.,
                    adapt_size=AdaptSizeConfig(enabled=True, initial_agents=2, decommission_hold_steps=1))
    reward = reward_config()
    reset, transition, _, _ = make_autoreset_fns(building, env, reward)
    states = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(22), 2))
    base = jnp.broadcast_to(jnp.array([2., 2., 2.]), (2, 3))
    pos = jnp.broadcast_to(jnp.array([[5., 2., 2.], [2., 2., 3.5]]), (2, 2, 3))
    states = states._replace(base_pos=base, target_pos=base + jnp.array([3., 0., 0.]),
        pos=pos, active=jnp.ones((2, 2), dtype=bool),
        directly_sees_target=jnp.array([[1, 0], [1, 0]], dtype=bool),
        target_known=jnp.ones((2, 2), dtype=bool), fully_connected=jnp.ones(2, dtype=bool),
        chain_held_steps=jnp.ones(2, dtype=jnp.int32),
        adaptive=states.adaptive._replace(zone_center=pos[:, 1], zone_valid=jnp.ones(2, dtype=bool)))
    fresh, rewards, done, info = jax.jit(transition.batched)(states, jnp.zeros((2, 2, 4)))
    np.testing.assert_array_equal(done, True)
    np.testing.assert_array_equal(info["non_contributor_decommission_reward"], [[0, 1500], [0, 1500]])
    np.testing.assert_array_equal(rewards, [[0, 1500], [0, 1500]])
    np.testing.assert_array_equal(info["agent_terminated"], [[False, True], [False, True]])
    np.testing.assert_array_equal(fresh.adaptive.decommissions, 0)
