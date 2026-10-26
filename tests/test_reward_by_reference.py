"""Small CPU checks for caller attribution and signed reference rewards.

Run in WSL: JAX_PLATFORMS=cpu uv run pytest -q tests/test_reward_by_reference.py
Uses the real vote protocol and reward calculation with 2-4 drone slots.
No map loading, physics rollout, model, gradients, training, or 200-step loop.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import AdaptSizeConfig, EnvConfig, RewardConfig
from swarmecho.env.adapt_size import initial_state, transition
from swarmecho.env.environment import EnvState, compute_rewards, success_contributors


@pytest.fixture(autouse=True)
def eager_checks():
    # These tiny checks do not need compilation of whole transitions/graphs.
    with jax.disable_jit():
        yield


def _protocol(n=4, initial=1, **settings):
    cfg = EnvConfig(
        num_agents=n, max_steps=256,
        adapt_size=AdaptSizeConfig(enabled=True, initial_agents=initial,
                                  **{"reward_by_reference_enabled": True, **settings}),
    )
    # Fewer than 32 YES emissions in every scenario: one receipt word suffices.
    adaptive = initial_state(n, initial, jnp.array([100., 100., 100.]), max_steps=32)
    return cfg, adaptive, jnp.arange(n) < initial


def _tick(cfg, adaptive, active, yes, *, communicate=True, pos=None, peers=None, base=None):
    n = cfg.num_agents
    commands = jnp.asarray(yes, dtype=jnp.float32)
    if cfg.adapt_size.vote_mode == "keep_yes_no":
        commands = jnp.where(commands > 0, 1, 2)
    return transition(
        adaptive, active, jnp.zeros((n, 3)) if pos is None else pos, commands,
        jnp.zeros((n, n), dtype=bool) if peers is None else peers,
        jnp.ones(n, dtype=bool) if base is None else base,
        jnp.bool_(communicate), cfg.adapt_size,
    )


def _reward(**settings):
    cfg = RewardConfig(
        exploration_reward=1., collision_reward=-100.,
        finder_reward=0., target_found_reward=0., success_reward=0.,
        time_presure_reward=0., gap_progress_reward_per_meter=0.,
        reinforcement_reward=0., decommission_reward=0., rejected_call_reward=0.,
        all_decommissioned_penalty=0., useless_agent_at_success_reward=0.,
    )
    return replace(cfg, **settings)


def _state(adaptive, active):
    n = len(active)
    zero = jnp.zeros(n, dtype=bool)
    return EnvState(
        pos=jnp.zeros((n, 3)), vel=jnp.zeros((n, 3)),
        base_pos=jnp.zeros(3), target_pos=jnp.array([100., 0., 0.]), active=active,
        coverage=jnp.zeros((1, 1, 1), dtype=bool), step=jnp.int32(1),
        key=jnp.zeros(2, dtype=jnp.uint32), directly_sees_target=zero,
        is_conn_base=zero, is_conn_target=zero, target_known=zero,
        success=jnp.bool_(False), fully_connected=jnp.bool_(False),
        chain_held_steps=jnp.int32(0), done=jnp.bool_(False), collided=zero,
        coverage_credit=jnp.zeros(n), adaptive=adaptive,
    )


def _score(cfg, adaptive, active, credit, *, collisions=None, reward=None):
    before = _state(adaptive, active)
    after = before._replace(step=jnp.int32(2), coverage_credit=jnp.asarray(credit, dtype=jnp.float32))
    if collisions is not None:
        after = after._replace(collided=jnp.asarray(collisions, dtype=bool))
    return compute_rewards(before, after, _reward() if reward is None else reward, cfg)


def _three_drones(**settings):
    cfg, adaptive, active = _protocol(n=3, **settings)
    adaptive, active = _tick(cfg, adaptive, active, [1, 0, 0])  # D1 calls D2.
    adaptive, active = _tick(cfg, adaptive, active, [0, 1, 0])  # D2 calls D3.
    # Deliver the latest grant announcement before the next voting batch.
    adaptive, active = _tick(cfg, adaptive, active, [0, 0, 0])
    return cfg, adaptive, active


def test_original_calling_sequence_assigns_full_shares_only_to_the_callers():
    cfg, adaptive, active = _protocol()
    adaptive, active = _tick(cfg, adaptive, active, [1, 0, 0, 0])
    total, terms = _score(cfg, adaptive, active, [20, 100, 0, 0])
    np.testing.assert_allclose(terms["reward_by_reference"], [10, 0, 0, 0])
    np.testing.assert_allclose(total, [30, 100, 0, 0])  # Recruit keeps all 100.

    # Jump to the final discounted age rather than simulate 200 transitions.
    adaptive = adaptive._replace(reference_age=adaptive.reference_age.at[1].set(199))
    adaptive, active = _tick(cfg, adaptive, active, [0, 1, 0, 0])
    total, terms = _score(cfg, adaptive, active, [20, 30, 100, 0])
    np.testing.assert_allclose(terms["reward_by_reference"], [0, 10, 0, 0])
    np.testing.assert_allclose(total, [20, 40, 100, 0])

    adaptive = adaptive._replace(reference_age=adaptive.reference_age.at[2].set(199))
    adaptive, active = _tick(cfg, adaptive, active, [0, 0, 0, 0])  # Acknowledge.
    adaptive, active = _tick(cfg, adaptive, active, [0, 1, 1, 0])
    np.testing.assert_array_equal(adaptive.reference_callers[:, 3], [0, 1, 1, 0])
    total, terms = _score(cfg, adaptive, active, [10, 20, 30, 100])
    np.testing.assert_allclose(terms["reward_by_reference"], [0, 10, 10, 0])
    np.testing.assert_allclose(total, [10, 30, 40, 100])  # Full 10 paid twice.
    assert int(adaptive.calls) == 3


@pytest.mark.parametrize("vote_mode", ["hold", "keep_yes_no"])
def test_yes_voter_below_delivered_credit_threshold_gets_no_reference(vote_mode):
    cfg, adaptive, active = _protocol(initial=3, vote_holding=2, vote_mode=vote_mode)
    adaptive, active = _tick(cfg, adaptive, active, [1, 1, 0, 0])
    adaptive, active = _tick(cfg, adaptive, active, [1, 1, 1, 0])
    assert int(adaptive.calls) == 1
    # D3 voted YES too, but only D1 and D2 supplied a qualifying batch.
    np.testing.assert_array_equal(adaptive.reference_callers[:, 3], [1, 1, 0, 0])
    np.testing.assert_array_equal(adaptive.packet_count[-1], 0)  # Grant cleared receipts.
    total, terms = _score(cfg, adaptive, active, [0, 0, 0, 100])
    np.testing.assert_allclose(terms["reward_by_reference"], [10, 10, 0, 0])
    np.testing.assert_allclose(total, [10, 10, 0, 100])


def test_relay_gets_no_credit_for_delivering_another_drones_votes():
    cfg, adaptive, active = _protocol(n=3, initial=2, vote_holding=2)
    peers = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    base = jnp.array([0, 1, 0], dtype=bool)
    for yes in ([1, 0, 0], [1, 0, 0], [0, 0, 0]):
        adaptive, active = _tick(cfg, adaptive, active, yes, peers=peers, base=base)
    assert int(adaptive.calls) == 1
    # The source's delivered votes count even though it is now voting NO.
    np.testing.assert_array_equal(adaptive.reference_callers[:, 2], [1, 0, 0])
    total, _ = _score(cfg, adaptive, active, [0, 0, 100])
    np.testing.assert_allclose(total, [10, 0, 100])


@pytest.mark.parametrize("fraction,steps,samples", [
    (.10, 200, [(0, 10.), (1, 9.95), (100, 5.), (199, .05), (200, 0.), (201, 0.)]),
    (.25, 4, [(0, 25.), (1, 18.75), (2, 12.5), (3, 6.25), (4, 0.), (5, 0.)]),
])
def test_linear_decay_boundaries_and_custom_parameters(fraction, steps, samples):
    cfg, adaptive, active = _protocol(n=2)
    adaptive, active = _tick(cfg, adaptive, active, [1, 0])
    assert int(adaptive.reference_age[1]) == 0
    cfg = replace(cfg, adapt_size=replace(cfg.adapt_size,
                  reward_by_reference_fraction=fraction, reward_by_reference_steps=steps))
    for age, expected in samples:
        aged = adaptive._replace(reference_age=adaptive.reference_age.at[1].set(age))
        total, terms = _score(cfg, aged, active, [0, 100])
        np.testing.assert_allclose(terms["reward_by_reference"], [expected, 0], atol=1e-5)
        np.testing.assert_allclose(total, [expected, 100], atol=1e-5)
    # Aging follows environment transitions even without communication.
    aged = adaptive._replace(reference_age=adaptive.reference_age.at[1].set(99))
    aged, _ = _tick(cfg, aged, active, [0, 0], communicate=False)
    assert int(aged.reference_age[1]) == 100


@pytest.mark.parametrize("sign", [1, -1])
def test_signed_rewards_do_not_cascade_and_toggle_preserves_normal_reward(sign):
    cfg, adaptive, active = _three_drones()
    credit = [0, 0, 100 if sign > 0 else 0]
    collisions = [0, 0, sign < 0]
    # D3's reference has aged one tick since D2's call.
    total, terms = _score(cfg, adaptive, active, credit, collisions=collisions)
    np.testing.assert_allclose(terms["reward_by_reference"], [0, sign * 9.95, 0])
    np.testing.assert_allclose(terms["reward_by_reference_positive"], [0, 9.95 if sign > 0 else 0, 0])
    np.testing.assert_allclose(terms["reward_by_reference_negative"], [0, -9.95 if sign < 0 else 0, 0])
    np.testing.assert_allclose(total, [0, sign * 9.95, sign * 100])
    # D1 called D2, but D2's reference bonus/penalty must not flow to D1.
    disabled, disabled_terms = _score(
        replace(cfg, adapt_size=replace(cfg.adapt_size, reward_by_reference_enabled=False)),
        adaptive, active, credit, collisions=collisions,
    )
    np.testing.assert_array_equal(disabled_terms["reward_by_reference"], 0)
    np.testing.assert_array_equal(disabled_terms["reward_by_reference_positive"], 0)
    np.testing.assert_array_equal(disabled_terms["reward_by_reference_negative"], 0)
    np.testing.assert_allclose(disabled, [0, 0, sign * 100])

    if sign > 0:
        # D1 calls both recruits. Their opposing payments on the same step
        # must remain visible even though D1's net reference reward is zero.
        mixed_cfg, mixed, mixed_active = _protocol(n=3)
        mixed, mixed_active = _tick(mixed_cfg, mixed, mixed_active, [1, 0, 0])
        mixed, mixed_active = _tick(mixed_cfg, mixed, mixed_active, [0, 0, 0])
        mixed, mixed_active = _tick(mixed_cfg, mixed, mixed_active, [1, 0, 0])
        mixed = mixed._replace(reference_age=jnp.zeros(3, dtype=jnp.int32))
        mixed_total, mixed_terms = _score(mixed_cfg, mixed, mixed_active,
                                          [0, 100, 0], collisions=[0, 0, 1])
        np.testing.assert_allclose(mixed_terms["reward_by_reference"], [0, 0, 0])
        np.testing.assert_allclose(mixed_terms["reward_by_reference_positive"], [10, 0, 0])
        np.testing.assert_allclose(mixed_terms["reward_by_reference_negative"], [-10, 0, 0])
        np.testing.assert_allclose(mixed_total, [0, 100, -100])  # No extra payout for diagnostics.


@pytest.mark.parametrize("recruit_contributes,age,extra_reward", [
    (False, 0, -1.), (False, 100, -.5), (False, 200, 0.), (True, 0, 5.),
])
def test_terminal_success_and_useless_recruit_rewards_go_to_each_caller(
    recruit_contributes, age, extra_reward,
):
    cfg, adaptive, active = _protocol(n=3, initial=2)
    cfg = replace(cfg, comm_radius=4.1, comm_radius_base=4.1, visual_radius=2.1)
    before = _state(adaptive, active)
    adaptive, active = _tick(cfg, adaptive, active, [1, 1, 0])
    adaptive = adaptive._replace(reference_age=adaptive.reference_age.at[2].set(age))
    positions = ([[5., 2., 2.], [17., 17., 10.], [9., 2., 2.]] if recruit_contributes
                 else [[5., 2., 2.], [9., 2., 2.], [17., 17., 10.]])
    contributors = [1, 0, 1] if recruit_contributes else [1, 1, 0]
    after = _state(adaptive, active)._replace(
        pos=jnp.asarray(positions),
        base_pos=jnp.array([2., 2., 2.]), target_pos=jnp.array([10., 2., 2.]),
        directly_sees_target=jnp.array([0, 0, 1] if recruit_contributes else [0, 1, 0], dtype=bool),
        is_conn_base=jnp.asarray(contributors, dtype=bool),
        is_conn_target=jnp.asarray(contributors, dtype=bool),
        fully_connected=jnp.bool_(True), success=jnp.bool_(True), done=jnp.bool_(True),
    )
    np.testing.assert_array_equal(success_contributors(after, cfg), contributors)
    reward = _reward(success_reward=50., useless_agent_at_success_reward=-60.)
    total, terms = compute_rewards(before, after, reward, cfg)
    np.testing.assert_allclose(terms["success_reward"], [50, 50, 50])
    np.testing.assert_allclose(terms["useless_agent_at_success_reward"], [0, -60, 0] if recruit_contributes else [0, 0, -60])
    np.testing.assert_allclose(terms["reward_by_reference"], [extra_reward, extra_reward, 0])
    expected = [55, -5, 50] if recruit_contributes else [50 + extra_reward, 50 + extra_reward, -10]
    np.testing.assert_allclose(total, expected)


@pytest.mark.parametrize("retired", [0, 1])
def test_decommissioning_either_lifetime_clears_its_references(retired):
    cfg, adaptive, active = _three_drones(decommission_hold_steps=1)
    pos = jnp.zeros((3, 3)).at[retired].set(adaptive.zone_center)
    adaptive, active = _tick(cfg, adaptive, active, [0, 0, 0], pos=pos, communicate=False)
    assert not bool(active[retired])
    np.testing.assert_array_equal(adaptive.reference_callers[retired], False)
    np.testing.assert_array_equal(adaptive.reference_callers[:, retired], False)
    expected = np.zeros((3, 3), dtype=bool)
    if retired == 0:
        expected[1, 2] = True  # D2's separate reference to D3 survives.
    np.testing.assert_array_equal(adaptive.reference_callers, expected)


@pytest.mark.parametrize("retired", [0, 1])
def test_immediate_slot_reuse_gets_fresh_callers_and_no_inherited_bonus(retired):
    cfg, adaptive, active = _three_drones(decommission_hold_steps=1)
    old_id = int(adaptive.ids[retired])
    pos = jnp.zeros((3, 3)).at[retired].set(adaptive.zone_center)
    yes = [int(i != retired) for i in range(3)]
    adaptive, active = _tick(cfg, adaptive, active, yes, pos=pos)
    assert bool(adaptive.decommissioned[retired]) and bool(adaptive.spawned[retired])
    assert int(adaptive.ids[retired]) != old_id
    assert int(adaptive.reference_age[retired]) == 0
    expected = np.zeros((3, 3), dtype=bool)
    expected[:, retired] = np.arange(3) != retired
    if retired == 0:
        expected[1, 2] = True
    np.testing.assert_array_equal(adaptive.reference_callers, expected)
    credit = [100 if i == retired else 0 for i in range(3)]
    total, terms = _score(cfg, adaptive, active, credit)
    expected_bonus = [0 if i == retired else 10 for i in range(3)]
    np.testing.assert_allclose(terms["reward_by_reference"], expected_bonus)
    np.testing.assert_allclose(total, [100 if i == retired else 10 for i in range(3)])


def test_rejected_call_does_not_reassign_or_restart_a_reference_and_reset_clears_it():
    cfg, adaptive, active = _protocol(n=2)
    np.testing.assert_array_equal(adaptive.reference_callers, False)  # Initial cohort.
    adaptive, active = _tick(cfg, adaptive, active, [1, 0])
    adaptive, active = _tick(cfg, adaptive, active, [0, 0])  # Acknowledge.
    adaptive, active = _tick(cfg, adaptive, active, [1, 1])  # Already at capacity.
    assert int(adaptive.calls) == 1 and int(adaptive.rejected_calls) == 1
    np.testing.assert_array_equal(adaptive.reference_callers, [[0, 1], [0, 0]])
    assert int(adaptive.reference_age[1]) == 2
    total, _ = _score(cfg, adaptive, active, [0, 100])
    np.testing.assert_allclose(total, [9.9, 100])
    fresh = initial_state(2, 1, adaptive.zone_center, max_steps=32)
    np.testing.assert_array_equal(fresh.reference_callers, False)
    np.testing.assert_array_equal(fresh.reference_age, 0)
