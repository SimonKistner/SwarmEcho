"""Chain-hold completion with independent surplus cleanup and caller penalties."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import (
    AdaptSizeConfig, EnvConfig, PrivelegedVoteRewardConfig, RewardConfig, load_level,
)
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import compute_rewards, make_autoreset_fns, make_env_fns, success_contributors
from swarmecho.training.artifacts import load_eval_info_csv, save_eval_info_csv
from swarmecho.training.evaluate import _replay_stage_label, _successful_stage_mask


BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"))
CLEANUP = "chain_held_and_cleand_up"
ACTION = jnp.zeros((2, 4))


def scenario(condition=CLEANUP, **overrides):
    cfg = replace(EnvConfig(
        num_agents=2, spawn_delay=0, success_condition=condition, hold_chain_for=2,
        max_steps=30, comm_radius_base=3.1, comm_radius=3.1, visual_radius=1.1,
        adapt_size=AdaptSizeConfig(enabled=True, initial_agents=2,
                                  vote_protocol="latest_decision", decommission_hold_steps=2),
    ), **overrides)
    reset, step, _, _ = make_env_fns(BUILDING, cfg)
    state = reset(jax.random.PRNGKey(12))
    state = state._replace(
        base_pos=jnp.array([2., 2., 2.]), target_pos=jnp.array([6., 2., 2.]),
        pos=jnp.array([[5., 2., 2.], [2., 5., 3.5]]), vel=jnp.zeros((2, 3)),
        active=jnp.ones(2, dtype=bool), target_known=jnp.ones(2, dtype=bool),
        base_target_known=jnp.bool_(True), fully_connected=jnp.bool_(True),
        directly_sees_target=jnp.array([True, False]),
        is_conn_base=jnp.array([True, False]), is_conn_target=jnp.array([True, False]),
        adaptive=state.adaptive._replace(zone_center=jnp.array([2., 2., 3.5]), zone_valid=jnp.bool_(True)),
    )
    return cfg, state, jax.jit(step)


def rewards(**overrides):
    return replace(RewardConfig(
        exploration_reward=0., collision_reward=0., finder_reward=0., target_found_reward=0.,
        time_presure_reward=0., gap_progress_reward_per_meter=0.,
        non_contributor_return_reward_per_meter=3., non_contributor_departure_reward_per_meter=-3.,
        non_contributor_decommission_reward=1500.,
    ), **overrides)


@pytest.mark.parametrize("condition,expected", [("chain_held", True), (CLEANUP, False)])
def test_hold_threshold_only_finishes_legacy_mode_with_surplus(condition, expected):
    cfg, state, step = scenario(condition)
    first = step(state, ACTION)
    assert not first.success and not first.done
    second = step(first, ACTION)
    assert int(second.chain_held_steps) == cfg.hold_chain_for
    assert bool(second.success) == bool(second.done) == expected
    _, terms = compute_rewards(first, second, rewards(), cfg)
    np.testing.assert_array_equal(terms["success_reward"], [3000., 3000.] if expected else [0., 0.])
    np.testing.assert_array_equal(terms["useless_agent_at_success_reward"], [0., -3000.] if expected else [0., 0.])


def test_cleanup_after_hold_finishes_on_removal_and_pays_departing_lifetime():
    cfg, state, step = scenario()
    state = step(step(state, ACTION), ACTION)
    assert not state.done
    state = state._replace(pos=state.pos.at[1].set(state.adaptive.zone_center))
    dwelling = step(state, ACTION)
    assert int(dwelling.chain_held_steps) > cfg.hold_chain_for
    assert not dwelling.done
    final = step(dwelling, ACTION)
    assert final.success and final.done and final.fully_connected
    np.testing.assert_array_equal(final.active, [True, False])
    _, terms = compute_rewards(dwelling, final, rewards(), cfg)
    np.testing.assert_array_equal(terms["non_contributor_decommission_reward"], [0., 1500.])
    np.testing.assert_array_equal(terms["success_reward"], [3000., 0.])
    np.testing.assert_array_equal(terms["useless_agent_at_success_all_agents_reward"], 0.)


def test_chain_break_during_cleanup_requires_a_fresh_full_hold():
    cfg, state, step = scenario(hold_chain_for=3)
    for _ in range(cfg.hold_chain_for):
        state = step(state, ACTION)
    assert not state.done
    broken = step(state._replace(pos=state.pos.at[0].set(jnp.array([12., 2., 2.]))), ACTION)
    assert not broken.fully_connected and not broken.success and not broken.done
    assert int(broken.chain_held_steps) == 0
    assert int(broken.best_chain_held_steps) == cfg.hold_chain_for
    restored = broken._replace(pos=broken.pos.at[0].set(jnp.array([5., 2., 2.]))
                               .at[1].set(broken.adaptive.zone_center))
    first = step(restored, ACTION)
    cleaned = step(first, ACTION)
    np.testing.assert_array_equal(cleaned.active, [True, False])
    assert int(cleaned.chain_held_steps) == 2
    assert not cleaned.success and not cleaned.done
    final = step(cleaned, ACTION)
    assert int(final.chain_held_steps) == cfg.hold_chain_for
    assert final.success and final.done


def test_new_surplus_recruit_on_completion_step_prevents_success():
    cfg, state, step = scenario(adapt_size=AdaptSizeConfig(
        enabled=True, initial_agents=1, vote_protocol="latest_decision"))
    state = state._replace(active=jnp.array([True, False]), chain_held_steps=jnp.int32(cfg.hold_chain_for))
    final = step(state, ACTION.at[0, 3].set(1.))
    assert final.fully_connected and int(final.chain_held_steps) > cfg.hold_chain_for
    np.testing.assert_array_equal(final.adaptive.spawned, [False, True])
    assert not final.success and not final.done


def test_hold_record_payments_do_not_finish_cleanup_or_repeat_after_threshold():
    cfg, state, step = scenario()
    reward = rewards(success_reward_as_hold_record=True)
    for expected in (1500., 1500., 0.):
        following = step(state, ACTION)
        _, terms = compute_rewards(state, following, reward, cfg)
        np.testing.assert_array_equal(terms["success_reward"], [expected, 0.])
        assert not following.success and not following.done
        state = following


@pytest.mark.parametrize("failure", ["timeout", "idle", "all_decommissioned"])
def test_failure_can_terminate_while_cleanup_is_pending_without_success(failure):
    overrides = {"max_steps": 2} if failure == "timeout" else {}
    if failure == "idle":
        overrides["no_movement_termination_steps"] = 2
    cfg, state, step = scenario(**overrides)
    if failure == "all_decommissioned":
        state = state._replace(pos=jnp.broadcast_to(state.adaptive.zone_center, (2, 3)))
    first = step(state, ACTION)
    final = step(first, ACTION)
    assert final.done and not final.success
    if failure == "all_decommissioned":
        assert final.adaptive.all_decommissioned
    _, terms = compute_rewards(first, final, rewards(), cfg)
    np.testing.assert_array_equal(terms["success_reward"], 0.)
    np.testing.assert_array_equal(terms["useless_agent_at_success_reward"], 0.)


def test_batched_autoreset_keeps_pending_lane_and_terminal_exit_rewards():
    cfg, state, step = scenario()
    pending = step(step(state, ACTION), ACTION)
    ready = pending._replace(pos=pending.pos.at[1].set(pending.adaptive.zone_center),
                             adaptive=pending.adaptive._replace(dwell=jnp.array([0, 1], dtype=jnp.int32)))
    states = jax.tree.map(lambda a, b: jnp.stack([a, b]), ready, pending)
    _, transition, _, _ = make_autoreset_fns(BUILDING, cfg, rewards())
    fresh, _, done, info = jax.jit(transition.batched)(states, jnp.stack([ACTION, ACTION]))
    np.testing.assert_array_equal(done, [True, False])
    np.testing.assert_array_equal(info["success"], [True, False])
    np.testing.assert_array_equal(info["non_contributor_decommission_reward"], [[0., 1500.], [0., 0.]])
    assert int(fresh.step[0]) == 0 and int(fresh.step[1]) == 3


@pytest.mark.parametrize("system", ["euclidean", "obstacle_geodesic"])
def test_cleanup_and_return_shaping_use_selected_contributor_membership(system):
    cfg, before, _ = scenario()
    _, step, _, _ = make_env_fns(BUILDING, cfg, chain_reward_system=system)
    # The tiny open building has the same selected shortest chain in either system.
    np.testing.assert_array_equal(success_contributors(before, cfg, chain_reward_system=system), [True, False])
    after = jax.jit(step)(before._replace(pos=before.pos.at[1, 1].set(4.)), ACTION)
    _, terms = compute_rewards(before, after, rewards(chain_reward_system=system), cfg)
    np.testing.assert_allclose(terms["non_contributor_return_reward"], [0., 3.])


@pytest.mark.parametrize("base_known,budget,connected,expected", [
    (False, 1, False, 10.),  # Search calls remain rewarded, even above the chain budget.
    (True, 2, False, 10.),   # A necessary reinforcement remains rewarded after delivery.
    (True, 1, False, -500.), # Sufficient population: responsible caller is penalized.
    (True, 2, True, -500.),  # An intact chain makes another deployment unnecessary.
])
def test_caller_penalty_is_delivery_gated_and_leaves_newborn_unpunished(base_known, budget, connected, expected):
    shaping = PrivelegedVoteRewardConfig(enabled=True, unnecessary_reinforcement_call_reward=-500.)
    cfg = EnvConfig(num_agents=2, spawn_delay=0, comm_radius_base=3.1, visual_radius=1.1,
        success_condition=CLEANUP,
        adapt_size=AdaptSizeConfig(enabled=True, initial_agents=1, vote_protocol="latest_decision"))
    reset, step, _, _ = make_env_fns(BUILDING, cfg, priveleged_vote_reward=shaping, plan_geodesic=False)
    before = reset(jax.random.PRNGKey(8))
    before = before._replace(
        base_pos=jnp.array([2., 2., 2.]), target_pos=jnp.array([16., 16., 16.]),
        pos=jnp.array([[5., 2., 2.], [2., 2., 2.]]), vel=jnp.zeros((2, 3)),
        target_known=jnp.array([True, False]), base_target_known=jnp.bool_(base_known),
        fully_connected=jnp.bool_(connected), target_required_agents=jnp.int32(budget),
        adaptive=before.adaptive._replace(zone_center=jnp.array([2., 2., 3.5]), zone_valid=jnp.bool_(True)),
    )
    after = jax.jit(step)(before, ACTION.at[0, 3].set(1.))
    np.testing.assert_array_equal(after.adaptive.spawned, [False, True])
    _, terms = compute_rewards(before, after, rewards(reinforcement_reward=10., priveleged_vote_reward=shaping), cfg)
    actual = terms["reinforcement_reward"] + terms["unnecessary_reinforcement_call_reward"]
    np.testing.assert_array_equal(actual, [expected, 0.])


def test_cluster_config_and_evaluation_override_accept_optional_cleanup():
    cluster = load_level("G03_CLU_random_buildings")
    local = load_level("G03_random_buildings")
    assert cluster.env.success_condition == CLEANUP
    assert cluster.reward.priveleged_vote_reward.unnecessary_reinforcement_call_reward == -500.
    assert local.env.success_condition == "chain_held"
    assert local.reward.priveleged_vote_reward.unnecessary_reinforcement_call_reward == -200.
    override = load_level("G03_random_buildings", [f"inspection_evaluation.success_condition={CLEANUP}"])
    assert override.evaluation.success_condition == CLEANUP


@pytest.mark.parametrize("validation_goal,expected", [(None, False), ("chain_held", True)])
def test_validation_environment_can_finish_before_training_cleanup(tmp_path, validation_goal, expected):
    from swarmecho.training.validation import ValidationRunner

    cfg, state, training_step = scenario()
    level = load_level("G03_CLU_random_buildings")
    level = replace(level, env=cfg, reward=RewardConfig(),
        training=replace(level.training,
            dynamic_max_geo_distance=replace(level.training.dynamic_max_geo_distance, enabled=False)),
        validation=replace(level.validation, num_maps=1, success_condition=validation_goal))
    runner = ValidationRunner(level, tmp_path,
        [Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")])
    _, validation_step, _, _ = runner._prepare()
    validation_step = jax.jit(validation_step)
    final = validation_step(validation_step(state, ACTION), ACTION)
    assert bool(final.success) == bool(final.done) == expected
    assert int(final.chain_held_steps) == cfg.hold_chain_for
    np.testing.assert_array_equal(final.active, [True, True])
    training_final = training_step(training_step(state, ACTION), ACTION)
    assert not training_final.success and not training_final.done
    assert level.env.success_condition == CLEANUP


def test_cleanup_success_survives_csv_and_replay_filtering(tmp_path):
    path = save_eval_info_csv(tmp_path / "eval.csv", np.zeros((2, 3)), np.zeros((2, 3)),
        successes=[True, False], delivered=[True, True], visually_found=[True, True], success_condition=CLEANUP)
    records = load_eval_info_csv(path)
    assert records["stages"].tolist() == ["chain_success", "found_and_delivered"]
    np.testing.assert_array_equal(_successful_stage_mask(records["stages"], CLEANUP), [True, False])
    assert _replay_stage_label("chain_success", CLEANUP) == "SUCCESS: CHAIN HELD AND CLEANED UP"
    assert _replay_stage_label("found_and_delivered", CLEANUP) == "DELIVERED"
