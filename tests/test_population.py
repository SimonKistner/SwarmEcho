"""Deployment, lifecycle, and relay-chain cleanup checks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from swarmecho.core.config import (
    AdaptSizeConfig,
    EnvConfig,
    EvaluationConfig,
    Level,
    load_level,
    LoggingConfig,
    NetworkConfig,
    PrivelegedVoteRewardConfig,
    RewardConfig,
    TrainingConfig,
    validate_adapt_size,
    validate_reward_config,
)
from swarmecho.env.adapt_size import (
    apply_vote,
    available_votes,
    initial_state,
    lifetime_ended,
    policy_resets,
    transition,
)
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import (
    _contributing_chain_agents,
    _return_progress_metres,
    chain_distance_matrix,
    compute_rewards,
    EnvState,
    make_autoreset_fns,
    make_env_fns,
    observation_dim,
    success_contributors,
)
from swarmecho.env.obstacles import roadmap_distances
from swarmecho.models.actions import physical_actions
from swarmecho.training.adapt_size_metrics import AdaptiveWindow, INFO_NAMES
from swarmecho.training.artifacts import load_eval_info_csv, save_eval_info_csv
from swarmecho.training.checkpoints import validate_checkpoint_contract
from swarmecho.training.evaluate import _replay_stage_label, _successful_stage_mask
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.ppo import loss, PPOTrainer, replay
from swarmecho.training.train import (
    build_model,
    evaluate_model,
    evaluate_suite_with_action_capture,
    replay_recorded_actions,
)
from swarmecho.visualize.replay import load_replay, planner_metadata, write_replay


# Population and vote handling

BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"))


def protocol(n=3, initial=2, **settings):
    cfg = AdaptSizeConfig(enabled=True, initial_agents=initial, **settings)
    a = initial_state(n, initial, jnp.array([100., 100., 100.]), max_steps=128)
    return cfg, a, jnp.arange(n) < initial


def tick(a, active, cfg, yes, *, peers=None, base=None, pos=None, due=True):
    n = len(active)
    return transition(a, active, jnp.zeros((n, 3)) if pos is None else pos,
                      jnp.asarray(yes, dtype=jnp.float32),
                      jnp.zeros((n, n), dtype=bool) if peers is None else peers,
                      jnp.ones(n, dtype=bool) if base is None else base, jnp.bool_(due), cfg)


def env_config(**overrides):
    return replace(EnvConfig(num_agents=3, max_steps=100, hold_chain_for=90,
                             no_movement_termination_steps=1000,
                             adapt_size=AdaptSizeConfig(enabled=True, initial_agents=2,
                                                       vote_holding=2)), **overrides)


def test_disabled_defaults_and_validation():
    cfg = EnvConfig()
    assert not cfg.adapt_size.enabled
    reset, step, obs, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(0))
    assert state.adaptive is None
    assert int(state.active.sum()) == 1
    assert obs(state).shape == (cfg.num_agents, observation_dim(cfg))
    assert step(state, jnp.zeros((cfg.num_agents, 3))).adaptive is None
    enabled = replace(cfg, adapt_size=replace(cfg.adapt_size, enabled=True))
    assert observation_dim(enabled) == observation_dim(cfg) + 2
    for field, value in (("initial_agents", 0), ("initial_agents", cfg.num_agents + 1),
                         ("vote_holding", 0), ("quorum_fraction", 1.01), ("quorum_hold", -1),
                         ("decommission_radius_m", 0), ("vote_mode", "toggle")):
        with pytest.raises(ValueError):
            validate_adapt_size(replace(cfg, adapt_size=replace(cfg.adapt_size, **{field: value})))


def test_decommission_zone_observation_toggle_and_membership():
    cfg = env_config()
    reset, _, observe, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(0))
    center = state.base_pos
    state = state._replace(
        active=jnp.array([True, True, False]),
        pos=jnp.stack([center, center + jnp.array([cfg.adapt_size.decommission_radius_m, 0., 0.]), center]),
        adaptive=state.adaptive._replace(zone_center=center, zone_valid=jnp.bool_(True)),
    )
    enabled_obs = jax.jit(observe)(state)
    # Vote is channel 6; zone membership follows it, including the sphere boundary.
    np.testing.assert_array_equal(enabled_obs[:, 7], [1., 1., 0.])
    outside = state._replace(pos=state.pos.at[0, 0].add(cfg.adapt_size.decommission_radius_m + 0.1))
    assert float(observe(outside)[0, 7]) == 0.
    invalid = state._replace(adaptive=state.adaptive._replace(zone_valid=jnp.bool_(False)))
    np.testing.assert_array_equal(observe(invalid)[:, 7], 0.)

    disabled = replace(cfg, observe_decommission_zone=False)
    _, _, observe_disabled, _ = make_env_fns(BUILDING, disabled, plan_geodesic=False)
    disabled_obs = jax.jit(observe_disabled)(state)
    assert disabled_obs.shape == (cfg.num_agents, observation_dim(disabled))
    assert observation_dim(disabled) == observation_dim(cfg) - 1
    np.testing.assert_allclose(disabled_obs, np.delete(enabled_obs, 7, axis=1))

    fixed = replace(cfg, adapt_size=AdaptSizeConfig())
    reset_fixed, _, observe_fixed, _ = make_env_fns(BUILDING, fixed, plan_geodesic=False)
    _, _, observe_fixed_disabled, _ = make_env_fns(
        BUILDING, replace(fixed, observe_decommission_zone=False), plan_geodesic=False)
    fixed_state = reset_fixed(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(observe_fixed(fixed_state), observe_fixed_disabled(fixed_state))
    with pytest.raises(ValueError, match="observe_decommission_zone must be boolean"):
        make_env_fns(BUILDING, replace(cfg, observe_decommission_zone=1), plan_geodesic=False)


def test_both_action_semantics_and_unsquashed_vote():
    previous = jnp.array([False, True, True])
    np.testing.assert_array_equal(apply_vote(previous, jnp.array([0, 1, 0]), "hold"), [0, 1, 0])
    np.testing.assert_array_equal(apply_vote(previous, jnp.array([1, 0, 2]), "keep_yes_no"), [1, 1, 0])
    actions = physical_actions(jnp.array([[20., -20., 0., 2.]]))
    np.testing.assert_allclose(actions, [[1., -1., 0., 2.]])


def test_additive_votes_no_pauses_and_noncommunication_steps_do_not_count():
    cfg, a, active = protocol(vote_holding=3)
    a, active = tick(a, active, cfg, [1, 0, 0])
    for _ in range(8):
        a, active = tick(a, active, cfg, [1, 0, 0], due=False)
    assert int(available_votes(a)[0]) == 1
    a, active = tick(a, active, cfg, [0, 0, 0])
    assert int(available_votes(a)[0]) == 1
    a, active = tick(a, active, cfg, [1, 0, 0])
    assert int(a.calls) == 0
    a, active = tick(a, active, cfg, [1, 0, 0])
    assert int(a.calls) == 1 and int(active.sum()) == 3
    assert int(a.round) == 1 and int(a.known_round[2]) == 1
    np.testing.assert_array_equal(available_votes(a), 0)
    assert bool(a.vote[0])  # Fulfillment must travel back, not reset globally.
    a, _ = tick(a, active, cfg, [1, 0, 0])
    assert not bool(a.vote[0]) and int(a.known_round[0]) == 1


def test_one_hop_transport_deduplicates_paths_and_resends():
    cfg, a, active = protocol(n=3, initial=3, vote_holding=20)
    peers = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    base = jnp.array([0, 0, 1], dtype=bool)
    for expected in (0, 0, 1):
        a, active = tick(a, active, cfg, [1, 0, 0] if expected == 0 and int(a.source_yes[0]) == 0 else [0, 0, 0],
                         peers=peers, base=base)
        assert int(available_votes(a)[0]) == expected
    for _ in range(8):
        a, active = tick(a, active, cfg, [0, 0, 0], peers=jnp.ones((3, 3), bool), base=base)
    assert int(available_votes(a)[0]) == 1  # Neither loops nor multiple paths multiply it.


def test_isolated_yes_votes_are_lost_and_relayed_votes_arrive_one_by_one():
    cfg, a, active = protocol(n=3, initial=3, vote_holding=20)
    for _ in range(6):
        a, active = tick(a, active, cfg, [1, 0, 0], base=jnp.zeros(3, bool))
    assert int(a.source_yes[0]) == 0
    assert int(available_votes(a)[0]) == 0

    peers = jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], bool)
    for expected in (1, 2, 3):
        a, active = tick(a, active, cfg, [1, 0, 0], peers=peers, base=jnp.zeros(3, bool))
        assert int(a.source_yes[0]) == expected
        assert int(a.packet_count[1, 0]) == expected
        assert int(a.packet_count[2, 0]) == expected
        assert int(available_votes(a)[0]) == 0

    # The source itself has no backlog to replay after reconnecting to base.
    a, active = tick(a, active, cfg, [0, 0, 0], base=jnp.array([1, 0, 0], bool))
    assert int(available_votes(a)[0]) == 0

    # The sender is now isolated. Two relays hold identical packets, yet the
    # base receives only one distinct credit per communication tick.
    for expected in (1, 2, 3):
        a, active = tick(a, active, cfg, [1, 0, 0], base=jnp.array([0, 1, 1], bool))
        assert int(a.source_yes[0]) == 3
        assert int(available_votes(a)[0]) == expected


def test_fulfillment_catches_up_multiple_missed_rounds_one_hop_at_a_time():
    cfg, a, active = protocol(n=3, initial=3, vote_holding=20)
    a = a._replace(round=jnp.int32(7), vote=jnp.ones(3, bool), source_yes=jnp.full(3, 8))
    peers = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    for expected in ([0, 0, 7], [0, 7, 7], [7, 7, 7]):
        a, active = tick(a, active, cfg, [0, 0, 0], peers=peers, base=jnp.array([0, 0, 1], bool))
        np.testing.assert_array_equal(a.known_round, expected)
    np.testing.assert_array_equal(a.vote, False)
    np.testing.assert_array_equal(available_votes(a), 0)


def test_nine_capacity_repeated_rejections_require_fresh_batches():
    cfg, a, active = protocol(n=9, initial=8, vote_holding=10)
    for _ in range(10):
        a, active = tick(a, active, cfg, [1] * 4 + [0] * 5)
    assert int(active.sum()) == 9 and int(a.calls) == 1
    a, active = tick(a, active, cfg, [0] * 9)  # Deliver fulfillment.
    for batch in (1, 2):
        for _ in range(9):
            a, active = tick(a, active, cfg, [1] * 5 + [0] * 4)
            assert int(a.rejected_calls) == batch - 1
        a, active = tick(a, active, cfg, [1] * 5 + [0] * 4)
        assert int(a.rejected_calls) == batch
        np.testing.assert_array_equal(a.rejected_voters, [1] * 5 + [0] * 4)
        assert int(a.round) == 1 and int(a.calls) == 1
        assert int(a.packet_count[-1, 0]) == batch * 10
        np.testing.assert_array_equal(available_votes(a), 0)


def test_already_delivered_surplus_cannot_buy_repeat_rejections():
    cfg, a, active = protocol(n=3, initial=3, vote_holding=3, quorum_fraction=.6)
    for _ in range(12):
        a, active = tick(a, active, cfg, [1, 0, 0])
    for _ in range(3):
        a, active = tick(a, active, cfg, [0, 1, 0])
    assert int(a.rejected_calls) == 1
    assert int(available_votes(a)[0]) == 0
    for _ in range(6):
        a, active = tick(a, active, cfg, [0, 1, 0])
    assert int(a.rejected_calls) == 1


def test_quorum_hold_and_strict_fraction():
    cfg, a, active = protocol(n=3, initial=2, vote_holding=1, quorum_fraction=.5, quorum_hold=3)
    a, active = tick(a, active, cfg, [1, 0, 0])
    assert int(a.calls) == 0  # Exactly 50% does not exceed 50%.
    for expected in (0, 0, 1):
        a, active = tick(a, active, cfg, [1, 1, 0])
        assert int(a.calls) == expected


def test_decommission_dwell_exit_denominator_and_fresh_slot_identity():
    cfg, a, active = protocol(n=3, initial=3, vote_holding=2, decommission_hold_steps=2,
                              quorum_fraction=.49)
    outside = jnp.zeros((3, 3))
    inside = outside.at[2].set(a.zone_center)
    a, active = tick(a, active, cfg, [1, 0, 0], pos=inside)
    a, active = tick(a, active, cfg, [0, 0, 0], pos=outside)
    assert int(a.dwell[2]) == 0
    a, active = tick(a, active, cfg, [0, 0, 0], pos=inside)
    previous = SimpleNamespace(active=active, adaptive=a)
    a, active = tick(a, active, cfg, [1, 0, 0], pos=inside)
    # Removing drone 3 changes 1/3 to 1/2: quorum succeeds immediately.
    assert int(a.decommissions) == 1 and int(a.calls) == 1
    assert int(a.ids[2]) == 3 and int(a.next_id) == 4
    assert bool(a.spawned[2]) and int(a.source_yes[2]) == 0 and not bool(a.vote[2])
    np.testing.assert_array_equal(lifetime_ended(previous, SimpleNamespace(active=active, adaptive=a)), [0, 0, 1])
    np.testing.assert_array_equal(policy_resets(SimpleNamespace(active=active, adaptive=a)), [0, 0, 1])


def test_jit_vmap_scan_population_churn_keeps_static_shapes():
    cfg, a, active = protocol(n=3, initial=2, vote_holding=1, decommission_hold_steps=1)

    def run(a, active):
        def body(carry, t):
            a, active = carry
            # Alternate removing the third drone and replacing it on subsequent ticks.
            pos = jnp.zeros((3, 3)).at[2].set(jnp.where(t % 3 == 1, a.zone_center, 0.))
            return tick(a, active, cfg, jnp.array([1, 1, 0]), pos=pos), None
        return jax.lax.scan(body, (a, active), jnp.arange(60))[0]

    batch_a = jax.tree.map(lambda x: jnp.stack([x, x]), a)
    result, alive = jax.jit(jax.vmap(run))(batch_a, jnp.stack([active, active]))
    assert result.vote.shape == (2, 3) and result.packet_count.shape == (2, 4, 3)
    assert result.packet_bits.shape == (2, 4, 3, 4)
    assert np.all(np.asarray(result.calls) > 9)
    assert np.all(np.asarray(alive.sum(-1)) <= 3)
    assert np.all(np.asarray(result.next_id) > 9)


def test_initial_spawn_delay_is_preserved_and_call_arrives_immediately():
    cfg = env_config(spawn_delay=5, adapt_size=replace(env_config().adapt_size,
                                                       initial_agents=2, vote_holding=1))
    reset, step, _, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False,
                                     memory_comm_every_k_steps=1)
    state = reset(jax.random.PRNGKey(40))
    np.testing.assert_array_equal(state.active, [1, 0, 0])
    np.testing.assert_array_equal(state.adaptive.ids, [0, 1, -1])
    # The lone active drone may call slot 3; slot 2 remains reserved for its
    # already scheduled initial deployment. The call incurs no spawn delay.
    state = jax.jit(step)(state, jnp.zeros((3, 4)).at[0, 3].set(1))
    np.testing.assert_array_equal(state.active, [1, 0, 1])
    np.testing.assert_array_equal(state.adaptive.ids, [0, 1, 2])
    assert int(state.adaptive.calls) == 1 and int(state.step) == 1
    for expected in (2, 3, 4):
        state = step(state, jnp.zeros((3, 4)))
        assert int(state.step) == expected
        assert not bool(state.active[1])
    state = step(state, jnp.zeros((3, 4)))
    np.testing.assert_array_equal(state.active, [1, 1, 1])
    assert bool(state.adaptive.spawned[1]) and int(state.adaptive.calls) == 1
    assert not bool(state.target_known[1]) and not bool(state.adaptive.vote[1])


def test_delayed_initial_drone_is_not_charged_as_a_reinforcement():
    cfg = env_config(spawn_delay=2)
    reset, step, _, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    reward_cfg = RewardConfig(reinforcement_reward=-10.)
    state = reset(jax.random.PRNGKey(41))
    state1 = step(state, jnp.zeros((3, 4)))
    state2 = step(state1, jnp.zeros((3, 4)))
    assert bool(state2.adaptive.spawned[1])
    assert int(state2.adaptive.calls) == 0
    _, terms = compute_rewards(state1, state2, reward_cfg, cfg)
    np.testing.assert_array_equal(terms["reinforcement_reward"], 0)
    yes = jnp.zeros((3, 4)).at[0, 3].set(1)
    voted1 = step(state, yes)
    voted2 = step(voted1, yes)
    assert int(voted2.adaptive.calls) == 1
    np.testing.assert_array_equal(voted2.adaptive.spawned, [0, 1, 1])
    _, voted_terms = compute_rewards(voted1, voted2, reward_cfg, cfg)
    assert float(voted_terms["reinforcement_reward"].sum()) == pytest.approx(-10.)


@pytest.mark.parametrize("terminated_lanes", [1, 3])
def test_all_decommissioned_terminal_info_survives_sparse_and_dense_autoreset(terminated_lanes):
    cfg = env_config(spawn_delay=0, adapt_size=replace(env_config().adapt_size,
                                                       decommission_hold_steps=1))
    reward = RewardConfig(all_decommissioned_penalty=-60., time_presure_reward=-3.)
    reset, step, _, _ = make_autoreset_fns(BUILDING, cfg, reward)
    states = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(2), 3))
    inside = jnp.broadcast_to(states.adaptive.zone_center[:, None, :], states.pos.shape)
    pos = jnp.where((jnp.arange(3) < terminated_lanes)[:, None, None], inside, states.pos)
    states = states._replace(pos=pos)
    fresh, _, done, info = jax.jit(step.batched)(states, jnp.zeros((3, 3, 4)))
    np.testing.assert_array_equal(done, np.arange(3) < terminated_lanes)
    np.testing.assert_array_equal(info["adapt_end_agents"][:terminated_lanes], 0)
    np.testing.assert_array_equal(info["adapt_decommissions"][:terminated_lanes], 2)
    np.testing.assert_array_equal(info["adapt_all_decommissioned"][:terminated_lanes], True)
    np.testing.assert_allclose(info["all_decommissioned_penalty"][:terminated_lanes].sum(-1), -120.)
    np.testing.assert_allclose(info["time_presure_reward"][:terminated_lanes].sum(-1), -6.)
    np.testing.assert_array_equal(fresh.active.sum(-1), 2)
    np.testing.assert_array_equal(fresh.adaptive.calls, 0)
    np.testing.assert_array_equal(fresh.adaptive.vote, False)


def test_capacity_and_useless_agent_rewards_have_correct_individual_recipients():
    cfg = env_config(comm_radius=4.1, comm_radius_base=4.1, visual_radius=2.1)
    reset, step, _, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(3))
    state = state._replace(base_pos=jnp.array([2., 2., 2.]), target_pos=jnp.array([10., 2., 2.]),
                           pos=jnp.array([[5., 2., 2.], [9., 2., 2.], [17., 17., 10.]]),
                           active=jnp.ones(3, bool), target_known=jnp.ones(3, bool),
                           adaptive=initial_state(3, 3, jnp.array([100., 100., 100.])))
    connected = step(state, jnp.zeros((3, 4)))
    assert bool(connected.fully_connected)
    np.testing.assert_array_equal(success_contributors(connected, cfg), [1, 1, 0])
    successful = connected._replace(success=jnp.bool_(True), done=jnp.bool_(True),
        adaptive=connected.adaptive._replace(rejected_voters=jnp.array([1, 1, 0], bool)))
    reward = RewardConfig(useless_agent_at_success_reward=-30., rejected_call_reward=-7., success_reward=120.)
    _, terms = compute_rewards(state, successful, reward, cfg)
    np.testing.assert_allclose(terms["useless_agent_at_success_reward"], [0., 0., -30.])
    np.testing.assert_allclose(terms["rejected_call_reward"], [-7., -7., 0.])
    np.testing.assert_allclose(terms["success_reward"], [120., 120., 120.])
    _, repeated = compute_rewards(successful, successful, reward, cfg)
    np.testing.assert_array_equal(repeated["useless_agent_at_success_reward"], 0)


def test_complete_chain_excludes_redundant_frontier_leader():
    cfg = EnvConfig(num_agents=5, comm_radius=3.1, comm_radius_base=3.1)
    state = SimpleNamespace(
        pos=jnp.array([[3., 0., 0.], [6., 0., 0.], [9., 0., 0.],
                       [12., 0., 0.], [0., 3., 0.]]),
        base_pos=jnp.array([0., 0., 0.]),
        active=jnp.ones(5, bool),
        directly_sees_target=jnp.array([0, 0, 0, 1, 0], bool),
        solid_min=jnp.zeros((0, 3)), solid_max=jnp.zeros((0, 3)),
        fully_connected=jnp.bool_(True),
    )
    # D5 is one hop from base but is a dead end. A frontier leader at D5
    # previously pulled the complete-chain reward onto that longer detour.
    contributors = _contributing_chain_agents(state, cfg, frontier_leaders=jnp.array([4, 4]))
    np.testing.assert_array_equal(contributors, [1, 1, 1, 1, 0])


def test_gae_does_not_cross_decommission_replacement_boundary():
    buffer = MAPPORolloutBuffer(2, 1, 1, 1, 4, gamma=1., gae_lambda=1., recurrent=True, mask_inactive=True)
    for reward, ended in ((2., True), (100., False)):
        buffer.add(MAPPOTransition(obs=np.zeros((1, 1, 1)), actions=np.zeros((1, 1, 4)),
            log_probs=np.zeros((1, 1)), values=np.zeros((1, 1)), rewards=np.array([[reward]]),
            dones=np.array([False]), active_masks=np.ones((1, 1), bool), agent_terminated=np.array([[ended]])))
    _, returns = buffer.compute_gae(np.array([[300.]]), np.array([False]))
    np.testing.assert_allclose(returns[:, 0, 0], [2., 400.])


def tiny_level(mode="hold"):
    # Use the small map directly; its historical level reward block is unsupported.
    level = Level(
        name="test", map_names=["M00_no_maze_open_cuboid"], building=BUILDING,
        env=EnvConfig(), reward=RewardConfig(), training=TrainingConfig(),
        network=NetworkConfig(), evaluation=EvaluationConfig(), logging=LoggingConfig(),
    )
    return replace(level, env=env_config(max_steps=6, adapt_size=AdaptSizeConfig(
        enabled=True, initial_agents=2, vote_mode=mode, vote_holding=1)),
        network=replace(level.network, hidden_dim=8, num_layers=1, actor_num_layers=1,
                        tarmac_sig_dim=4, tarmac_val_dim=4, memory_comm_every_k_steps=3),
        training=replace(level.training, num_epochs=1, value_normalization="none",
                         randomize_base=False, minimum_geodesic_separation=False),
        evaluation=replace(level.evaluation, num_agents=None, success_condition=None,
                           eval_differes_from_training_map=False, minimum_geodesic_separation=False))


def test_evaluation_useless_agents_use_success_terminal_state(monkeypatch, tmp_path):
    import sys
    from swarmecho.training import train

    level = tiny_level()
    cfg = replace(level.env, max_steps=3, hold_chain_for=1, comm_radius=4.1,
                  comm_radius_base=4.1, visual_radius=2.1,
                  adapt_size=replace(level.env.adapt_size, initial_agents=3))
    level = replace(level, env=cfg)
    reset, step, observations, env_metrics = make_env_fns(BUILDING, cfg, plan_geodesic=False)

    def fixed_reset(key, **kwargs):
        return reset(key)._replace(
            base_pos=jnp.array([2., 2., 2.]), target_pos=jnp.array([10., 2., 2.]),
            pos=jnp.array([[5., 2., 2.], [9., 2., 2.], [17., 17., 10.]]),
            active=jnp.ones(3, bool), target_known=jnp.ones(3, bool),
            target_required_agents=jnp.int32(2))

    def terminal_then_empty(state, actions):
        # Later scan steps must not overwrite the first successful population.
        state = state._replace(active=jnp.where(state.done, False, state.active))
        return step(state, jnp.zeros((3, 4)))

    monkeypatch.setattr(train, "_evaluation_env_fns",
                        lambda level: (fixed_reset, terminal_then_empty, observations, env_metrics))
    model = build_model(level)
    metrics, info = train.evaluate_suite(model, level, episodes=2,
                                        return_episode_info=True, return_required_agents=True)
    assert metrics["eval_success"] == 1
    assert metrics["eval_useless_agents_at_success"] == 1
    assert metrics["eval_informed_count"] == 6
    assert metrics["eval_peer-informed-pct"] == 0
    assert metrics["eval_base-informed-pct"] == 0
    np.testing.assert_array_equal(info["required_agents"], [2, 2])
    validation_batch = train._run_parallel_evaluation_jit(
        model, jax.random.PRNGKey(0), jax.random.PRNGKey(1),
        reset=fixed_reset, step=terminal_then_empty, observations=observations,
        env_metrics=env_metrics, reward_cfg=level.reward, cfg=cfg,
        num_envs=2, horizon=3, return_population=True,
        return_required_agents=True, return_useless_agents=True,
        return_information=True, return_geo=True)
    np.testing.assert_array_equal(validation_batch[15], [3, 3])
    np.testing.assert_array_equal(validation_batch[16], [2, 2])
    np.testing.assert_array_equal(validation_batch[17], [1, 1])
    np.testing.assert_array_equal(validation_batch[18], [[3, 0, 0], [3, 0, 0]])
    assert validation_batch[-2].shape == (2, 3, model.hidden_dim)
    assert validation_batch[-1].shape == (2, 3)
    captured, _, capture = train.evaluate_suite_with_action_capture(
        model, level, episodes=2, result="success", offset=0)
    assert captured["eval_useless_agents_at_success"] == 1
    assert captured["eval_informed_count"] == 6
    assert capture["length"] == 1
    assert capture["actions"].shape == (3, 3, 4)
    logged = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(log=lambda values, **kwargs: logged.append(values)))
    monkeypatch.setattr(train, "evaluation_row", lambda *args, **kwargs: None)
    train._report_evaluation(metrics, level, update=1, total=1, steps=6, phase="eval",
                             wandb_run=object(), history_path=tmp_path / "history.jsonl",
                             hold=SimpleNamespace(consecutive=0))
    assert logged[0]["eval/useless_agents_at_success"] == 1
    assert logged[0]["eval/peer-informed-pct"] == 0
    assert logged[0]["eval/base-informed-pct"] == 0


@pytest.mark.parametrize("mode", ["hold", "keep_yes_no"])
def test_recurrent_ppo_exact_replay_gradients_and_fresh_memory(mode):
    level = tiny_level(mode)
    model = build_model(level)
    n, d, h, s = 3, model.obs_dim, 8, 4
    obs = jnp.ones((n, d)) * .2
    active = jnp.array([1, 1, 0], bool)
    reset = jnp.array([1, 0, 1], bool)
    hidden = jnp.ones((n, h)) * 12
    sig = val = jnp.ones((n, s)) * 12
    keys = jax.random.split(jax.random.PRNGKey(4), n)
    kwargs = dict(reset=reset, active=active, comm_mask=jnp.zeros((n, n), bool),
                  base_signature=jnp.zeros(s), base_value=jnp.zeros(s), base_memory_mask=jnp.zeros(n, bool))
    out = model.actor.act_team(obs, hidden, sig, val, keys, **kwargs)
    fresh = model.actor.act_team(obs, hidden.at[0].set(0), sig.at[0].set(0), val.at[0].set(0), keys, **kwargs)
    for a, b in zip(out, fresh):
        np.testing.assert_allclose(a, b)
    np.testing.assert_array_equal(out[0][2], 0)
    assert out[3].shape == (3, 4)
    mb = {"obs": obs[None, None], "actions": out[3][None, None],
          "old_log_probs": out[4][None, None], "old_values": jnp.zeros((1, 1, n)),
          "returns": jnp.ones((1, 1, n)), "advantages": jnp.ones((1, 1, n)),
          "rnn_resets": reset[None, None], "active_masks": active[None, None],
          "comm_masks": kwargs["comm_mask"][None, None],
          "base_signatures": jnp.zeros((1, 1, s)), "base_values": jnp.zeros((1, 1, s)),
          "base_memory_masks": jnp.zeros((1, 1, n), bool),
          "initial_actor_h": hidden[None], "initial_actor_signature": sig[None],
          "initial_actor_value": val[None], "initial_critic_h": jnp.zeros((1, n, h))}
    _, lp, *_ = replay(model, mb, jax.random.PRNGKey(5))
    # Vmapped replay and scalar sampling use slightly different fp32 reduction
    # orders on XLA; the categorical/Gaussian density must still closely match.
    np.testing.assert_allclose(lp[0, 0, :2], out[4][:2], rtol=1e-4, atol=5e-4)
    objective = lambda m: loss(m, mb, jax.random.PRNGKey(5), actor_clip_eps=.2,
                               value_clip_eps=None, entropy_mode="squashed", vf_coef=.5, ent_coef=.01)[0]
    value, grads = nnx.value_and_grad(objective)(model)
    assert np.isfinite(value)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    assert np.any(np.asarray(grads["actor"]["vote_head"]["bias"][...]) != 0)
    before = np.array(model.actor.vote_head.bias[...])
    trainer = PPOTrainer(model, level.training)
    stats = trainer.update([mb])
    assert np.isfinite(stats["total_loss"])
    assert np.any(np.asarray(model.actor.vote_head.bias[...]) != before)


def test_metrics_full_window_definitions_and_undefined_ratios():
    window = AdaptiveWindow(2)
    def row(c, d, end, rejected=0, extinct=0, useless=0):
        return dict(zip(INFO_NAMES, (c, d, rejected, end, extinct, useless, True)))
    window.append(row(5, 2, 5, useless=1), success=True)
    assert window.metrics() == {}
    window.append(row(0, 2, 0, rejected=2, extinct=1), success=False)
    m = window.metrics()
    assert m["adapt_size/active_agents_at_EP_end_mean/all"] == 2.5
    assert m["adapt_size/active_agents_at_EP_end_mean/success"] == 5
    assert m["adapt_size/active_agents_at_EP_end_mean/fail"] == 0
    assert m["adapt_size/called_decommissioned_mean"] == 1
    assert m["adapt_size/called_decommissioned_fraction"] == .4
    assert m["adapt_size/called_decommissioned_fraction_pooled"] == .4
    assert m["adapt_size/total_calls"] == 5 and m["adapt_size/total_decommissions"] == 4
    assert m["adapt_size/total_calls_to_decommissions_ratio"] == 1.25
    assert m["adapt_size/decommission_termination_rate"] == .5
    assert m["adapt_size/rejected_calls_mean"] == 1
    assert m["adapt_size/useless_agents_at_success"] == 1
    for _ in range(2):
        window.append(row(0, 0, 2), success=False)
    m = window.metrics()
    assert m["adapt_size/total_calls"] == 0  # Rolling totals, not lifetime totals.
    assert "adapt_size/called_decommissioned_fraction" not in m
    assert "adapt_size/active_agents_at_EP_end_mean/success" not in m


@pytest.mark.parametrize("mode", ["hold", "keep_yes_no"])
def test_evaluation_capture_replay_and_inspector_payload(mode, tmp_path):
    level = tiny_level(mode)
    model = build_model(level)
    frames, rewards = evaluate_model(model, level, max_steps=3)
    assert len(frames) >= 2 and frames[0].adaptive is not None
    _, manifest = write_replay(tmp_path / "replay", frames, map_name="test", dt=level.env.dt,
                               metadata=planner_metadata(level.env), reward_terms=rewards, progress=False)
    meta, data = load_replay(manifest)
    assert meta["adapt_size"]["vote_mode"] == mode
    assert data["adapt_vote"].shape == data["active"].shape
    assert data["adapt_available_votes"].shape == data["active"].shape
    metrics, info, capture = evaluate_suite_with_action_capture(model, level, episodes=2,
                                                                max_steps=3, result="fail", offset=0)
    assert capture["actions"].shape[-1] == 4
    replay_frames, _ = replay_recorded_actions(level, capture["actions"], capture_lane=capture["lane"],
                                              batch_size=2, max_steps=3)
    np.testing.assert_allclose(replay_frames[0].target_pos, info["target_positions"][capture["lane"]])
    assert len(replay_frames) == capture["length"] + 1
    assert np.isfinite(metrics["eval_return"])


def test_checkpoints_reject_enabled_and_vote_mode_mismatches(tmp_path):
    model = build_model(tiny_level())
    with pytest.raises(ValueError, match="Adaptive voting"):
        validate_checkpoint_contract(model, tmp_path)
    contract = dict(model.checkpoint_contract)
    (tmp_path / "training_contract.json").write_text(json.dumps(contract))
    validate_checkpoint_contract(model, tmp_path)
    contract["vote_mode"] = "keep_yes_no"
    (tmp_path / "training_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="vote_mode"):
        validate_checkpoint_contract(model, tmp_path)


def test_decommission_zone_checkpoint_legacy_default_and_toggle(tmp_path):
    level = tiny_level()
    model = build_model(level)
    legacy = dict(model.checkpoint_contract)
    assert legacy.pop("observe_decommission_zone") is True
    (tmp_path / "training_contract.json").write_text(json.dumps(legacy))
    validate_checkpoint_contract(model, tmp_path)
    disabled = build_model(replace(level, env=replace(level.env, observe_decommission_zone=False)))
    assert dict(disabled.checkpoint_contract)["observe_decommission_zone"] is False
    with pytest.raises(ValueError, match="observe_decommission_zone"):
        validate_checkpoint_contract(disabled, tmp_path)


# Chain completion and cleanup

CLEANUP_BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"))
CLEANUP = "chain_held_and_cleand_up"
ACTION = jnp.zeros((2, 4))


def cleanup_scenario(condition=CLEANUP, **overrides):
    cfg = replace(EnvConfig(
        num_agents=2, spawn_delay=0, success_condition=condition, hold_chain_for=2,
        max_steps=30, comm_radius_base=3.1, comm_radius=3.1, visual_radius=1.1,
        adapt_size=AdaptSizeConfig(enabled=True, initial_agents=2,
                                  vote_protocol="latest_decision", decommission_hold_steps=2),
    ), **overrides)
    reset, step, _, _ = make_env_fns(CLEANUP_BUILDING, cfg)
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
    cfg, state, step = cleanup_scenario(condition)
    first = step(state, ACTION)
    assert not first.success and not first.done
    second = step(first, ACTION)
    assert int(second.chain_held_steps) == cfg.hold_chain_for
    assert bool(second.success) == bool(second.done) == expected
    _, terms = compute_rewards(first, second, rewards(), cfg)
    np.testing.assert_array_equal(terms["success_reward"], [3000., 3000.] if expected else [0., 0.])
    np.testing.assert_array_equal(terms["useless_agent_at_success_reward"], [0., -3000.] if expected else [0., 0.])


def test_cleanup_after_hold_finishes_on_removal_and_pays_departing_lifetime():
    cfg, state, step = cleanup_scenario()
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
    cfg, state, step = cleanup_scenario(hold_chain_for=3)
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
    cfg, state, step = cleanup_scenario(adapt_size=AdaptSizeConfig(
        enabled=True, initial_agents=1, vote_protocol="latest_decision"))
    state = state._replace(active=jnp.array([True, False]), chain_held_steps=jnp.int32(cfg.hold_chain_for))
    final = step(state, ACTION.at[0, 3].set(1.))
    assert final.fully_connected and int(final.chain_held_steps) > cfg.hold_chain_for
    np.testing.assert_array_equal(final.adaptive.spawned, [False, True])
    assert not final.success and not final.done


def test_hold_record_payments_do_not_finish_cleanup_or_repeat_after_threshold():
    cfg, state, step = cleanup_scenario()
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
    cfg, state, step = cleanup_scenario(**overrides)
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
    cfg, state, step = cleanup_scenario()
    pending = step(step(state, ACTION), ACTION)
    ready = pending._replace(pos=pending.pos.at[1].set(pending.adaptive.zone_center),
                             adaptive=pending.adaptive._replace(dwell=jnp.array([0, 1], dtype=jnp.int32)))
    states = jax.tree.map(lambda a, b: jnp.stack([a, b]), ready, pending)
    _, transition, _, _ = make_autoreset_fns(CLEANUP_BUILDING, cfg, rewards())
    fresh, _, done, info = jax.jit(transition.batched)(states, jnp.stack([ACTION, ACTION]))
    np.testing.assert_array_equal(done, [True, False])
    np.testing.assert_array_equal(info["success"], [True, False])
    np.testing.assert_array_equal(info["non_contributor_decommission_reward"], [[0., 1500.], [0., 0.]])
    assert int(fresh.step[0]) == 0 and int(fresh.step[1]) == 3


@pytest.mark.parametrize("system", ["euclidean", "obstacle_geodesic"])
def test_cleanup_and_return_shaping_use_selected_contributor_membership(system):
    cfg, before, _ = cleanup_scenario()
    _, step, _, _ = make_env_fns(CLEANUP_BUILDING, cfg, chain_reward_system=system)
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
    reset, step, _, _ = make_env_fns(CLEANUP_BUILDING, cfg, priveleged_vote_reward=shaping, plan_geodesic=False)
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
    local = load_level("G03_legacy_random_buildings")
    assert cluster.env.success_condition == CLEANUP
    assert cluster.reward.priveleged_vote_reward.unnecessary_reinforcement_call_reward == -500.
    assert local.env.success_condition == "chain_held"
    assert local.reward.priveleged_vote_reward.unnecessary_reinforcement_call_reward == -200.
    override = load_level("G03_legacy_random_buildings", [f"inspection_evaluation.success_condition={CLEANUP}"])
    assert override.evaluation.success_condition == CLEANUP


@pytest.mark.parametrize("validation_goal,expected", [(None, False), ("chain_held", True)])
def test_validation_environment_can_finish_before_training_cleanup(tmp_path, validation_goal, expected):
    from swarmecho.training.validation import ValidationRunner

    cfg, state, training_step = cleanup_scenario()
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


# Non-contributor rewards

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


@pytest.mark.parametrize("name", ["G03_legacy_random_buildings", "G03_CLU_random_buildings"])
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


# Recruitment reward attribution

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
