"""Adaptive population contracts. Run in WSL: uv run pytest -q tests/test_adapt_size.py.

Uses tiny authored environments; no building bank, W&B account or training run.
Includes compiled JAX transitions, recurrent PPO gradients and replay round trips.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from swarmecho.core.config import AdaptSizeConfig, EnvConfig, RewardConfig, load_level, validate_adapt_size
from swarmecho.env.adapt_size import initial_state, transition, available_votes, apply_vote, lifetime_ended, policy_resets
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import make_env_fns, make_autoreset_fns, observation_dim, compute_rewards, success_contributors, _contributing_chain_agents
from swarmecho.models.actions import physical_actions
from swarmecho.training.adapt_size_metrics import AdaptiveWindow, INFO_NAMES
from swarmecho.training.checkpoints import validate_checkpoint_contract
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.ppo import replay, loss, PPOTrainer
from swarmecho.training.train import build_model, evaluate_model, evaluate_suite_with_action_capture, replay_recorded_actions
from swarmecho.visualize.replay import write_replay, load_replay, planner_metadata


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
                         ("vote_holding", 0), ("quorum_fraction", 1), ("quorum_hold", -1),
                         ("decommission_radius_m", 0), ("vote_mode", "toggle")):
        with pytest.raises(ValueError):
            validate_adapt_size(replace(cfg, adapt_size=replace(cfg.adapt_size, **{field: value})))


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


def test_environment_birth_has_zero_knowledge_and_only_two_new_observations():
    cfg = env_config(adapt_size=replace(env_config().adapt_size, vote_holding=1))
    reset, step, obs, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False, memory_comm_every_k_steps=3)
    state = reset(jax.random.PRNGKey(1))
    assert bool(state.adaptive.zone_valid)
    assert np.all(np.linalg.norm(np.asarray(state.pos) - state.adaptive.zone_center, axis=-1) > cfg.adapt_size.decommission_radius_m)
    np.testing.assert_allclose(state.adaptive.zone_center[:2], state.base_pos[:2])
    assert float(state.adaptive.zone_center[2] - state.pos[0, 2]) > cfg.adapt_size.decommission_radius_m
    state = state._replace(target_known=jnp.array([1, 1, 0], bool), base_target_known=jnp.bool_(True))
    born = jax.jit(step)(state, jnp.zeros((3, 4)).at[0, 3].set(1))
    assert bool(born.adaptive.spawned[2])
    assert not bool(born.target_known[2])
    np.testing.assert_array_equal(born.vel[2], 0)
    np.testing.assert_allclose(born.pos[2], state.base_pos + jnp.array([0., 0., cfg.drone_radius]))
    assert obs(born).shape == (3, observation_dim(cfg))
    for _ in range(2):
        born = step(born, jnp.zeros((3, 4)).at[:, 3].set(1))
    assert int(born.adaptive.rejected_calls) == 0


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
    reward_cfg = RewardConfig(reinforcement_cost=10.)
    state = reset(jax.random.PRNGKey(41))
    state1 = step(state, jnp.zeros((3, 4)))
    state2 = step(state1, jnp.zeros((3, 4)))
    assert bool(state2.adaptive.spawned[1])
    assert int(state2.adaptive.calls) == 0
    _, terms = compute_rewards(state1, state2, reward_cfg, cfg)
    np.testing.assert_array_equal(terms["reinforcement_cost"], 0)
    yes = jnp.zeros((3, 4)).at[0, 3].set(1)
    voted1 = step(state, yes)
    voted2 = step(voted1, yes)
    assert int(voted2.adaptive.calls) == 1
    np.testing.assert_array_equal(voted2.adaptive.spawned, [0, 1, 1])
    _, voted_terms = compute_rewards(voted1, voted2, reward_cfg, cfg)
    assert float(voted_terms["reinforcement_cost"].sum()) == pytest.approx(-10.)


@pytest.mark.parametrize("terminated_lanes", [1, 3])
def test_all_decommissioned_terminal_info_survives_sparse_and_dense_autoreset(terminated_lanes):
    cfg = env_config(spawn_delay=0, adapt_size=replace(env_config().adapt_size,
                                                       decommission_hold_steps=1))
    reward = RewardConfig(all_decommissioned_penalty=60., time_penalty_per_step=3.)
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
    np.testing.assert_allclose(info["all_decommissioned_penalty"][:terminated_lanes].sum(-1), -60.)
    np.testing.assert_allclose(info["time_penalty"][:terminated_lanes].sum(-1), -3.)
    np.testing.assert_array_equal(fresh.active.sum(-1), 2)
    np.testing.assert_array_equal(fresh.adaptive.calls, 0)
    np.testing.assert_array_equal(fresh.adaptive.vote, False)


def test_capacity_and_success_penalties_have_correct_recipients_and_team_total():
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
    reward = RewardConfig(success_unused_agent_penalty=30., rejected_call_penalty=7.)
    _, terms = compute_rewards(state, successful, reward, cfg)
    np.testing.assert_allclose(terms["useless_agetnst_at_success_penalty"], [-10., -10., -10.])
    np.testing.assert_allclose(terms["rejected_call_penalty"], [-7., -7., 0.])
    contributor_bonus = replace(reward, success_bonus=120., success_bonus_contributors_only=True)
    _, bonus_terms = compute_rewards(state, successful, contributor_bonus, cfg)
    np.testing.assert_allclose(bonus_terms["success"], [60., 60., 0.])
    _, repeated = compute_rewards(successful, successful, reward, cfg)
    np.testing.assert_array_equal(repeated["useless_agetnst_at_success_penalty"], 0)


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
    level = load_level("M00_no_maze_open_cuboid")
    return replace(level, env=env_config(max_steps=6, adapt_size=AdaptSizeConfig(
        enabled=True, initial_agents=2, vote_mode=mode, vote_holding=1)),
        network=replace(level.network, hidden_dim=8, num_layers=1, actor_num_layers=1,
                        tarmac_sig_dim=4, tarmac_val_dim=4, memory_comm_every_k_steps=3),
        training=replace(level.training, num_epochs=1, value_normalization="none",
                         randomize_base=False, minimum_geodesic_separation=False),
        evaluation=replace(level.evaluation, num_agents=None, success_condition=None,
                           eval_differes_from_training_map=False, minimum_geodesic_separation=False))


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
