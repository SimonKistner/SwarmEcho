"""Deployment, lifecycle, and relay-chain cleanup checks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import math

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
    RewardConfig,
    TrainingConfig,
    validate_adapt_size,
    validate_reward_config,
)
from swarmecho.env.adapt_size import (
    base_population_feedback,
    initial_state,
    transition,
)
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import (
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


# Automatic deployment and population observations

BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml"))


def env_config(**overrides):
    return replace(EnvConfig(num_agents=3, max_steps=100, hold_chain_for=90,
                             adapt_size=AdaptSizeConfig(enabled=True)), **overrides)


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
    assert observation_dim(enabled) == observation_dim(cfg) + 1
    for field, value in (("decommission_hold_steps", 0), ("decommission_radius_m", 0)):
        with pytest.raises(ValueError):
            validate_adapt_size(replace(cfg, adapt_size=replace(cfg.adapt_size, **{field: value})))


def test_population_observation_is_the_only_optional_channel_when_the_zone_is_off():
    cfg = env_config(observe_base_population=True, observe_decommission_zone=False)
    reset, step, observe, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(0))._replace(
        active=jnp.array([True, True, False]), is_conn_base=jnp.array([True, False, True]),
    )
    off = replace(cfg, observe_base_population=False)
    _, _, observe_off, _ = make_env_fns(BUILDING, off, plan_geodesic=False)
    enabled = observe(state)
    assert enabled.shape[-1] == observation_dim(off) + 1
    expected = base_population_feedback(state.active, state.is_conn_base, 10.)
    np.testing.assert_allclose(expected, [.2, 0., 0.])
    np.testing.assert_allclose(enabled[:, 6], expected)
    np.testing.assert_allclose(np.delete(enabled, 6, axis=1), observe_off(state))
    with pytest.raises(ValueError, match="Expected actions shaped"):
        step(state, jnp.zeros((3, 4)))


def test_automatic_launches_are_staggered_and_removed_drones_stay_retired():
    cfg = AdaptSizeConfig(enabled=True, decommission_hold_steps=1)
    state = initial_state(3, jnp.zeros(3))
    active = jnp.array([True, False, False])
    positions = jnp.array([[0., 0., 0.], [5., 0., 0.], [5., 0., 0.]])
    state, active = transition(state, active, positions, cfg, spawn_delay=2)
    np.testing.assert_array_equal(active, [False, False, False])
    assert not state.all_decommissioned
    assert int(state.ids[0]) == -1
    state, active = transition(state, active, positions, cfg, spawn_delay=2)
    np.testing.assert_array_equal(active, [False, True, False])
    state, active = transition(state, active, positions, cfg, spawn_delay=2)
    np.testing.assert_array_equal(active, [False, True, False])
    state, active = transition(state, active, positions, cfg, spawn_delay=2)
    np.testing.assert_array_equal(active, [False, True, True])
    assert int(state.decommissions) == 1


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
    # Zone membership includes the sphere boundary.
    np.testing.assert_array_equal(enabled_obs[:, 6], [1., 1., 0.])
    outside = state._replace(pos=state.pos.at[0, 0].add(cfg.adapt_size.decommission_radius_m + 0.1))
    assert float(observe(outside)[0, 6]) == 0.
    invalid = state._replace(adaptive=state.adaptive._replace(zone_valid=jnp.bool_(False)))
    np.testing.assert_array_equal(observe(invalid)[:, 6], 0.)

    disabled = replace(cfg, observe_decommission_zone=False)
    _, _, observe_disabled, _ = make_env_fns(BUILDING, disabled, plan_geodesic=False)
    disabled_obs = jax.jit(observe_disabled)(state)
    assert disabled_obs.shape == (cfg.num_agents, observation_dim(disabled))
    assert observation_dim(disabled) == observation_dim(cfg) - 1
    np.testing.assert_allclose(disabled_obs, np.delete(enabled_obs, 6, axis=1))

    fixed = replace(cfg, adapt_size=AdaptSizeConfig())
    reset_fixed, _, observe_fixed, _ = make_env_fns(BUILDING, fixed, plan_geodesic=False)
    _, _, observe_fixed_disabled, _ = make_env_fns(
        BUILDING, replace(fixed, observe_decommission_zone=False), plan_geodesic=False)
    fixed_state = reset_fixed(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(observe_fixed(fixed_state), observe_fixed_disabled(fixed_state))
    with pytest.raises(ValueError, match="observe_decommission_zone must be boolean"):
        make_env_fns(BUILDING, replace(cfg, observe_decommission_zone=1), plan_geodesic=False)



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
    fresh, _, done, info = jax.jit(step.batched)(states, jnp.zeros((3, 3, 3)))
    np.testing.assert_array_equal(done, np.arange(3) < terminated_lanes)
    np.testing.assert_array_equal(info["adapt_end_agents"][:terminated_lanes], 0)
    np.testing.assert_array_equal(info["adapt_decommissions"][:terminated_lanes], 3)
    np.testing.assert_array_equal(info["adapt_all_decommissioned"][:terminated_lanes], True)
    np.testing.assert_allclose(info["all_decommissioned_penalty"][:terminated_lanes].sum(-1), -180.)
    np.testing.assert_allclose(info["time_presure_reward"][:terminated_lanes].sum(-1), -9.)
    np.testing.assert_array_equal(fresh.active.sum(-1), 3)


def test_capacity_and_useless_agent_rewards_have_correct_individual_recipients():
    cfg = env_config(comm_radius=4.1, comm_radius_base=4.1, visual_radius=2.1)
    reset, step, _, _ = make_env_fns(BUILDING, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(3))
    state = state._replace(base_pos=jnp.array([2., 2., 2.]), target_pos=jnp.array([10., 2., 2.]),
                           pos=jnp.array([[5., 2., 2.], [9., 2., 2.], [17., 17., 10.]]),
                           active=jnp.ones(3, bool), target_known=jnp.ones(3, bool),
                           adaptive=initial_state(3, jnp.array([100., 100., 100.])))
    connected = step(state, jnp.zeros((3, 3)))
    assert bool(connected.fully_connected)
    np.testing.assert_array_equal(success_contributors(connected, cfg), [1, 1, 0])
    successful = connected._replace(success=jnp.bool_(True), done=jnp.bool_(True))
    reward = RewardConfig(useless_agent_at_success_reward=-30., success_reward=120.)
    _, terms = compute_rewards(state, successful, reward, cfg)
    np.testing.assert_allclose(terms["useless_agent_at_success_reward"], [0., 0., -30.])
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
    contributors = success_contributors(state, cfg)
    np.testing.assert_array_equal(contributors, [1, 1, 1, 1, 0])


def test_gae_stops_at_decommission():
    buffer = MAPPORolloutBuffer(2, 1, 1, 1, 3, gamma=1., gae_lambda=1.)
    for reward, ended in ((2., True), (100., False)):
        buffer.add(MAPPOTransition(obs=np.zeros((1, 1, 1)), actions=np.zeros((1, 1, 3)),
            log_probs=np.zeros((1, 1)), values=np.zeros((1, 1)), rewards=np.array([[reward]]),
            dones=np.array([False]), active_masks=np.ones((1, 1), bool), agent_terminated=np.array([[ended]])))
    _, returns = buffer.compute_gae(np.array([[300.]]), np.array([False]))
    np.testing.assert_allclose(returns[:, 0, 0], [2., 400.])


def tiny_level():
    # Use the small map directly; its historical level reward block is unsupported.
    level = Level(
        name="test", map_names=["open_cuboid"], building=BUILDING,
        env=EnvConfig(), reward=RewardConfig(), training=TrainingConfig(),
        network=NetworkConfig(), evaluation=EvaluationConfig(), logging=LoggingConfig(),
    )
    return replace(level, env=env_config(max_steps=6, adapt_size=AdaptSizeConfig(
        enabled=True)),
        network=replace(level.network, hidden_dim=8, num_layers=1, actor_num_layers=1,
                        tarmac_sig_dim=4, tarmac_val_dim=4),
        training=replace(level.training, num_epochs=1,
                         randomize_base=False),
        evaluation=replace(level.evaluation, num_agents=None, success_condition=None,
                           eval_differes_from_training_map=False))


def test_evaluation_useless_agents_use_success_terminal_state(monkeypatch, tmp_path):
    import sys
    from swarmecho.training import train

    level = tiny_level()
    cfg = replace(level.env, max_steps=3, hold_chain_for=1, comm_radius=4.1,
                  comm_radius_base=4.1, visual_radius=2.1)
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
        return step(state, jnp.zeros((3, 3)))

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
        model, jax.random.PRNGKey(0),
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
    assert capture["actions"].shape == (3, 3, 3)
    logged = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(log=lambda values, **kwargs: logged.append(values)))
    monkeypatch.setattr(train, "evaluation_row", lambda *args, **kwargs: None)
    train._report_evaluation(metrics, level, update=1, total=1, steps=6, phase="eval",
                             wandb_run=object(), history_path=tmp_path / "history.jsonl")
    assert logged[0]["eval/useless_agents_at_success"] == 1
    assert logged[0]["eval/peer-informed-pct"] == 0
    assert logged[0]["eval/base-informed-pct"] == 0



def test_recurrent_ppo_exact_replay_gradients_and_fresh_memory():
    level = tiny_level()
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
    assert out[3].shape == (3, 3)
    mb = {"obs": obs[None, None], "actions": out[3][None, None],
          "old_log_probs": out[4][None, None], "old_values": jnp.zeros((1, 1, n)),
          "returns": jnp.ones((1, 1, n)), "advantages": jnp.ones((1, 1, n)),
          "rnn_resets": reset[None, None], "active_masks": active[None, None],
          "comm_masks": kwargs["comm_mask"][None, None],
          "base_signatures": jnp.zeros((1, 1, s)), "base_values": jnp.zeros((1, 1, s)),
          "base_memory_masks": jnp.zeros((1, 1, n), bool),
          "base_write_indices": jnp.full((1, 1), -1, dtype=jnp.int32),
          "episode_dones": jnp.zeros((1, 1), dtype=bool),
          "initial_actor_h": hidden[None], "initial_actor_signature": sig[None],
          "initial_actor_value": val[None], "initial_critic_h": jnp.zeros((1, n, h))}
    _, lp, *_ = replay(model, mb, jax.random.PRNGKey(5))
    # Vmapped replay and scalar sampling use slightly different fp32 reduction
    # orders on XLA; the Gaussian density must still closely match.
    np.testing.assert_allclose(lp[0, 0, :2], out[4][:2], rtol=1e-4, atol=5e-4)
    objective = lambda m: loss(m, mb, jax.random.PRNGKey(5), actor_clip_eps=.2,
                               value_clip_eps=.2, vf_coef=.5, ent_coef=.01)[0]
    value, grads = nnx.value_and_grad(objective)(model)
    assert np.isfinite(value)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    assert np.any(np.asarray(grads["actor"]["mu_head"]["bias"][...]) != 0)
    before = np.array(model.actor.mu_head.bias[...])
    aux_valid = active[None, :, None]
    grid_width = math.prod(model.geo_probe_head.grid_shape)
    mb.update(
        env_indices=jnp.array([0]),
        aux_times=jnp.where(aux_valid, 0, -1),
        aux_targets=jnp.zeros((1, n, 1, grid_width), dtype=bool),
        aux_valid=aux_valid,
        aux_weights=aux_valid.astype(jnp.float32) / aux_valid.sum(),
    )
    trainer = PPOTrainer(model, level.training)
    stats = trainer.update([mb])
    assert np.isfinite(stats["total_loss"])
    assert np.any(np.asarray(model.actor.mu_head.bias[...]) != before)



def test_evaluation_capture_replay_and_inspector_payload(tmp_path):
    level = tiny_level()
    model = build_model(level)
    frames, rewards = evaluate_model(model, level, max_steps=3)
    assert len(frames) >= 2 and frames[0].adaptive is not None
    _, manifest = write_replay(tmp_path / "replay", frames, map_name="test", dt=level.env.dt,
                               metadata=planner_metadata(level.env), reward_terms=rewards, progress=False)
    meta, data = load_replay(manifest)
    assert meta["adapt_size"]["enabled"]
    assert data["adapt_decommissioned"].shape == data["active"].shape
    assert "adapt_vote" not in data
    metrics, info, capture = evaluate_suite_with_action_capture(model, level, episodes=2,
                                                                max_steps=3, result="fail", offset=0)
    assert capture["actions"].shape[-1] == 3
    replay_frames, _ = replay_recorded_actions(level, capture["actions"], capture_lane=capture["lane"],
                                              batch_size=2, max_steps=3)
    np.testing.assert_allclose(replay_frames[0].target_pos, info["target_positions"][capture["lane"]])
    assert len(replay_frames) == capture["length"] + 1
    assert np.isfinite(metrics["eval_return"])



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

CLEANUP_BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml"))
CLEANUP = "chain_held_and_cleand_up"
ACTION = jnp.zeros((2, 3))


def cleanup_scenario(condition=CLEANUP, **overrides):
    cfg = replace(EnvConfig(
        num_agents=2, spawn_delay=0, success_condition=condition, hold_chain_for=2,
        max_steps=30, comm_radius_base=3.1, comm_radius=3.1, visual_radius=1.1,
        adapt_size=AdaptSizeConfig(enabled=True, decommission_hold_steps=2),
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



@pytest.mark.parametrize("failure", ["timeout", "all_decommissioned"])
def test_failure_can_terminate_while_cleanup_is_pending_without_success(failure):
    overrides = {"max_steps": 2} if failure == "timeout" else {}
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


def test_cleanup_and_return_shaping_use_selected_contributor_membership():
    cfg, before, _ = cleanup_scenario()
    _, step, _, _ = make_env_fns(CLEANUP_BUILDING, cfg)
    np.testing.assert_array_equal(success_contributors(before, cfg), [True, False])
    after = jax.jit(step)(before._replace(pos=before.pos.at[1, 1].set(4.)), ACTION)
    _, terms = compute_rewards(before, after, rewards(), cfg)
    np.testing.assert_allclose(terms["non_contributor_return_reward"], [0., 3.])


def test_g03_training_and_inspection_have_independent_success_conditions():
    level = load_level("G03_CLU_random_buildings")
    assert level.env.success_condition == CLEANUP
    assert level.evaluation.success_condition == "chain_held"
    override = load_level("G03_CLU_random_buildings", [f"inspection_evaluation.success_condition={CLEANUP}"])
    assert override.evaluation.success_condition == CLEANUP


@pytest.mark.parametrize("validation_goal,expected", [(None, False), ("chain_held", True)])
def test_validation_environment_can_finish_before_training_cleanup(tmp_path, validation_goal, expected):
    from swarmecho.training.validation import ValidationRunner

    cfg, state, training_step = cleanup_scenario()
    level = load_level("G03_CLU_random_buildings")
    level = replace(level, env=cfg, reward=RewardConfig(),
        validation=replace(level.validation, num_maps=1, success_condition=validation_goal))
    runner = ValidationRunner(level, tmp_path,
        [Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml")])
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
                    adapt_size=AdaptSizeConfig(enabled=True,
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
        adaptive=initial_state(4, jnp.array([0., 0., 1.])),
    )
    return env, state


def reward_config(**overrides):
    return replace(RewardConfig(
        exploration_reward=0., finder_reward=0., target_found_reward=0., success_reward=0.,
        collision_reward=0., time_presure_reward=0., gap_progress_reward_per_meter=0.,
        all_decommissioned_penalty=0.,
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
                                        "automatic_launch", "invalid_zone", "becomes_contributor"])
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
    elif change == "automatic_launch":
        after = after._replace(adaptive=after.adaptive._replace(spawned=after.active.at[:3].set(False)))
    elif change == "invalid_zone":
        after = after._replace(adaptive=after.adaptive._replace(zone_valid=jnp.bool_(False)))
    else:
        after = after._replace(pos=after.pos.at[2].set([20., 0., 0.]).at[3].set([9., 0., 0.]),
                               directly_sees_target=jnp.array([0, 0, 0, 1], dtype=bool))
    _, terms = compute_rewards(before, after, reward_config(), env)
    np.testing.assert_array_equal(terms["non_contributor_return_reward"], 0.)
    np.testing.assert_array_equal(terms["non_contributor_departure_reward"], 0.)


def removed_state(state, slot=3):
    removed = jnp.zeros(4, dtype=bool).at[slot].set(True)
    return state._replace(
        pos=state.pos.at[slot].set(state.base_pos),
        active=state.active.at[slot].set(False), step=state.step + 1,
        adaptive=state.adaptive._replace(
            decommissioned=removed, decommissions=jnp.int32(1),
            ids=state.adaptive.ids.at[slot].set(-1),
        ),
    )


def test_qualified_decommission_is_paid_once_to_departing_lifetime():
    env, before = scenario(1.)
    after = removed_state(before)
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


def test_g03_configures_surplus_agent_cleanup_rewards():
    name = "G03_CLU_random_buildings"
    data = yaml.safe_load(Path(f"src/swarmecho/curriculum_config/levels/G/{name}.yaml").read_text())
    cfg = data["reward"]
    assert data["env"]["observe_chain_contributor"] is True
    assert cfg["non_contributor_return_reward_per_meter"] == 3.
    assert cfg["non_contributor_departure_reward_per_meter"] == -3.
    assert cfg["non_contributor_decommission_reward"] == cfg["success_reward"] == 3000.
    assert cfg["decommission_reward"] == 0.
    level = load_level(name)
    assert level.reward.non_contributor_return_reward_per_meter == 3.
    assert level.reward.non_contributor_departure_reward_per_meter == -3.
    assert level.reward.non_contributor_decommission_reward == 3000.


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
    building = load_building(Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml"))
    env = EnvConfig(num_agents=2, spawn_delay=0, hold_chain_for=2, comm_radius_base=10.,
                    adapt_size=AdaptSizeConfig(enabled=True, decommission_hold_steps=1))
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
    fresh, rewards, done, info = jax.jit(transition.batched)(states, jnp.zeros((2, 2, 3)))
    np.testing.assert_array_equal(done, True)
    np.testing.assert_array_equal(info["non_contributor_decommission_reward"], [[0, 1500], [0, 1500]])
    np.testing.assert_array_equal(rewards, [[0, 1500], [0, 1500]])
    np.testing.assert_array_equal(info["agent_terminated"], [[False, True], [False, True]])
    np.testing.assert_array_equal(fresh.adaptive.decommissions, 0)
