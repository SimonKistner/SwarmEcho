"""Focused regression checks for the 3D PPO controls. No training jobs required."""
import json
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from swarmecho.core.config import TrainingConfig, load_level
from swarmecho.env.environment import EnvConfig, RewardConfig, observation_dim
from swarmecho.models.critic import RecurrentAgentCentricCritic
from swarmecho.training.checkpoints import validate_checkpoint_contract
from swarmecho.training.ppo import RunningValueNormalizer, loss
from swarmecho.training.train import communication_due


class Actor(nnx.Module):
    def __init__(self):
        self.mean = nnx.Param(jnp.float32(3.0))

    def __call_team__(self, obs, h, sig, val, **kwargs):
        shape = (*obs.shape[:-1], 3)
        return h, sig, val, jnp.broadcast_to(self.mean[...], shape), jnp.full(shape, -2.0)


class Critic(nnx.Module):
    def __init__(self):
        self.prediction = nnx.Param(jnp.float32(10.0))

    def values_sequence(self, obs, hidden, resets, **kwargs):
        return hidden, jnp.broadcast_to(self.prediction[...], obs.shape[:-1])


class Model(nnx.Module):
    def __init__(self):
        self.actor, self.critic = Actor(), Critic()


def minibatch():
    shape = (2, 1, 2)
    return {
        "obs": jnp.zeros((*shape, 1)), "critic_obs": jnp.zeros((*shape, 1)),
        "actions": jnp.full((*shape, 3), 3.0), "old_log_probs": jnp.zeros(shape),
        "old_values": jnp.zeros(shape), "returns": jnp.full(shape, 20.0),
        "advantages": jnp.zeros(shape), "rnn_resets": jnp.zeros(shape, dtype=bool),
        "active_masks": jnp.broadcast_to(jnp.array([True, False]), shape),
        "comm_masks": jnp.zeros((*shape, 2), dtype=bool),
        "base_signatures": jnp.zeros((2, 1, 1)), "base_values": jnp.zeros((2, 1, 1)),
        "base_memory_masks": jnp.zeros(shape, dtype=bool),
        "initial_actor_h": jnp.zeros((1, 2, 1)),
        "initial_actor_signature": jnp.zeros((1, 2, 1)),
        "initial_actor_value": jnp.zeros((1, 2, 1)),
        "initial_critic_h": jnp.zeros((1, 2, 1)),
    }


def objective(model, batch, **overrides):
    settings = dict(actor_clip_eps=0.2, value_clip_eps=0.2, entropy_mode="legacy", vf_coef=0.5, ent_coef=0.01)
    settings.update(overrides)
    return loss(model, batch, jax.random.PRNGKey(0), **settings)


def test_clip_none_is_unclipped_and_zero_is_not():
    model, batch = Model(), minibatch()
    assert float(objective(model, batch)[1]["value_loss"]) == pytest.approx(196.02)
    assert float(objective(model, batch, value_clip_eps=None)[1]["value_loss"]) == pytest.approx(50.0)
    assert float(objective(model, batch, value_clip_eps=0.0)[1]["value_loss"]) == pytest.approx(200.0)
    assert float(objective(model, batch, actor_clip_eps=0.9)[1]["value_loss"]) == pytest.approx(196.02)


def test_inactive_targets_do_not_change_loss_or_gradients():
    model, batch = Model(), minibatch()
    changed = {**batch, "returns": batch["returns"].at[:, :, 1].set(1e10),
               "advantages": batch["advantages"].at[:, :, 1].set(1e10)}
    for mode in ("legacy", "squashed"):
        value, grad = nnx.value_and_grad(lambda m: objective(m, batch, entropy_mode=mode)[0])(model)
        other, other_grad = nnx.value_and_grad(lambda m: objective(m, changed, entropy_mode=mode)[0])(model)
        np.testing.assert_allclose(value, other)
        for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(other_grad)):
            np.testing.assert_allclose(a, b)


def test_squashed_entropy_pushes_saturated_mean_back_toward_center():
    model, batch = Model(), minibatch()
    def gradient(mode):
        return nnx.grad(lambda m: objective(m, batch, entropy_mode=mode, vf_coef=0.0, ent_coef=1.0)[0])(model)
    assert float(gradient("legacy")["actor"]["mean"].value) == pytest.approx(0.0)
    assert float(gradient("squashed")["actor"]["mean"].value) > 0.0


def test_value_normalizer_merges_moments_and_round_trips():
    normalizer = RunningValueNormalizer()
    normalizer.update(jnp.float32(2), jnp.float32(2), jnp.float32(1))  # samples 1, 3
    normalizer.update(jnp.float32(2), jnp.float32(6), jnp.float32(1))  # samples 5, 7
    assert float(normalizer.mean[...]) == pytest.approx(4.0)
    assert float(normalizer.variance[...]) == pytest.approx(5.0)
    raw = jnp.array([-100000.0, 0.0, 125.0])
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(raw)), raw, atol=0.01)
    normalizer.update(jnp.float32(0), jnp.float32(0), jnp.float32(0))
    assert float(normalizer.mean[...]) == pytest.approx(4.0)


def test_episode_clock_does_not_restart_at_rollout_boundary():
    steps = jnp.array([98, 99, 100, 101, 102, 0, 1])
    np.testing.assert_array_equal(communication_due(steps, 3), [False, True, False, False, True, True, False])


def test_inactive_critic_tokens_are_invisible_and_all_inactive_is_finite():
    critic = RecurrentAgentCentricCritic(3, 8, 1, nnx.Rngs(0))
    obs, hidden = jnp.ones((1, 2, 3)), jnp.zeros((1, 2, 8))
    active = jnp.array([[True, False]])
    _, expected = critic(obs, hidden, active=active)
    _, actual = critic(obs.at[:, 1].set(1e6), hidden.at[:, 1].set(1e6), active=active)
    np.testing.assert_allclose(actual, expected)
    h, values = critic(obs, hidden, active=jnp.zeros_like(active))
    np.testing.assert_array_equal(h, 0.0)
    np.testing.assert_array_equal(values, 0.0)


def test_3d_defaults_and_find_only_overrides():
    defaults = TrainingConfig()
    assert defaults.entropy_mode == "legacy" and defaults.value_normalization == "none"
    assert RewardConfig().no_movement_termination_penalty == -1000.0
    cfg = EnvConfig()
    assert not cfg.observe_current_timestep
    assert observation_dim(replace(cfg, observe_current_timestep=True)) == observation_dim(cfg) + 1
    level = load_level("B01a_office_find_only", [
        "training.value_clip_eps=null",
        "reward.no_movement_termination_penalty=-100000.0",
    ])
    assert level.training.value_clip_eps is None
    assert level.training.entropy_mode == "squashed" and level.training.value_normalization == "running"
    assert level.env.observe_current_timestep
    assert level.reward.no_movement_termination_penalty == -100000.0


def test_checkpoint_rejects_silent_value_unit_changes(tmp_path):
    model = SimpleNamespace(checkpoint_contract=(("format", 1), ("value_normalization", "running")))
    with pytest.raises(ValueError, match="no value-normalization metadata"):
        validate_checkpoint_contract(model, tmp_path)
    (tmp_path / "training_contract.json").write_text(json.dumps({"format": 1, "value_normalization": "none"}))
    with pytest.raises(ValueError, match="value_normalization"):
        validate_checkpoint_contract(model, tmp_path)
