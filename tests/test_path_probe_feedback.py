"""Proof of concept for the previous-step G03 actor observation."""

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.core.config import load_level
from swarmecho.training.train import build_model


def test_previous_path_probe_feedback():
    level = load_level("G03_CLU_random_buildings")
    enabled = load_level("G03_CLU_random_buildings", [
        "training.geo_probe_auxiliary.observe_prev_path_probe=true",
    ])
    control = build_model(level, hidden_dim=8)
    model = build_model(enabled, hidden_dim=8)
    assert (control.obs_dim, model.obs_dim, model.critic_obs_dim) == (141, 266, 141)

    n = level.env.num_agents
    raw = jax.random.uniform(jax.random.PRNGKey(1), (n, 141))
    active = jnp.arange(n) < 5
    known = active.at[2].set(False)
    no_resets = jnp.zeros(n, dtype=bool)
    hidden = model.initial_actor_hidden()
    critic_hidden = model.initial_critic_hidden()
    initial_obs = model.actor_observations(raw, model.initial_path_probe(), active, known, no_resets)
    np.testing.assert_array_equal(initial_obs[:, 141:], np.zeros((n, 125)))
    np.testing.assert_array_equal(
        control.actor_observations(raw, control.initial_path_probe(), active, known, no_resets), raw,
    )

    hidden, _, _, _, _, _, _ = model.rollout_step_recurrent(
        initial_obs, jax.random.split(jax.random.PRNGKey(2), n), hidden, critic_hidden,
        no_resets, active=active,
    )
    emitted = model.next_path_probe(hidden, active, known)
    expected = jax.nn.sigmoid(model.geo_probe_head(hidden))
    np.testing.assert_allclose(emitted[0], expected[0])
    np.testing.assert_array_equal(emitted[2], np.zeros(125))
    np.testing.assert_array_equal(emitted[5], np.zeros(125))

    # Slot 1 starts a new lifetime, slot 2 just learned the target, slot 3
    # retired, and slot 4 currently has no target knowledge.
    next_active = active.at[3].set(False)
    next_known = known.at[2].set(True).at[4].set(False)
    resets = no_resets.at[1].set(True)
    next_obs = model.actor_observations(raw, emitted, next_active, next_known, resets)
    np.testing.assert_array_equal(next_obs[:, :141], raw)
    np.testing.assert_allclose(next_obs[0, 141:], expected[0])
    np.testing.assert_array_equal(next_obs[1:, 141:], np.zeros((n - 1, 125)))

    _, values_with_feedback = model.get_value_recurrent(next_obs, critic_hidden, resets, active=next_active)
    _, values_without_feedback = model.get_value_recurrent(raw, critic_hidden, resets, active=next_active)
    np.testing.assert_allclose(values_with_feedback, values_without_feedback)
