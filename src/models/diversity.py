"""
swarmecho/models/diversity.py
=============================
Forward-history diversity model for CDS-style MAPPO.

The model compares a shared next-observation predictor against a role-aware
predictor. If the role-aware predictor explains the next observation better,
the transition carries role-specific information and can produce intrinsic
reward during rollout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """LayerNorm + Tanh MLP."""

    def __init__(
        self,
        in_features:  int,
        hidden_dim:   int,
        num_layers:   int,
        out_features: int,
        rngs:         nnx.Rngs,
    ) -> None:
        assert num_layers >= 1
        layers, norms = [], []
        prev = in_features
        for _ in range(num_layers):
            layers.append(nnx.Linear(prev, hidden_dim, rngs=rngs))
            norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            prev = hidden_dim
        self.layers     = nnx.List(layers)
        self.norms      = nnx.List(norms)
        self.out_linear = nnx.Linear(hidden_dim, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for lin, norm in zip(self.layers, self.norms):
            x = jnp.tanh(norm(lin(x)))
        return self.out_linear(x)


class ForwardHistoryDiversity(nnx.Module):
    """
    Shared-vs-role forward predictor.

    history has shape (..., history_len, obs_dim + act_dim), where the action is
    the squashed normalized action that was actually sent to the physics step
    before max_force scaling.
    """

    def __init__(
        self,
        history_len:         int,
        history_feature_dim: int,
        obs_dim:             int,
        num_roles:           int,
        hidden_dim:          int,
        num_layers:          int,
        rngs:                nnx.Rngs,
    ) -> None:
        self.history_len = history_len
        self.history_feature_dim = history_feature_dim
        self.history_dim = history_len * history_feature_dim
        self.obs_dim = obs_dim
        self.num_roles = num_roles

        self.shared_predictor = MLP(
            self.history_dim, hidden_dim, num_layers, obs_dim, rngs
        )
        self.role_predictor = MLP(
            self.history_dim + num_roles, hidden_dim, num_layers, obs_dim, rngs
        )

    def _flat_history(self, history: jax.Array) -> jax.Array:
        return history.reshape(*history.shape[:-2], self.history_dim)

    def predict(
        self,
        history:  jax.Array,
        role_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        h = self._flat_history(history)
        shared_mu = self.shared_predictor(h)
        role_oh = jax.nn.one_hot(role_ids, self.num_roles, dtype=h.dtype)
        role_mu = self.role_predictor(jnp.concatenate([h, role_oh], axis=-1))
        return shared_mu, role_mu

    def squared_errors(
        self,
        history:  jax.Array,
        next_obs: jax.Array,
        role_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        shared_mu, role_mu = self.predict(history, role_ids)
        shared_se = jnp.mean((shared_mu - next_obs) ** 2, axis=-1)
        role_se = jnp.mean((role_mu - next_obs) ** 2, axis=-1)
        return shared_se, role_se

    def intrinsic_reward(
        self,
        history:  jax.Array,
        next_obs: jax.Array,
        role_ids: jax.Array,
        mask:     jax.Array,
    ) -> jax.Array:
        shared_se, role_se = self.squared_errors(history, next_obs, role_ids)
        # Fixed-variance Gaussian log q_role - log q_shared.
        return jax.lax.stop_gradient(0.5 * (shared_se - role_se) * mask)

    def loss(
        self,
        history:  jax.Array,
        next_obs: jax.Array,
        role_ids: jax.Array,
        mask:     jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        shared_se, role_se = self.squared_errors(history, next_obs, role_ids)
        mask = mask.astype(shared_se.dtype)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        shared_loss = jnp.sum(shared_se * mask) / denom
        role_loss = jnp.sum(role_se * mask) / denom
        return shared_loss + role_loss, shared_loss, role_loss
