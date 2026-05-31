"""
swarmecho/models/actor.py
==========================
Decentralised actor network for SwarmEcho (CTDE — Centralised Training,
Decentralised Execution).

Network
-------
  Local obs (obs_dim,) → MLP trunk → mu, log_std

Action space
------------
  Uses tanh squashing: action = tanh(u), where u ~ N(mu, std).
  The squashed action is stored in the buffer. During PPO updates we invert the
  squash with atanh(action) and include the tanh Jacobian term in the log-prob.
  This prevents boundary gradient explosion issues.
  Physical force = action × max_force is applied by the runner, NOT here.

Parameter sharing
-----------------
  One actor instance is shared across all N agents (called via vmap).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0

# Small epsilon added inside log(1 - tanh²(x)) to prevent log(0)
_TANH_EPS = 1e-6


# ---------------------------------------------------------------------------
# Building block: MLP with LayerNorm + Tanh activations
# ---------------------------------------------------------------------------

class MLP(nnx.Module):
    """
    Multi-layer perceptron with LayerNorm + Tanh activations.

    Parameters
    ----------
    in_features  : input dimension
    hidden_dim   : width of each hidden layer
    num_layers   : number of hidden layers (≥ 1)
    out_features : output dimension (no activation on final layer)
    rngs         : Flax NNX random-number generators
    """

    def __init__(
        self,
        in_features:  int,
        hidden_dim:   int,
        num_layers:   int,
        out_features: int,
        rngs:         nnx.Rngs,
    ) -> None:
        assert num_layers >= 1, "Need at least one hidden layer."
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


# ---------------------------------------------------------------------------
# Optional role adapter for CDS-style specialization
# ---------------------------------------------------------------------------

class RoleAdapter(nnx.Module):
    """
    Zero-initialized role-specific residual head.

    The module predicts one residual action mean per role and selects the slot
    addressed by role_ids. Because the projection starts at zero, enabling the
    adapter does not change the initial shared policy.
    """

    def __init__(
        self,
        hidden_dim: int,
        act_dim:    int,
        num_roles:  int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.act_dim   = act_dim
        self.num_roles = num_roles
        self.out = nnx.Linear(
            hidden_dim,
            num_roles * act_dim,
            kernel_init = nnx.initializers.zeros,
            bias_init   = nnx.initializers.zeros,
            rngs        = rngs,
        )

    def __call__(self, features: jax.Array, role_ids: jax.Array | None) -> jax.Array:
        out_shape = (*features.shape[:-1], self.act_dim)
        if role_ids is None:
            return jnp.zeros(out_shape, dtype=features.dtype)

        all_deltas = self.out(features).reshape(*features.shape[:-1], self.num_roles, self.act_dim)
        role_oh = jax.nn.one_hot(role_ids, self.num_roles, dtype=features.dtype)
        return jnp.sum(all_deltas * role_oh[..., None], axis=-2)


# ---------------------------------------------------------------------------
# DecentralizedActor
# ---------------------------------------------------------------------------

class DecentralizedActor(nnx.Module):
    """
    Shared actor network — every agent runs the same weights on its local obs.

    Input : obs_i  (obs_dim,)    — agent i's local observation
    Output: mu, log_std          — parameters of a diagonal Gaussian policy
                                   BOTH clipped to give actions in [-1, 1]

    Parameters
    ----------
    obs_dim          : observation dimension
    act_dim          : action dimension (2 for SwarmEcho)
    hidden_dim       : hidden layer width
    actor_num_layers : number of hidden layers (recommended: 2)
    rngs             : Flax NNX RNG state
    """

    def __init__(
        self,
        obs_dim:          int,
        act_dim:          int,
        hidden_dim:       int,
        actor_num_layers: int,
        rngs:             nnx.Rngs,
        use_role_adapters: bool  = False,
        num_roles:         int   = 0,
        adapter_scale:     float = 1.0,
    ) -> None:
        self.act_dim           = act_dim
        self.use_role_adapters = use_role_adapters
        self.num_roles         = num_roles
        self.adapter_scale     = adapter_scale
        self.trunk        = MLP(obs_dim, hidden_dim, actor_num_layers, hidden_dim, rngs)
        self.mu_head      = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        if use_role_adapters:
            if num_roles <= 0:
                raise ValueError("num_roles must be positive when role adapters are enabled.")
            self.role_mu = RoleAdapter(hidden_dim, act_dim, num_roles, rngs)

    def __call__(
        self,
        obs:      jax.Array,
        role_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """obs: (..., obs_dim) → mu (..., act_dim), log_std (..., act_dim)"""
        feat    = self.trunk(obs)
        mu      = self.mu_head(feat)
        if self.use_role_adapters:
            mu = mu + self.adapter_scale * self.role_mu(feat, role_ids)
        log_std = jnp.clip(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def act(
        self,
        obs:           jax.Array,   # (obs_dim,)
        key:           jax.Array,
        deterministic: bool = False,
        role_id:       jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Sample a tanh-squashed action for one agent.

        Action space
        ------------
        We use tanh squashing (not hard clip) to keep the action in (-1, 1):
          1. Sample pre-squash noise: u ~ N(mu, std)
          2. Squash:  action = tanh(u)  ∈ (-1, 1)

        This is numerically stable (no hard boundaries → no gradient explosions)
        and keeps old_log_probs / new_log_probs consistent, preventing NaN entropy.

        The buffer stores the squashed action. evaluate_actions clips it away
        from the exact boundary, maps it back with atanh, and uses the same
        squashed-Gaussian log-prob.

        Returns
        -------
        action   : (act_dim,)  — squashed normalised action in (-1, 1)
        log_prob : ()         — squashed Gaussian log probability
        entropy  : ()         — Gaussian entropy on u (for logging)
        """
        mu, log_std = self(obs, role_id)
        std = jnp.exp(log_std)

        if deterministic:
            u = mu
        else:
            u = mu + std * jax.random.normal(key, mu.shape)

        action = jnp.tanh(u)   # squash to (-1, 1)

        gaussian_log_prob = -0.5 * jnp.sum(
            ((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_det = jnp.sum(jnp.log(1.0 - action ** 2 + _TANH_EPS))
        log_prob = gaussian_log_prob - log_det

        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std)

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs:     jax.Array,   # (..., obs_dim)
        actions: jax.Array,   # (..., act_dim)  — squashed normalised actions
        role_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Evaluate log-probs and entropy for stored squashed actions.
        Used in the PPO update step.

        `actions` must be the squashed normalised values returned by act().

        Returns
        -------
        log_prob : (...,)
        entropy  : (...,)
        """
        mu, log_std = self(obs, role_ids)
        std = jnp.exp(log_std)

        clipped_actions = jnp.clip(actions, -1.0 + _TANH_EPS, 1.0 - _TANH_EPS)
        u = jnp.arctanh(clipped_actions)

        gaussian_log_prob = -0.5 * jnp.sum(
            ((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        log_det = jnp.sum(jnp.log(1.0 - clipped_actions ** 2 + _TANH_EPS), axis=-1)
        log_prob = gaussian_log_prob - log_det

        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        return log_prob, entropy

    def role_adapter_l1(self, obs: jax.Array, role_ids: jax.Array | None) -> jax.Array:
        """Mean absolute role residual on the supplied observations."""
        if not self.use_role_adapters:
            return jnp.array(0.0, dtype=obs.dtype)
        feat = self.trunk(obs)
        delta = self.role_mu(feat, role_ids)
        return jnp.mean(jnp.abs(delta))
