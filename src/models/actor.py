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
  Uses tanh squashing: physical_action = tanh(u) × max_force, where u ~ N(mu, std).
  The pre-squash sample u is stored in the buffer. The policy log-probabilities
  and updates are computed directly on the pre-squash values u as standard Gaussian
  densities (without the need for explicit Tanh Jacobian correction as the correction
  terms cancel out in the PPO ratio).
  This prevents boundary gradient explosion issues.
  Physical force = tanh(u) × max_force is applied by the runner, NOT here.

Parameter sharing
-----------------
  One actor instance is shared across all N agents (called via vmap).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from models.recurrent import GRUCell

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0

# Small epsilon added inside log(1 - tanh²(x)) to prevent log(0)
_TANH_EPS = 1e-6


def tarmac_aggregate(
    query: jax.Array,
    signature: jax.Array,
    value: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Return TarMAC targeted communication context for one team.

    Parameters
    ----------
    query     : (N, S)
        Receiver queries projected from each agent's previous GRU hidden state.
    signature : (M, S)
        Sender keys/signatures from agents plus optional base replay token.
    value     : (M, V)
        Sender message values matching ``signature``.
    mask      : (N, M)
        True where receiver ``i`` may attend to sender token ``j``. The caller
        owns all environment communication rules, including walls, base range,
        active-agent masks, self-attention policy, and sparse comm steps.

    Returns
    -------
    context : (N, V)
        One value-space communication context per receiver. Receivers with no
        valid sender get an all-zero context.
    """
    sig_dim = query.shape[-1]
    scores = jnp.einsum("ns,ms->nm", query, signature) / jnp.sqrt(jnp.asarray(sig_dim, query.dtype))
    has_any = jnp.any(mask, axis=-1, keepdims=True)
    first_key = jnp.arange(mask.shape[-1]) == 0
    safe_mask = mask | ((~has_any) & first_key[None, :])
    scores = jnp.where(safe_mask, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("nm,mv->nv", weights, value)
    return jnp.where(has_any, context, jnp.zeros_like(context))


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
    ) -> None:
        self.act_dim = act_dim
        self.trunk        = MLP(obs_dim, hidden_dim, actor_num_layers, hidden_dim, rngs)
        self.mu_head      = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """obs: (..., obs_dim) → mu (..., act_dim), log_std (..., act_dim)"""
        feat    = self.trunk(obs)
        mu      = self.mu_head(feat)
        log_std = jnp.clip(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def act(
        self,
        obs:           jax.Array,   # (obs_dim,)
        key:           jax.Array,
        deterministic: bool = False,
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

        The buffer stores the PRE-SQUASH sample `u` so that evaluate_actions can
        recompute the exact same log-prob on u without needing to invert tanh.
        PPO updates are computed directly in the pre-squash space (omitting explicit
        Jacobian corrections since they cancel out in the PPO ratio).

        Returns
        -------
        action   : (act_dim,)  — pre-squash sample u (runner squashes and scales)
        log_prob : ()                       — Gaussian log probability on u
        entropy  : ()                       — Gaussian entropy on u (for logging)
        """
        mu, log_std = self(obs)
        std = jnp.exp(log_std)

        if deterministic:
            u = mu
        else:
            u = mu + std * jax.random.normal(key, mu.shape)

        action = jnp.tanh(u)   # squash to (-1, 1)

        # Gaussian log-prob at pre-squash sample u
        log_prob = -0.5 * jnp.sum(
            ((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )

        # True squashed entropy = Gaussian entropy + log(1 - tanh^2(u))
        ent_gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std)
        jacobian = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        entropy = ent_gaussian + jnp.sum(jacobian)

        # Return the pre-squash sample as the "stored action" so evaluate_actions
        # can recompute the same log-prob exactly (buffer stores u, not tanh(u)).
        return u, log_prob, entropy

    def evaluate_actions(
        self,
        obs:     jax.Array,   # (..., obs_dim)
        actions: jax.Array,   # (..., act_dim)  — PRE-SQUASH samples u from buffer
    ) -> tuple[jax.Array, jax.Array]:
        """
        Evaluate log-probs and entropy for stored pre-squash actions.
        Used in the PPO update step.

        `actions` must be the PRE-SQUASH values u (as returned by act()),
        NOT the tanh-squashed actions.

        Returns
        -------
        log_prob : (...,)
        entropy  : (...,)
        """
        mu, log_std = self(obs)
        std = jnp.exp(log_std)

        log_prob = -0.5 * jnp.sum(
            ((actions - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )

        ent_gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        jacobian = 2.0 * (jnp.log(2.0) - actions - jax.nn.softplus(-2.0 * actions))
        entropy = ent_gaussian + jnp.sum(jacobian, axis=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# RecurrentDecentralizedActor
# ---------------------------------------------------------------------------

class RecurrentDecentralizedActor(nnx.Module):
    """
    Shared recurrent actor for decentralized execution.

    Flow per agent and timestep:

        obs_i -> encoder MLP -> GRU_i -> policy MLP -> mu/log_std

    The GRU state is per environment and per agent. It is reset by the runner
    at episode boundaries and while an agent is inactive, so memory remains
    episode-local and agent-local.
    """

    def __init__(
        self,
        obs_dim:          int,
        act_dim:          int,
        hidden_dim:       int,
        actor_num_layers: int,
        rngs:             nnx.Rngs,
        memory_comm_enabled: bool = False,
        memory_comm_every_k_steps: int = 5,
        tarmac_sig_dim: int = 64,
        tarmac_val_dim: int = 128,
        tarmac_include_self: bool = True,
    ) -> None:
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.memory_comm_enabled = memory_comm_enabled
        self.memory_comm_every_k_steps = memory_comm_every_k_steps
        self.tarmac_sig_dim = tarmac_sig_dim
        self.tarmac_val_dim = tarmac_val_dim
        self.tarmac_include_self = tarmac_include_self
        self.encoder = MLP(obs_dim, hidden_dim, 1, hidden_dim, rngs)

        gru_input_dim = hidden_dim + tarmac_val_dim if memory_comm_enabled else hidden_dim
        self.gru = GRUCell(gru_input_dim, hidden_dim, rngs)
        if memory_comm_enabled:
            self.tarmac_query = nnx.Linear(hidden_dim, tarmac_sig_dim, rngs=rngs)
            self.tarmac_signature = nnx.Linear(hidden_dim, tarmac_sig_dim, rngs=rngs)
            self.tarmac_value = nnx.Linear(hidden_dim, tarmac_val_dim, rngs=rngs)
        self.policy_trunk = MLP(hidden_dim, hidden_dim, actor_num_layers, hidden_dim, rngs)
        self.mu_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)


    def initial_tarmac_signature(self, batch_shape=(), num_agents: int | None = None) -> jax.Array:
        """Return zero TarMAC signatures with shape batch_shape + (N, S)."""
        n = num_agents if num_agents is not None else 0
        return jnp.zeros((*tuple(batch_shape), n, self.tarmac_sig_dim), dtype=jnp.float32)

    def initial_tarmac_value(self, batch_shape=(), num_agents: int | None = None) -> jax.Array:
        """Return zero TarMAC values with shape batch_shape + (N, V)."""
        n = num_agents if num_agents is not None else 0
        return jnp.zeros((*tuple(batch_shape), n, self.tarmac_val_dim), dtype=jnp.float32)

    def _tarmac_context(
        self,
        prev_hidden: jax.Array,
        prev_signature: jax.Array,
        prev_value: jax.Array,
        comm_mask: jax.Array | None = None,
        active: jax.Array | None = None,
        base_signature: jax.Array | None = None,
        base_value: jax.Array | None = None,
        base_memory_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Return TarMAC value-space context from previous sender tokens.

        Agent messages are the previous step's learned ``(signature, value)``
        pair. A saved base replay token, when supplied, is appended as an
        indistinguishable extra sender for receivers in base range. The caller
        supplies the wall-aware communication graph and sparse communication
        cadence; this method only applies the learned query/signature matching.
        """
        if not self.memory_comm_enabled:
            return jnp.zeros((*prev_hidden.shape[:-1], self.tarmac_val_dim), dtype=prev_hidden.dtype)

        N = prev_hidden.shape[-2]
        if comm_mask is None:
            comm_mask = jnp.ones((N, N), dtype=bool)
        if active is None:
            active = jnp.ones((N,), dtype=bool)

        active_pair = active[:, None] & active[None, :]
        share_mask = comm_mask & active_pair
        if not self.tarmac_include_self:
            share_mask = share_mask & ~jnp.eye(N, dtype=bool)

        sender_signature = prev_signature
        sender_value = prev_value
        mask = share_mask
        if base_signature is not None and base_value is not None and base_memory_mask is not None:
            sender_signature = jnp.concatenate([sender_signature, base_signature[None, :]], axis=-2)
            sender_value = jnp.concatenate([sender_value, base_value[None, :]], axis=-2)
            mask = jnp.concatenate([mask, base_memory_mask[:, None] & active[:, None]], axis=-1)

        query = self.tarmac_query(prev_hidden)
        return tarmac_aggregate(query, sender_signature, sender_value, mask)

    def __call_team__(
        self,
        obs: jax.Array,
        hidden: jax.Array,
        signature: jax.Array,
        value: jax.Array,
        reset: jax.Array | None = None,
        comm_mask: jax.Array | None = None,
        active: jax.Array | None = None,
        base_signature: jax.Array | None = None,
        base_value: jax.Array | None = None,
        base_memory_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Team actor pass with TarMAC pre-GRU communication.

        The GRU consumes the local observation embedding concatenated with a
        TarMAC context built from previous ``(signature, value)`` sender tokens.
        The updated hidden state emits the next signature/value pair and feeds
        SwarmEcho's existing Gaussian policy MLP.
        """
        if reset is not None:
            hidden = jnp.where(reset[..., None], jnp.zeros_like(hidden), hidden)
            signature = jnp.where(reset[..., None], jnp.zeros_like(signature), signature)
            value = jnp.where(reset[..., None], jnp.zeros_like(value), value)
        prev_hidden = hidden
        encoded = self.encoder(obs)
        if self.memory_comm_enabled:
            comm_context = self._tarmac_context(
                prev_hidden, signature, value, comm_mask, active,
                base_signature, base_value, base_memory_mask,
            )
            gru_input = jnp.concatenate([encoded, comm_context], axis=-1)
        else:
            gru_input = encoded
        hidden = self.gru(prev_hidden, gru_input)
        signature = self.tarmac_signature(hidden) if self.memory_comm_enabled else signature
        value = self.tarmac_value(hidden) if self.memory_comm_enabled else value
        feat = self.policy_trunk(hidden)
        mu = self.mu_head(feat)
        log_std = jnp.clip(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return hidden, signature, value, mu, log_std

    def act_team(
        self,
        obs: jax.Array,
        hidden: jax.Array,
        signature: jax.Array,
        value: jax.Array,
        keys: jax.Array,
        reset: jax.Array | None = None,
        comm_mask: jax.Array | None = None,
        active: jax.Array | None = None,
        base_signature: jax.Array | None = None,
        base_value: jax.Array | None = None,
        base_memory_mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        hidden, signature, value, mu, log_std = self.__call_team__(
            obs, hidden, signature, value, reset, comm_mask, active,
            base_signature, base_value, base_memory_mask, deterministic,
        )
        std = jnp.exp(log_std)
        u = jnp.where(deterministic, mu, mu + std * jax.vmap(lambda k: jax.random.normal(k, mu.shape[-1:]))(keys))
        log_prob = -0.5 * jnp.sum(((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        ent_gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        jacobian = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        entropy = ent_gaussian + jnp.sum(jacobian, axis=-1)
        return hidden, signature, value, u, log_prob, entropy

    def __call__(
        self,
        obs:    jax.Array,
        hidden: jax.Array,
        reset:  jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """obs (..., D), hidden (..., H), reset (...) -> hidden, mu, log_std."""
        if reset is not None:
            hidden = jnp.where(reset[..., None], jnp.zeros_like(hidden), hidden)

        encoded = self.encoder(obs)
        hidden = self.gru(hidden, encoded)
        feat = self.policy_trunk(hidden)
        mu = self.mu_head(feat)
        log_std = jnp.clip(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return hidden, mu, log_std

    def act(
        self,
        obs:           jax.Array,
        hidden:        jax.Array,
        key:           jax.Array,
        deterministic: bool = False,
        reset:         jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Recurrent action sampling for one agent.

        Returns the updated hidden state and the pre-squash sample `u`, matching
        the stateless actor's buffer contract.
        """
        hidden, mu, log_std = self(obs, hidden, reset)
        std = jnp.exp(log_std)

        if deterministic:
            u = mu
        else:
            u = mu + std * jax.random.normal(key, mu.shape)

        log_prob = -0.5 * jnp.sum(
            ((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        ent_gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std)
        jacobian = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        entropy = ent_gaussian + jnp.sum(jacobian)
        return hidden, u, log_prob, entropy

    def evaluate_actions_sequence(
        self,
        obs:          jax.Array,
        actions:      jax.Array,
        init_hidden:  jax.Array,
        resets:       jax.Array,
        init_signature: jax.Array | None = None,
        init_value:   jax.Array | None = None,
        comm_masks:   jax.Array | None = None,
        actives:      jax.Array | None = None,
        base_signatures: jax.Array | None = None,
        base_values:   jax.Array | None = None,
        base_memory_masks: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Replay a rollout sequence for recurrent PPO updates.

        When TarMAC communication is enabled, the scan carries hidden,
        signature, and value state so policy replay matches rollout exactly.
        """

        if self.memory_comm_enabled:
            if init_signature is None:
                init_signature = jnp.zeros((*init_hidden.shape[:-1], self.tarmac_sig_dim), dtype=init_hidden.dtype)
            if init_value is None:
                init_value = jnp.zeros((*init_hidden.shape[:-1], self.tarmac_val_dim), dtype=init_hidden.dtype)

            def _step(carry, xs):
                hidden, signature, value = carry
                obs_t, act_t, reset_t, mask_t, active_t, base_sig_t, base_val_t, base_mask_t = xs
                hidden, signature, value, mu, log_std = self.__call_team__(
                    obs_t, hidden, signature, value, reset_t, mask_t, active_t,
                    base_sig_t, base_val_t, base_mask_t
                )
                std = jnp.exp(log_std)
                log_prob = -0.5 * jnp.sum(((act_t - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
                ent_gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
                jacobian = 2.0 * (jnp.log(2.0) - act_t - jax.nn.softplus(-2.0 * act_t))
                entropy = ent_gaussian + jnp.sum(jacobian, axis=-1)
                return (hidden, signature, value), (log_prob, entropy)
            (final_hidden, final_signature, final_value), (log_probs, entropy) = jax.lax.scan(
                _step,
                (init_hidden, init_signature, init_value),
                (obs, actions, resets, comm_masks, actives, base_signatures, base_values, base_memory_masks),
            )
            return final_hidden, log_probs, entropy

        def _step(hidden, xs):
            obs_t, act_t, reset_t = xs
            hidden, mu, log_std = self(obs_t, hidden, reset_t)
            std = jnp.exp(log_std)
            log_prob = -0.5 * jnp.sum(((act_t - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
            ent_gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
            jacobian = 2.0 * (jnp.log(2.0) - act_t - jax.nn.softplus(-2.0 * act_t))
            entropy = ent_gaussian + jnp.sum(jacobian, axis=-1)
            return hidden, (log_prob, entropy)

        final_hidden, (log_probs, entropy) = jax.lax.scan(_step, init_hidden, (obs, actions, resets))
        return final_hidden, log_probs, entropy
