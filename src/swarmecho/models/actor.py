"""Shared recurrent actor for decentralized execution with TarMAC.

Each drone encodes its local observation and attends to available teammate
and base messages. The encoder output and message context feed its GRU. The
policy head then parameterizes a Gaussian over three pre-squash commands.
PPO stores those commands and uses Gaussian log-probabilities for its ratio.
The entropy bonus accounts for tanh squashing. The environment scales the
squashed commands by its force limit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from swarmecho.models.recurrent import GRUCell

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0


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
class RecurrentDecentralizedActor(nnx.Module):
    """
    Shared recurrent actor for decentralized execution.

    Flow per agent and timestep:

        [encoded local observation, TarMAC context] -> GRU -> policy head

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
        tarmac_sig_dim: int = 64,
        tarmac_val_dim: int = 128,
    ) -> None:
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.tarmac_sig_dim = tarmac_sig_dim
        self.tarmac_val_dim = tarmac_val_dim
        self.encoder = MLP(obs_dim, hidden_dim, 1, hidden_dim, rngs)

        self.gru = GRUCell(hidden_dim + tarmac_val_dim, hidden_dim, rngs)
        self.tarmac_query = nnx.Linear(hidden_dim, tarmac_sig_dim, rngs=rngs)
        self.tarmac_signature = nnx.Linear(hidden_dim, tarmac_sig_dim, rngs=rngs)
        self.tarmac_value = nnx.Linear(hidden_dim, tarmac_val_dim, rngs=rngs)
        self.policy_trunk = MLP(hidden_dim, hidden_dim, actor_num_layers, hidden_dim, rngs)
        self.mu_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)


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
        N = prev_hidden.shape[-2]
        if comm_mask is None:
            comm_mask = jnp.ones((N, N), dtype=bool)
        if active is None:
            active = jnp.ones((N,), dtype=bool)

        active_pair = active[:, None] & active[None, :]
        eye = jnp.eye(N, dtype=bool)
        mask = comm_mask & active_pair & ~eye
        sender_signature = prev_signature
        sender_value = prev_value
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
        if active is not None:
            prev_hidden = jnp.where(active[..., None], prev_hidden, 0.0)
            signature = jnp.where(active[..., None], signature, 0.0)
            value = jnp.where(active[..., None], value, 0.0)
            obs = jnp.where(active[..., None], obs, 0.0)
        encoded = self.encoder(obs)
        comm_context = self._tarmac_context(
            prev_hidden, signature, value, comm_mask, active,
            base_signature, base_value, base_memory_mask,
        )
        gru_input = jnp.concatenate([encoded, comm_context], axis=-1)
        hidden = self.gru(prev_hidden, gru_input)
        if active is not None:
            hidden = jnp.where(active[..., None], hidden, 0.0)
        signature = self.tarmac_signature(hidden)
        value = self.tarmac_value(hidden)
        feat = self.policy_trunk(hidden)
        mu = self.mu_head(feat)
        log_std = jnp.clip(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        if active is not None:
            signature = jnp.where(active[..., None], signature, 0.0)
            value = jnp.where(active[..., None], value, 0.0)
            mu = jnp.where(active[..., None], mu, 0.0)
            log_std = jnp.where(active[..., None], log_std, 0.0)
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
        if active is not None:
            u = jnp.where(active[..., None], u, 0.0)
            log_prob = jnp.where(active, log_prob, 0.0)
            entropy = jnp.where(active, entropy, 0.0)
        return hidden, signature, value, u, log_prob, entropy
