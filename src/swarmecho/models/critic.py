"""Recurrent agent-centric critic with cross-agent attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from swarmecho.models.recurrent import GRUCell


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
class RecurrentAgentCentricCritic(nnx.Module):
    """
    Agent-centric critic with per-agent episode memory.

    Flow per timestep:

        obs_i -> encoder -> e_i
        (h_i, e_i) -> GRU -> h_i'
        token_i = [e_i | h_i']
        tokens -> masked cross-agent attention -> x_i
        [token_i | x_i] -> value head -> V_i

    Memory is per agent and independent from the actor memory. Attention is
    applied after the GRU so the critic can attend over remembered agent
    histories, not only the current observation frame.
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.encoder = MLP(obs_dim, hidden_dim, 1, hidden_dim, rngs)
        self.gru = GRUCell(hidden_dim, hidden_dim, rngs)
        token_dim = hidden_dim * 2
        self.attention = nnx.MultiHeadAttention(
            num_heads   = 4,
            in_features = token_dim,
            rngs        = rngs,
        )
        self.value_head = MLP(token_dim * 2, hidden_dim, num_layers, 1, rngs)

    def __call__(
        self,
        obs:           jax.Array,
        hidden:        jax.Array,
        resets:        jax.Array | None = None,
        deterministic: bool = True,
        active:        jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        obs (..., N, obs_dim), hidden (..., N, H) -> hidden, values (..., N).
        """
        N = obs.shape[-2]

        if resets is not None:
            hidden = jnp.where(resets[..., None], jnp.zeros_like(hidden), hidden)

        if active is not None:
            obs = jnp.where(active[..., None], obs, 0.0)
            hidden = jnp.where(active[..., None], hidden, 0.0)
        e = self.encoder(obs)
        hidden = self.gru(hidden, e)
        if active is not None:
            e = jnp.where(active[..., None], e, 0.0)
            hidden = jnp.where(active[..., None], hidden, 0.0)
        token = jnp.concatenate([e, hidden], axis=-1)

        mask = ~jnp.eye(N, dtype=bool)
        if active is not None:
            mask = mask & active[..., :, None] & active[..., None, :]
            has_other = jnp.any(mask, axis=-1, keepdims=True)
            # A dummy key prevents all-masked softmax; its output is discarded.
            mask = mask | (~has_other & (jnp.arange(N) == 0))
        x = self.attention(
            token,
            token,
            mask          = mask[..., None, :, :] if active is not None else mask,
            decode        = False,
            deterministic = deterministic,
        )

        if active is not None:
            x = jnp.where(has_other, x, 0.0)
        combined = jnp.concatenate([token, x], axis=-1)
        values = self.value_head(combined).squeeze(-1)
        if active is not None:
            values = jnp.where(active, values, 0.0)
        return hidden, values

    def values_sequence(
        self,
        obs:           jax.Array,  # (T, B, N, D)
        init_hidden:   jax.Array,  # (B, N, H)
        resets:        jax.Array,  # (T, B, N)
        deterministic: bool = True,
        active:        jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Replay a critic sequence for recurrent PPO updates."""

        def _step(hidden, xs):
            obs_t, reset_t, active_t = xs
            hidden, values = self(obs_t, hidden, reset_t, deterministic=deterministic,
                                  **({"active": active_t} if active is not None else {}))
            return hidden, values

        final_hidden, values = jax.lax.scan(
            _step,
            init_hidden,
            (obs, resets, jnp.ones_like(resets, dtype=bool) if active is None else active),
        )
        return final_hidden, values
