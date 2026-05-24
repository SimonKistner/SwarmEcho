"""
swarmecho/models/critic.py
===========================
Centralised critic networks for SwarmEcho MAPPO.

Both critics receive the full team observation stack (..., N, obs_dim) and
are therefore permutation-invariant by construction.

GlobalMeanCritic   (kept for ablations)
  MLP encoder → Self-Attention → Global Mean Pool → scalar V
  Output: (...,)   — ONE value for the entire team

AgentCentricCritic  (MAAC-style, recommended)
  MLP encoder → Masked Cross-Agent Attention → Concat → MLP head → per-agent V_i
  Output: (..., N) — one value per agent, computed from every other agent's
                     perspective (agent i does NOT attend to itself)

The AgentCentricCritic preserves permutation invariance because:
  - The encoder is shared (weight tying across agents)
  - The attention is symmetric in the key/value space
  - The diagonal mask is the only asymmetry and it is structurally consistent
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


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
# GlobalMeanCritic  (kept for ablations)
# ---------------------------------------------------------------------------

class GlobalMeanCritic(nnx.Module):
    """
    Centralised critic that pools the team into a single value.

    Architecture
    ------------
    1. Shared MLP encoder          : (..., N, obs_dim) → (..., N, H)
    2. Multi-Head Self-Attention   : (..., N, H)       → (..., N, H)  + residual + LN
    3. Global Mean Pool            : (..., N, H)       → (..., H)
    4. Value MLP head              : (..., H)           → (...,)

    Output shape: (...,)  — one scalar per environment
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.encoder   = MLP(obs_dim, hidden_dim, 1, hidden_dim, rngs)
        self.mha       = nnx.MultiHeadAttention(num_heads=4, in_features=hidden_dim, rngs=rngs)
        self.ln        = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.trunk     = MLP(hidden_dim, hidden_dim, num_layers, 1, rngs)

    def __call__(self, obs: jax.Array, deterministic: bool = True) -> jax.Array:
        """obs: (..., N, obs_dim) → value (...,)"""
        h = self.encoder(obs)                                              # (..., N, H)
        h = self.ln(h + self.mha(h, decode=False, deterministic=deterministic))  # residual + LN
        g = jnp.mean(h, axis=-2)                                           # (..., H)
        return self.trunk(g).squeeze(-1)                                   # (...,)


# ---------------------------------------------------------------------------
# AgentCentricCritic  (MAAC-style — recommended)
# ---------------------------------------------------------------------------

class AgentCentricCritic(nnx.Module):
    """
    Agent-centric critic — outputs a unique V_i for each agent.

    Architecture (MAAC-style, adapted for V(o) not Q(o,a))
    -------------------------------------------------------
    1. Encoder MLP  : obs_i → e_i          shape (..., N, H)
    2. Masked MHA   : agent i attends to ALL OTHER agents (diagonal masked out)
                      e → x_i              shape (..., N, H)
    3. Concatenate  : [e_i | x_i]          shape (..., N, 2H)
    4. Value head   : 2H → 1 → squeeze     shape (..., N)

    The off-diagonal mask ensures agent i's value estimate is formed from the
    perspectives of every teammate EXCEPT itself, which encourages the critic
    to learn credit assignment over the team rather than self-loop bias.

    Permutation invariance is preserved: the encoder weights are shared and
    the attention is symmetric in key/value space.

    Output shape: (..., N)  — one value per agent per environment
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.encoder    = MLP(obs_dim, hidden_dim, 1, hidden_dim, rngs)
        self.attention  = nnx.MultiHeadAttention(
            num_heads   = 4,
            in_features = hidden_dim,
            rngs        = rngs,
        )
        self.value_head = MLP(hidden_dim * 2, hidden_dim, num_layers, 1, rngs)

    def __call__(self, obs: jax.Array, deterministic: bool = True) -> jax.Array:
        """
        obs: (..., N, obs_dim) → values (..., N)

        Parameters
        ----------
        obs           : team observations, shape (..., N, obs_dim)
        deterministic : passed to MHA dropout (True = no dropout at inference)
        """
        N = obs.shape[-2]

        # 1. Encode each agent's obs into a latent vector
        e = self.encoder(obs)            # (..., N, H)

        # 2. Cross-agent attention — agent i queries all agents EXCEPT itself
        #    mask[i, j] = True  means "attend to j"
        #    mask[i, i] = False means "do NOT attend to self"
        mask = ~jnp.eye(N, dtype=bool)  # (N, N) — True off-diagonal
        x = self.attention(
            e,                    # inputs_q
            e,                    # inputs_kv (cross-attention: same source)
            mask          = mask,
            decode        = False,
            deterministic = deterministic,
        )                         # (..., N, H)

        # 3. Concatenate local encoding with cross-agent context
        combined = jnp.concatenate([e, x], axis=-1)  # (..., N, 2H)

        # 4. Per-agent value head
        values = self.value_head(combined).squeeze(-1)  # (..., N)
        return values
