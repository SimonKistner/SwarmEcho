"""
swarmecho/models/critic.py
===========================
Centralised critic networks for SwarmEcho MAPPO.

The critic receives the full team observation stack (..., N, obs_dim) and is
permutation-invariant by construction.

AgentCentricCritic  (MAAC-style)
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
# AgentCentricCritic  (MAAC-style)
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


# ---------------------------------------------------------------------------
# RecurrentAgentCentricCritic
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
    ) -> tuple[jax.Array, jax.Array]:
        """
        obs (..., N, obs_dim), hidden (..., N, H) -> hidden, values (..., N).
        """
        N = obs.shape[-2]

        if resets is not None:
            hidden = jnp.where(resets[..., None], jnp.zeros_like(hidden), hidden)

        e = self.encoder(obs)
        hidden = self.gru(hidden, e)
        token = jnp.concatenate([e, hidden], axis=-1)

        mask = ~jnp.eye(N, dtype=bool)
        x = self.attention(
            token,
            token,
            mask          = mask,
            decode        = False,
            deterministic = deterministic,
        )

        combined = jnp.concatenate([token, x], axis=-1)
        values = self.value_head(combined).squeeze(-1)
        return hidden, values

    def values_sequence(
        self,
        obs:           jax.Array,  # (T, B, N, D)
        init_hidden:   jax.Array,  # (B, N, H)
        resets:        jax.Array,  # (T, B, N)
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Replay a critic sequence for recurrent PPO updates."""

        def _step(hidden, xs):
            obs_t, reset_t = xs
            hidden, values = self(obs_t, hidden, reset_t, deterministic=deterministic)
            return hidden, values

        final_hidden, values = jax.lax.scan(
            _step,
            init_hidden,
            (obs, resets),
        )
        return final_hidden, values


class SemanticMapEncoder(nnx.Module):
    """Small feature-pyramid CNN for five-channel top-down state maps."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.conv1 = nnx.Conv(5, 16, (5, 5), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(32, 32, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv4 = nnx.Conv(32, 32, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv5 = nnx.Conv(32, 32, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)

    def __call__(self, image: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jax.nn.silu(self.conv1(image))
        local_map = jax.nn.silu(self.conv2(x))
        x = jax.nn.silu(self.conv3(local_map))
        x = jax.nn.silu(self.conv4(x))
        x = jax.nn.silu(self.conv5(x))
        return local_map, jnp.mean(x, axis=(-3, -2))


class PrivilegedAgentCentricCritic(nnx.Module):
    """CNN-backed centralized critic with unrestricted agent attention."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        rngs: nnx.Rngs,
        wall_map: jax.Array,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.wall_map = nnx.Variable(jnp.asarray(wall_map, dtype=jnp.float32))
        self.map_encoder = SemanticMapEncoder(rngs)
        self.encoder = MLP(obs_dim + 64, hidden_dim, 1, hidden_dim, rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=4, in_features=hidden_dim, rngs=rngs
        )
        self.value_head = MLP(hidden_dim * 2, hidden_dim, num_layers, 1, rngs)

    def _semantic_image(
        self, tokens: jax.Array, coverage_map: jax.Array
    ) -> jax.Array:
        leading = tokens.shape[:-2]
        n = tokens.shape[-2]
        h, w = coverage_map.shape[-2:]
        flat_tokens = tokens.reshape((-1, n, tokens.shape[-1]))

        def rasterize(team):
            cells = jnp.rint(
                team[:, :2] * jnp.array([h - 1, w - 1])
            ).astype(jnp.int32)
            active_agents = jnp.zeros((h, w), dtype=jnp.float32).at[
                cells[:, 0], cells[:, 1]
            ].add(team[:, 9])
            base = jnp.rint(
                (team[0, :2] + team[0, 4:6]) * jnp.array([h - 1, w - 1])
            ).astype(jnp.int32)
            target = jnp.rint(
                (team[0, :2] + team[0, 6:8]) * jnp.array([h - 1, w - 1])
            ).astype(jnp.int32)
            base_map = jnp.zeros((h, w), dtype=jnp.float32).at[base[0], base[1]].set(1.0)
            target_map = jnp.zeros((h, w), dtype=jnp.float32).at[
                target[0], target[1]
            ].set(1.0)
            return jnp.stack([active_agents, base_map, target_map], axis=-1)

        dynamic = jax.vmap(rasterize)(flat_tokens).reshape((*leading, h, w, 3))
        wall = jnp.broadcast_to(self.wall_map.get_value(), (*leading, h, w))
        return jnp.concatenate(
            [wall[..., None], coverage_map[..., None], dynamic], axis=-1
        )

    @staticmethod
    def _sample(local_map: jax.Array, positions: jax.Array) -> jax.Array:
        leading = positions.shape[:-2]
        n = positions.shape[-2]
        h, w = local_map.shape[-3:-1]
        flat_map = local_map.reshape((-1, h, w, local_map.shape[-1]))
        flat_pos = positions.reshape((-1, n, 2))

        def sample_one(feature_map, pos):
            coord = pos * jnp.array([h - 1, w - 1])
            lo = jnp.floor(coord).astype(jnp.int32)
            hi = jnp.minimum(lo + 1, jnp.array([h - 1, w - 1]))
            frac = coord - lo
            f00 = feature_map[lo[:, 0], lo[:, 1]]
            f10 = feature_map[hi[:, 0], lo[:, 1]]
            f01 = feature_map[lo[:, 0], hi[:, 1]]
            f11 = feature_map[hi[:, 0], hi[:, 1]]
            fx0 = f00 * (1 - frac[:, :1]) + f10 * frac[:, :1]
            fx1 = f01 * (1 - frac[:, :1]) + f11 * frac[:, :1]
            return fx0 * (1 - frac[:, 1:]) + fx1 * frac[:, 1:]

        sampled = jax.vmap(sample_one)(flat_map, flat_pos)
        return sampled.reshape((*leading, n, local_map.shape[-1]))

    def encode(self, tokens: jax.Array, coverage_map: jax.Array) -> jax.Array:
        semantic = self._semantic_image(tokens, coverage_map)
        local_map, global_features = self.map_encoder(semantic)
        local_features = self._sample(local_map, tokens[..., :2])
        global_features = jnp.broadcast_to(
            global_features[..., None, :], local_features.shape
        )
        return self.encoder(
            jnp.concatenate([tokens, local_features, global_features], axis=-1)
        )

    def __call__(
        self,
        tokens: jax.Array,
        coverage_map: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        encoded = self.encode(tokens, coverage_map)
        context = self.attention(
            encoded, encoded, decode=False, deterministic=deterministic
        )
        return self.value_head(
            jnp.concatenate([encoded, context], axis=-1)
        ).squeeze(-1)


class RecurrentPrivilegedAgentCentricCritic(PrivilegedAgentCentricCritic):
    """Semantic-map critic with per-agent memory before global attention."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        rngs: nnx.Rngs,
        wall_map: jax.Array,
    ) -> None:
        super().__init__(obs_dim, hidden_dim, num_layers, rngs, wall_map)
        self.gru = GRUCell(hidden_dim, hidden_dim, rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=4, in_features=hidden_dim * 2, rngs=rngs
        )
        self.value_head = MLP(
            hidden_dim * 4,
            hidden_dim,
            num_layers,
            1,
            rngs,
        )

    def __call__(
        self,
        tokens: jax.Array,
        coverage_map: jax.Array,
        hidden: jax.Array,
        resets: jax.Array | None = None,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        if resets is not None:
            hidden = jnp.where(resets[..., None], jnp.zeros_like(hidden), hidden)
        encoded = self.encode(tokens, coverage_map)
        hidden = self.gru(hidden, encoded)
        recurrent_tokens = jnp.concatenate([encoded, hidden], axis=-1)
        context = self.attention(
            recurrent_tokens,
            recurrent_tokens,
            decode=False,
            deterministic=deterministic,
        )
        values = self.value_head(
            jnp.concatenate([recurrent_tokens, context], axis=-1)
        ).squeeze(-1)
        return hidden, values

    def values_sequence(
        self,
        tokens: jax.Array,
        coverage_maps: jax.Array,
        init_hidden: jax.Array,
        resets: jax.Array,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        def step(hidden, inputs):
            token_t, map_t, reset_t = inputs
            return self(token_t, map_t, hidden, reset_t, deterministic)

        return jax.lax.scan(step, init_hidden, (tokens, coverage_maps, resets))
