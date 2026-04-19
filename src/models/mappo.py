"""
swarmecho/models/mappo.py
==========================
MAPPO (Multi-Agent PPO with Centralised Critic) model for 

Architecture — Centralised Training, Decentralised Execution (CTDE)
--------------------------------------------------------------------
  MAPPOActor  (shared across agents, uses LOCAL observation)
    └── MLP trunk → mu, log_std  → Gaussian policy

  MAPPOCritic (one per team, uses GLOBAL observation = concat of all agents' obs)
    └── MLP trunk → scalar V(s_global)

At rollout time:
  - Each agent i calls actor(obs_i)  → action_i
  - One critic call per env: critic(concat(obs_0,...,obs_{N-1})) → V

At update time:
  - Actor loss: PPO-clip on per-agent log ratios, shared advantage
  - Value loss: MSE on centralised V vs GAE returns

This is the canonical MAPPO setup from Yu et al. (2022):
  "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class MLP(nnx.Module):
    """LayerNorm + Tanh MLP — identical to the one in actor_critic.py."""

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
# Attention block (N-invariant summary)
# ---------------------------------------------------------------------------

class SelfAttention(nnx.Module):
    """
    Standard Multi-Head Attention followed by Norm and Residual connection.
    This is the core for Agent-Invariance and high-fidelity swarm understanding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rngs:      nnx.Rngs,
    ) -> None:
        self.mha  = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """x: (..., N, D) -> (..., N, D)"""
        h = self.mha(x, decode=False, deterministic=deterministic)
        return self.norm(x + h)


# ---------------------------------------------------------------------------
# Decentralised actor (local obs → action)
# ---------------------------------------------------------------------------

class MAPPOActor(nnx.Module):
    """
    Shared actor network used by every agent independently.

    Input : obs_i  (obs_dim,)    — agent i's local observation
    Output: mu, log_std          — parameters of a diagonal Gaussian
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.act_dim = act_dim
        self.trunk   = MLP(obs_dim, hidden_dim, num_layers, hidden_dim, rngs)
        self.mu_head      = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

    def __call__(self, obs: jax.Array):
        """obs: (..., obs_dim) → mu, log_std each (..., act_dim)"""
        feat    = self.trunk(obs)
        mu      = self.mu_head(feat)
        log_std = jnp.clip(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def act(
        self,
        obs:           jax.Array,   # (obs_dim,)
        key:           jax.Array,
        max_force:     float = 50.0,
        deterministic: bool  = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Sample an action for one agent.
        Returns: action (act_dim,), log_prob (), entropy ()
        """
        mu, log_std = self(obs)
        std = jnp.exp(log_std)

        action = mu if deterministic else mu + std * jax.random.normal(key, mu.shape)
        action = jnp.clip(action, -max_force, max_force)

        log_prob = -0.5 * jnp.sum(
            ((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs:     jax.Array,   # (..., obs_dim)
        actions: jax.Array,   # (..., act_dim)
    ) -> tuple[jax.Array, jax.Array]:
        """Returns log_prob (...,) and entropy (...,) for given actions."""
        mu, log_std = self(obs)
        std = jnp.exp(log_std)
        log_prob = -0.5 * jnp.sum(
            ((actions - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Centralised critic (global obs → scalar value)
# ---------------------------------------------------------------------------

class MAPPOCritic(nnx.Module):
    """
    Centralised critic for the team.
    
    Architecture:
      1. Shared Encoder (MLP) projects local obs to latent features
      2. Self-Attention allows agents to 'talk' and capture geometry
      3. Global Pooling (Mean) reduces N agents to a single vector
      4. Value Head produces scalar V
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.encoder    = MLP(obs_dim, hidden_dim, 1, hidden_dim, rngs)
        self.attention  = SelfAttention(hidden_dim, num_heads=4, rngs=rngs)
        self.trunk      = MLP(hidden_dim, hidden_dim, num_layers, 1, rngs)

    def __call__(self, obs: jax.Array, deterministic: bool = True) -> jax.Array:
        """obs: (..., N, obs_dim) → value (...,)"""
        # 1. Project to latent space
        h = self.encoder(obs)               # (..., N, H)
        
        # 2. Self-Attention (N-invariant)
        h = self.attention(h, deterministic=deterministic)              # (..., N, H)
        
        # 3. Global Pooling (Mean)
        g = jnp.mean(h, axis=-2)            # (..., H)
        
        # 4. Value Trunk
        return self.trunk(g).squeeze(-1)


# ---------------------------------------------------------------------------
# Combined MAPPO model (actor + critic together for easy checkpointing)
# ---------------------------------------------------------------------------

class MAPPOModel(nnx.Module):
    """
    Wrapper holding both the decentralised actor and centralised critic.

    Designed for easy checkpointing via nnx.split / standard Orbax.
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        num_agents: int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim

        self.actor  = MAPPOActor(obs_dim, act_dim, hidden_dim, num_layers, rngs)
        self.critic = MAPPOCritic(obs_dim, hidden_dim, num_layers, rngs)

    # -- Convenience wrappers ---------------------------------------------------

    def act(
        self,
        obs_i:         jax.Array,   # (obs_dim,)
        key:           jax.Array,
        max_force:     float = 50.0,
        deterministic: bool  = False,
    ):
        """Decentralised act — only uses the local actor."""
        return self.actor.act(obs_i, key, max_force, deterministic)

    def get_value(self, all_obs: jax.Array, deterministic: bool = True) -> jax.Array:
        """Centralised value estimate. all_obs: (..., N, D) -> (...,)"""
        return self.critic(all_obs, deterministic=deterministic)

    def rollout_step(
        self,
        all_obs:       jax.Array,   # (N, obs_dim)
        keys:          jax.Array,   # (N, 2) — per-agent PRNGKeys
        max_force:     float = 50.0,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Full CTDE forward pass for one env during rollout.

        Returns
        -------
        actions   : (N, act_dim)
        log_probs : (N,)
        value     : ()   — single centralised value
        """
        def _act_one(obs_i, key_i):
            a, lp, _ = self.actor.act(obs_i, key_i, max_force)
            return a, lp
        actions, log_probs = jax.vmap(_act_one)(all_obs, keys)
        value = self.get_value(all_obs, deterministic=False)
        return actions, log_probs, value


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim

    print("── MAPPO Self-Test ──────────────────────────────────────────")
    cfg     = load_config(cli_overrides=False)
    validate_config(cfg)
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N       = int(cfg.env.num_agents)

    print(f"  obs_dim        : {obs_dim}")
    print(f"  act_dim        : {act_dim}")
    print(f"  num_agents N   : {N}")
    print(f"  global_obs_dim : {N * obs_dim}")
    print(f"  hidden         : {cfg.network.hidden_dim} × {cfg.network.num_layers} layers")

    rngs  = nnx.Rngs(0)
    model = MAPPOModel(obs_dim, act_dim, N, cfg.network.hidden_dim, cfg.network.num_layers, rngs)

    _, params = nnx.split(model)
    n_params  = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total params   : {n_params:,}")

    # Single env rollout step
    obs   = jnp.zeros((N, obs_dim))
    keys  = jax.random.split(jax.random.PRNGKey(0), N)
    acts, lps, val = model.rollout_step(obs, keys, max_force=float(cfg.env.max_force))
    assert acts.shape == (N, act_dim), f"actions shape {acts.shape}"
    assert lps.shape  == (N,),         f"log_probs shape {lps.shape}"
    assert val.shape  == (),           f"value shape {val.shape}"
    print(f"  rollout_step shapes: actions={acts.shape}, log_probs={lps.shape}, value={val.shape}  ✓")

    # Evaluate actions (used during PPO update)
    obs_flat  = obs.reshape(N, obs_dim)
    acts_flat = acts
    log_probs_eval, entropy_eval = model.actor.evaluate_actions(obs_flat, acts_flat)
    assert log_probs_eval.shape == (N,), f"eval log_probs shape {log_probs_eval.shape}"
    print(f"  evaluate_actions shapes: log_probs={log_probs_eval.shape}, entropy={entropy_eval.shape}  ✓")

    # Centralised value batch
    obs_batch = jnp.zeros((8, N, obs_dim))
    vals_batch = model.get_value(obs_batch)
    assert vals_batch.shape == (8,), f"batched value shape {vals_batch.shape}"
    print(f"  get_value batch shape: {vals_batch.shape}  ✓")

    print("\nMAPPO self-test passed ✓")


