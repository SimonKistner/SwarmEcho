"""
swarmecho/models/mappo.py
==========================
MAPPO model wrapper for SwarmEcho.

Holds a DecentralizedActor and one of two critic variants, selected via config:
  - "agent_centric"  (default) : AgentCentricCritic → V_i per agent  (..., N)
  - "global_mean"              : GlobalMeanCritic   → scalar V        (...,)

All heavy lifting (network definitions) lives in actor.py and critic.py.
This file is the single construction point used by runner.py.

CTDE contract
-------------
  Rollout  : actor(obs_i) per agent — decentralised
  Training : critic(all_obs)        — centralised (sees full team)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from models.actor import DecentralizedActor
from models.critic import AgentCentricCritic, GlobalMeanCritic


# ---------------------------------------------------------------------------
# MAPPOModel — actor + critic wrapper
# ---------------------------------------------------------------------------

class MAPPOModel(nnx.Module):
    """
    Combined actor-critic wrapper for MAPPO.

    Parameters
    ----------
    obs_dim          : per-agent observation dimension
    act_dim          : action dimension (2 for SwarmEcho)
    num_agents       : N — swarm size (used for shape assertions only)
    hidden_dim       : hidden layer width (shared by actor and critic)
    num_layers       : number of hidden layers in the CRITIC
    actor_num_layers : number of hidden layers in the ACTOR (recommended: 2)
    critic_type      : "agent_centric" | "global_mean"
    rngs             : Flax NNX RNG state
    """

    def __init__(
        self,
        obs_dim:          int,
        act_dim:          int,
        num_agents:       int,
        hidden_dim:       int,
        num_layers:       int,
        actor_num_layers: int,
        critic_type:      str,
        rngs:             nnx.Rngs,
    ) -> None:
        self.num_agents  = num_agents
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.critic_type = critic_type

        self.actor = DecentralizedActor(
            obs_dim          = obs_dim,
            act_dim          = act_dim,
            hidden_dim       = hidden_dim,
            actor_num_layers = actor_num_layers,
            rngs             = rngs,
        )

        if critic_type == "agent_centric":
            self.critic = AgentCentricCritic(
                obs_dim    = obs_dim,
                hidden_dim = hidden_dim,
                num_layers = num_layers,
                rngs       = rngs,
            )
        elif critic_type == "global_mean":
            self.critic = GlobalMeanCritic(
                obs_dim    = obs_dim,
                hidden_dim = hidden_dim,
                num_layers = num_layers,
                rngs       = rngs,
            )
        else:
            raise ValueError(
                f"Unknown critic_type '{critic_type}'. "
                "Expected 'agent_centric' or 'global_mean'."
            )

    # ── Convenience wrappers ────────────────────────────────────────────────

    def get_value(self, all_obs: jax.Array, deterministic: bool = True) -> jax.Array:
        """
        Centralised value estimate.

        Parameters
        ----------
        all_obs : (..., N, obs_dim)

        Returns
        -------
        agent_centric : (..., N)  — one value per agent
        global_mean   : (...,)    — one value for the team
        """
        return self.critic(all_obs, deterministic=deterministic)

    def rollout_step(
        self,
        all_obs:   jax.Array,   # (N, obs_dim)
        keys:      jax.Array,   # (N, 2) — per-agent PRNGKeys
        max_force: float = 50.0,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Full CTDE forward pass for ONE environment during rollout.

        Actions are returned as PRE-SQUASH samples u (Gaussian samples before tanh).
        The caller (runner) is responsible for:
          1. Squashing:  tanh(u)  → bounded (-1, 1)
          2. Scaling:    tanh(u) × max_force → physics space
        The buffer stores u directly so evaluate_actions can recompute the
        exact same log-prob on u.

        Returns
        -------
        actions   : (N, act_dim)  — pre-squash u values
        log_probs : (N,)          — Gaussian log-prob on u
        value     : (N,) for agent_centric  |  () for global_mean
        """
        def _act_one(obs_i, key_i):
            a, lp, _ = self.actor.act(obs_i, key_i, deterministic=False)
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

    print("── MAPPOModel Self-Test ─────────────────────────────────────")
    cfg     = load_config(cli_overrides=False)
    validate_config(cfg)
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N       = int(cfg.env.num_agents)

    print(f"  obs_dim        : {obs_dim}")
    print(f"  act_dim        : {act_dim}")
    print(f"  N              : {N}")
    print(f"  critic_type    : {cfg.network.critic_type}")
    print(f"  actor_layers   : {cfg.network.actor_num_layers}")
    print(f"  critic_layers  : {cfg.network.num_layers}")

    rngs  = nnx.Rngs(0)
    model = MAPPOModel(
        obs_dim          = obs_dim,
        act_dim          = act_dim,
        num_agents       = N,
        hidden_dim       = int(cfg.network.hidden_dim),
        num_layers       = int(cfg.network.num_layers),
        actor_num_layers = int(cfg.network.actor_num_layers),
        critic_type      = str(cfg.network.critic_type),
        rngs             = rngs,
    )

    _, params = nnx.split(model)
    n_params  = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total params   : {n_params:,}")

    # rollout_step (single env)
    import jax.numpy as jnp
    obs   = jnp.zeros((N, obs_dim))
    keys  = jax.random.split(jax.random.PRNGKey(0), N)
    acts, lps, val = model.rollout_step(obs, keys)

    assert acts.shape == (N, act_dim), f"actions shape {acts.shape}"
    assert lps.shape  == (N,),         f"log_probs shape {lps.shape}"
    if str(cfg.network.critic_type) == "agent_centric":
        assert val.shape == (N,), f"value shape {val.shape} (expected ({N},))"
    else:
        assert val.shape == (), f"value shape {val.shape} (expected ())"
    print(f"  rollout_step : acts={acts.shape} lps={lps.shape} val={val.shape}  ✓")

    # Action normalisation check — acts are pre-squash u; squashed = tanh(u) must be in (-1, 1)
    squashed_acts = jnp.tanh(acts)
    assert jnp.all(squashed_acts > -1.0) and jnp.all(squashed_acts < 1.0), "tanh(actions) out of (-1, 1)!"
    print("  tanh(actions) in (-1, 1)  ✓")

    # Batched get_value
    obs_batch  = jnp.zeros((8, N, obs_dim))
    vals_batch = model.get_value(obs_batch)
    if str(cfg.network.critic_type) == "agent_centric":
        assert vals_batch.shape == (8, N), f"batched value shape {vals_batch.shape}"
    else:
        assert vals_batch.shape == (8,), f"batched value shape {vals_batch.shape}"
    print(f"  get_value batch: {vals_batch.shape}  ✓")

    print("\nMAPPOModel self-test passed ✓")
