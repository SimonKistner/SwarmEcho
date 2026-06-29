"""
swarmecho/models/mappo.py
==========================
MAPPO model wrapper for SwarmEcho.

Holds a DecentralizedActor and one of two critic variants, selected via config:
  - "agent_centric"  (default) : AgentCentricCritic → V_i per agent  (..., N)
  - "global_mean"              : GlobalMeanCritic   → scalar V        (...,)

Optional recurrent actor and critic paths are selected by config flags. When
disabled, the architecture and call contract are the original feed-forward MAPPO.

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

from models.actor import DecentralizedActor, RecurrentDecentralizedActor
from models.critic import AgentCentricCritic, GlobalMeanCritic, RecurrentAgentCentricCritic


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
    actor_memory     : if True, use a per-agent GRU actor
    critic_memory    : if True, use a per-agent GRU agent-centric critic
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
        actor_memory:     bool = False,
        critic_memory:    bool = False,
        memory_comm_enabled: bool = False,
        memory_comm_every_k_steps: int = 5,
        memory_comm_frequency_control: str = "static",
        ic3_comm_gate_entropy_coef: float = 0.001,
        ic3_comm_gate_cost: float = 0.0,
        tarmac_sig_dim: int = 64,
        tarmac_val_dim: int = 128,
        tarmac_include_self: bool = True,
    ) -> None:
        self.num_agents  = num_agents
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.hidden_dim  = hidden_dim
        self.critic_type = critic_type
        self.actor_memory = actor_memory
        self.critic_memory = critic_memory
        self.memory_comm_enabled = memory_comm_enabled
        self.memory_comm_every_k_steps = memory_comm_every_k_steps
        self.memory_comm_frequency_control = memory_comm_frequency_control
        self.ic3_comm_enabled = memory_comm_enabled and memory_comm_frequency_control == "ic3"
        self.ic3_comm_gate_entropy_coef = ic3_comm_gate_entropy_coef
        self.ic3_comm_gate_cost = ic3_comm_gate_cost
        self.tarmac_sig_dim = tarmac_sig_dim
        self.tarmac_val_dim = tarmac_val_dim
        self.tarmac_include_self = tarmac_include_self

        if actor_memory:
            self.actor = RecurrentDecentralizedActor(
                obs_dim          = obs_dim,
                act_dim          = act_dim,
                hidden_dim       = hidden_dim,
                actor_num_layers = actor_num_layers,
                rngs             = rngs,
                memory_comm_enabled = memory_comm_enabled,
                memory_comm_every_k_steps = memory_comm_every_k_steps,
                memory_comm_frequency_control = memory_comm_frequency_control,
                tarmac_sig_dim = tarmac_sig_dim,
                tarmac_val_dim = tarmac_val_dim,
                tarmac_include_self = tarmac_include_self,
            )
        else:
            self.actor = DecentralizedActor(
                obs_dim          = obs_dim,
                act_dim          = act_dim,
                hidden_dim       = hidden_dim,
                actor_num_layers = actor_num_layers,
                rngs             = rngs,
            )

        if critic_memory and critic_type != "agent_centric":
            raise ValueError("critic_memory=True requires critic_type='agent_centric'.")

        if critic_type == "agent_centric" and critic_memory:
            self.critic = RecurrentAgentCentricCritic(
                obs_dim    = obs_dim,
                hidden_dim = hidden_dim,
                num_layers = num_layers,
                rngs       = rngs,
            )
        elif critic_type == "agent_centric":
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
        if self.critic_memory:
            hidden = self.initial_critic_hidden(all_obs.shape[:-2])
            _, values = self.critic(all_obs, hidden, deterministic=deterministic)
            return values
        return self.critic(all_obs, deterministic=deterministic)

    def initial_actor_hidden(self, batch_shape=()) -> jax.Array:
        """Return zero actor memory with shape batch_shape + (N, H)."""
        return jnp.zeros((*tuple(batch_shape), self.num_agents, self.hidden_dim), dtype=jnp.float32)

    def initial_actor_signature(self, batch_shape=()) -> jax.Array:
        """Return zero TarMAC actor signatures with shape batch_shape + (N, S)."""
        return jnp.zeros((*tuple(batch_shape), self.num_agents, self.tarmac_sig_dim), dtype=jnp.float32)

    def initial_actor_value(self, batch_shape=()) -> jax.Array:
        """Return zero TarMAC actor values with shape batch_shape + (N, V)."""
        return jnp.zeros((*tuple(batch_shape), self.num_agents, self.tarmac_val_dim), dtype=jnp.float32)

    def initial_critic_hidden(self, batch_shape=()) -> jax.Array:
        """Return zero critic memory with shape batch_shape + (N, H)."""
        return jnp.zeros((*tuple(batch_shape), self.num_agents, self.hidden_dim), dtype=jnp.float32)

    def get_value_recurrent(
        self,
        all_obs:        jax.Array,
        critic_hidden:  jax.Array | None,
        resets:         jax.Array | None,
        deterministic:  bool = True,
    ) -> tuple[jax.Array | None, jax.Array]:
        """Centralised value call that carries critic memory when enabled."""
        if self.critic_memory:
            if critic_hidden is None:
                critic_hidden = self.initial_critic_hidden(all_obs.shape[:-2])
            return self.critic(all_obs, critic_hidden, resets, deterministic=deterministic)
        return critic_hidden, self.critic(all_obs, deterministic=deterministic)

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

    def rollout_step_recurrent(
        self,
        all_obs:        jax.Array,   # (N, obs_dim)
        keys:           jax.Array,   # (N, 2)
        actor_hidden:   jax.Array | None,
        critic_hidden:  jax.Array | None,
        resets:         jax.Array,   # (N,)
        max_force:      float = 50.0,
        actor_signature: jax.Array | None = None,
        actor_value:    jax.Array | None = None,
        comm_mask:      jax.Array | None = None,
        active:         jax.Array | None = None,
        base_signature: jax.Array | None = None,
        base_value:     jax.Array | None = None,
        base_memory_mask: jax.Array | None = None,
        comm_gate:      jax.Array | None = None,
    ) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None, jax.Array, jax.Array, jax.Array]:
        """
        Rollout step that carries optional actor and critic recurrent states.

        The returned actions are pre-squash Gaussian samples, matching the
        feed-forward rollout contract.
        """
        if self.actor_memory:
            if actor_hidden is None:
                actor_hidden = self.initial_actor_hidden(())

            if self.memory_comm_enabled:
                if actor_signature is None:
                    actor_signature = self.initial_actor_signature(())
                if actor_value is None:
                    actor_value = self.initial_actor_value(())
                actor_hidden, actor_signature, actor_value, actions, log_probs, _ = self.actor.act_team(
                    all_obs,
                    actor_hidden,
                    actor_signature,
                    actor_value,
                    keys,
                    reset=resets,
                    comm_mask=comm_mask,
                    active=active,
                    base_signature=base_signature,
                    base_value=base_value,
                    base_memory_mask=base_memory_mask,
                    deterministic=False,
                    comm_gate=comm_gate,
                )
            else:
                def _act_one(obs_i, key_i, h_i, reset_i):
                    h_i, a, lp, _ = self.actor.act(obs_i, h_i, key_i, deterministic=False, reset=reset_i)
                    return h_i, a, lp

                actor_hidden, actions, log_probs = jax.vmap(_act_one)(
                    all_obs, keys, actor_hidden, resets
                )
        else:
            def _act_one(obs_i, key_i):
                a, lp, _ = self.actor.act(obs_i, key_i, deterministic=False)
                return a, lp

            actions, log_probs = jax.vmap(_act_one)(all_obs, keys)

        critic_hidden, value = self.get_value_recurrent(
            all_obs,
            critic_hidden,
            resets,
            deterministic=False,
        )
        return actor_hidden, actor_signature, actor_value, critic_hidden, actions, log_probs, value


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
    print(f"  actor_memory   : {cfg.network.get('actor_memory', False)}")
    print(f"  critic_memory  : {cfg.network.get('critic_memory', False)}")

    rngs  = nnx.Rngs(0)
    model = MAPPOModel(
        obs_dim          = obs_dim,
        act_dim          = act_dim,
        num_agents       = N,
        hidden_dim       = int(cfg.network.hidden_dim),
        num_layers       = int(cfg.network.num_layers),
        actor_num_layers = int(cfg.network.actor_num_layers),
        critic_type      = str(cfg.network.critic_type),
        actor_memory     = bool(cfg.network.get("actor_memory", False)),
        critic_memory    = bool(cfg.network.get("critic_memory", False)),
        rngs             = rngs,
        memory_comm_enabled = bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_every_k_steps = int(cfg.network.get("memory_comm_every_k_steps", 5)),
        memory_comm_frequency_control = str(cfg.network.get("memory_comm_frequency_control", "static")),
        ic3_comm_gate_entropy_coef = float(cfg.network.get("ic3_comm_gate_entropy_coef", 0.001)),
        ic3_comm_gate_cost = float(cfg.network.get("ic3_comm_gate_cost", 0.0)),
        tarmac_sig_dim = int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim = int(cfg.network.get("tarmac_val_dim", 128)),
        tarmac_include_self = bool(cfg.network.get("tarmac_include_self", True)),
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
