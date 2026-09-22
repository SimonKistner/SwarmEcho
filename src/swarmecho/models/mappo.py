"""
swarmecho/models/mappo.py
==========================
MAPPO model wrapper for SwarmEcho.

Holds a DecentralizedActor and an AgentCentricCritic that returns one value
per agent.

Optional recurrent actor and critic paths are selected by config flags. When
disabled, the architecture and call contract are the original feed-forward MAPPO.

All heavy lifting (network definitions) lives in actor.py and critic.py.
The trainer and workflow validator construct this shared model.

CTDE contract
-------------
  Rollout  : actor(obs_i) per agent — decentralised
  Training : critic(all_obs)        — centralised (sees full team)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from swarmecho.models.actor import DecentralizedActor, RecurrentDecentralizedActor
from swarmecho.models.critic import (
    AgentCentricCritic,
    RecurrentAgentCentricCritic,
)


# ---------------------------------------------------------------------------
# MAPPOModel — actor + critic wrapper
# ---------------------------------------------------------------------------

class MAPPOModel(nnx.Module):
    """
    Combined actor-critic wrapper for MAPPO.

    Parameters
    ----------
    obs_dim          : per-agent observation dimension
    act_dim          : action dimension (3 for the SwarmEcho runtime)
    num_agents       : N — swarm size (used for shape assertions only)
    hidden_dim       : hidden layer width (shared by actor and critic)
    num_layers       : number of hidden layers in the CRITIC
    actor_num_layers : number of hidden layers in the ACTOR (recommended: 2)
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
        rngs:             nnx.Rngs,
        actor_memory:     bool = False,
        critic_memory:    bool = False,
        critic_type:      str = "observation",
        critic_input_dim: int | None = None,
        memory_comm_enabled: bool = False,
        memory_comm_every_k_steps: int = 5,
        tarmac_sig_dim: int = 64,
        tarmac_val_dim: int = 128,
        tarmac_include_self: bool = True,
    ) -> None:
        self.num_agents  = num_agents
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.hidden_dim  = hidden_dim
        self.actor_memory = actor_memory
        self.critic_memory = critic_memory
        self.critic_type = critic_type
        self.critic_input_dim = obs_dim if critic_input_dim is None else critic_input_dim
        self.memory_comm_enabled = memory_comm_enabled
        self.memory_comm_every_k_steps = memory_comm_every_k_steps
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

        if critic_memory:
            self.critic = RecurrentAgentCentricCritic(
                obs_dim    = self.critic_input_dim,
                hidden_dim = hidden_dim,
                num_layers = num_layers,
                rngs       = rngs,
            )
        else:
            self.critic = AgentCentricCritic(
                obs_dim    = self.critic_input_dim,
                hidden_dim = hidden_dim,
                num_layers = num_layers,
                rngs       = rngs,
            )

    # ── Convenience wrappers ────────────────────────────────────────────────

    def get_value(self, all_obs: jax.Array, deterministic: bool = True, critic_obs: jax.Array | None = None,
                  active: jax.Array | None = None) -> jax.Array:
        """
        Centralised value estimate.

        Parameters
        ----------
        all_obs : (..., N, obs_dim)

        Returns
        -------
        (..., N) — one value per agent
        """
        critic_input = all_obs if critic_obs is None else critic_obs
        if self.critic_memory:
            hidden = self.initial_critic_hidden(critic_input.shape[:-2])
            if getattr(self, "mask_inactive", False):
                return self.get_value_recurrent(
                    all_obs, hidden, None, deterministic=deterministic, critic_obs=critic_obs, active=active
                )[1]
            _, values = self.critic(critic_input, hidden, deterministic=deterministic)
            return values
        return self.critic(critic_input, deterministic=deterministic)

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
        critic_obs:      jax.Array | None = None,
        active:          jax.Array | None = None,
    ) -> tuple[jax.Array | None, jax.Array]:
        """Centralised value call that carries critic memory when enabled."""
        critic_input = all_obs if critic_obs is None else critic_obs
        if self.critic_memory:
            if critic_hidden is None:
                critic_hidden = self.initial_critic_hidden(critic_input.shape[:-2])
            if getattr(self, "mask_inactive", False):
                hidden, values = self.critic(critic_input, critic_hidden, resets,
                                             deterministic=deterministic, active=active)
                if hasattr(self, "value_normalizer"):
                    values = self.value_normalizer.denormalize(values)
                if active is not None:
                    values = jnp.where(active, values, 0.0)
                return hidden, values
            return self.critic(critic_input, critic_hidden, resets, deterministic=deterministic)
        return critic_hidden, self.critic(critic_input, deterministic=deterministic)

    def rollout_step(
        self,
        all_obs:   jax.Array,   # (N, obs_dim)
        keys:      jax.Array,   # (N, 2) — per-agent PRNGKeys
        max_force: float = 50.0,
        critic_obs: jax.Array | None = None,
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
        value     : (N,)
        """
        def _act_one(obs_i, key_i):
            a, lp, _ = self.actor.act(obs_i, key_i, deterministic=False)
            return a, lp

        actions, log_probs = jax.vmap(_act_one)(all_obs, keys)
        value = self.get_value(all_obs, deterministic=False, critic_obs=critic_obs)
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
        critic_obs:     jax.Array | None = None,
    ) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None, jax.Array, jax.Array, jax.Array]:
        """
        Rollout step that carries optional actor and critic recurrent states.

        The returned actions are pre-squash Gaussian samples, matching the
        feed-forward rollout contract.
        """
        if getattr(self, "mask_inactive", False) and active is not None:
            resets = resets | ~active
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
            critic_obs=critic_obs,
            active=active,
        )
        if getattr(self, "mask_inactive", False) and active is not None:
            actor_hidden = jnp.where(active[..., None], actor_hidden, 0.0)
            if actor_signature is not None:
                actor_signature = jnp.where(active[..., None], actor_signature, 0.0)
            if actor_value is not None:
                actor_value = jnp.where(active[..., None], actor_value, 0.0)
            actions = jnp.where(active[..., None], actions, 0.0)
            log_probs = jnp.where(active, log_probs, 0.0)
        return actor_hidden, actor_signature, actor_value, critic_hidden, actions, log_probs, value
