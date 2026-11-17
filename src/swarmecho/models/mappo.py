"""Recurrent MAPPO actor and agent-centric critic for SwarmEcho."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from swarmecho.models.actor import RecurrentDecentralizedActor
from swarmecho.models.critic import RecurrentAgentCentricCritic


class MAPPOModel(nnx.Module):
    """
    Combined actor-critic wrapper for MAPPO.

    Parameters
    ----------
    obs_dim          : per-agent observation dimension
    act_dim          : action dimension (3 for the SwarmEcho runtime)
    num_agents       : default swarm size for recurrent state allocation
    hidden_dim       : hidden layer width (shared by actor and critic)
    num_layers       : number of hidden layers in the critic value head
    actor_num_layers : number of hidden layers in the actor policy trunk
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
        tarmac_sig_dim: int = 64,
        tarmac_val_dim: int = 128,
    ) -> None:
        self.num_agents  = num_agents
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.hidden_dim  = hidden_dim
        self.tarmac_sig_dim = tarmac_sig_dim
        self.tarmac_val_dim = tarmac_val_dim

        self.actor = RecurrentDecentralizedActor(
            obs_dim          = self.obs_dim,
            act_dim          = act_dim,
            hidden_dim       = hidden_dim,
            actor_num_layers = actor_num_layers,
            rngs             = rngs,
            tarmac_sig_dim = tarmac_sig_dim,
            tarmac_val_dim = tarmac_val_dim,
        )
        self.critic = RecurrentAgentCentricCritic(
            obs_dim    = obs_dim,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            rngs       = rngs,
        )

    def initial_actor_hidden(self, batch_shape=(), *, num_agents: int | None = None) -> jax.Array:
        """Return zero actor memory; N defaults to training, or the deployment count."""
        n = self.num_agents if num_agents is None else num_agents
        return jnp.zeros((*tuple(batch_shape), n, self.hidden_dim), dtype=jnp.float32)

    def initial_actor_signature(self, batch_shape=(), *, num_agents: int | None = None) -> jax.Array:
        """Return zero TarMAC signatures for the training or deployment count."""
        n = self.num_agents if num_agents is None else num_agents
        return jnp.zeros((*tuple(batch_shape), n, self.tarmac_sig_dim), dtype=jnp.float32)

    def initial_actor_value(self, batch_shape=(), *, num_agents: int | None = None) -> jax.Array:
        """Return zero TarMAC values for the training or deployment count."""
        n = self.num_agents if num_agents is None else num_agents
        return jnp.zeros((*tuple(batch_shape), n, self.tarmac_val_dim), dtype=jnp.float32)

    def initial_critic_hidden(self, batch_shape=()) -> jax.Array:
        """Return zero critic memory with shape batch_shape + (N, H)."""
        return jnp.zeros((*tuple(batch_shape), self.num_agents, self.hidden_dim), dtype=jnp.float32)

    def get_value_recurrent(
        self,
        all_obs:        jax.Array,
        critic_hidden:  jax.Array | None,
        resets:         jax.Array | None,
        deterministic:  bool = True,
        active:          jax.Array | None = None,
    ) -> tuple[jax.Array | None, jax.Array]:
        """Return the recurrent critic state and values in reward units."""
        if critic_hidden is None:
            critic_hidden = self.initial_critic_hidden(all_obs.shape[:-2])
        hidden, values = self.critic(all_obs, critic_hidden, resets,
                                     deterministic=deterministic, active=active)
        values = self.value_normalizer.denormalize(values)
        if active is not None:
            values = jnp.where(active, values, 0.0)
        return hidden, values

    def rollout_step_recurrent(
        self,
        all_obs:        jax.Array,   # (N, obs_dim)
        keys:           jax.Array,   # (N, 2)
        actor_hidden:   jax.Array | None,
        critic_hidden:  jax.Array | None,
        resets:         jax.Array,   # (N,)
        actor_signature: jax.Array | None = None,
        actor_value:    jax.Array | None = None,
        comm_mask:      jax.Array | None = None,
        active:         jax.Array | None = None,
        base_signature: jax.Array | None = None,
        base_value:     jax.Array | None = None,
        base_memory_mask: jax.Array | None = None,
    ) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None, jax.Array, jax.Array, jax.Array]:
        """Sample pre-squash movement while updating actor and critic memory."""
        if active is not None:
            resets = resets | ~active
        if actor_hidden is None:
            actor_hidden = self.initial_actor_hidden()
        if actor_signature is None:
            actor_signature = self.initial_actor_signature()
        if actor_value is None:
            actor_value = self.initial_actor_value()
        actor_hidden, actor_signature, actor_value, actions, log_probs, _ = self.actor.act_team(
            all_obs, actor_hidden, actor_signature, actor_value, keys,
            reset=resets, comm_mask=comm_mask, active=active,
            base_signature=base_signature, base_value=base_value, base_memory_mask=base_memory_mask,
            deterministic=False,
        )

        critic_hidden, value = self.get_value_recurrent(
            all_obs,
            critic_hidden,
            resets,
            deterministic=False,
            active=active,
        )
        if active is not None:
            actor_hidden = jnp.where(active[..., None], actor_hidden, 0.0)
            if actor_signature is not None:
                actor_signature = jnp.where(active[..., None], actor_signature, 0.0)
            if actor_value is not None:
                actor_value = jnp.where(active[..., None], actor_value, 0.0)
            actions = jnp.where(active[..., None], actions, 0.0)
            log_probs = jnp.where(active, log_probs, 0.0)
        return actor_hidden, actor_signature, actor_value, critic_hidden, actions, log_probs, value
