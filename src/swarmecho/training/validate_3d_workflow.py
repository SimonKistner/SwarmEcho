"""Small environment→MAPPO-update validation for the 3D migration path."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from swarmecho.core.config3d import load_level_3d
from swarmecho.env.baseline3d import (
    make_autoreset_3d_fns,
)
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.mappo_trainer import MAPPOTrainer


def validate_3d_update(
    *,
    num_envs: int = 4,
    num_steps: int = 4,
    hidden_dim: int = 32,
    recurrent: bool = True,
) -> dict[str, float]:
    """Collect a real 3D rollout and complete one PPO update."""
    level = load_level_3d()
    cfg = replace(level.env, max_steps=max(2, num_steps - 1))
    reset, step, observations, _ = make_autoreset_3d_fns(level.building, cfg)
    obs_dim = 6 + cfg.radar_bins * 4
    model = MAPPOModel(
        obs_dim=obs_dim,
        act_dim=3,
        num_agents=cfg.num_agents,
        hidden_dim=hidden_dim,
        num_layers=2,
        actor_num_layers=2,
        actor_memory=recurrent,
        critic_memory=recurrent,
        memory_comm_enabled=recurrent,
        memory_comm_every_k_steps=1,
        tarmac_sig_dim=8,
        tarmac_val_dim=8,
        tarmac_include_self=False,
        rngs=nnx.Rngs(0),
    )
    trainer = MAPPOTrainer(
        model,
        num_epochs=1,
        actor_memory=recurrent,
        critic_memory=recurrent,
    )
    buffer = MAPPORolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        act_dim=3,
        recurrent=recurrent,
        hidden_dim=hidden_dim,
        tarmac_sig_dim=8,
        tarmac_val_dim=8,
        actor_memory=recurrent,
        critic_memory=recurrent,
    )
    keys = jax.random.split(jax.random.PRNGKey(1), num_envs)
    states = jax.vmap(reset)(keys)

    actor_hidden = model.initial_actor_hidden((num_envs,)) if recurrent else None
    actor_signature = model.initial_actor_signature((num_envs,)) if recurrent else None
    actor_value = model.initial_actor_value((num_envs,)) if recurrent else None
    critic_hidden = model.initial_critic_hidden((num_envs,)) if recurrent else None
    if recurrent:
        buffer.reset(
            actor_h=actor_hidden,
            critic_h=critic_hidden,
            actor_signature=actor_signature,
            actor_value=actor_value,
        )
    resets = jnp.zeros((num_envs, cfg.num_agents), dtype=jnp.bool_)

    for _ in range(num_steps):
        obs = jax.vmap(observations)(states)
        def split_key(key):
            parts = jax.random.split(key)
            return parts[0], parts[1]

        keys, action_roots = jax.vmap(split_key)(keys)
        agent_keys = jax.vmap(lambda key: jax.random.split(key, cfg.num_agents))(action_roots)
        delta = states.pos[:, :, None, :] - states.pos[:, None, :, :]
        comm_masks = (
            (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
            & states.active[:, :, None]
            & states.active[:, None, :]
            & ~jnp.eye(cfg.num_agents, dtype=jnp.bool_)[None, :, :]
        )
        base_signatures = jnp.zeros((num_envs, 8), dtype=jnp.float32)
        base_values = jnp.zeros((num_envs, 8), dtype=jnp.float32)
        base_memory_masks = jnp.zeros((num_envs, cfg.num_agents), dtype=jnp.bool_)
        if recurrent:
            def policy_one(
                obs_one,
                keys_one,
                actor_h_one,
                actor_sig_one,
                actor_val_one,
                critic_h_one,
                comm_one,
                active_one,
                base_sig_one,
                base_val_one,
                base_mask_one,
                resets_one,
            ):
                return model.rollout_step_recurrent(
                    obs_one,
                    keys_one,
                    actor_h_one,
                    critic_h_one,
                    resets_one,
                    actor_signature=actor_sig_one,
                    actor_value=actor_val_one,
                    comm_mask=comm_one,
                    active=active_one,
                    base_signature=base_sig_one,
                    base_value=base_val_one,
                    base_memory_mask=base_mask_one,
                )

            (
                actor_hidden,
                actor_signature,
                actor_value,
                critic_hidden,
                actions,
                log_probs,
                values,
            ) = jax.vmap(policy_one)(
                obs,
                agent_keys,
                actor_hidden,
                actor_signature,
                actor_value,
                critic_hidden,
                comm_masks,
                states.active,
                base_signatures,
                base_values,
                base_memory_masks,
                resets,
            )
        else:
            actions, log_probs, values = jax.vmap(model.rollout_step)(obs, agent_keys)
        next_states, rewards, dones, _ = jax.vmap(step)(states, jnp.tanh(actions))
        buffer.add(
            MAPPOTransition(
                obs=np.asarray(obs),
                actions=np.asarray(actions),
                log_probs=np.asarray(log_probs),
                values=np.asarray(values),
                rewards=np.asarray(rewards),
                dones=np.asarray(dones),
                rnn_resets=np.asarray(resets),
                comm_masks=np.asarray(comm_masks),
                active_masks=np.asarray(states.active),
                base_signatures=np.asarray(base_signatures),
                base_values=np.asarray(base_values),
                base_memory_masks=np.asarray(base_memory_masks),
                critic_obs=np.asarray(obs),
            )
        )
        states = next_states
        resets = jnp.broadcast_to(dones[:, None], (num_envs, cfg.num_agents))

    final_obs = jax.vmap(observations)(states)
    if recurrent:
        def final_value_one(obs_one, critic_h_one):
            _, values = model.get_value_recurrent(
                obs_one,
                critic_h_one,
                jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
            )
            return values

        final_values = jax.vmap(final_value_one)(final_obs, critic_hidden)
    else:
        final_values = jax.vmap(model.get_value)(final_obs)
    advantages, returns = buffer.compute_gae(final_values, states.done)
    minibatches = buffer.get_minibatches(
        advantages,
        returns,
        n_minibatches=1,
        key=jax.random.PRNGKey(2),
    )
    stats = trainer.update(minibatches)
    if not all(np.isfinite(value) for value in stats.values()):
        raise FloatingPointError(f"Non-finite 3D PPO update statistics: {stats}")
    return stats


def main() -> None:
    stats = validate_3d_update()
    print("3D environment → recurrent TarMAC MAPPO update validation passed.")
    for name, value in stats.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
