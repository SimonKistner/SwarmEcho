"""Small environment→MAPPO-update validation for the 3D migration path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from swarmecho.env.baseline3d import (
    Baseline3DConfig,
    make_baseline_3d_fns,
    rewards_3d,
)
from swarmecho.env.buildings import load_building
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.benchmark_3d import DEFAULT_BUILDING
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.mappo_trainer import MAPPOTrainer


def validate_3d_update(
    *,
    num_envs: int = 4,
    num_steps: int = 4,
    hidden_dim: int = 32,
) -> dict[str, float]:
    """Collect a real 3D rollout and complete one feed-forward PPO update."""
    building = load_building(DEFAULT_BUILDING)
    cfg = Baseline3DConfig()
    reset, step, observations, _ = make_baseline_3d_fns(building, cfg)
    obs_dim = 6 + cfg.radar_bins * 4
    model = MAPPOModel(
        obs_dim=obs_dim,
        act_dim=3,
        num_agents=cfg.num_agents,
        hidden_dim=hidden_dim,
        num_layers=2,
        actor_num_layers=2,
        actor_memory=False,
        critic_memory=False,
        rngs=nnx.Rngs(0),
    )
    trainer = MAPPOTrainer(model, num_epochs=1)
    buffer = MAPPORolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        act_dim=3,
    )
    keys = jax.random.split(jax.random.PRNGKey(1), num_envs)
    states = jax.vmap(reset)(keys)

    def policy_one(obs, agent_keys):
        return model.rollout_step(obs, agent_keys)

    for _ in range(num_steps):
        obs = jax.vmap(observations)(states)
        def split_key(key):
            parts = jax.random.split(key)
            return parts[0], parts[1]

        keys, action_roots = jax.vmap(split_key)(keys)
        agent_keys = jax.vmap(lambda key: jax.random.split(key, cfg.num_agents))(action_roots)
        actions, log_probs, values = jax.vmap(policy_one)(obs, agent_keys)
        next_states = jax.vmap(step)(states, jnp.tanh(actions))
        rewards, _ = jax.vmap(rewards_3d)(states, next_states)
        dones = next_states.success
        buffer.add(
            MAPPOTransition(
                obs=np.asarray(obs),
                actions=np.asarray(actions),
                log_probs=np.asarray(log_probs),
                values=np.asarray(values),
                rewards=np.asarray(rewards),
                dones=np.asarray(dones),
                critic_obs=np.asarray(obs),
            )
        )
        states = next_states

    final_obs = jax.vmap(observations)(states)
    final_values = jax.vmap(model.get_value)(final_obs)
    advantages, returns = buffer.compute_gae(final_values, states.success)
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
    print("3D environment → MAPPO update validation passed.")
    for name, value in stats.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
