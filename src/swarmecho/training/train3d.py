"""Config-driven recurrent MAPPO training runner for the 3D baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from swarmecho.core.config3d import Level3D, load_level_3d
from swarmecho.env.baseline3d import make_autoreset_3d_fns, make_baseline_3d_fns, rewards_3d
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.checkpoints import save_model_checkpoint
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.mappo_trainer import MAPPOTrainer
from swarmecho.visualize.replay3d import write_replay


def build_model_3d(level: Level3D, hidden_dim: int | None = None) -> MAPPOModel:
    """Construct the canonical recurrent 3D MAPPO/TarMAC model."""
    cfg = level.env
    return MAPPOModel(
        obs_dim=6 + cfg.radar_bins * 4,
        act_dim=3,
        num_agents=cfg.num_agents,
        hidden_dim=hidden_dim or level.training.hidden_dim,
        num_layers=3,
        actor_num_layers=3,
        actor_memory=True,
        critic_memory=True,
        memory_comm_enabled=True,
        memory_comm_every_k_steps=1,
        tarmac_sig_dim=16,
        tarmac_val_dim=32,
        tarmac_include_self=False,
        rngs=nnx.Rngs(level.training.seed),
    )


def _communication_inputs(states, cfg, base_valid, base_signature, base_value):
    delta = states.pos[:, :, None, :] - states.pos[:, None, :, :]
    comm_masks = (
        (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
        & states.active[:, :, None]
        & states.active[:, None, :]
        & ~jnp.eye(cfg.num_agents, dtype=jnp.bool_)[None]
    )
    in_base_range = (
        jnp.linalg.norm(states.pos - states.base_pos[:, None, :], axis=-1)
        <= cfg.base_comm_radius
    ) & states.active
    base_memory_masks = base_valid[:, None] & in_base_range & ~states.target_known
    return comm_masks, in_base_range, base_memory_masks, base_signature, base_value


def _update_base_memory(
    states,
    in_base_range,
    emitted_signature,
    emitted_value,
    base_valid,
    base_signature,
    base_value,
    dones,
):
    reporters = states.target_known & in_base_range
    has_reporter = jnp.any(reporters, axis=-1)
    reporter_index = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
    env_index = jnp.arange(states.pos.shape[0])
    should_store = ~base_valid & has_reporter
    base_signature = jnp.where(
        should_store[:, None], emitted_signature[env_index, reporter_index], base_signature
    )
    base_value = jnp.where(
        should_store[:, None], emitted_value[env_index, reporter_index], base_value
    )
    base_valid = base_valid | should_store
    base_valid = jnp.where(dones, False, base_valid)
    base_signature = jnp.where(dones[:, None], 0.0, base_signature)
    base_value = jnp.where(dones[:, None], 0.0, base_value)
    return base_valid, base_signature, base_value


def train_3d(
    level: Level3D,
    *,
    updates: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, float]]:
    """Train the strict 3D level and return its final checkpoint and metrics."""
    training = level.training
    if updates is not None:
        training = replace(training, updates=updates)
    destination = Path(output_dir or training.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cfg = level.env
    reset, env_step, observations, _ = make_autoreset_3d_fns(level.building, cfg)
    obs_dim = 6 + cfg.radar_bins * 4
    sig_dim = 16
    val_dim = 32
    model = build_model_3d(level, training.hidden_dim)
    trainer = MAPPOTrainer(
        model,
        lr=training.learning_rate,
        num_epochs=training.num_epochs,
        actor_memory=True,
        critic_memory=True,
    )
    buffer = MAPPORolloutBuffer(
        num_steps=training.num_steps,
        num_envs=training.num_envs,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        act_dim=3,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        recurrent=True,
        hidden_dim=training.hidden_dim,
        tarmac_sig_dim=sig_dim,
        tarmac_val_dim=val_dim,
        actor_memory=True,
        critic_memory=True,
    )
    keys = jax.random.split(jax.random.PRNGKey(training.seed + 1), training.num_envs)
    states = jax.vmap(reset)(keys)
    actor_hidden = model.initial_actor_hidden((training.num_envs,))
    actor_signature = model.initial_actor_signature((training.num_envs,))
    actor_value = model.initial_actor_value((training.num_envs,))
    critic_hidden = model.initial_critic_hidden((training.num_envs,))
    base_valid = jnp.zeros(training.num_envs, dtype=jnp.bool_)
    base_signature = jnp.zeros((training.num_envs, sig_dim), dtype=jnp.float32)
    base_value = jnp.zeros((training.num_envs, val_dim), dtype=jnp.float32)
    resets = jnp.zeros((training.num_envs, cfg.num_agents), dtype=jnp.bool_)
    latest_stats: dict[str, float] = {}
    episode_successes = 0
    episode_count = 0

    for update in range(1, training.updates + 1):
        buffer.reset(actor_hidden, critic_hidden, actor_signature, actor_value)
        for _ in range(training.num_steps):
            obs = jax.vmap(observations)(states)
            split = jax.vmap(lambda key: jax.random.split(key))(keys)
            keys, action_roots = split[:, 0], split[:, 1]
            agent_keys = jax.vmap(lambda key: jax.random.split(key, cfg.num_agents))(action_roots)
            comm_masks, in_base_range, base_masks, base_sig_in, base_val_in = (
                _communication_inputs(states, cfg, base_valid, base_signature, base_value)
            )

            def policy_one(obs_e, keys_e, ah, sig, val, ch, cm, active, bs, bv, bm, rst):
                return model.rollout_step_recurrent(
                    obs_e,
                    keys_e,
                    ah,
                    ch,
                    rst,
                    actor_signature=sig,
                    actor_value=val,
                    comm_mask=cm,
                    active=active,
                    base_signature=bs,
                    base_value=bv,
                    base_memory_mask=bm,
                )

            (
                actor_hidden,
                emitted_signature,
                emitted_value,
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
                base_sig_in,
                base_val_in,
                base_masks,
                resets,
            )
            next_states, rewards, dones, info = jax.vmap(env_step)(states, jnp.tanh(actions))
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
                    base_signatures=np.asarray(base_sig_in),
                    base_values=np.asarray(base_val_in),
                    base_memory_masks=np.asarray(base_masks),
                    critic_obs=np.asarray(obs),
                )
            )
            base_valid, base_signature, base_value = _update_base_memory(
                states,
                in_base_range,
                emitted_signature,
                emitted_value,
                base_valid,
                base_signature,
                base_value,
                dones,
            )
            actor_signature, actor_value = emitted_signature, emitted_value
            episode_successes += int(np.asarray(info["success"]).sum())
            episode_count += int(np.asarray(dones).sum())
            states = next_states
            resets = jnp.broadcast_to(dones[:, None], resets.shape)

        final_obs = jax.vmap(observations)(states)

        def value_one(obs_e, hidden_e):
            return model.get_value_recurrent(obs_e, hidden_e, jnp.zeros(cfg.num_agents))[1]

        final_values = jax.vmap(value_one)(final_obs, critic_hidden)
        advantages, returns = buffer.compute_gae(final_values, states.done)
        minibatches = buffer.get_minibatches(
            advantages,
            returns,
            training.num_minibatches,
            jax.random.fold_in(jax.random.PRNGKey(training.seed), update),
        )
        latest_stats = trainer.update(minibatches)
        latest_stats["episode_success_rate"] = episode_successes / max(episode_count, 1)
        latest_stats["completed_episodes"] = float(episode_count)
        print(
            f"3D update {update}/{training.updates}: "
            f"loss={latest_stats['total_loss']:.4f} "
            f"success={latest_stats['episode_success_rate']:.3f}"
        )

    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint = save_model_checkpoint(
        model,
        destination / "checkpoints" / f"ckpt_{training.updates:06d}",
        run_name=level.name,
        update=training.updates,
        num_envs=training.num_envs,
        num_steps=training.num_steps,
        prior_history=[],
    )
    (destination / "metrics.json").write_text(
        json.dumps(latest_stats, indent=2) + "\n", encoding="utf-8"
    )
    replay_states, replay_rewards = evaluate_model_3d(
        model,
        level,
        max_steps=min(128, training.num_steps, cfg.max_steps),
    )
    write_replay(
        destination / "replays" / "latest",
        replay_states,
        map_name=level.building_name,
        dt=cfg.dt,
        reward_terms=replay_rewards,
    )
    return checkpoint, latest_stats


def evaluate_model_3d(
    model: MAPPOModel,
    level: Level3D,
    *,
    max_steps: int | None = None,
) -> tuple[list, np.ndarray]:
    """Run one deterministic trained-policy episode for replay/inspection."""
    cfg = level.env
    reset, step, observations, _ = make_baseline_3d_fns(level.building, cfg)
    state = reset(jax.random.PRNGKey(level.training.seed + 10_000))
    hidden = model.initial_actor_hidden(())
    signature = model.initial_actor_signature(())
    value = model.initial_actor_value(())
    base_valid = jnp.zeros(1, dtype=jnp.bool_)
    base_signature = jnp.zeros((1, model.tarmac_sig_dim), dtype=jnp.float32)
    base_value = jnp.zeros((1, model.tarmac_val_dim), dtype=jnp.float32)
    states = [state]
    reward_frames = [np.zeros((cfg.num_agents, 1), dtype=np.float32)]
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps)

    for _ in range(horizon):
        batched_state = jax.tree_util.tree_map(lambda item: item[None], state)
        comm, in_base, base_masks, base_sig_in, base_val_in = _communication_inputs(
            batched_state,
            cfg,
            base_valid,
            base_signature,
            base_value,
        )
        obs = observations(state)
        hidden, signature, value, mu, _ = model.actor.__call_team__(
            obs,
            hidden,
            signature,
            value,
            reset=jnp.zeros(cfg.num_agents, dtype=jnp.bool_),
            comm_mask=comm[0],
            active=state.active,
            base_signature=base_sig_in[0],
            base_value=base_val_in[0],
            base_memory_mask=base_masks[0],
            deterministic=True,
        )
        next_state = step(state, jnp.tanh(mu))
        reward, _ = rewards_3d(state, next_state, level.reward)
        done = next_state.done[None]
        base_valid, base_signature, base_value = _update_base_memory(
            batched_state,
            in_base,
            signature[None],
            value[None],
            base_valid,
            base_signature,
            base_value,
            done,
        )
        states.append(next_state)
        reward_frames.append(np.asarray(reward)[:, None])
        state = next_state
        if bool(state.done):
            break
    return states, np.stack(reward_frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="B00_3d_baseline")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint, _ = train_3d(
        load_level_3d(args.level),
        updates=args.updates,
        output_dir=args.output,
    )
    print(f"3D training complete: {checkpoint}")


if __name__ == "__main__":
    main()
