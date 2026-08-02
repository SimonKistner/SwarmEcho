"""Config-driven recurrent MAPPO training runner for the 3D baseline."""

from __future__ import annotations

import argparse
import os

# XLA/absl verbosity must be configured before importing JAX. Doing this in the
# training function is too late because plugin discovery happens at import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import json
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx

from swarmecho.core.config3d import Level3D, load_level_3d
from swarmecho.env.baseline3d import make_autoreset_3d_fns, make_baseline_3d_fns, rewards_3d
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.artifacts import artifact_suffix, train_replay_root
from swarmecho.training.checkpoints import restore_model_checkpoint, save_model_checkpoint
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
        hidden_dim=hidden_dim or level.network.hidden_dim,
        num_layers=level.network.num_layers,
        actor_num_layers=level.network.actor_num_layers,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
        critic_type=level.network.critic_type,
        memory_comm_enabled=level.network.memory_comm_enabled,
        memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
        tarmac_sig_dim=level.network.tarmac_sig_dim,
        tarmac_val_dim=level.network.tarmac_val_dim,
        tarmac_include_self=level.network.tarmac_include_self,
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
        <= cfg.comm_radius_base
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
    checkpoint_path: str | Path | None = None,
) -> tuple[Path, dict[str, float]]:
    """Train the strict 3D level and return its final checkpoint and metrics."""
    training = level.training
    network = level.network
    evaluation = level.evaluation
    logging = level.logging
    num_updates = updates if updates is not None else level.num_updates
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_name = logging.run_name
    run_name = (
        f"run_{timestamp}"
        if not configured_name
        else (
            f"{configured_name}_{timestamp}"
            if logging.use_timestamp_postfix
            else configured_name
        )
    )
    default_output = Path(logging.log_dir) / run_name
    destination = Path(output_dir) if output_dir is not None else default_output
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(evaluation.checkpoint_dir).absolute()
        if evaluation.checkpoint_dir
        else destination / "checkpoints"
    )
    replay_dir = train_replay_root(destination)
    cfg = level.env
    reset, env_step, observations, _ = make_autoreset_3d_fns(level.building, cfg)
    obs_dim = 6 + cfg.radar_bins * 4
    sig_dim = network.tarmac_sig_dim
    val_dim = network.tarmac_val_dim
    model = build_model_3d(level)
    resume_from = checkpoint_path or training.checkpoint_path
    if resume_from:
        restored = restore_model_checkpoint(model, resume_from)
        print(f"  Restored weights  ✓  {restored}")
    trainer = MAPPOTrainer(
        model,
        lr=training.lr,
        max_grad_norm=training.max_grad_norm,
        clip_eps=training.clip_eps,
        vf_coef=training.vf_coef,
        ent_coef=training.ent_coef,
        num_epochs=training.num_epochs,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
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
        hidden_dim=network.hidden_dim,
        tarmac_sig_dim=sig_dim,
        tarmac_val_dim=val_dim,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
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
    episode_returns = np.zeros(training.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(training.num_envs, dtype=np.int32)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)
    recent_success: deque[float] = deque(maxlen=100)
    recent_coverage: deque[float] = deque(maxlen=100)
    history_path = destination / "training_history.jsonl"
    config_snapshot = {
        "name": level.name,
        "building": level.building_name,
        "env": asdict(level.env),
        "reward": asdict(level.reward),
        "training": asdict(training),
        "network": asdict(network),
        "evaluation": asdict(evaluation),
        "logging": asdict(logging),
    }
    (destination / "config.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8"
    )
    wandb_run = None
    if logging.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=logging.wandb_project,
            entity=logging.wandb_entity,
            group=logging.wandb_group,
            name=run_name,
            mode=logging.wandb_mode,
            config=config_snapshot,
            dir=str(destination),
        )

    _, params = nnx.split(model)
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
    total_steps = num_updates * training.num_envs * training.num_steps
    print("\n══════════════════════════════════════════════════════")
    print("  SwarmEcho 3D — Recurrent MAPPO + TarMAC")
    print("══════════════════════════════════════════════════════")
    print(f"  level / building : {level.name} / {level.building_name}")
    print(f"  devices          : {jax.devices()}")
    print(f"  agents           : {cfg.num_agents}")
    print(f"  observation      : {obs_dim}  (radar bins: {cfg.radar_bins})")
    print("  action           : 3D continuous force")
    print(f"  model parameters : {parameter_count:,}")
    print(f"  environments     : {training.num_envs:,}")
    print(f"  rollout / updates: {training.num_steps} / {num_updates}")
    print(f"  total env steps  : {total_steps:,}")
    print(f"  output           : {destination.resolve()}")
    if wandb_run is not None:
        print(f"  W&B              : {wandb_run.url}")
    print("──────────────────────────────────────────────────────")
    start_time = time.perf_counter()

    for update in range(1, num_updates + 1):
        buffer.reset(actor_hidden, critic_hidden, actor_signature, actor_value)
        reward_totals = {
            "coverage": 0.0,
            "collision": 0.0,
            "finder": 0.0,
            "target_found": 0.0,
            "success": 0.0,
        }
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
            rewards_host = np.asarray(rewards)
            dones_host = np.asarray(dones)
            for reward_name in reward_totals:
                reward_totals[reward_name] += float(np.asarray(info[reward_name]).sum())
            episode_returns += rewards_host.sum(axis=-1)
            episode_lengths += 1
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
            completed_indices = np.flatnonzero(dones_host)
            for index in completed_indices:
                recent_returns.append(float(episode_returns[index]))
                recent_lengths.append(int(episode_lengths[index]))
                recent_success.append(float(np.asarray(info["success"])[index]))
                recent_coverage.append(float(np.asarray(info["terminal_coverage_fraction"])[index]))
            episode_count += len(completed_indices)
            episode_returns[completed_indices] = 0.0
            episode_lengths[completed_indices] = 0
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
        elapsed = time.perf_counter() - start_time
        steps_done = update * training.num_envs * training.num_steps
        sps = steps_done / max(elapsed, 1e-6)
        eta_seconds = (num_updates - update) * elapsed / update
        latest_stats.update(
            {
                "global_step": float(steps_done),
                "sps": float(sps),
                "mean_episode_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "mean_episode_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                "rolling_success_rate": float(np.mean(recent_success)) if recent_success else 0.0,
                "mean_terminal_coverage": float(np.mean(recent_coverage)) if recent_coverage else 0.0,
                "live_episode_return": float(np.mean(episode_returns)),
                "live_coverage": float(np.asarray(states.coverage).mean()),
                "live_target_known_rate": float(np.asarray(states.target_known).mean()),
                "live_chain_rate": float(np.asarray(states.fully_connected).mean()),
                "live_episode_step": float(np.asarray(states.step).mean()),
            }
        )
        reward_denominator = training.num_envs * training.num_steps * cfg.num_agents
        latest_stats.update(
            {
                f"reward_{name}": total / reward_denominator
                for name, total in reward_totals.items()
            }
        )
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps({"update": update, **latest_stats}) + "\n")
        if update % logging.log_every == 0 or update == 1 or update == num_updates:
            eta_m, eta_s = divmod(int(eta_seconds), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"
            warmup = "warmup" if not recent_returns else f"ep={len(recent_returns):3d}"
            display_return = (
                latest_stats["mean_episode_return"]
                if recent_returns
                else latest_stats["live_episode_return"]
            )
            display_coverage = (
                latest_stats["mean_terminal_coverage"]
                if recent_coverage
                else latest_stats["live_coverage"]
            )
            display_length = (
                latest_stats["mean_episode_length"]
                if recent_lengths
                else latest_stats["live_episode_step"]
            )
            print(
                f"[{datetime.now():%H:%M:%S}] [{update:>4}/{num_updates}] "
                f"steps={steps_done:>10,} sps={sps:>9,.0f} "
                f"return={display_return:>8.2f} "
                f"len={display_length:>6.1f} "
                f"cov={display_coverage:>6.1%} "
                f"known={latest_stats['live_target_known_rate']:>6.1%} "
                f"chain={latest_stats['live_chain_rate']:>6.1%} "
                f"succ={latest_stats['rolling_success_rate']:>6.1%} "
                f"loss={latest_stats['total_loss']:>8.3f} {warmup} eta={eta}"
            )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "ppo/policy_loss": latest_stats["policy_loss"],
                    "ppo/value_loss": latest_stats["value_loss"],
                    "ppo/entropy": latest_stats["entropy"],
                    "ppo/approx_kl": latest_stats["approx_kl"],
                    "ppo/clip_fraction": latest_stats["clip_fraction"],
                    "train/episode_return": latest_stats["mean_episode_return"],
                    "train/episode_length": latest_stats["mean_episode_length"],
                    "train/success_rate": latest_stats["rolling_success_rate"],
                    "train/coverage": latest_stats["mean_terminal_coverage"],
                    "train/live_coverage": latest_stats["live_coverage"],
                    "train/live_target_known_rate": latest_stats["live_target_known_rate"],
                    "train/live_chain_rate": latest_stats["live_chain_rate"],
                    "perf/sps": sps,
                    **{
                        f"rewards/{name}": latest_stats[f"reward_{name}"]
                        for name in reward_totals
                    },
                },
                step=steps_done,
            )
        checkpoint_due = (
            evaluation.save_model
            and update > evaluation.checkpoint_offset
            and (update - evaluation.checkpoint_offset) % evaluation.checkpoint_freq == 0
        )
        if checkpoint_due and update < num_updates:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_model_checkpoint(
                model,
                checkpoint_dir / f"ckpt_{update:06d}",
                run_name=run_name,
                update=update,
                num_envs=training.num_envs,
                num_steps=training.num_steps,
                prior_history=[],
            )
        threshold_met = (
            latest_stats["rolling_success_rate"] >= evaluation.eval_min_train_success
        )
        eval_due = (
            update > evaluation.eval_offset
            and (update - evaluation.eval_offset) % evaluation.eval_freq == 0
            and threshold_met
        )
        replay_due = (
            evaluation.eval_video
            and update > evaluation.eval_video_offset
            and (update - evaluation.eval_video_offset) % evaluation.eval_video_freq == 0
            and threshold_met
        )
        if update == num_updates:
            eval_due = True
            replay_due = evaluation.eval_video
        if eval_due or replay_due:
            eval_metrics, eval_states, eval_rewards = evaluate_suite_3d(
                model,
                level,
                episodes=evaluation.eval_parallel_envs,
                max_steps=min(128, cfg.max_steps),
            )
            latest_stats.update(eval_metrics)
            if replay_due:
                suffix = artifact_suffix(update, steps_done)
                write_replay(
                    replay_dir / f"eval_{suffix}",
                    eval_states,
                    map_name=level.building_name,
                    dt=cfg.dt,
                    reward_terms=eval_rewards,
                    metadata={
                        "world_size_m": level.building.world_size_m.tolist(),
                        "cell_size_m": level.building.cell_size_m,
                        "comm_radius_m": cfg.comm_radius,
                        "comm_radius_base_m": cfg.comm_radius_base,
                        "visual_radius_m": cfg.visual_radius,
                        "training_update": update,
                        "environment_steps": steps_done,
                        "artifact_scope": "train",
                    },
                )
            print(
                f"         evaluation: return={latest_stats['eval_return']:.2f} "
                f"coverage={latest_stats['eval_coverage']:.1%} "
                f"success={latest_stats['eval_success']:.0%}"
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_model_checkpoint(
        model,
        checkpoint_dir / f"ckpt_{num_updates:06d}",
        run_name=run_name,
        update=num_updates,
        num_envs=training.num_envs,
        num_steps=training.num_steps,
        prior_history=[],
    )
    (destination / "metrics.json").write_text(
        json.dumps(latest_stats, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.finish()
    print("──────────────────────────────────────────────────────")
    print(f"  Training complete ✓  checkpoint: {checkpoint}")
    if evaluation.eval_video:
        final_suffix = artifact_suffix(num_updates, total_steps)
        print(f"  Replay ready      ✓  {replay_dir / f'eval_{final_suffix}.json'}")
    return checkpoint, latest_stats


def evaluate_model_3d(
    model: MAPPOModel,
    level: Level3D,
    *,
    max_steps: int | None = None,
    seed: int | None = None,
) -> tuple[list, np.ndarray]:
    """Run one deterministic trained-policy episode for replay/inspection."""
    cfg = level.env
    reset, step, observations, _ = make_baseline_3d_fns(level.building, cfg)
    state = reset(jax.random.PRNGKey(seed if seed is not None else level.training.seed + 10_000))
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


def evaluate_suite_3d(
    model: MAPPOModel,
    level: Level3D,
    *,
    episodes: int,
    max_steps: int | None = None,
) -> tuple[dict[str, float], list, np.ndarray]:
    """Evaluate several reproducible targets and retain the first replay."""
    returns, successes, coverages, lengths = [], [], [], []
    first_states = None
    first_rewards = None
    for episode in range(episodes):
        states, rewards = evaluate_model_3d(
            model,
            level,
            max_steps=max_steps,
            seed=level.training.seed + 10_000 + episode,
        )
        if first_states is None:
            first_states, first_rewards = states, rewards
        returns.append(float(rewards.sum()))
        successes.append(float(states[-1].success))
        coverages.append(float(jnp.mean(states[-1].coverage)))
        lengths.append(len(states) - 1)
    metrics = {
        "eval_return": float(np.mean(returns)),
        "eval_success": float(np.mean(successes)),
        "eval_coverage": float(np.mean(coverages)),
        "eval_episode_length": float(np.mean(lengths)),
    }
    return metrics, first_states, first_rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="M00_no_maze_open_cuboid_3D")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint, _ = train_3d(
        load_level_3d(args.level),
        updates=args.updates,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
    )
    print(f"3D training complete: {checkpoint}")


if __name__ == "__main__":
    main()
