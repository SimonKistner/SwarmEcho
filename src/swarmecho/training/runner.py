"""
swarmecho/training/runner.py
==============================
Main MAPPO training loop for SwarmEcho.

Auto-reset design
-----------------
Rather than a gymnasium-style wrapper, we integrate auto-reset directly into
the rollout step (_make_autoreset_step).  When an episode terminates (time
limit *or* success), the state is transparently replaced with a fresh reset
state before being returned.  From the buffer's perspective each transition is
stored as (obs, action, reward, done, ...) where done=True marks a boundary.
GAE correctly bootstraps at episode boundaries via the done flag.

Action normalisation
--------------------
The actor always outputs actions in [-1, 1].  Before passing to the physics
engine, the runner scales: physical_action = normalised_action × max_force.
The buffer always stores the normalised [-1, 1] actions, keeping the PPO
log-ratio computation consistent.

Episode tracking
----------------
Inside the rollout we maintain per-env accumulators:
  ep_ret_accum, ep_len_accum, ep_success_accum, ep_found_accum, ep_gap_accum

When done fires for env e we append the completed episode's stats to lists.
After the rollout we report episode metrics using a 100-episode sliding window
deque (completed episodes only).  This avoids the spikes caused by partial-
rollout returns being logged mid-episode.  If the deque is empty (e.g. very
long first episode), we skip the ep_return log for that update.

W&B x-axis
-----------
All wandb.log() calls use step=steps_done (environment timesteps), not the PPO
update index.

Run organisation
-----------------
    outputs/{run_name}_{timestamp}/
         checkpoints/ckpt_{update:06d}/  ← Orbax-owned, left unchanged
         artifacts/
             train/   ← mid-training eval videos, heatmaps, point data, manifests
             eval/    ← manual/checkpoint-scoped evaluation artifacts
"""

from __future__ import annotations

import functools
import gc
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from swarmecho.core.config import compute_action_dim, compute_obs_dim
from swarmecho.env.grid_utils import has_inner_obstacles
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.artifacts import artifact_suffix, train_artifact_root
from swarmecho.training.checkpoints import (
    restore_model_checkpoint,
    save_model_checkpoint,
)
from swarmecho.training.evaluation import (
    collect_video_episode,
    evaluate_parallel,
    release_video_evaluation_trajectory,
)
from swarmecho.training.evaluation_artifacts import write_training_evaluation_artifacts
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.mappo_trainer import MAPPOTrainer
from swarmecho.training.runtime import build_environment_runtime, build_model
from swarmecho.training.video_worker import render_eval_video


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig, run_name: str, run_dir: Path) -> Optional[object]:
    mode = cfg.logging.get("wandb_mode", "disabled")
    if mode == "disabled":
        return None

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    wandb_kwargs = {
        "project": cfg.logging.get("wandb_project", "swarmecho"),
        "entity": cfg.logging.get("wandb_entity", None),
        "name": run_name,
        "mode": mode,
        "dir": str(run_dir),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    wandb_group = cfg.logging.get("wandb_group", None)
    if wandb_group:
        wandb_kwargs["group"] = str(wandb_group)

    # Auto-resume WandB run if resuming training and a previous run directory exists
    if cfg.training.checkpoint_path:
        wandb_dir = run_dir / "wandb"
        if wandb_dir.exists():
            run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
            if run_dirs:
                # Sort by modification time to find the most recent run folder
                run_dirs.sort(key=lambda d: d.stat().st_mtime)
                latest_run_dir = run_dirs[-1]
                parts = latest_run_dir.name.split("-")
                if len(parts) >= 3:
                    run_id = parts[-1]
                    wandb_kwargs["id"] = run_id
                    wandb_kwargs["resume"] = "allow"
                    print(f"  [W&B] Auto-detected previous run ID '{run_id}' from '{latest_run_dir.name}'. Resuming run.")

    import wandb
    run = wandb.init(**wandb_kwargs)
    return run


def _broadcast_eval_metrics_to_remaining_wandb_steps(
    *,
    wandb_run,
    cfg: DictConfig,
    eval_logs: dict[str, float],
    trigger_update: int,
    n_updates: int,
    steps_per_update: int,
    total_timesteps: int,
) -> None:
    """Carry the final early-stop eval point forward for grouped W&B charts.

    Curriculum stages can stop early once an eval crosses the configured
    threshold. Without extra points, W&B grouped averages at later x-axis
    steps only include slower/worse runs that are still training. This helper
    logs the exact threshold-hitting eval metrics at each remaining scheduled
    eval step and at the configured final timestep, without performing more
    training or evaluation.
    """
    if wandb_run is None:
        return
    if not bool(cfg.evaluation.get("eval_broadcast_on_curriculum_early_stop", True)):
        return
    if not eval_logs:
        return

    eval_every = max(1, int(cfg.evaluation.get("eval_freq", 50)))
    eval_offset = int(cfg.evaluation.get("eval_offset", 1))
    future_steps: list[int] = []

    for future_update in range(trigger_update + 1, n_updates + 1):
        is_eval_step = (
            ((future_update - eval_offset) % eval_every == 0)
            and future_update > eval_offset
        )
        if future_update == n_updates:
            is_eval_step = True
        if is_eval_step:
            future_steps.append(min(int(future_update * steps_per_update), int(total_timesteps)))

    # Always include the configured budget endpoint, even when total_timesteps
    # is not exactly divisible by the PPO update size or the final update was
    # already included above.
    if future_steps and future_steps[-1] != int(total_timesteps):
        future_steps.append(int(total_timesteps))
    elif not future_steps and int(trigger_update * steps_per_update) < int(total_timesteps):
        future_steps.append(int(total_timesteps))

    if not future_steps:
        return

    broadcast_logs = {
        **eval_logs,
        "eval/curriculum_early_stop_broadcast": 1.0,
    }
    unique_steps = sorted(set(future_steps))
    for step in unique_steps:
        wandb_run.log(broadcast_logs, step=step)

    print(
        f"  [wandb-carry-forward] broadcast final eval metrics to "
        f"{len(unique_steps)} future W&B step(s), through step {unique_steps[-1]:,}."
    )


# ---------------------------------------------------------------------------
# Auto-reset step factory
# ---------------------------------------------------------------------------

def _make_autoreset_step(env_step_fn, reset_fn, reward_fn, max_steps: int, hold_chain_for: int = 0, terminate_on_target_found: bool = False):
    """
    Wrap env_step to auto-reset on episode termination.

    Done conditions (either triggers reset):
      * time_up        : new_state.physics.step >= max_steps
      * fully_connected: the chain is closed and held for `hold_chain_for` timesteps (success)
      * target_found   : target is found/delivered (if terminate_on_target_found is True)

    Returns
    -------
    (next_state, reward, done, info)
      next_state : already reset if done; otherwise new_state
    """
    max_steps_jnp = jnp.int32(max_steps)
    hold_chain_for_jnp = jnp.int32(hold_chain_for)
    terminate_on_target_found_jnp = jnp.bool_(terminate_on_target_found)

    def step(state, actions):
        new_state = env_step_fn(state, actions)

        time_up = new_state.physics.step >= max_steps_jnp

        # Call reward_fn with is_done=False so success bonus is not prematurely added
        reward, info = reward_fn(state, new_state, jnp.bool_(False))

        fully_connected = info["fully_connected"] > jnp.float32(0.5)

        # Track steps held
        new_chain_held_steps = jnp.where(
            fully_connected,
            state.relay.chain_held_steps + jnp.int32(1),
            jnp.int32(0)
        )
        new_state = new_state.replace(
            relay=new_state.relay.replace(
                chain_held_steps=new_chain_held_steps,
            ),
        )

        success_achieved = new_chain_held_steps >= (hold_chain_for_jnp + jnp.int32(1))
        
        target_found = info["global_target_found"] > jnp.float32(0.5)
        done = time_up | success_achieved | (terminate_on_target_found_jnp & target_found)

        _, info_terminal = reward_fn(state, new_state, jnp.bool_(True))
        reward_terminal  = info_terminal["r_success"] / reward.shape[0]  # Get per-agent bonus
        extra_bonus = jnp.where(
            success_achieved,
            reward_terminal,
            jnp.float32(0.0),
        )
        reward = reward + extra_bonus

        reset_state = reset_fn(new_state.physics.key)
        next_state = jax.tree_util.tree_map(
            lambda r, c: jnp.where(done, r, c),
            reset_state, new_state,
        )

        # Override key fields in info dict to reflect hold status and terminal target data
        info = {
            **info,
            "fully_connected": success_achieved.astype(jnp.float32),
            "terminal_target_pos": new_state.physics.target_pos,
            "terminal_delivered": new_state.communication.base_target_known,
            "terminal_visually_found": jnp.any(
                new_state.communication.target_known,
                axis=-1,
            ),
        }

        return next_state, reward, done, info

    return step


# ---------------------------------------------------------------------------
# Batched Rollout Step helper (JIT Compiled)
# ---------------------------------------------------------------------------

def _batched_rollout_and_memory_step_impl(
    model: MAPPOModel,
    obs_batch,
    act_keys,
    actor_h_in,
    actor_signature_in,
    actor_value_in,
    critic_h_in,
    reset_agents_b,
    max_force,
    states_adj_matrix=None,
    states_active=None,
    states_target_known=None,
    base_signature=None,
    base_value=None,
    base_memory_valid=None,
    t=None,
    *,
    recurrent: bool,
    memory_comm_enabled: bool,
):
    E_ = obs_batch.shape[0]
    N = obs_batch.shape[1]

    if recurrent:
        if memory_comm_enabled:
            raw_adj_b = states_adj_matrix[:, :N, :N] if states_adj_matrix.shape[-1] > 0 else jnp.zeros((E_, N, N), dtype=bool)
            step_share = (int(model.memory_comm_every_k_steps) <= 1) | ((t % int(model.memory_comm_every_k_steps)) == jnp.int32(0))
            comm_mask_b = raw_adj_b & step_share
            active_mask_b = states_active
            base_signature_b = base_signature if base_signature is not None else jnp.zeros((E_, model.tarmac_sig_dim), dtype=jnp.float32)
            base_value_b = base_value if base_value is not None else jnp.zeros((E_, model.tarmac_val_dim), dtype=jnp.float32)
            
            base_receiver_mask_b = (
                states_adj_matrix[:, :N, N]
                if states_adj_matrix.shape[-1] > N
                else jnp.zeros((E_, N), dtype=bool)
            )
            base_receiver_mask_b = (
                base_receiver_mask_b
                & ~states_target_known[:, :N]
                & (base_memory_valid[:, None] if base_memory_valid is not None else jnp.zeros((E_, 1), dtype=bool))
                & step_share
            )

            def _rollout_one_env(obs_n, keys_n, actor_h_n, actor_sig_n, actor_val_n, critic_h_n, resets_n, comm_mask_n, active_n, base_sig_n, base_val_n, base_memory_mask_n):
                actor_h_out, actor_sig_out, actor_val_out, critic_h_out, actions_n, log_probs_n, values_n = model.rollout_step_recurrent(
                    obs_n, keys_n, actor_h_n, critic_h_n, resets_n, max_force,
                    actor_signature=actor_sig_n,
                    actor_value=actor_val_n,
                    comm_mask=comm_mask_n,
                    active=active_n,
                    base_signature=base_sig_n,
                    base_value=base_val_n,
                    base_memory_mask=base_memory_mask_n,
                )
                return actor_h_out, actor_sig_out, actor_val_out, critic_h_out, actions_n, log_probs_n, values_n

            actor_h, actor_signature, actor_value, critic_h, actions_b, log_probs_b, values_b = jax.vmap(_rollout_one_env)(
                obs_batch, act_keys, actor_h_in, actor_signature_in, actor_value_in, critic_h_in, reset_agents_b,
                comm_mask_b, active_mask_b, base_signature_b, base_value_b, base_receiver_mask_b,
            )
            
            # Base memory update logic
            connected_to_base_b = states_adj_matrix[:, :N, N] if states_adj_matrix.shape[-1] > N else jnp.zeros((E_, N), dtype=bool)
            reporters_b = states_target_known[:, :N] & connected_to_base_b & states_active[:, :N]
            first_idx_b = jnp.argmax(reporters_b.astype(jnp.int32), axis=-1)
            has_reporter_b = jnp.any(reporters_b, axis=-1)
            reported_signature_b = jnp.take_along_axis(actor_signature, first_idx_b[:, None, None], axis=1).squeeze(axis=1)
            reported_value_b = jnp.take_along_axis(actor_value, first_idx_b[:, None, None], axis=1).squeeze(axis=1)
            should_store_b = has_reporter_b & ~base_memory_valid
            base_signature = jnp.where(should_store_b[:, None], reported_signature_b, base_signature)
            base_value = jnp.where(should_store_b[:, None], reported_value_b, base_value)
            base_memory_valid = base_memory_valid | should_store_b
            
            return (
                actor_h, actor_signature, actor_value, critic_h, actions_b, log_probs_b, values_b, base_signature, base_value, base_memory_valid,
                comm_mask_b, active_mask_b, base_signature_b, base_value_b, base_receiver_mask_b
            )
            
        else:
            def _rollout_one_env(obs_n, keys_n, actor_h_n, critic_h_n, resets_n):
                return model.rollout_step_recurrent(
                    obs_n, keys_n, actor_h_n, critic_h_n, resets=resets_n, max_force=max_force,
                )

            actor_h, _, _, critic_h, actions_b, log_probs_b, values_b = jax.vmap(_rollout_one_env)(
                obs_batch, act_keys, actor_h_in, critic_h_in, reset_agents_b,
            )
            return actor_h, None, None, critic_h, actions_b, log_probs_b, values_b, None, None, None, None, None, None, None, None
    else:
        def _rollout_one_env(obs_n, keys_n):
            actions, log_probs, value = model.rollout_step(obs_n, keys_n, max_force)
            return actions, log_probs, value

        actions_b, log_probs_b, values_b = jax.vmap(_rollout_one_env)(obs_batch, act_keys)
        return None, None, None, None, actions_b, log_probs_b, values_b, None, None, None, None, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Rollout collection — MAPPO
# ---------------------------------------------------------------------------

def _collect_rollout_mappo(
    states,
    model:            MAPPOModel,
    buf:              MAPPORolloutBuffer,
    autoreset_step_v,
    obs_fn_v,
    batched_rollout_step_jit,
    key:              jax.Array,
    max_force:        float,
    T:                int,
    ep_trackers:      dict,
    actor_h = None,
    actor_signature = None,
    actor_value = None,
    critic_h = None,
    last_dones = None,
    base_signature = None,
    base_value = None,
    base_memory_valid = None,
) -> tuple:
    """
    Collect T steps across all envs, storing normalised actions in the buffer.
    Actions sent to the physics engine are scaled by max_force.
    """
    recurrent = bool(model.actor_memory or model.critic_memory)
    if last_dones is None:
        last_dones = np.zeros(buf.E, dtype=bool)
    buf.reset(actor_h, critic_h, actor_signature, actor_value)
    E = buf.E
    if model.actor_memory and model.memory_comm_enabled:
        if actor_signature is None:
            actor_signature = model.initial_actor_signature((E,))
        if actor_value is None:
            actor_value = model.initial_actor_value((E,))
        if base_signature is None:
            base_signature = jnp.zeros((E, model.tarmac_sig_dim), dtype=jnp.float32)
        if base_value is None:
            base_value = jnp.zeros((E, model.tarmac_val_dim), dtype=jnp.float32)
        if base_memory_valid is None:
            base_memory_valid = jnp.zeros((E,), dtype=bool)

    ep_ret_accum     = ep_trackers["ret"]
    ep_len_accum     = ep_trackers["len"]
    ep_success_accum = ep_trackers["success"]
    ep_found_accum   = ep_trackers["found"]
    ep_gap_accum     = ep_trackers["gap"]
    ep_prog_pct_accum = ep_trackers.get("prog_pct", np.zeros(E))
    r_coverage_accum = ep_trackers.get("r_coverage", np.zeros(E))
    r_gap_accum      = ep_trackers.get("r_gap",      np.zeros(E))
    r_coll_accum     = ep_trackers.get("r_coll",     np.zeros(E))
    r_found_accum    = ep_trackers.get("r_found",    np.zeros(E))
    r_succ_accum     = ep_trackers.get("r_succ",     np.zeros(E))
    cov_accum        = ep_trackers.get("coverage",   np.zeros(E))

    completed_returns = []
    completed_lengths  = []
    completed_success  = []
    completed_found    = []
    completed_gaps     = []
    completed_prog_pcts = []
    completed_r_coverage = []
    completed_r_gap      = []
    completed_r_coll     = []
    completed_r_found    = []
    completed_r_succ     = []
    completed_coverage   = []

    # Keep the complete rollout device-resident while its kernels are queued.
    # Converting individual step results to NumPy here would synchronize the
    # host with the accelerator once (or several times) per environment step.
    rollout_device = {
        "obs": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
        "rnn_resets": [],
        "comm_masks": [],
        "active_masks": [],
        "base_signatures": [],
        "base_values": [],
        "base_memory_masks": [],
    }
    info_device = {
        name: [] for name in (
            "fully_connected",
            "global_target_found",
            "chain_gap_dist",
            "chain_progress_pct",
            "r_coverage",
            "r_chain_gap",
            "r_collision",
            "r_target_found",
            "r_success",
            "global_coverage",
        )
    }
    for t in range(T):
        key, act_key = jax.random.split(key)

        obs_batch = obs_fn_v(states)          # (E, N, D)
        E_, N_, D_ = obs_batch.shape
        act_keys = jax.random.split(act_key, E_ * N_).reshape(E_, N_, 2)
        reset_agents_b = (
            jnp.asarray(last_dones)[:, None]
            | jnp.logical_not(states.physics.active)
        )

        if recurrent:
            if model.actor_memory and actor_h is None:
                actor_h = model.initial_actor_hidden((E_,))
            if model.actor_memory and model.memory_comm_enabled and actor_signature is None:
                actor_signature = model.initial_actor_signature((E_,))
            if model.actor_memory and model.memory_comm_enabled and actor_value is None:
                actor_value = model.initial_actor_value((E_,))
            if model.critic_memory and critic_h is None:
                critic_h = model.initial_critic_hidden((E_,))

            actor_h_in = actor_h if actor_h is not None else jnp.zeros((E_, N_, model.hidden_dim), dtype=jnp.float32)
            actor_signature_in = actor_signature if actor_signature is not None else jnp.zeros((E_, N_, model.tarmac_sig_dim), dtype=jnp.float32)
            actor_value_in = actor_value if actor_value is not None else jnp.zeros((E_, N_, model.tarmac_val_dim), dtype=jnp.float32)
            critic_h_in = critic_h if critic_h is not None else jnp.zeros((E_, N_, model.hidden_dim), dtype=jnp.float32)

            if model.actor_memory and model.memory_comm_enabled:
                (actor_h, actor_signature, actor_value, critic_h, actions_b, log_probs_b, values_b, base_signature, base_value, base_memory_valid,
                 comm_mask_b, active_mask_b, base_signature_b, base_value_b, base_receiver_mask_b) = batched_rollout_step_jit(
                    model,
                    obs_batch,
                    act_keys,
                    actor_h_in,
                    actor_signature_in,
                    actor_value_in,
                    critic_h_in,
                    reset_agents_b,
                    max_force,
                    states.communication.adj_matrix,
                    states.physics.active,
                    states.communication.target_known,
                    base_signature,
                    base_value,
                    base_memory_valid,
                    jnp.int32(t),
                )
            else:
                (actor_h, _, _, critic_h, actions_b, log_probs_b, values_b, _, _, _,
                 comm_mask_b, active_mask_b, base_signature_b, base_value_b, base_receiver_mask_b) = batched_rollout_step_jit(
                    model,
                    obs_batch,
                    act_keys,
                    actor_h_in,
                    actor_signature_in,
                    actor_value_in,
                    critic_h_in,
                    reset_agents_b,
                    max_force,
                )

            if not model.actor_memory:
                actor_h = None
            if not model.critic_memory:
                critic_h = None
        else:
            (_, _, _, _, actions_b, log_probs_b, values_b, _, _, _,
             comm_mask_b, active_mask_b, base_signature_b, base_value_b, base_receiver_mask_b) = batched_rollout_step_jit(
                model,
                obs_batch,
                act_keys,
                None,
                None,
                None,
                None,
                None,
                max_force,
            )

        # actions_b: (E, N, A) — PRE-SQUASH samples u from actor.act()
        # Apply tanh squashing before scaling for the physics engine.
        # The buffer stores the raw pre-squash u for consistent PPO log-prob re-evaluation.
        squashed_b = jnp.tanh(actions_b)  # (E, N, A) in (-1, 1)

        # Scale to physical space ONLY for the env step
        states, rewards_b, dones_b, info = autoreset_step_v(states, squashed_b * max_force)

        rollout_device["obs"].append(obs_batch)
        rollout_device["actions"].append(actions_b)
        rollout_device["log_probs"].append(log_probs_b)
        rollout_device["values"].append(values_b)
        rollout_device["rewards"].append(rewards_b)
        rollout_device["dones"].append(dones_b)
        if recurrent:
            rollout_device["rnn_resets"].append(reset_agents_b)
        if recurrent and model.actor_memory and model.memory_comm_enabled:
            rollout_device["comm_masks"].append(comm_mask_b)
            rollout_device["active_masks"].append(active_mask_b)
            rollout_device["base_signatures"].append(base_signature_b)
            rollout_device["base_values"].append(base_value_b)
            rollout_device["base_memory_masks"].append(base_receiver_mask_b)
        for name in info_device:
            info_device[name].append(info[name])

        if recurrent and model.actor_memory and model.memory_comm_enabled:
            done_j = dones_b.astype(bool)
            base_signature = jnp.where(done_j[:, None], jnp.zeros_like(base_signature), base_signature)
            base_value = jnp.where(done_j[:, None], jnp.zeros_like(base_value), base_value)
            base_memory_valid = jnp.where(done_j, False, base_memory_valid)
        last_dones = dones_b

    # One bulk transfer replaces the per-array, per-step NumPy conversions.
    # Episode accounting and the existing CPU rollout buffer retain exactly the
    # same values and ordering as before.
    rollout_host, info_host = jax.device_get((
        {
            name: jnp.stack(values)
            for name, values in rollout_device.items()
            if values
        },
        {name: jnp.stack(values) for name, values in info_device.items()},
    ))

    for t in range(T):
        rewards_np = rollout_host["rewards"][t]
        dones_np = rollout_host["dones"][t].astype(bool)

        ep_ret_accum += rewards_np
        ep_len_accum += 1
        ep_success_accum = np.maximum(ep_success_accum, info_host["fully_connected"][t])
        ep_found_accum = np.maximum(ep_found_accum, info_host["global_target_found"][t])
        ep_gap_accum = info_host["chain_gap_dist"][t]
        ep_prog_pct_accum = info_host["chain_progress_pct"][t]
        r_coverage_accum += info_host["r_coverage"][t]
        r_gap_accum += info_host["r_chain_gap"][t]
        r_coll_accum += info_host["r_collision"][t]
        r_found_accum += info_host["r_target_found"][t]
        r_succ_accum += info_host["r_success"][t]
        cov_accum = info_host["global_coverage"][t]

        for e in np.where(dones_np)[0]:
            completed_returns.append(float(ep_ret_accum[e].sum()))
            completed_lengths.append(int(ep_len_accum[e]))
            completed_success.append(float(ep_success_accum[e]))
            completed_found.append(float(ep_found_accum[e]))
            completed_gaps.append(float(ep_gap_accum[e]))
            completed_prog_pcts.append(float(ep_prog_pct_accum[e]))
            completed_r_coverage.append(float(r_coverage_accum[e]))
            completed_r_gap.append(float(r_gap_accum[e]))
            completed_r_coll.append(float(r_coll_accum[e]))
            completed_r_found.append(float(r_found_accum[e]))
            completed_r_succ.append(float(r_succ_accum[e]))
            completed_coverage.append(float(cov_accum[e]))

        ep_ret_accum = np.where(dones_np[:, None], 0.0, ep_ret_accum)
        ep_len_accum = np.where(dones_np, 0, ep_len_accum)
        ep_success_accum = np.where(dones_np, 0.0, ep_success_accum)
        ep_found_accum = np.where(dones_np, 0.0, ep_found_accum)
        ep_gap_accum = np.where(dones_np, 0.0, ep_gap_accum)
        ep_prog_pct_accum = np.where(dones_np, 0.0, ep_prog_pct_accum)
        r_coverage_accum = np.where(dones_np, 0.0, r_coverage_accum)
        r_gap_accum = np.where(dones_np, 0.0, r_gap_accum)
        r_coll_accum = np.where(dones_np, 0.0, r_coll_accum)
        r_found_accum = np.where(dones_np, 0.0, r_found_accum)
        r_succ_accum = np.where(dones_np, 0.0, r_succ_accum)
        cov_accum = np.where(dones_np, 0.0, cov_accum)

        buf.add(MAPPOTransition(
            obs=rollout_host["obs"][t],
            actions=rollout_host["actions"][t],
            log_probs=rollout_host["log_probs"][t],
            values=rollout_host["values"][t],
            rewards=rewards_np,
            dones=dones_np.astype(np.float32),
            rnn_resets=rollout_host["rnn_resets"][t] if recurrent else None,
            comm_masks=rollout_host["comm_masks"][t] if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            active_masks=rollout_host["active_masks"][t] if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_signatures=rollout_host["base_signatures"][t] if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_values=rollout_host["base_values"][t] if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_memory_masks=rollout_host["base_memory_masks"][t] if recurrent and model.actor_memory and model.memory_comm_enabled else None,
        ))

    # Bootstrap value for last state
    last_obs    = obs_fn_v(states)
    if recurrent and model.critic_memory:
        reset_agents_b = (
            jnp.asarray(last_dones)[:, None]
            | jnp.logical_not(states.physics.active)
        )

        def _value_one_env(obs_n, h_n, resets_n):
            _, value = model.get_value_recurrent(obs_n, h_n, resets_n)
            return value

        last_values = jax.vmap(_value_one_env)(last_obs, critic_h, reset_agents_b)
    else:
        last_values = model.get_value(last_obs)    # (E,) or (E, N)
    bootstrap_dones = jnp.zeros(E, dtype=jnp.float32)

    # Save persistent accumulators back
    ep_trackers["ret"]     = ep_ret_accum
    ep_trackers["len"]     = ep_len_accum
    ep_trackers["success"] = ep_success_accum
    ep_trackers["found"]   = ep_found_accum
    ep_trackers["gap"]     = ep_gap_accum
    ep_trackers["prog_pct"] = ep_prog_pct_accum

    ep_trackers["r_coverage"] = r_coverage_accum
    ep_trackers["r_gap"]      = r_gap_accum
    ep_trackers["r_coll"]     = r_coll_accum
    ep_trackers["r_found"]    = r_found_accum
    ep_trackers["r_succ"]     = r_succ_accum
    ep_trackers["coverage"]   = cov_accum


    comm_summary = {}


    return (
        states, key, last_values, bootstrap_dones, actor_h, actor_signature, actor_value, critic_h, np.asarray(last_dones, dtype=bool), base_signature, base_value, base_memory_valid,
        comm_summary,
        completed_returns, completed_lengths, completed_success,
        completed_found, completed_gaps, completed_prog_pcts,
        completed_r_coverage, completed_r_gap, completed_r_coll,
        completed_r_found, completed_r_succ,
        completed_coverage,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: DictConfig):
    """
    Main MAPPO training loop.

    Parameters
    ----------
    cfg : DictConfig
        Fully-merged configuration.
    """
    # ── Logging config ────────────────────────────────────────────────────
    if cfg.logging.get("suppress_xla_warnings", True):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NVIDIA_TF32_OVERRIDE"]  = "0"

    actor_memory = bool(cfg.network.get("actor_memory", False))
    critic_memory = bool(cfg.network.get("critic_memory", False))
    recurrent   = actor_memory or critic_memory

    print("\n══════════════════════════════════════════════════════")
    print("  SwarmEcho — MAPPO  [agent-centric critic]")
    print("══════════════════════════════════════════════════════")

    N         = int(cfg.env.num_agents)
    E         = int(cfg.training.num_envs)
    T         = int(cfg.training.num_steps)
    MB        = int(cfg.training.num_minibatches)
    max_steps = int(cfg.env.max_steps)
    total_ts  = int(cfg.training.total_timesteps)
    n_updates = total_ts // (E * T)
    obs_dim   = compute_obs_dim(cfg)
    act_dim   = compute_action_dim(cfg)
    max_force = float(cfg.env.max_force)

    transitions_per_mb = (E * T) // MB

    print(f"  Devices          : {jax.devices()}")
    print()
    print(f"  num_agents N     : {N}")
    print(f"  obs_dim          : {obs_dim}")
    print(f"  max_force        : {max_force}")
    print()
    print(f"  max_steps        : {max_steps}")
    print(f"  num_envs E       : {E}")
    print(f"  rollout T        : {T}")
    print(f"  num_minibatches  : {MB}")
    print(f"  transitions/mb   : {transitions_per_mb:,}")
    print()
    print(f"  actor_layers     : {cfg.network.actor_num_layers}")
    print(f"  actor_memory     : {actor_memory}")
    print(f"  critic_layers    : {cfg.network.num_layers}")
    print(f"  critic_memory    : {critic_memory}")
    print()
    print(f"  ppo updates      : {n_updates:,}  ({total_ts:,} total timesteps)")
    print()

    # ── Environment ───────────────────────────────────────────────────────
    environment = build_environment_runtime(cfg)
    env_step = environment.env_step
    reset = environment.reset
    compute_obs = environment.compute_obs
    compute_reward = environment.compute_reward
    resolved_W = environment.width
    resolved_H = environment.height
    occ_grid = environment.occupancy_grid
    comm_has_inner_obstacles = has_inner_obstacles(occ_grid)
    if comm_has_inner_obstacles:
        print("  [comm-los] inner communication obstacles detected; using wall-aware raycasts")
    else:
        print("  [comm-los] no inner communication obstacles detected; skipping communication LOS raycasts")

    hold_chain_for   = int(cfg.env.get("hold_chain_for", 0))
    terminate_on_target_found = bool(cfg.env.get("terminate_on_target_found", False))
    autoreset_step   = _make_autoreset_step(env_step, reset, compute_reward, max_steps, hold_chain_for, terminate_on_target_found)
    autoreset_step_v = jax.jit(jax.vmap(autoreset_step))
    obs_fn_v         = jax.jit(jax.vmap(compute_obs))
    reset_v          = jax.jit(jax.vmap(reset))
    reset_s          = jax.jit(reset)
    env_step_jit       = jax.jit(env_step)
    compute_obs_jit    = jax.jit(compute_obs)
    compute_reward_jit = jax.jit(compute_reward)
    # ── Model ─────────────────────────────────────────────────────────────
    master_key = jax.random.PRNGKey(int(cfg.training.seed))
    model_key, env_key, master_key = jax.random.split(master_key, 3)
    model = build_model(cfg, rng_seed=int(model_key[0]))
    trainer = MAPPOTrainer(
        model         = model,
        lr            = float(cfg.training.lr),
        max_grad_norm = float(cfg.training.max_grad_norm),
        clip_eps      = float(cfg.training.clip_eps),
        vf_coef       = float(cfg.training.vf_coef),
        ent_coef      = float(cfg.training.ent_coef),
        num_epochs    = int(cfg.training.num_epochs),
        actor_memory  = actor_memory,
        critic_memory = critic_memory,
    )
    buf = MAPPORolloutBuffer(
        num_steps  = T,
        num_envs   = E,
        num_agents = N,
        obs_dim    = obs_dim,
        act_dim    = act_dim,
        gamma      = float(cfg.training.gamma),
        gae_lambda = float(cfg.training.gae_lambda),
        recurrent  = actor_memory or critic_memory,
        hidden_dim = int(cfg.network.hidden_dim),
        tarmac_sig_dim = int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim = int(cfg.network.get("tarmac_val_dim", 128)),
        actor_memory  = actor_memory,
        critic_memory = critic_memory,
    )

    _, params = nnx.split(model)
    n_params  = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model params : {n_params:,}")

    # ── Checkpoint Resumption ─────────────────────────────────────────────
    if cfg.training.checkpoint_path:
        checkpoint_path = Path(str(cfg.training.checkpoint_path).replace("\\", "/")).absolute()
        print(f"  Resuming from: {checkpoint_path}")
        restore_model_checkpoint(model, checkpoint_path)
        print("  Weights restored ✓")

    # ── Environments init ─────────────────────────────────────────────────
    env_keys = jax.random.split(env_key, E)
    states   = reset_v(env_keys)
    print("  Environments initialised ✓")

    use_memory_comm = actor_memory and bool(cfg.network.get("memory_comm_enabled", False))
    actor_h = model.initial_actor_hidden((E,)) if actor_memory else None
    actor_signature = model.initial_actor_signature((E,)) if use_memory_comm else None
    actor_value = model.initial_actor_value((E,)) if use_memory_comm else None
    critic_h = model.initial_critic_hidden((E,)) if critic_memory else None
    base_signature = jnp.zeros((E, int(cfg.network.get("tarmac_sig_dim", 64))), dtype=jnp.float32) if use_memory_comm else None
    base_value = jnp.zeros((E, int(cfg.network.get("tarmac_val_dim", 128))), dtype=jnp.float32) if use_memory_comm else None
    base_memory_valid = jnp.zeros((E,), dtype=bool) if use_memory_comm else None
    rollout_last_dones = np.zeros(E, dtype=bool)

    batched_rollout_step_fn = functools.partial(
        _batched_rollout_and_memory_step_impl,
        recurrent=recurrent,
        memory_comm_enabled=use_memory_comm,
    )
    batched_rollout_step_jit = nnx.jit(batched_rollout_step_fn)

    # ── Run directory & Name ──────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_run_name = cfg.logging.get("run_name", None)
    use_ts = cfg.logging.get("use_timestamp_postfix", False)

    if not cfg_run_name:
        run_name = f"run_{ts}"
    else:
        run_name = f"{cfg_run_name}_{ts}" if use_ts else cfg_run_name

    log_root = Path(cfg.logging.get("log_dir", "outputs")).absolute()
    run_dir  = log_root / run_name
    configured_ckpt_dir = cfg.evaluation.get("checkpoint_dir", None)
    ckpt_dir = (
        Path(str(configured_ckpt_dir).replace("\\", "/")).absolute()
        if configured_ckpt_dir
        else run_dir / "checkpoints"
    )
    train_artifacts = train_artifact_root(run_dir)
    train_video_dir = train_artifacts / "vids"
    train_data_dir = train_artifacts / "data"
    train_manifest_dir = train_artifacts / "manifests"
    eval_video_dir = run_dir / "artifacts" / "eval" / "early" / "vids"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Update config with resolved world dimensions for the snapshot
    OmegaConf.set_readonly(cfg, False)
    cfg.env.box_width  = float(resolved_W)
    cfg.env.box_height = float(resolved_H)

    OmegaConf.set_readonly(cfg, True)

    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"  Run dir      : {run_dir}")

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_run = _init_wandb(cfg, run_name, run_dir)
    if wandb_run:
        print(f"  W&B run      : {wandb_run.url}")

    # ── Renderer config ───────────────────────────────────────────────────
    eval_video   = bool(cfg.evaluation.get("eval_video", True))
    save_model   = bool(cfg.evaluation.get("save_model", True))
    is_benchmark = bool(cfg.logging.get("benchmark_mode", False))

    if not save_model:
        print("  [ckpt] Scheduled and final checkpoint saving disabled")
    if not eval_video:
        print("  [render] Eval video rendering disabled")
    else:
        print("  [render] OpenCV mode | one episode per video step")

    # ── Training loop ─────────────────────────────────────────────────────
    eval_every  = int(cfg.evaluation.get("eval_freq", 50))
    eval_offset = int(cfg.evaluation.get("eval_offset", 0) or 0)
    eval_video_every = int(cfg.evaluation.get("eval_video_freq", eval_every))
    eval_video_offset = int(cfg.evaluation.get("eval_video_offset", 0) or 0)

    checkpoint_freq = int(cfg.evaluation.get("checkpoint_freq", 50))
    checkpoint_offset = int(cfg.evaluation.get("checkpoint_offset", 0) or 0)
    ckpt_every = checkpoint_freq
    ckpt_offset = checkpoint_offset
    if save_model:
        print(f"  [ckpt] Saving checkpoints every {ckpt_every} updates (offset: {ckpt_offset})")

    log_every   = max(1, n_updates // 200)

    # ── Persistent Accumulators & Windows ────────────────────────────────
    ep_trackers = {
        "ret":     np.zeros((E, N), dtype=np.float32),
        "len":     np.zeros(E, dtype=np.int32),
        "success": np.zeros(E, dtype=np.float32),
        "found":   np.zeros(E, dtype=np.float32),
        "gap":     np.zeros(E, dtype=np.float32),
    }

    # Sliding window for stable logging metrics
    window_ret  = deque(maxlen=E)
    window_len  = deque(maxlen=E)
    window_succ = deque(maxlen=E)
    window_fnd  = deque(maxlen=E)
    window_gap  = deque(maxlen=E)
    window_prog_pct = deque(maxlen=E)

    window_r_cov   = deque(maxlen=E)
    window_r_gap   = deque(maxlen=E)
    window_r_coll  = deque(maxlen=E)
    window_r_found = deque(maxlen=E)
    window_r_succ  = deque(maxlen=E)
    window_cov     = deque(maxlen=E)

    completed_eps_count = 0

    start_update = 0
    loading_mode = cfg.training.get("ckpt_loading_mode", "branch").lower()

    loaded_history = []
    total_checkpoint_steps = 0
    prior_history = []
    prior_offset = 0

    if cfg.training.checkpoint_path:
        checkpoint_path = Path(str(cfg.training.checkpoint_path).replace("\\", "/")).absolute()
        
        # Determine start_update from checkpoint name if resuming
        try:
            import re
            path_name = checkpoint_path.name
            match = re.search(r'ckpt_(?:early_|final_)?(\d+)', path_name, re.IGNORECASE)
            if match:
                if loading_mode == "resume":
                    start_update = int(match.group(1))
                    print(f"  [resumption] Resuming from update {start_update} (step {start_update * E * T:,})")
                else:
                    print(f"  [checkpoint-load] Loaded weights from {path_name}, but training will start at update 0 (loading_mode=branch)")
        except Exception as e:
            print(f"  [resumption] Failed to parse starting update from checkpoint path: {e}")

        # Load history
        history_file = checkpoint_path / "step_history.json"
        if history_file.exists():
            try:
                import json
                with open(history_file, "r") as f:
                    hist_data = json.load(f)
                loaded_history = hist_data.get("history", [])
                total_checkpoint_steps = hist_data.get("total_steps", 0)
                print(f"  [checkpoint-history] Loaded history from {history_file.name}. Total steps: {total_checkpoint_steps:,}")
            except Exception as e:
                print(f"  [checkpoint-history] Failed to load step_history.json: {e}")
                
        if not loaded_history or total_checkpoint_steps == 0:
            # Fallback to name parsing if no history file exists
            try:
                import re
                path_name = checkpoint_path.name
                match = re.search(r'ckpt_(?:early_|final_)?(\d+)', path_name, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    total_checkpoint_steps = val * E * T
                elif path_name.isdigit():
                    val = int(path_name)
                    total_checkpoint_steps = val * E * T if val < 100000 else val
                
                if total_checkpoint_steps > 0:
                    parent_run_name = checkpoint_path.parents[1].name
                    loaded_history = [{"run_name": parent_run_name, "steps": total_checkpoint_steps}]
                    print(f"  [checkpoint-history] Reconstructed history from checkpoint path '{path_name}'. Total steps: {total_checkpoint_steps:,}")
            except Exception as e:
                print(f"  [checkpoint-history] Failed to reconstruct history: {e}")

        # Determine prior_history and prior_offset
        if loading_mode == "resume" and start_update > 0:
            if loaded_history and loaded_history[-1]["run_name"] == run_name:
                prior_history = loaded_history[:-1]
            else:
                prior_history = loaded_history[:-1] if len(loaded_history) > 0 else []
            prior_offset = max(0, total_checkpoint_steps - start_update * E * T)
        else:
            prior_history = loaded_history
            prior_offset = total_checkpoint_steps

    # Manual offset override
    manual_offset = cfg.training.get("checkpoint_step_offset", None)
    if manual_offset is not None:
        step_offset = int(manual_offset)
        print(f"  [ckpt] Using manual checkpoint_step_offset override: {step_offset:,}")
    else:
        step_offset = prior_offset
        print(f"  [ckpt] Auto-detected cumulative step offset: {step_offset:,}")

    t_start = time.perf_counter()

    try:
        for update in range(start_update + 1, n_updates + 1):
            master_key, collect_key = jax.random.split(master_key)
            steps_done = update * E * T + step_offset

            # ── Rollout ───────────────────────────────────────────────────────
            (states, collect_key,
             last_values, last_dones, actor_h, actor_signature, actor_value, critic_h, rollout_last_dones, base_signature, base_value, base_memory_valid,
             comm_summary,
             raw_ret, raw_len, raw_success, raw_found, raw_gap, raw_prog_pct,
             raw_r_cov, raw_r_gap, raw_r_coll, raw_r_found, raw_r_succ,
             raw_cov) = _collect_rollout_mappo(
                states, model, buf, autoreset_step_v, obs_fn_v, batched_rollout_step_jit,
                collect_key, max_force, T, ep_trackers,
                actor_h, actor_signature, actor_value, critic_h, rollout_last_dones, base_signature, base_value, base_memory_valid,
            )

            # ── Update sliding window from COMPLETED episodes only ───────────────
            n_eps = len(raw_ret)
            if n_eps > 0:
                completed_eps_count += n_eps
                window_ret.extend(raw_ret)
                window_len.extend(raw_len)
                window_succ.extend(raw_success)
                window_fnd.extend(raw_found)
                window_gap.extend(raw_gap)
                window_prog_pct.extend(raw_prog_pct)

                window_r_cov.extend(raw_r_cov)
                window_r_gap.extend(raw_r_gap)
                window_r_coll.extend(raw_r_coll)
                window_r_found.extend(raw_r_found)
                window_r_succ.extend(raw_r_succ)
                window_cov.extend(raw_cov)

            # ── GAE + minibatches ─────────────────────────────────────────────
            advs, rets = buf.compute_gae(last_values, last_dones)
            mbs        = buf.get_minibatches(advs, rets, int(cfg.training.num_minibatches), collect_key)

            # ── PPO update ────────────────────────────────────────────────────
            ppo_stats = trainer.update(mbs)

            elapsed = time.perf_counter() - t_start
            sps     = ((update - start_update) * E * T) / max(1e-6, elapsed)

            # ── Stdout ───────────────────────────────────────────────────────
            if update % log_every == 0 or update == start_update + 1:
                window_full = len(window_ret) == window_ret.maxlen
                _cov = f"{np.mean(window_cov):>5.1%}" if window_full else " ----"
                _l = f"{np.mean(window_len):>6.0f}" if window_full else "  ----"
                _g = f"{np.mean(window_gap):>6.1f}" if window_full else "  ----"
                _prog = f"{np.mean(window_prog_pct):>5.1f}%" if window_full else "  ---%"
                _red  = f"{(1.0 - (np.mean(window_len) / max_steps)) * 100.0:>5.1f}%" if window_full else "  ---%"
                _s = f"{np.mean(window_succ):>5.1%}" if window_full else " ----"
                _f = f"{np.mean(window_fnd):>5.1%}"  if window_full else " ----"

                now_str = datetime.now().strftime("%H:%M:%S")

                # Calculate ETA
                time_per_update = elapsed / max(1, update - start_update)
                eta_sec = int(time_per_update * (n_updates - update))
                eta_m, eta_s = divmod(eta_sec, 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m{eta_s:02d}s"

                print(
                    f"[{now_str}]"
                    f"  [{update:>4}/{n_updates}]  "
                    f"steps={steps_done:>12,}  "
                    f"sps={sps:>6,.0f}  "
                    f"ep_len={_l}  "
                    f"cov={_cov}  "
                    f"found={_f}  "
                    f"chain={_prog}  "
                    f"succ={_s}  "
                    f"eta={eta_str}"
                )

                if not window_full:
                    current_steps = np.array(states.physics.step)
                    min_step = int(current_steps.min())
                    max_step = int(current_steps.max())
                    mean_step = float(current_steps.mean())
                    print(
                        f"         [warmup] completed_episodes={len(window_ret)}/{window_ret.maxlen} | "
                        f"env_steps: min={min_step} mean={mean_step:.1f} max={max_step}"
                    )

            # ── W&B logging ──────────────────────────────────────────────────
            if wandb_run:
                import wandb
                logs = {
                    "ppo/policy_loss":          ppo_stats["policy_loss"],
                    "ppo/value_loss":           ppo_stats["value_loss"],
                    "ppo/entropy":              ppo_stats["entropy"],
                    "ppo/approx_kl":            ppo_stats["approx_kl"],
                    "ppo/clip_fraction":        ppo_stats["clip_fraction"],
                    "perf/sps":                 sps,
                    "perf/ppo_updates":         update,
                    "perf/global_step":         steps_done,
                }
                if len(window_ret) == window_ret.maxlen:
                    logs.update({
                        "train/ep_return":          float(np.mean(window_ret)),
                        "train/ep_length":          float(np.mean(window_len)),
                        "train/success_rate":       float(np.mean(window_succ)),
                        "train/target_found_rate":  float(np.mean(window_fnd)),
                        "train/chain_progress_pct": float(np.mean(window_prog_pct)),
                        "train/ep_length_reduction": (1.0 - (float(np.mean(window_len)) / max_steps)) * 100.0,
                        "train/map_coverage_pct":   float(np.mean(window_cov)) * 100.0,
                        "train/episodes_completed": completed_eps_count,

                        "rewards/exploration":      float(np.mean(window_r_cov)),
                        "rewards/chain_gap":        float(np.mean(window_r_gap)),
                        "rewards/collision":        float(np.mean(window_r_coll)),
                        "rewards/target_found":     float(np.mean(window_r_found)),
                        "rewards/success_bonus":    float(np.mean(window_r_succ)),
                    })
                logs.update(comm_summary)
                wandb.log(logs, step=steps_done)

            # Calculate rolling training success rate and check threshold (always allow evaluation on final update)
            train_succ_rate = np.mean(window_succ) if len(window_succ) > 0 else 0.0
            eval_min_succ = float(cfg.evaluation.get("eval_min_train_success", 0.0))
            succ_threshold_met = (train_succ_rate >= eval_min_succ) or (update == n_updates)

            is_eval_step = ((update - eval_offset) % eval_every == 0) and update > eval_offset and succ_threshold_met
            is_video_step = (eval_video and ((update - eval_video_offset) % eval_video_every == 0) and update > eval_video_offset) and succ_threshold_met
            if update == n_updates:
                # Run the same train-eval artifact path one final time instead of
                # using a separate special final-eval renderer/output folder.
                is_eval_step = True
                is_video_step = bool(eval_video)

            # ── Training eval + optional single video ─────────────────────────
            if is_eval_step or is_video_step:
                if is_eval_step or is_video_step:
                    if is_eval_step:
                        master_key, eval_key = jax.random.split(master_key)
                        eval_result = evaluate_parallel(
                            model, reset, env_step, compute_obs, compute_reward,
                            cfg, eval_key, num_envs=int(cfg.evaluation.eval_parallel_envs),
                        )
                        eval_ret = float(jnp.mean(eval_result.returns))
                        eval_len = float(jnp.mean(eval_result.lengths))
                        eval_gap = float(jnp.mean(eval_result.chain_gaps))
                        eval_prog_pct = float(jnp.mean(eval_result.chain_progress))
                        eval_success = float(jnp.mean(eval_result.successes))
                        eval_found = float(jnp.mean(eval_result.target_found))
                        eval_cov = float(jnp.mean(
                            eval_result.final_state.exploration.coverage_grid.astype(
                                jnp.float32
                            )
                        ))

                        eval_prefix = "[EVAL]" + " " * 50
                        _l_eval = f"{eval_len:>6.0f}"
                        _cov_eval = f"{eval_cov:>5.1%}"
                        _f_eval = f"{eval_found:>5.1%}"
                        _prog_eval = f"{eval_prog_pct:>5.1f}%"
                        _s_eval = f"{eval_success:>5.1%}"

                        num_envs = int(cfg.evaluation.eval_parallel_envs)
                        if num_envs >= 1000:
                            if num_envs % 1000 == 0:
                                par_envs_val = f"{num_envs // 1000}k"
                            else:
                                par_envs_val = f"{num_envs / 1000:.1f}k"
                        else:
                            par_envs_val = str(num_envs)

                        print(
                            f"{eval_prefix}"
                            f"ep_len={_l_eval}  "
                            f"cov={_cov_eval}  "
                            f"found={_f_eval}  "
                            f"chain={_prog_eval}  "
                            f"succ={_s_eval}  "
                            f"par_envs={par_envs_val}"
                        )

                        eval_wandb_logs = {
                            "eval/ep_return":           eval_ret,
                            "eval/ep_length":           eval_len,
                            "eval/chain_progress_pct":  eval_prog_pct,
                            "eval/ep_length_reduction": (1.0 - (eval_len / max_steps)) * 100.0,
                            "eval/success_rate":        eval_success,
                            "eval/target_found_rate":   eval_found,
                            "eval/map_coverage_pct":    eval_cov * 100.0,
                        }
                        if wandb_run:
                            import wandb
                            wandb.log(eval_wandb_logs, step=steps_done)

                        write_training_evaluation_artifacts(
                            eval_result,
                            cfg,
                            run_dir=run_dir,
                            data_dir=train_data_dir,
                            manifest_dir=train_manifest_dir,
                            artifact_root=train_artifacts,
                            update=update,
                            steps_done=steps_done,
                        )
                        # ── Free GPU memory from parallel eval ──────────────
                        # The vmapped eval returns full EnvState for all envs
                        # plus the JIT-cached XLA executable.  Both consume
                        # significant GPU memory that the PPO backward pass
                        # needs.  Delete result arrays immediately and clear
                        # the XLA cache to prevent late OOM.
                        del eval_result
                        gc.collect()
                        jax.clear_caches()
                        jax.block_until_ready(jnp.asarray(0, dtype=jnp.int32))
                        time.sleep(1.0)

                        early_exit_thresh = float(cfg.evaluation.get("early_exit_threshold", 0.95))
                        if bool(cfg.evaluation.get("early_exit", False)) and eval_success >= early_exit_thresh:
                            print(
                                f"\n  [eval-early-exit] Evaluation success {eval_success:.1%} >= "
                                f"threshold {early_exit_thresh:.1%} -- concluding training early."
                            )
                            early_ckpt = (ckpt_dir / f"ckpt_early_{update:06d}").absolute()
                            save_model_checkpoint(
                                model,
                                early_ckpt,
                                run_name=run_name,
                                update=update,
                                num_envs=E,
                                num_steps=T,
                                prior_history=prior_history,
                            )
                            print(f"  [ckpt-early] saved -> {early_ckpt}")

                            if eval_video:
                                master_key, video_key = jax.random.split(master_key)
                                ep_states_list, ep_rewards_list, all_metrics_list = collect_video_episode(
                                    model, reset_s, env_step_jit, compute_obs_jit,
                                    compute_reward_jit, cfg, video_key,
                                )
                                render_eval_video(
                                    ep_states=ep_states_list[0],
                                    ep_rewards=ep_rewards_list[0],
                                    ep_metrics=all_metrics_list[0],
                                    cfg=cfg,
                                    out_dir=eval_video_dir,
                                    filename_stem=f"eval_early_{update:06d}",
                                )
                                del ep_states_list, ep_rewards_list, all_metrics_list
                                release_video_evaluation_trajectory()
                                time.sleep(1.0)
                            _broadcast_eval_metrics_to_remaining_wandb_steps(
                                wandb_run=wandb_run,
                                cfg=cfg,
                                eval_logs=eval_wandb_logs,
                                trigger_update=update,
                                n_updates=n_updates,
                                steps_per_update=E * T,
                                total_timesteps=total_ts,
                            )
                            return str(early_ckpt)

                    if is_video_step:
                        master_key, video_key = jax.random.split(master_key)
                        ep_states_list, ep_rewards_list, all_metrics_list = collect_video_episode(
                            model, reset_s,
                            env_step_jit, compute_obs_jit, compute_reward_jit,
                            cfg, video_key,
                        )
                        stem = f"eval_{artifact_suffix(update, steps_done)}"
                        render_eval_video(
                            ep_states  = ep_states_list[0],
                            ep_rewards = ep_rewards_list[0],
                            ep_metrics = all_metrics_list[0],
                            cfg        = cfg,
                            out_dir    = train_video_dir,
                            filename_stem = stem,
                        )
                        del ep_states_list, ep_rewards_list, all_metrics_list
                        release_video_evaluation_trajectory()
                        time.sleep(1.0)
                # Small sleep to allow XLA to settle after the eval/render spike
                time.sleep(1.0)

            # Final training progress is now represented by the normal train eval
            # artifacts. The special final-eval render path is intentionally
            # removed so the last output uses the same artifact contract as every
            # other scheduled training eval.

            # ── Checkpoint ────────────────────────────────────────────────────
            is_ckpt_step = ((update - ckpt_offset) % ckpt_every == 0)
            if save_model and not is_benchmark and is_ckpt_step:
                ckpt_path = (ckpt_dir / f"ckpt_{update:06d}").absolute()
                save_model_checkpoint(
                    model,
                    ckpt_path,
                    run_name=run_name,
                    update=update,
                    num_envs=E,
                    num_steps=T,
                    prior_history=prior_history,
                )
                print(f"  [ckpt] saved -> {ckpt_path}")

        total_time = time.perf_counter() - t_start
        print(f"\n  Training complete in {total_time:.1f}s  ({total_time/60:.1f} min)")

        # ── Final Checkpoint ─────────────────────────────────────────────────
        if save_model and not is_benchmark:
            final_ckpt = (ckpt_dir / f"ckpt_{n_updates:06d}").absolute()
            save_model_checkpoint(
                model,
                final_ckpt,
                run_name=run_name,
                update=n_updates,
                num_envs=E,
                num_steps=T,
                prior_history=prior_history,
            )
            print(f"  [ckpt-final] saved -> {final_ckpt}")
            return str(final_ckpt)

        return ""

    finally:
        if wandb_run:
            import wandb
            wandb.finish()
