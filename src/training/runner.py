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
         checkpoints/ckpt_{update:06d}/
         videos/
             train/   ← mid-training eval videos and heatmaps (every eval_freq updates)
             eval/    ← final-eval videos (end of training or early exit)
"""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from core.config import compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from training.mappo_trainer import MAPPOTrainer
from training.video_worker import render_eval_video
from visualize.renderer import render_video


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig, run_name: str, run_dir: Path) -> Optional[object]:
    mode = cfg.logging.get("wandb_mode", "disabled")
    if mode == "disabled":
        return None

    env_path = Path(__file__).resolve().parents[2] / ".env"
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

    import wandb
    run = wandb.init(**wandb_kwargs)
    return run


# ---------------------------------------------------------------------------
# Auto-reset step factory
# ---------------------------------------------------------------------------

def _make_autoreset_step(env_step_fn, reset_fn, reward_fn, max_steps: int, hold_chain_for: int = 0):
    """
    Wrap env_step to auto-reset on episode termination.

    Done conditions (either triggers reset):
      * time_up        : new_state.step >= max_steps
      * fully_connected: the chain is closed and held for `hold_chain_for` timesteps (success)

    Returns
    -------
    (next_state, reward, done, info)
      next_state : already reset if done; otherwise new_state
    """
    max_steps_jnp = jnp.int32(max_steps)
    hold_chain_for_jnp = jnp.int32(hold_chain_for)

    def step(state, actions):
        new_state = env_step_fn(state, actions)

        time_up = new_state.step >= max_steps_jnp

        # Call reward_fn with is_done=False so success bonus is not prematurely added
        reward, info = reward_fn(state, new_state, jnp.bool_(False))

        fully_connected = info["fully_connected"] > jnp.float32(0.5)

        # Track steps held
        new_chain_held_steps = jnp.where(
            fully_connected,
            state.chain_held_steps + jnp.int32(1),
            jnp.int32(0)
        )
        new_state = dataclasses.replace(new_state, chain_held_steps=new_chain_held_steps)

        success_achieved = new_chain_held_steps >= (hold_chain_for_jnp + jnp.int32(1))
        done = time_up | success_achieved

        _, info_terminal = reward_fn(state, new_state, jnp.bool_(True))
        reward_terminal  = info_terminal["r_success"] / reward.shape[0]  # Get per-agent bonus
        extra_bonus = jnp.where(
            success_achieved,
            reward_terminal,
            jnp.float32(0.0),
        )
        reward = reward + extra_bonus

        reset_state = reset_fn(new_state.key)
        next_state = jax.tree_util.tree_map(
            lambda r, c: jnp.where(done, r, c),
            reset_state, new_state,
        )

        # Override key fields in info dict to reflect hold status and terminal target data
        info = {
            **info,
            "fully_connected": success_achieved.astype(jnp.float32),
            "terminal_target_pos": new_state.target_pos,
            "terminal_delivered": new_state.base_target_known,
            "terminal_visually_found": jnp.any(new_state.target_known, axis=-1)
        }

        return next_state, reward, done, info

    return step


# ---------------------------------------------------------------------------
# Rollout collection — MAPPO
# ---------------------------------------------------------------------------

def _collect_rollout_mappo(
    states,
    model:            MAPPOModel,
    buf:              MAPPORolloutBuffer,
    autoreset_step_v,
    obs_fn_v,
    key:              jax.Array,
    max_force:        float,
    T:                int,
    ep_trackers:      dict,
    actor_h = None,
    critic_h = None,
    last_dones = None,
    base_memory = None,
    base_memory_valid = None,
    track_heatmap_data: bool = False,
) -> tuple:
    """
    Collect T steps across all envs, storing normalised actions in the buffer.
    Actions sent to the physics engine are scaled by max_force.
    """
    recurrent = bool(model.actor_memory or model.critic_memory)
    if last_dones is None:
        last_dones = np.zeros(buf.E, dtype=bool)
    buf.reset(actor_h, critic_h)
    E, N = buf.E, buf.N
    if model.actor_memory and model.memory_comm_enabled:
        if base_memory is None:
            base_memory = jnp.zeros((E, model.hidden_dim), dtype=jnp.float32)
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
    r_prox_accum     = ep_trackers.get("r_prox",     np.zeros(E))
    r_found_accum    = ep_trackers.get("r_found",    np.zeros(E))
    r_succ_accum     = ep_trackers.get("r_succ",     np.zeros(E))
    cov_accum        = ep_trackers.get("coverage",   np.zeros(E))

    completed_returns = []
    completed_lengths  = []
    completed_success  = []
    completed_found    = []
    completed_gaps     = []
    completed_prog_pcts = []
    completed_target_pos = []
    completed_target_success = []
    completed_target_delivered = []
    completed_target_visually_found = []

    completed_r_coverage = []
    completed_r_gap      = []
    completed_r_coll     = []
    completed_r_prox     = []
    completed_r_found    = []
    completed_r_succ     = []
    completed_coverage   = []

    for t in range(T):
        key, act_key = jax.random.split(key)

        obs_batch = obs_fn_v(states)          # (E, N, D)
        E_, N_, D_ = obs_batch.shape
        act_keys = jax.random.split(act_key, E_ * N_).reshape(E_, N_, 2)
        reset_agents_b = jnp.asarray(last_dones)[:, None] | jnp.logical_not(states.active)

        if recurrent:
            if model.actor_memory and actor_h is None:
                actor_h = model.initial_actor_hidden((E_,))
            if model.critic_memory and critic_h is None:
                critic_h = model.initial_critic_hidden((E_,))

            actor_h_in = actor_h if actor_h is not None else jnp.zeros((E_, N_, model.hidden_dim), dtype=jnp.float32)
            critic_h_in = critic_h if critic_h is not None else jnp.zeros((E_, N_, model.hidden_dim), dtype=jnp.float32)

            if model.actor_memory and model.memory_comm_enabled:
                def _rollout_one_env(obs_n, keys_n, actor_h_n, critic_h_n, resets_n, comm_mask_n, active_n, base_memory_n, base_memory_mask_n):
                    return model.rollout_step_recurrent(
                        obs_n, keys_n, actor_h_n, critic_h_n, resets_n, max_force,
                        comm_mask=comm_mask_n,
                        active=active_n,
                        base_memory=base_memory_n,
                        base_memory_mask=base_memory_mask_n,
                    )

                raw_adj_b = states.adj_matrix[:, :N_, :N_] if states.adj_matrix.shape[-1] else jnp.zeros((E_, N_, N_), dtype=bool)
                step_share = (int(model.memory_comm_every_k_steps) <= 1) or ((t % int(model.memory_comm_every_k_steps)) == 0)
                comm_mask_b = raw_adj_b & jnp.asarray(step_share)
                active_mask_b = states.active
                base_memory_b = base_memory if base_memory is not None else jnp.zeros((E_, model.hidden_dim), dtype=jnp.float32)
                base_receiver_mask_b = (
                    states.adj_matrix[:, :N_, N_]
                    if states.adj_matrix.shape[-1]
                    else jnp.zeros((E_, N_), dtype=bool)
                )
                # Base-memory relay is only offered to drones that do not already know the target.
                # Once the relay makes target_known true, later base-memory shares to that drone are masked out.
                base_receiver_mask_b = (
                    base_receiver_mask_b
                    & ~states.target_known[:, :N_]
                    & (base_memory_valid[:, None] if base_memory_valid is not None else jnp.zeros((E_, 1), dtype=bool))
                    & jnp.asarray(step_share)
                )
                def _rollout_one_env_no_comm(obs_n, keys_n, actor_h_n, critic_h_n, resets_n):
                    return model.rollout_step_recurrent(
                        obs_n, keys_n, actor_h_n, critic_h_n, resets_n, max_force,
                    )

                def _rollout_with_comm(_):
                    return jax.vmap(_rollout_one_env)(
                        obs_batch, act_keys, actor_h_in, critic_h_in, reset_agents_b,
                        comm_mask_b, active_mask_b, base_memory_b, base_receiver_mask_b,
                    )

                def _rollout_without_comm(_):
                    return jax.vmap(_rollout_one_env_no_comm)(
                        obs_batch, act_keys, actor_h_in, critic_h_in, reset_agents_b,
                    )

                has_drone_receiver = jnp.any(comm_mask_b)
                # base_receiver_mask_b already includes base_memory_valid, so drones connected to
                # an empty base do not force the attention path at episode start.
                has_base_memory_receiver = jnp.any(base_receiver_mask_b)
                can_skip_empty_comm = str(model.memory_comm_variant) != "self_attention"
                use_attention_path = (has_drone_receiver | has_base_memory_receiver) | jnp.asarray(not can_skip_empty_comm)
                actor_h, critic_h, actions_b, log_probs_b, values_b = jax.lax.cond(
                    use_attention_path,
                    _rollout_with_comm,
                    _rollout_without_comm,
                    operand=None,
                )
            else:
                def _rollout_one_env(obs_n, keys_n, actor_h_n, critic_h_n, resets_n):
                    return model.rollout_step_recurrent(
                        obs_n, keys_n, actor_h_n, critic_h_n, resets_n, max_force,
                    )

                actor_h, critic_h, actions_b, log_probs_b, values_b = jax.vmap(_rollout_one_env)(
                    obs_batch, act_keys, actor_h_in, critic_h_in, reset_agents_b,
                )

            if not model.actor_memory:
                actor_h = None
            if not model.critic_memory:
                critic_h = None
        else:
            def _rollout_one_env(obs_n, keys_n):
                # Returns normalised actions [-1,1], log_probs, value
                actions, log_probs, value = model.rollout_step(obs_n, keys_n, max_force)
                return actions, log_probs, value

            actions_b, log_probs_b, values_b = jax.vmap(_rollout_one_env)(obs_batch, act_keys)
        if recurrent and model.actor_memory and model.memory_comm_enabled:
            connected_to_base_b = states.adj_matrix[:, :N, N] if states.adj_matrix.shape[-1] else jnp.zeros((E, N), dtype=bool)
            reporters_b = states.target_known & connected_to_base_b & states.active
            # If multiple target-knowing drones reconnect on the same first step, choose the lowest-index reporter.
            # After base_memory_valid becomes true, later reporters cannot overwrite the first stored memory.
            first_idx_b = jnp.argmax(reporters_b.astype(jnp.int32), axis=-1)
            has_reporter_b = jnp.any(reporters_b, axis=-1)
            reported_memory_b = jnp.take_along_axis(actor_h, first_idx_b[:, None, None], axis=1).squeeze(axis=1)
            should_store_b = has_reporter_b & ~base_memory_valid
            # TODO: Future extension: give the base its own learned attention/MLP aggregation module. The base could receive observations or memories from drones as they connect, learn a larger-picture task representation, and share that representation with later incoming agents. For v1, keep the simpler behavior: store the memory of the first target-knowing agent that reconnects to base and relay that saved memory until episode end.
            base_memory = jnp.where(should_store_b[:, None], reported_memory_b, base_memory)
            base_memory_valid = base_memory_valid | should_store_b

        # actions_b: (E, N, A) — PRE-SQUASH samples u from actor.act()
        # Apply tanh squashing before scaling for the physics engine.
        # The buffer stores the raw pre-squash u for consistent PPO log-prob re-evaluation.
        squashed_b = jnp.tanh(actions_b)  # (E, N, A) in (-1, 1)

        # Scale to physical space ONLY for the env step
        states, rewards_b, dones_b, info = autoreset_step_v(states, squashed_b * max_force)

        rewards_np = np.array(rewards_b)
        dones_np   = np.array(dones_b).astype(bool)

        ep_ret_accum     += rewards_np
        ep_len_accum     += 1
        ep_success_accum  = np.maximum(ep_success_accum, np.array(info["fully_connected"]))
        # MEM_T8-only diagnostic fallback: normal levels report global_target_found.
        found_metric = np.array(info.get("target_found_fraction", info["global_target_found"]))
        ep_found_accum    = np.maximum(ep_found_accum, found_metric)
        ep_gap_accum      = np.array(info["chain_gap_dist"])
        ep_prog_pct_accum  = np.array(info["chain_progress_pct"])

        # Track reward components
        r_coverage_accum += np.array(info["r_coverage"])
        r_gap_accum      += np.array(info["r_chain_gap"])
        r_coll_accum     += np.array(info["r_collision"])
        r_prox_accum     += np.array(info["r_proximity"])
        r_found_accum    += np.array(info["r_target_found"])
        r_succ_accum     += np.array(info["r_success"])
        cov_accum         = np.array(info["global_coverage"])

        done_envs = np.where(dones_np)[0]
        for e in done_envs:
            completed_returns.append(float(ep_ret_accum[e].sum()))
            completed_lengths.append(int(ep_len_accum[e]))
            completed_success.append(float(ep_success_accum[e]))
            completed_found.append(float(ep_found_accum[e]))
            completed_gaps.append(float(ep_gap_accum[e]))
            completed_prog_pcts.append(float(ep_prog_pct_accum[e]))

            completed_r_coverage.append(float(r_coverage_accum[e]))
            completed_r_gap.append(float(r_gap_accum[e]))
            completed_r_coll.append(float(r_coll_accum[e]))
            completed_r_prox.append(float(r_prox_accum[e]))
            completed_r_found.append(float(r_found_accum[e]))
            completed_r_succ.append(float(r_succ_accum[e]))
            completed_coverage.append(float(cov_accum[e]))

            # Append terminal target data
            if track_heatmap_data:
                t_pos = np.array(info["terminal_target_pos"][e])
                if t_pos.ndim == 2:
                    t_pos = t_pos[0]
                completed_target_pos.append(t_pos.tolist())
                completed_target_success.append(bool(ep_success_accum[e] > 0.5))
                completed_target_delivered.append(bool(info["terminal_delivered"][e]))
                completed_target_visually_found.append(bool(info["terminal_visually_found"][e]))

        ep_ret_accum     = np.where(dones_np[:, None], 0.0, ep_ret_accum)
        ep_len_accum     = np.where(dones_np, 0,   ep_len_accum)
        ep_success_accum = np.where(dones_np, 0.0, ep_success_accum)
        ep_found_accum   = np.where(dones_np, 0.0, ep_found_accum)
        ep_gap_accum     = np.where(dones_np, 0.0, ep_gap_accum)  # reset so next ep starts clean
        ep_prog_pct_accum = np.where(dones_np, 0.0, ep_prog_pct_accum)

        r_coverage_accum = np.where(dones_np, 0.0, r_coverage_accum)
        r_gap_accum      = np.where(dones_np, 0.0, r_gap_accum)
        r_coll_accum     = np.where(dones_np, 0.0, r_coll_accum)
        r_prox_accum     = np.where(dones_np, 0.0, r_prox_accum)
        r_found_accum    = np.where(dones_np, 0.0, r_found_accum)
        r_succ_accum     = np.where(dones_np, 0.0, r_succ_accum)
        cov_accum        = np.where(dones_np, 0.0, cov_accum)

        # Store normalised actions in the buffer
        buf.add(MAPPOTransition(
            obs       = np.array(obs_batch),
            actions   = np.array(actions_b),   # [-1, 1]
            log_probs = np.array(log_probs_b),
            values    = np.array(values_b),
            rewards   = rewards_np,
            dones     = dones_np.astype(np.float32),
            rnn_resets = np.array(reset_agents_b) if recurrent else None,
            comm_masks = np.array(comm_mask_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            active_masks = np.array(active_mask_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_memories = np.array(base_memory_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_memory_masks = np.array(base_receiver_mask_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
        ))
        if recurrent and model.actor_memory and model.memory_comm_enabled:
            done_j = jnp.asarray(dones_np, dtype=bool)
            base_memory = jnp.where(done_j[:, None], jnp.zeros_like(base_memory), base_memory)
            base_memory_valid = jnp.where(done_j, False, base_memory_valid)
        last_dones = dones_np

    # Bootstrap value for last state
    last_obs    = obs_fn_v(states)
    if recurrent and model.critic_memory:
        reset_agents_b = jnp.asarray(last_dones)[:, None] | jnp.logical_not(states.active)

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
    ep_trackers["r_prox"]     = r_prox_accum
    ep_trackers["r_found"]    = r_found_accum
    ep_trackers["r_succ"]     = r_succ_accum
    ep_trackers["coverage"]   = cov_accum

    return (
        states, key, last_values, bootstrap_dones, actor_h, critic_h, np.asarray(last_dones, dtype=bool), base_memory, base_memory_valid,
        completed_returns, completed_lengths, completed_success,
        completed_found, completed_gaps, completed_prog_pcts,
        completed_r_coverage, completed_r_gap, completed_r_coll,
        completed_r_prox, completed_r_found, completed_r_succ,
        completed_coverage,
        completed_target_pos, completed_target_success, completed_target_delivered, completed_target_visually_found
    )


# ---------------------------------------------------------------------------
# Deterministic eval rollout (single env)
# ---------------------------------------------------------------------------

def _evaluate(
    model:        MAPPOModel,
    reset_fn,
    env_step_fn,
    obs_fn,
    reward_fn,
    cfg:          DictConfig,
    key:          jax.Array,
    num_episodes: int = 1,
    episode_callback: Optional[Callable] = None,
) -> tuple:
    """
    Run deterministic evaluation episodes.

    Parameters
    ----------
    episode_callback : optional callable(ep_idx, success, ep_states, ep_rewards, ep_metrics) -> bool
        Called after each episode completes.  Return True to stop early
        (e.g. when selective render buckets are full).
        If None, all num_episodes are run and data is collected normally.
    """
    max_force = float(cfg.env.max_force)
    max_steps = int(cfg.env.max_steps)

    all_states, all_rewards, all_metrics = [], [], []
    total_ret = total_len = total_gap = total_prog_pct = total_success = total_found = 0.0

    for ep_idx in range(num_episodes):
        key, rk = jax.random.split(key)
        state   = reset_fn(rk)
        actor_h = model.initial_actor_hidden(()) if model.actor_memory else None
        base_memory = jnp.zeros((model.hidden_dim,), dtype=jnp.float32) if model.actor_memory else None
        base_memory_valid = jnp.bool_(False)
        ep_ret  = ep_gap = ep_prog_pct = 0.0
        ep_success = ep_found = False
        ep_states, ep_rewards = [], []
        ep_metrics = {"r_explor": [], "r_gap": [], "r_coll": [], "chain_pct": [], "chain_gap": [], "r_total": [], "obs": []}

        for t in range(max_steps):
            ep_states.append(jax.device_get(state))
            obs = obs_fn(state)

            # Deterministic: take mean action (pre-squash = mu), squash then scale
            if model.actor_memory:
                resets = jnp.logical_not(state.active)

                if model.memory_comm_enabled:
                    N_eval = obs.shape[0]
                    step_share = (int(model.memory_comm_every_k_steps) <= 1) or ((t % int(model.memory_comm_every_k_steps)) == 0)
                    comm_mask = (state.adj_matrix[:N_eval, :N_eval] if state.adj_matrix.shape[-1] else jnp.zeros((N_eval, N_eval), dtype=bool)) & jnp.asarray(step_share)
                    # Base-memory relay is only offered to drones that do not already know the target.
                    # Once the relay makes target_known true, later base-memory shares to that drone are masked out.
                    base_mask = (
                        (
                            state.adj_matrix[:N_eval, N_eval]
                            if state.adj_matrix.shape[-1]
                            else jnp.zeros((N_eval,), dtype=bool)
                        )
                        & ~state.target_known[:N_eval]
                        & base_memory_valid
                        & jnp.asarray(step_share)
                    )
                    actor_h, actions, _ = model.actor.__call_team__(
                        obs, actor_h, resets, comm_mask, state.active, base_memory, base_mask
                    )
                    connected_to_base = state.adj_matrix[:N_eval, N_eval] if state.adj_matrix.shape[-1] else jnp.zeros((N_eval,), dtype=bool)
                    reporters = state.target_known & connected_to_base & state.active
                    # If multiple target-knowing drones reconnect on the same first step, choose the lowest-index reporter.
                    # After base_memory_valid becomes true, later reporters cannot overwrite the first stored memory.
                    first_idx = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
                    has_reporter = jnp.any(reporters)
                    reported_memory = actor_h[first_idx]
                    should_store = has_reporter & ~base_memory_valid
                    base_memory = jnp.where(should_store, reported_memory, base_memory)
                    base_memory_valid = base_memory_valid | should_store
                else:
                    def _act_eval(o, h, r):
                        h, mu, _ = model.actor(o, h, r)
                        return h, mu

                    actor_h, actions = jax.vmap(_act_eval)(obs, actor_h, resets)
            else:
                actions = jax.vmap(lambda o: model.actor(o)[0])(obs)  # (N, A) mu in pre-squash space
            actions = jnp.tanh(actions) * max_force                # squash + scale to physics

            old_state = state
            state     = env_step_fn(state, actions)

            hold_chain_for = int(cfg.env.get("hold_chain_for", 0))

            # Call reward_fn with is_done=False so success bonus is not prematurely added
            rew, info = reward_fn(old_state, state, jnp.bool_(False))

            fully_connected = info["fully_connected"] > 0.5
            new_chain_held_steps = jnp.where(
                fully_connected,
                state.chain_held_steps + jnp.int32(1),
                jnp.int32(0)
            )
            state = dataclasses.replace(state, chain_held_steps=new_chain_held_steps)
            success_achieved = new_chain_held_steps >= (hold_chain_for + 1)

            # If success is achieved, we add the terminal success bonus
            if bool(success_achieved):
                _, info_terminal = reward_fn(old_state, state, jnp.bool_(True))
                success_bonus_per_agent = np.array(info_terminal["r_success"]) / rew.shape[0]
                rew = rew + success_bonus_per_agent

            ep_ret   += float(rew.sum())
            ep_len    = float(t + 1)
            ep_gap    = float(info["chain_gap_dist"])
            ep_prog_pct = float(info["chain_progress_pct"])
            ep_success = ep_success or bool(success_achieved)
            # MEM_T8-only diagnostic fallback: normal levels report global_target_found.
            found_metric = info.get("target_found_fraction", info["global_target_found"])
            ep_found = max(float(ep_found), float(found_metric))
            ep_rewards.append(np.array(rew))

            ep_metrics["r_explor"].append(float(info["r_coverage"]))
            ep_metrics["r_gap"].append(float(info["r_chain_gap"]))
            ep_metrics["r_coll"].append(float(info["r_collision"]))
            ep_metrics["chain_pct"].append(float(info["chain_progress_pct"]))
            ep_metrics["chain_gap"].append(float(info["chain_gap_dist"]))
            ep_metrics["r_total"].append(float(rew.sum()))
            ep_metrics["obs"].append(np.array(obs))

            if bool(success_achieved):
                ep_states.append(jax.device_get(state))  # include the connected frame as freeze frame
                # duplicate last metric to match states length
                ep_metrics["r_explor"].append(ep_metrics["r_explor"][-1])
                ep_metrics["r_gap"].append(ep_metrics["r_gap"][-1])
                ep_metrics["r_coll"].append(ep_metrics["r_coll"][-1])
                ep_metrics["chain_pct"].append(ep_metrics["chain_pct"][-1])
                ep_metrics["chain_gap"].append(ep_metrics["chain_gap"][-1])
                ep_metrics["r_total"].append(float(rew.sum()))
                ep_metrics["obs"].append(ep_metrics["obs"][-1])
                break

        final_metrics = {k: np.array(v) for k, v in ep_metrics.items()}
        all_states.append(ep_states)
        all_rewards.append(ep_rewards)
        all_metrics.append(final_metrics)
        total_ret     += ep_ret
        total_len     += ep_len
        total_gap     += ep_gap
        total_prog_pct += ep_prog_pct
        total_success += float(ep_success)
        total_found   += float(ep_found)

        # Fire per-episode callback (used for streaming selective renders)
        if episode_callback is not None:
            stop = episode_callback(ep_idx, ep_success, ep_states, ep_rewards, final_metrics)
            if stop:
                break

    n = max(1, len(all_states))
    return (
        all_states, all_rewards, all_metrics,
        total_ret / n,
        total_len / n,
        total_gap / n,
        total_prog_pct / n,
        total_success / n,
        total_found / n,
    )


# ---------------------------------------------------------------------------
# Selective eval render callback factory
# ---------------------------------------------------------------------------

def _make_selective_eval_callback(
    n_success:    int,
    n_fail:       int,
    out_dir:      Path,
    ckpt_name:    str,
    renderer:     str,
    cfg,
) -> Callable:
    """
    Returns a closure for use as episode_callback in _evaluate().

    Behaviour per episode
    ---------------------
    * success=True  and rendered_success < n_success  → render with SUCCESS_ prefix
    * success=False and rendered_fail    < n_fail      → render with FAIL_ prefix
    * otherwise                                        → skip rendering
    Returns True (early-exit signal) when both buckets are full.

    Note: both n_success=0 and n_fail=0 is valid — episodes are computed and
    counted but nothing is rendered (useful for stats-only mode).
    """
    rendered_success     = [0]
    rendered_fail        = [0]
    cumulative_successes = [0]

    def callback(ep_idx: int, success: bool, ep_states, ep_rewards, ep_metrics) -> bool:
        nonlocal rendered_success, rendered_fail, cumulative_successes

        if success:
            cumulative_successes[0] += 1

        should_render = False
        prefix = ""

        if success and rendered_success[0] < n_success:
            should_render = True
            prefix = "SUCCESS_"
        elif not success and rendered_fail[0] < n_fail:
            should_render = True
            prefix = "FAIL_"

        # Calculate running stats
        total_eps = ep_idx + 1
        success_rate = (cumulative_successes[0] / total_eps) * 100.0
        steps = len(ep_states)
        ep_ret = float(ep_metrics["r_total"].sum())

        status_str = f"Ep {ep_idx:>2}: success={str(success):<5} steps={steps:>3} return={ep_ret:>7.1f} | Success Rate={success_rate:>5.1f}% ({cumulative_successes[0]}/{total_eps})"

        if should_render:
            if success:
                rendered_success[0] += 1
            else:
                rendered_fail[0] += 1

            ep_num = rendered_success[0] + rendered_fail[0] - 1
            stem = f"{prefix}{ckpt_name}_ep{ep_num:02d}"
            render_status = f"Render: {rendered_success[0]}/{n_success} Success, {rendered_fail[0]}/{n_fail} Fail"

            print(f"  [eval] {status_str} | {render_status} | RENDERED {stem}.mp4")

            render_eval_video(
                ep_states  = ep_states,
                ep_rewards = ep_rewards,
                ep_metrics = ep_metrics,
                cfg        = cfg,
                out_dir    = out_dir,
                filename_stem = stem,
                renderer   = renderer,
            )
        else:
            reason = "Bucket Full" if (success and n_success > 0) or (not success and n_fail > 0) else "Render Target is 0"
            render_status = f"Render: {rendered_success[0]}/{n_success} Success, {rendered_fail[0]}/{n_fail} Fail"
            print(f"  [eval] {status_str} | {render_status} | SKIPPED ({reason})")

        # Stop early if both buckets are full
        buckets_full = (rendered_success[0] >= n_success) and (rendered_fail[0] >= n_fail)
        return buckets_full

    return callback


# ---------------------------------------------------------------------------
# Utility: print eval stats summary
# ---------------------------------------------------------------------------

def _print_eval_stats(
    label: str,
    num_computed: int,
    mean_ret: float,
    mean_len: float,
    mean_prog: float,
    success_rate: float,
    found_rate: float,
) -> None:
    print(
        f"  [{label}] episodes={num_computed}  "
        f"ep_return={mean_ret:.2f}  "
        f"chain={mean_prog:.1f}%  "
        f"success={success_rate:.1%}  "
        f"found={found_rate:.1%}  "
        f"ep_len={mean_len:.0f}"
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: DictConfig, success_threshold: Optional[float] = None):
    """
    Main MAPPO training loop.

    Parameters
    ----------
    cfg : DictConfig
        Fully-merged configuration.
    success_threshold : float or None
        If set, stop training early once the 2000-episode sliding-window
        success rate reaches this value (e.g. 0.95 for 95%).
        None (default) = always train for the full total_timesteps.
    """
    # ── Logging config ────────────────────────────────────────────────────
    if cfg.logging.get("suppress_xla_warnings", True):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NVIDIA_TF32_OVERRIDE"]  = "0"

    critic_type = str(cfg.network.critic_type)
    per_agent   = (critic_type == "agent_centric")
    actor_memory = bool(cfg.network.get("actor_memory", False))
    critic_memory = bool(cfg.network.get("critic_memory", False))

    print("\n══════════════════════════════════════════════════════")
    print(f"  SwarmEcho — MAPPO  [{critic_type} critic]")
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

    transitions_per_mb  = (E * T) // MB
    estimated_vram_gib  = transitions_per_mb * 0.00063

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
    print(f"  allocating       : {estimated_vram_gib:.2f}GiB")
    print()
    print(f"  actor_layers     : {cfg.network.actor_num_layers}")
    print(f"  actor_memory     : {actor_memory}")
    print(f"  critic_layers    : {cfg.network.num_layers}")
    print(f"  critic_memory    : {critic_memory}")
    print()
    print(f"  ppo updates      : {n_updates:,}  ({total_ts:,} total timesteps)")
    print()

    # ── Environment ───────────────────────────────────────────────────────
    env_step, reset, _, (resolved_W, resolved_H, occ_grid, comm_occ_grid) = make_env_fns(cfg)
    compute_obs, _     = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid, comm_occ_grid)
    compute_reward     = make_reward_fn(cfg)

    hold_chain_for   = int(cfg.env.get("hold_chain_for", 0))
    autoreset_step   = _make_autoreset_step(env_step, reset, compute_reward, max_steps, hold_chain_for)
    autoreset_step_v = jax.jit(jax.vmap(autoreset_step))
    obs_fn_v         = jax.jit(jax.vmap(compute_obs))
    reset_v          = jax.jit(jax.vmap(reset))
    reset_s          = jax.jit(reset)

    # ── Model ─────────────────────────────────────────────────────────────
    master_key = jax.random.PRNGKey(int(cfg.training.seed))
    model_key, env_key, master_key = jax.random.split(master_key, 3)
    rngs = nnx.Rngs(int(model_key[0]))

    model = MAPPOModel(
        obs_dim          = obs_dim,
        act_dim          = act_dim,
        num_agents       = N,
        hidden_dim       = int(cfg.network.hidden_dim),
        num_layers       = int(cfg.network.num_layers),
        actor_num_layers = int(cfg.network.actor_num_layers),
        critic_type      = critic_type,
        actor_memory     = actor_memory,
        critic_memory    = critic_memory,
        rngs             = rngs,
        memory_comm_enabled = bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_gradient_mode = str(cfg.network.get("memory_comm_gradient_mode", "rial")),
        memory_comm_variant = str(cfg.network.get("memory_comm_variant", "cross_attention_residual")),
        memory_comm_every_k_steps = int(cfg.network.get("memory_comm_every_k_steps", 10)),
        memory_comm_num_heads = int(cfg.network.get("memory_comm_num_heads", 4)),
    )
    trainer = MAPPOTrainer(
        model         = model,
        lr            = float(cfg.training.lr),
        max_grad_norm = float(cfg.training.max_grad_norm),
        clip_eps      = float(cfg.training.clip_eps),
        vf_coef       = float(cfg.training.vf_coef),
        ent_coef      = float(cfg.training.ent_coef),
        num_epochs    = int(cfg.training.num_epochs),
        per_agent     = per_agent,
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
        per_agent  = per_agent,
        recurrent  = actor_memory or critic_memory,
        hidden_dim = int(cfg.network.hidden_dim),
        actor_memory  = actor_memory,
        critic_memory = critic_memory,
    )

    _, params = nnx.split(model)
    n_params  = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model params : {n_params:,}")

    # ── Checkpoint Resumption ─────────────────────────────────────────────
    import orbax.checkpoint as ocp
    if cfg.training.checkpoint_path:
        checkpoint_path = Path(cfg.training.checkpoint_path).absolute()
        print(f"  Resuming from: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        graphdef, empty_state = nnx.split(model)
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        restored_state = checkpointer.restore(
            str(checkpoint_path),
            args=ocp.args.StandardRestore(empty_state),
        )
        nnx.update(model, restored_state)
        print("  Weights restored ✓")

    # ── Environments init ─────────────────────────────────────────────────
    env_keys = jax.random.split(env_key, E)
    states   = reset_v(env_keys)
    print("  Environments initialised ✓")

    actor_h = model.initial_actor_hidden((E,)) if actor_memory else None
    critic_h = model.initial_critic_hidden((E,)) if critic_memory else None
    use_memory_comm = actor_memory and bool(cfg.network.get("memory_comm_enabled", False))
    base_memory = jnp.zeros((E, int(cfg.network.hidden_dim)), dtype=jnp.float32) if use_memory_comm else None
    base_memory_valid = jnp.zeros((E,), dtype=bool) if use_memory_comm else None
    rollout_last_dones = np.zeros(E, dtype=bool)

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
    ckpt_dir       = run_dir / "checkpoints"
    train_video_dir = run_dir / "videos" / "train"   # mid-training evals
    eval_video_dir  = run_dir / "videos" / "eval"    # final / early-exit evals
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_video_dir.mkdir(parents=True, exist_ok=True)
    eval_video_dir.mkdir(parents=True, exist_ok=True)

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
    eval_video   = bool(cfg.logging.get("eval_video", True))
    save_model   = bool(cfg.logging.get("save_model", True))
    is_benchmark = bool(cfg.logging.get("benchmark_mode", False))

    train_eval_renderer = str(cfg.visualize.get("train_eval_renderer", "fast"))
    final_eval_renderer = str(cfg.visualize.get("final_eval_renderer", "slow"))
    # is_benchmark forces fast rendering everywhere regardless of config
    effective_train_renderer = "fast" if is_benchmark else train_eval_renderer
    effective_final_renderer = "fast" if is_benchmark else final_eval_renderer

    # Selective render config
    selective_eval_render    = bool(cfg.visualize.get("selective_eval_render", False))
    eval_render_videos       = int(cfg.visualize.get("eval_render_videos", 1))
    eval_max_compute_episodes = int(cfg.visualize.get("eval_max_compute_episodes", 50))
    eval_render_successes    = int(cfg.visualize.get("eval_render_successes", 3))
    eval_render_failures     = int(cfg.visualize.get("eval_render_failures", 3))

    if not save_model:
        print("  [ckpt] Checkpoint saving disabled")
    if not eval_video:
        print("  [render] Eval video rendering disabled")
    else:
        mode_str = "selective" if selective_eval_render else "legacy"
        print(f"  [render] Sequential mode | train={effective_train_renderer} | final={effective_final_renderer} | eval_mode={mode_str}")

    # ── Training loop ─────────────────────────────────────────────────────
    eval_every  = int(cfg.logging.get("eval_freq", cfg.logging.get("video_freq", 30) or 30))
    num_ckpt    = int(cfg.logging.get("num_checkpoints", 20))
    ckpt_every  = max(1, n_updates // num_ckpt)
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
    window_ret  = deque(maxlen=2000)
    window_len  = deque(maxlen=2000)
    window_succ = deque(maxlen=2000)
    window_fnd  = deque(maxlen=2000)
    window_gap  = deque(maxlen=2000)
    window_prog_pct = deque(maxlen=2000)

    # Sliding window for evaluation heatmaps
    window_target_pos = deque(maxlen=2000)
    window_target_success = deque(maxlen=2000)
    window_target_delivered = deque(maxlen=2000)
    window_target_visually_found = deque(maxlen=2000)

    window_r_cov   = deque(maxlen=2000)
    window_r_gap   = deque(maxlen=2000)
    window_r_coll  = deque(maxlen=2000)
    window_r_prox  = deque(maxlen=2000)
    window_r_found = deque(maxlen=2000)
    window_r_succ  = deque(maxlen=2000)
    window_cov     = deque(maxlen=2000)

    completed_eps_count = 0

    track_heatmap_data = bool(
        cfg.logging.get("eval_failed_chain_heatmap", False) or
        cfg.logging.get("eval_not_delivered_heatmap", False) or
        cfg.logging.get("eval_not_visually_found_heatmap", False)
    )

    t_start = time.perf_counter()

    try:
        for update in range(1, n_updates + 1):
            master_key, collect_key = jax.random.split(master_key)
            steps_done = update * E * T

            # ── Rollout ───────────────────────────────────────────────────────
            (states, collect_key,
             last_values, last_dones, actor_h, critic_h, rollout_last_dones, base_memory, base_memory_valid,
             raw_ret, raw_len, raw_success, raw_found, raw_gap, raw_prog_pct,
             raw_r_cov, raw_r_gap, raw_r_coll, raw_r_prox, raw_r_found, raw_r_succ,
             raw_cov,
             raw_target_pos, raw_target_success, raw_target_delivered, raw_target_visually_found) = _collect_rollout_mappo(
                states, model, buf, autoreset_step_v, obs_fn_v,
                collect_key, max_force, T, ep_trackers,
                actor_h, critic_h, rollout_last_dones, base_memory, base_memory_valid,
                track_heatmap_data=track_heatmap_data,
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

                window_target_pos.extend(raw_target_pos)
                window_target_success.extend(raw_target_success)
                window_target_delivered.extend(raw_target_delivered)
                window_target_visually_found.extend(raw_target_visually_found)

                window_r_cov.extend(raw_r_cov)
                window_r_gap.extend(raw_r_gap)
                window_r_coll.extend(raw_r_coll)
                window_r_prox.extend(raw_r_prox)
                window_r_found.extend(raw_r_found)
                window_r_succ.extend(raw_r_succ)
                window_cov.extend(raw_cov)

            # -- Success-rate curriculum transition check -------------------
            if (
                success_threshold is not None
                and len(window_succ) == window_succ.maxlen
                and float(np.mean(window_succ)) >= success_threshold
            ):
                print(
                    f"  [curriculum] Success rate {float(np.mean(window_succ)):.1%} >= "
                    f"threshold {success_threshold:.1%} -- advancing to next level."
                )
                # Save an intermediate checkpoint before breaking
                early_ckpt_str = ""
                if save_model:
                    import orbax.checkpoint as ocp
                    import shutil
                    early_ckpt = (ckpt_dir / f"ckpt_early_{update:06d}").absolute()
                    if early_ckpt.exists():
                        shutil.rmtree(early_ckpt)
                    _, state_dict = nnx.split(model)
                    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
                    checkpointer.save(str(early_ckpt), args=ocp.args.StandardSave(state_dict))
                    print(f"  [ckpt-early] saved -> {early_ckpt}")
                    early_ckpt_str = str(early_ckpt)
                else:
                    print("  [ckpt-early] skipped (save_model=false)")

                # ── Early-exit eval + video (same logic as final eval) ────────
                if eval_video:
                    master_key, eval_key = jax.random.split(master_key)
                    _run_eval_with_render(
                        model=model, reset_s=reset_s, env_step=env_step,
                        compute_obs=compute_obs, compute_reward=compute_reward,
                        cfg=cfg, eval_key=eval_key,
                        out_dir=eval_video_dir,
                        ckpt_name=f"early_{update:06d}",
                        renderer=effective_final_renderer,
                        selective=selective_eval_render,
                        eval_render_videos=eval_render_videos,
                        eval_max_compute=eval_max_compute_episodes,
                        n_success=eval_render_successes,
                        n_fail=eval_render_failures,
                        wandb_run=wandb_run,
                        steps_done=steps_done,
                        max_steps=max_steps,
                        label="eval-early",
                    )

                return early_ckpt_str

            # ── GAE + minibatches ─────────────────────────────────────────────
            advs, rets = buf.compute_gae(last_values, last_dones)
            mbs        = buf.get_minibatches(advs, rets, int(cfg.training.num_minibatches), collect_key)

            # ── PPO update ────────────────────────────────────────────────────
            ppo_stats = trainer.update(mbs)

            elapsed = time.perf_counter() - t_start
            sps     = steps_done / elapsed

            # ── Stdout ───────────────────────────────────────────────────────
            if update % log_every == 0 or update == 1:
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
                time_per_update = elapsed / update
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
                    current_steps = np.array(states.step)
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
                }
                if len(window_ret) == window_ret.maxlen:
                    logs.update({
                        "train/ep_return":          float(np.mean(window_ret)),
                        "train/ep_length":          float(np.mean(window_len)),
                        "train/success_rate":       float(np.mean(window_succ)),
                        "train/target_found_rate":  float(np.mean(window_fnd)),
                        "train/chain_gap_dist":     float(np.mean(window_gap)),
                        "train/chain_progress_pct": float(np.mean(window_prog_pct)),
                        "train/ep_length_reduction": (1.0 - (float(np.mean(window_len)) / max_steps)) * 100.0,
                        "train/map_coverage_pct":   float(np.mean(window_cov)) * 100.0,
                        "train/episodes_completed": completed_eps_count,

                        "rewards/exploration":      float(np.mean(window_r_cov)),
                        "rewards/chain_gap":        float(np.mean(window_r_gap)),
                        "rewards/collision":        float(np.mean(window_r_coll)),
                        "rewards/proximity":        float(np.mean(window_r_prox)),
                        "rewards/target_found":     float(np.mean(window_r_found)),
                        "rewards/success_bonus":    float(np.mean(window_r_succ)),
                    })
                wandb.log(logs, step=steps_done)

            # ── Mid-training eval + single video ─────────────────────────────
            if update % eval_every == 0 and update != n_updates:
                master_key, eval_key = jax.random.split(master_key)
                (ep_states_list, ep_rewards_list, all_metrics_list,
                 eval_ret, eval_len, eval_gap, eval_prog_pct, eval_success, eval_found) = _evaluate(
                    model, reset_s,
                    jax.jit(env_step), jax.jit(compute_obs), jax.jit(compute_reward),
                    cfg, eval_key, num_episodes=1,
                )

                eval_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

                if eval_video:
                    stem = f"{eval_timestamp}_eval_update_{update:06d}"
                    render_eval_video(
                        ep_states  = ep_states_list[0],
                        ep_rewards = ep_rewards_list[0],
                        ep_metrics = all_metrics_list[0],
                        cfg        = cfg,
                        out_dir    = train_video_dir,
                        filename_stem = stem,
                        renderer   = effective_train_renderer,
                    )

                print(
                    f"  [eval-train] update={update}  "
                    f"steps={steps_done:,}  "
                    f"ep_return={eval_ret:.2f}  "
                    f"chain={eval_prog_pct:.1f}%  "
                    f"success={eval_success:.1%}"
                )

                # Generate mid-run evaluation heatmaps if toggled in LoggingConfig
                generate_any_heatmap = bool(
                    cfg.logging.get("eval_failed_chain_heatmap", False) or
                    cfg.logging.get("eval_not_delivered_heatmap", False) or
                    cfg.logging.get("eval_not_visually_found_heatmap", False)
                )

                if generate_any_heatmap and len(window_target_success) > 0:
                    try:
                        # Load map data once (locally imported to prevent circular dependencies)
                        from training.evaluate_pipeline import (
                            load_map_data,
                            render_and_save_failed_chain_heatmap,
                            render_and_save_not_found_heatmap,
                            render_and_save_merged_heatmap,
                            MERGE_TARGET_FOUND_HEATMAPS
                        )
                        _, map_data, map_def = load_map_data(cfg)
                        
                        target_pos_arr = np.array(window_target_pos)
                        target_success_arr = np.array(window_target_success)
                        target_delivered_arr = np.array(window_target_delivered)
                        target_visually_found_arr = np.array(window_target_visually_found)
                        
                        n_completed = len(window_target_success)
                        run_timestamp = f"{eval_timestamp}_update_{update:06d}"
                        
                        # 1. Failed Chain Heatmap
                        if cfg.logging.get("eval_failed_chain_heatmap", False):
                            failed_positions = target_pos_arr[~target_success_arr]
                            success_rate = (np.sum(target_success_arr) / n_completed * 100.0)
                            num_fail = len(failed_positions)
                            _ = render_and_save_failed_chain_heatmap(
                                failed_positions=failed_positions,
                                map_data=map_data,
                                map_def=map_def,
                                success_rate=success_rate,
                                num_fail=num_fail,
                                run_dir=run_dir,
                                video_dir=train_video_dir,
                                run_timestamp=run_timestamp,
                                save_csv=False,
                                save_png=True,
                                total_episodes=n_completed
                            )
                            
                        # 2 & 3. Merged or Separate Target Found/Delivered Heatmaps
                        if MERGE_TARGET_FOUND_HEATMAPS:
                            if cfg.logging.get("eval_not_delivered_heatmap", False) or cfg.logging.get("eval_not_visually_found_heatmap", False):
                                not_delivered_positions = target_pos_arr[~target_delivered_arr]
                                delivered_rate = (np.sum(target_delivered_arr) / n_completed * 100.0)
                                num_not_delivered = len(not_delivered_positions)

                                not_visually_found_positions = target_pos_arr[~target_visually_found_arr]
                                visually_found_rate = (np.sum(target_visually_found_arr) / n_completed * 100.0)
                                num_not_visually_found = len(not_visually_found_positions)

                                render_and_save_merged_heatmap(
                                    not_delivered_positions=not_delivered_positions,
                                    not_visually_found_positions=not_visually_found_positions,
                                    map_data=map_data,
                                    map_def=map_def,
                                    delivered_rate=delivered_rate,
                                    visually_found_rate=visually_found_rate,
                                    num_not_delivered=num_not_delivered,
                                    num_not_visually_found=num_not_visually_found,
                                    run_dir=run_dir,
                                    video_dir=train_video_dir,
                                    run_timestamp=run_timestamp,
                                    total_episodes=n_completed
                                )
                        else:
                            # 2. Delivered-to-base Heatmap
                            if cfg.logging.get("eval_not_delivered_heatmap", False):
                                not_delivered_positions = target_pos_arr[~target_delivered_arr]
                                delivered_rate = (np.sum(target_delivered_arr) / n_completed * 100.0)
                                num_not_delivered = len(not_delivered_positions)
                                render_and_save_not_found_heatmap(
                                    not_found_positions=not_delivered_positions,
                                    map_data=map_data,
                                    map_def=map_def,
                                    found_rate=delivered_rate,
                                    num_not_found=num_not_delivered,
                                    run_dir=run_dir,
                                    video_dir=train_video_dir,
                                    run_timestamp=run_timestamp,
                                    filename_prefix="not_delivered_targets_heatmap",
                                    label="Not Delivered",
                                    total_episodes=n_completed
                                )
                                
                            # 3. Visually Found Heatmap
                            if cfg.logging.get("eval_not_visually_found_heatmap", False):
                                not_visually_found_positions = target_pos_arr[~target_visually_found_arr]
                                visually_found_rate = (np.sum(target_visually_found_arr) / n_completed * 100.0)
                                num_not_visually_found = len(not_visually_found_positions)
                                render_and_save_not_found_heatmap(
                                    not_found_positions=not_visually_found_positions,
                                    map_data=map_data,
                                    map_def=map_def,
                                    found_rate=visually_found_rate,
                                    num_not_found=num_not_visually_found,
                                    run_dir=run_dir,
                                    video_dir=train_video_dir,
                                    run_timestamp=run_timestamp,
                                    filename_prefix="not_visually_found_targets_heatmap",
                                    label="Not Visually Found",
                                    total_episodes=n_completed
                                )
                    except Exception as heatmap_err:
                        print(f"  [heatmap-error] Failed to render evaluation heatmaps: {heatmap_err}")

                # TODO: Mid-training eval W&B metrics are based on a single episode and carry
                #       little statistical weight. Replace with a proper multi-episode test
                #       harness before re-enabling.
                # if wandb_run:
                #     import wandb
                #     wandb.log({
                #         "eval/ep_return":           eval_ret,
                #         "eval/ep_length":           eval_len,
                #         "eval/chain_gap_dist":      eval_gap,
                #         "eval/chain_progress_pct":  eval_prog_pct,
                #         "eval/ep_length_reduction": (1.0 - (eval_len / max_steps)) * 100.0,
                #         "eval/success_rate":        eval_success,
                #         "eval/target_found":        eval_found,
                #     }, step=steps_done)

                # Small sleep to allow XLA to settle after the eval/render spike
                time.sleep(1.0)

            # ── Final eval (last update) ──────────────────────────────────────
            if update == n_updates and eval_video:
                master_key, eval_key = jax.random.split(master_key)
                _run_eval_with_render(
                    model=model, reset_s=reset_s, env_step=env_step,
                    compute_obs=compute_obs, compute_reward=compute_reward,
                    cfg=cfg, eval_key=eval_key,
                    out_dir=eval_video_dir,
                    ckpt_name=f"update_{update:06d}",
                    renderer=effective_final_renderer,
                    selective=selective_eval_render,
                    eval_render_videos=eval_render_videos,
                    eval_max_compute=eval_max_compute_episodes,
                    n_success=eval_render_successes,
                    n_fail=eval_render_failures,
                    wandb_run=wandb_run,
                    steps_done=steps_done,
                    max_steps=max_steps,
                    label="eval-final",
                )
                time.sleep(1.0)

            # ── Checkpoint ────────────────────────────────────────────────────
            if save_model and not is_benchmark and (update % ckpt_every == 0 or update == n_updates):
                import orbax.checkpoint as ocp
                import shutil
                ckpt_path = (ckpt_dir / f"ckpt_{update:06d}").absolute()
                if ckpt_path.exists():
                    shutil.rmtree(ckpt_path)
                _, state_dict = nnx.split(model)
                checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
                checkpointer.save(str(ckpt_path), args=ocp.args.StandardSave(state_dict))
                print(f"  [ckpt] saved -> {ckpt_path}")

        total_time = time.perf_counter() - t_start
        print(f"\n  Training complete in {total_time:.1f}s  ({total_time/60:.1f} min)")

        # ── Final Checkpoint ─────────────────────────────────────────────────
        if save_model and not is_benchmark:
            final_ckpt = (ckpt_dir / f"ckpt_{n_updates:06d}").absolute()
            import orbax.checkpoint as ocp
            import shutil
            if final_ckpt.exists():
                shutil.rmtree(final_ckpt)
            _, state_dict = nnx.split(model)
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
            checkpointer.save(str(final_ckpt), args=ocp.args.StandardSave(state_dict))
            print(f"  [ckpt-final] saved -> {final_ckpt}")
            return str(final_ckpt)

        return ""

    finally:
        if wandb_run:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Shared eval + render helper (used for both final and early-exit evals)
# ---------------------------------------------------------------------------

def _run_eval_with_render(
    model, reset_s, env_step, compute_obs, compute_reward,
    cfg, eval_key,
    out_dir: Path,
    ckpt_name: str,
    renderer: str,
    selective: bool,
    eval_render_videos: int,
    eval_max_compute: int,
    n_success: int,
    n_fail: int,
    wandb_run,
    steps_done: int,
    max_steps: int,
    label: str = "eval",
) -> None:
    """Run evaluation and render videos to out_dir. Handles both legacy and selective modes."""

    if selective:
        # Selective mode: stream episodes, fill SUCCESS_/FAIL_ buckets first-come-first-served
        callback = _make_selective_eval_callback(
            n_success = n_success,
            n_fail    = n_fail,
            out_dir   = out_dir,
            ckpt_name = ckpt_name,
            renderer  = renderer,
            cfg       = cfg,
        )
        (_, _, _,
         eval_ret, eval_len, eval_gap, eval_prog_pct,
         eval_success, eval_found) = _evaluate(
            model, reset_s,
            jax.jit(env_step), jax.jit(compute_obs), jax.jit(compute_reward),
            cfg, eval_key,
            num_episodes=eval_max_compute,
            episode_callback=callback,
        )
        n_computed = callback.__closure__[0].cell_contents[0] + callback.__closure__[1].cell_contents[0]
        # Note: n_computed above is approximate (only rendered counts); full episode count from _evaluate
    else:
        # Legacy mode: compute and render eval_render_videos episodes sequentially
        (ep_states_list, ep_rewards_list, all_metrics_list,
         eval_ret, eval_len, eval_gap, eval_prog_pct,
         eval_success, eval_found) = _evaluate(
            model, reset_s,
            jax.jit(env_step), jax.jit(compute_obs), jax.jit(compute_reward),
            cfg, eval_key,
            num_episodes=eval_render_videos,
        )
        for idx in range(eval_render_videos):
            stem = f"eval_{ckpt_name}_ep{idx:02d}"
            render_eval_video(
                ep_states  = ep_states_list[idx],
                ep_rewards = ep_rewards_list[idx],
                ep_metrics = all_metrics_list[idx],
                cfg        = cfg,
                out_dir    = out_dir,
                filename_stem = stem,
                renderer   = renderer,
            )

    _print_eval_stats(
        label=label,
        num_computed=eval_max_compute if selective else eval_render_videos,
        mean_ret=eval_ret,
        mean_len=eval_len,
        mean_prog=eval_prog_pct,
        success_rate=eval_success,
        found_rate=eval_found,
    )

    # TODO: Mid-training eval W&B metrics are based on very few episodes and carry
    #       little statistical weight. Replace with a proper multi-episode test
    #       harness before re-enabling. For now only training window metrics are
    #       logged to W&B.
    # if wandb_run:
    #     import wandb
    #     wandb.log({
    #         "eval/ep_return":           eval_ret,
    #         "eval/ep_length":           eval_len,
    #         "eval/chain_gap_dist":      eval_gap,
    #         "eval/chain_progress_pct":  eval_prog_pct,
    #         "eval/ep_length_reduction": (1.0 - (eval_len / max_steps)) * 100.0,
    #         "eval/success_rate":        eval_success,
    #         "eval/target_found":        eval_found,
    #     }, step=steps_done)
