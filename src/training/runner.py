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

import functools
import gc
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


def _save_checkpoint_history(ckpt_path: Path, run_name: str, update: int, E: int, T: int, prior_history: list):
    try:
        import json
        current_session_steps = update * E * T
        updated_history = list(prior_history)
        updated_history.append({"run_name": run_name, "steps": current_session_steps})
        total_steps = sum(x["steps"] for x in updated_history)
        
        history_file = ckpt_path / "step_history.json"
        with open(history_file, "w") as f:
            json.dump({
                "total_steps": total_steps,
                "history": updated_history
            }, f, indent=2)
    except Exception as e:
        print(f"  [checkpoint-history] Failed to save step_history.json to {ckpt_path}: {e}")


# ---------------------------------------------------------------------------
# Resolve step offset from checkpoint path
# ---------------------------------------------------------------------------

def _get_checkpoint_step_offset(cfg: DictConfig, E: int, T: int) -> int:
    """
    Resolve step offset for W&B logging and print statements.
    If checkpoint_step_offset is configured, use it directly.
    Otherwise, if resuming from a checkpoint, try to parse the number of steps/updates.
    """
    offset = cfg.training.get("checkpoint_step_offset", None)
    if offset is not None:
        return int(offset)

    ckpt_path_str = cfg.training.get("checkpoint_path", None)
    if ckpt_path_str:
        try:
            import re
            path_name = Path(ckpt_path_str).name
            
            # Case 1: Standard checkpoint folder name: ckpt_000500, ckpt_early_000500, ckpt_final_000500
            match = re.search(r'ckpt_(?:early_|final_)?(\d+)', path_name, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                steps_per_update = E * T
                calculated_steps = val * steps_per_update
                print(f"  [W&B step offset] Auto-detected update {val} from checkpoint name '{path_name}'. "
                      f"Using calculated step offset: {calculated_steps:,} ({val} updates x {steps_per_update:,} steps/update)")
                return calculated_steps

            # Case 2: Explicit step count: steps_100000000, step_5000000
            match = re.search(r'step(?:s)?_?(\d+)', path_name, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                print(f"  [W&B step offset] Auto-detected step count {val} from checkpoint name '{path_name}'. Using as step offset.")
                return val

            # Case 3: Folder name is just a number
            if path_name.isdigit():
                val = int(path_name)
                if val < 100000:
                    steps_per_update = E * T
                    calculated_steps = val * steps_per_update
                    print(f"  [W&B step offset] Auto-detected update index {val} from checkpoint name '{path_name}'. "
                          f"Using calculated step offset: {calculated_steps:,}")
                    return calculated_steps
                else:
                    print(f"  [W&B step offset] Auto-detected step count {val} from checkpoint name '{path_name}'. Using as step offset.")
                    return val

        except Exception as e:
            print(f"  [W&B step offset] Failed to parse step offset from checkpoint path '{ckpt_path_str}': {e}")

    return 0


# ---------------------------------------------------------------------------
# Auto-reset step factory
# ---------------------------------------------------------------------------

def _make_autoreset_step(env_step_fn, reset_fn, reward_fn, max_steps: int, hold_chain_for: int = 0, terminate_on_target_found: bool = False):
    """
    Wrap env_step to auto-reset on episode termination.

    Done conditions (either triggers reset):
      * time_up        : new_state.step >= max_steps
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
                return model.rollout_step_recurrent(
                    obs_n, keys_n, actor_h_n, critic_h_n, resets_n, max_force,
                    actor_signature=actor_sig_n,
                    actor_value=actor_val_n,
                    comm_mask=comm_mask_n,
                    active=active_n,
                    base_signature=base_sig_n,
                    base_value=base_val_n,
                    base_memory_mask=base_memory_mask_n,
                )

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
        return None, None, None, None, actions_b, log_probs_b, values_b, None, None, None, None, None, None, None, None


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
    E, N = buf.E, buf.N
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
    ep_finder_return_accum = ep_trackers.get("finder_return", np.zeros(E))
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
    completed_finder_returns = []
    completed_diag_memories = []
    completed_diag_targets = []
    completed_diag_valids = []

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
                    states.adj_matrix,
                    states.active,
                    states.target_known,
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
        ep_finder_return_accum = np.maximum(
            ep_finder_return_accum,
            np.array(info.get("finder_returned_to_target_after_delivery", 0.0)),
        )

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
            completed_finder_returns.append(float(ep_finder_return_accum[e]))

            completed_r_coverage.append(float(r_coverage_accum[e]))
            completed_r_gap.append(float(r_gap_accum[e]))
            completed_r_coll.append(float(r_coll_accum[e]))
            completed_r_prox.append(float(r_prox_accum[e]))
            completed_r_found.append(float(r_found_accum[e]))
            completed_r_succ.append(float(r_succ_accum[e]))
            completed_coverage.append(float(cov_accum[e]))

            is_diag_valid = (
                model.actor_memory and model.memory_comm_enabled
                and base_signature is not None
                and bool(np.array(info["terminal_delivered"])[e])
                and np.any(np.array(base_value)[e] != 0.0)
            )
            completed_diag_valids.append(is_diag_valid)
            if is_diag_valid:
                t_pos_diag = np.array(info["terminal_target_pos"][e])
                if t_pos_diag.ndim == 2:
                    t_pos_diag = t_pos_diag[0]
                completed_diag_memories.append(
                    np.concatenate([np.array(base_signature)[e], np.array(base_value)[e]], axis=0)
                )
                completed_diag_targets.append(t_pos_diag)

        ep_ret_accum     = np.where(dones_np[:, None], 0.0, ep_ret_accum)
        ep_len_accum     = np.where(dones_np, 0,   ep_len_accum)
        ep_success_accum = np.where(dones_np, 0.0, ep_success_accum)
        ep_found_accum   = np.where(dones_np, 0.0, ep_found_accum)
        ep_gap_accum     = np.where(dones_np, 0.0, ep_gap_accum)  # reset so next ep starts clean
        ep_prog_pct_accum = np.where(dones_np, 0.0, ep_prog_pct_accum)
        ep_finder_return_accum = np.where(dones_np, 0.0, ep_finder_return_accum)

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
            base_signatures = np.array(base_signature_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_values = np.array(base_value_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
            base_memory_masks = np.array(base_receiver_mask_b) if recurrent and model.actor_memory and model.memory_comm_enabled else None,
        ))
        if recurrent and model.actor_memory and model.memory_comm_enabled:
            done_j = jnp.asarray(dones_np, dtype=bool)
            base_signature = jnp.where(done_j[:, None], jnp.zeros_like(base_signature), base_signature)
            base_value = jnp.where(done_j[:, None], jnp.zeros_like(base_value), base_value)
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
    ep_trackers["finder_return"] = ep_finder_return_accum

    ep_trackers["r_coverage"] = r_coverage_accum
    ep_trackers["r_gap"]      = r_gap_accum
    ep_trackers["r_coll"]     = r_coll_accum
    ep_trackers["r_prox"]     = r_prox_accum
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
        completed_r_prox, completed_r_found, completed_r_succ,
        completed_coverage,
        completed_finder_returns,
        np.array(completed_diag_memories, dtype=np.float32),
        np.array(completed_diag_targets, dtype=np.float32),
        completed_diag_valids,
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
    collect_obs_logs = bool(cfg.logging.get("obs_log", cfg.logging.get("log_obs", True)))

    all_states, all_rewards, all_metrics = [], [], []
    total_ret = total_len = total_gap = total_prog_pct = total_success = total_found = 0.0

    # Pre-build vmapped action functions to avoid recreation and compilation triggers inside the loop
    if model.actor_memory:
        if model.memory_comm_enabled:
            act_team_fn = model.actor.__call_team__
        else:
            def _act_eval(o, h, r):
                h, mu, _ = model.actor(o, h, r)
                return h, mu
            vmapped_act = jax.vmap(_act_eval)
    else:
        def _act_eval_ff(o):
            return model.actor(o)[0]
        vmapped_act = jax.vmap(_act_eval_ff)

    for ep_idx in range(num_episodes):
        key, rk = jax.random.split(key)
        state   = reset_fn(rk)
        actor_h = model.initial_actor_hidden(()) if model.actor_memory else None
        actor_signature = model.initial_actor_signature(()) if (model.actor_memory and model.memory_comm_enabled) else None
        actor_value = model.initial_actor_value(()) if (model.actor_memory and model.memory_comm_enabled) else None
        base_signature = jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32) if (model.actor_memory and model.memory_comm_enabled) else None
        base_value = jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32) if (model.actor_memory and model.memory_comm_enabled) else None
        base_memory_valid = jnp.bool_(False)
        ep_ret  = ep_gap = ep_prog_pct = 0.0
        ep_success = ep_found = False
        ep_states, ep_rewards = [], []
        ep_metrics = {"r_explor": [], "r_gap": [], "r_coll": [], "chain_pct": [], "chain_gap": [], "r_total": []}
        if collect_obs_logs:
            ep_metrics["obs"] = []

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
                    actor_h, actor_signature, actor_value, actions, _ = act_team_fn(
                        obs, actor_h, actor_signature, actor_value, resets, comm_mask, state.active, base_signature, base_value, base_mask
                    )
                    connected_to_base = state.adj_matrix[:N_eval, N_eval] if state.adj_matrix.shape[-1] else jnp.zeros((N_eval,), dtype=bool)
                    reporters = state.target_known & connected_to_base & state.active
                    # If multiple target-knowing drones reconnect on the same first step, choose the lowest-index reporter.
                    # After base_memory_valid becomes true, later reporters cannot overwrite the first stored memory.
                    first_idx = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
                    has_reporter = jnp.any(reporters)
                    reported_signature = actor_signature[first_idx]
                    reported_value = actor_value[first_idx]
                    should_store = has_reporter & ~base_memory_valid
                    base_signature = jnp.where(should_store, reported_signature, base_signature)
                    base_value = jnp.where(should_store, reported_value, base_value)
                    base_memory_valid = base_memory_valid | should_store
                else:
                    actor_h, actions = vmapped_act(obs, actor_h, resets)
            else:
                actions = vmapped_act(obs)  # (N, A) mu in pre-squash space
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
            if collect_obs_logs:
                ep_metrics["obs"].append(np.array(obs))

            target_found = bool(info["global_target_found"] > 0.5)
            terminate_on_target_found = bool(cfg.env.get("terminate_on_target_found", False))
            episode_ended = success_achieved or (terminate_on_target_found and target_found)

            if bool(episode_ended):
                ep_states.append(jax.device_get(state))  # include the connected frame as freeze frame
                # duplicate last metric to match states length
                ep_metrics["r_explor"].append(ep_metrics["r_explor"][-1])
                ep_metrics["r_gap"].append(ep_metrics["r_gap"][-1])
                ep_metrics["r_coll"].append(ep_metrics["r_coll"][-1])
                ep_metrics["chain_pct"].append(ep_metrics["chain_pct"][-1])
                ep_metrics["chain_gap"].append(ep_metrics["chain_gap"][-1])
                ep_metrics["r_total"].append(float(rew.sum()))
                if collect_obs_logs:
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


@functools.partial(
    nnx.jit,
    static_argnames=("reset", "env_step", "obs_fn", "reward_fn", "cfg", "num_envs")
)
def _run_parallel_eval_jit(
    model:        MAPPOModel,
    key:          jax.Array,
    reset,
    env_step,
    obs_fn,
    reward_fn,
    cfg:          DictConfig,
    num_envs:     int = 4000,
) -> tuple:
    max_steps = int(cfg.env.max_steps)
    hold_chain_for = int(cfg.env.get("hold_chain_for", 0))
    actor_memory = bool(cfg.network.get("actor_memory", False))
    memory_comm_enabled = bool(cfg.network.get("memory_comm_enabled", False))
    memory_comm_every_k_steps = int(cfg.network.get("memory_comm_every_k_steps", 5))
    max_force = float(cfg.env.max_force)

    env_keys = jax.random.split(key, num_envs)

    def run_single_env_rollout(env_key):
        state = reset(env_key)

        actor_h = model.initial_actor_hidden(()) if actor_memory else None
        actor_signature = model.initial_actor_signature(()) if (actor_memory and memory_comm_enabled) else None
        actor_value = model.initial_actor_value(()) if (actor_memory and memory_comm_enabled) else None
        base_signature = jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32) if (actor_memory and memory_comm_enabled) else None
        base_value = jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32) if (actor_memory and memory_comm_enabled) else None
        base_memory_valid = jnp.bool_(False) if (actor_memory and memory_comm_enabled) else None

        init_carry = (
            state,
            actor_h,
            actor_signature,
            actor_value,
            base_signature,
            base_value,
            base_memory_valid,
            jnp.bool_(False),  # has_succeeded
            jnp.bool_(False),  # has_found_delivered
            jnp.bool_(False),  # has_found_visual
            jnp.float32(0.0),  # max_found
            jnp.float32(0.0),  # returns
            jnp.int32(0),      # lengths
        )

        def eval_step(carry, t):
            state, actor_h, actor_signature, actor_value, base_signature, base_value, base_memory_valid, has_succeeded, has_found_delivered, has_found_visual, max_found, returns, lengths = carry

            obs = obs_fn(state)

            if actor_memory:
                resets = jnp.logical_not(state.active)
                if memory_comm_enabled:
                    N_eval = obs.shape[0]
                    step_share = (memory_comm_every_k_steps <= 1) | ((t % memory_comm_every_k_steps) == 0)
                    comm_mask = (state.adj_matrix[:N_eval, :N_eval] if state.adj_matrix.shape[-1] else jnp.zeros((N_eval, N_eval), dtype=bool)) & step_share
                    base_mask = (
                        (
                            state.adj_matrix[:N_eval, N_eval]
                            if state.adj_matrix.shape[-1]
                            else jnp.zeros((N_eval,), dtype=bool)
                        )
                        & ~state.target_known[:N_eval]
                        & base_memory_valid
                        & step_share
                    )
                    actor_h, actor_signature, actor_value, actions, _ = model.actor.__call_team__(
                        obs, actor_h, actor_signature, actor_value, resets, comm_mask, state.active, base_signature, base_value, base_mask
                    )
                    connected_to_base = state.adj_matrix[:N_eval, N_eval] if state.adj_matrix.shape[-1] else jnp.zeros((N_eval,), dtype=bool)
                    reporters = state.target_known & connected_to_base & state.active
                    first_idx = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
                    has_reporter = jnp.any(reporters)
                    reported_signature = actor_signature[first_idx]
                    reported_value = actor_value[first_idx]
                    should_store = has_reporter & ~base_memory_valid
                    base_signature = jnp.where(should_store, reported_signature, base_signature)
                    base_value = jnp.where(should_store, reported_value, base_value)
                    base_memory_valid = base_memory_valid | should_store
                else:
                    def _act_eval(o, h, r):
                        h, mu, _ = model.actor(o, h, r)
                        return h, mu
                    actor_h, actions = jax.vmap(_act_eval)(obs, actor_h, resets)
            else:
                actions = jax.vmap(lambda o: model.actor(o)[0])(obs)

            actions = jnp.tanh(actions) * max_force
            old_state = state
            state = env_step(state, actions)

            rew, info = reward_fn(old_state, state, jnp.bool_(False))

            fully_connected = info["fully_connected"] > 0.5
            new_chain_held_steps = jnp.where(
                fully_connected,
                state.chain_held_steps + jnp.int32(1),
                jnp.int32(0)
            )
            state = dataclasses.replace(state, chain_held_steps=new_chain_held_steps)
            success_achieved = new_chain_held_steps >= (hold_chain_for + 1)

            _, info_terminal = reward_fn(old_state, state, jnp.bool_(True))
            success_bonus_per_agent = info_terminal["r_success"] / rew.shape[0]
            rew = jnp.where(success_achieved & ~has_succeeded, rew + success_bonus_per_agent, rew)

            new_has_succeeded = has_succeeded | success_achieved
            new_has_found_delivered = has_found_delivered | state.base_target_known
            new_has_found_visual = has_found_visual | jnp.any(state.target_known, axis=-1)

            found_metric = info.get("target_found_fraction", info["global_target_found"])
            new_max_found = jnp.maximum(max_found, found_metric)

            # Check for early episode termination (target found when terminate_on_target_found is True)
            terminate_on_target_found = bool(cfg.env.get("terminate_on_target_found", False))
            target_found = info["global_target_found"] > 0.5
            episode_ended_prev = has_succeeded | (jnp.bool_(terminate_on_target_found) & has_found_delivered)

            new_returns = jnp.where(episode_ended_prev, returns, returns + rew.sum())
            new_lengths = jnp.where(episode_ended_prev, lengths, lengths + 1)

            new_carry = (state, actor_h, actor_signature, actor_value, base_signature, base_value, base_memory_valid, new_has_succeeded, new_has_found_delivered, new_has_found_visual, new_max_found, new_returns, new_lengths)
            return new_carry, (info["chain_gap_dist"], info["chain_progress_pct"])

        final_carry, scan_outs = jax.lax.scan(eval_step, init_carry, jnp.arange(max_steps))
        state_f, _, _, _, _, _, _, succ, delivered, visual, fnd, ret, length = final_carry
        gap_dists, progress_pcts = scan_outs

        return ret, length, gap_dists[-1], progress_pcts[-1], succ, fnd, state_f, succ, delivered, visual

    return jax.vmap(run_single_env_rollout)(env_keys)


def _evaluate_parallel(
    model:        MAPPOModel,
    reset,
    env_step,
    obs_fn,
    reward_fn,
    cfg:          DictConfig,
    key:          jax.Array,
    num_envs:     int = 4000,
) -> tuple:
    """
    Run deterministic evaluation episodes in parallel using jax.vmap and jax.lax.scan.

    Returns:
      rets, lengths, gaps, progs, succs, fnds, final_state, final_succs, final_delivered, final_visual
    """
    return _run_parallel_eval_jit(model, key, reset, env_step, obs_fn, reward_fn, cfg, num_envs)


def _release_video_eval_trajectory() -> None:
    """
    Encourage prompt cleanup after synchronous video rendering.

    The video path materializes complete episode trajectories on the host so
    the renderer can operate without JAX/chex dependencies.  Drop cyclic Python
    garbage immediately before training resumes, and force a tiny JAX
    synchronization point so pending dispatches do not overlap the next PPO
    update's large allocations.
    """
    gc.collect()
    jax.block_until_ready(jnp.asarray(0, dtype=jnp.int32))


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
            _release_video_eval_trajectory()
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
    if success_threshold is not None:
        OmegaConf.set_readonly(cfg, False)
        cfg.curriculum.success_threshold = success_threshold
        OmegaConf.set_readonly(cfg, True)

    if cfg.logging.get("suppress_xla_warnings", True):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NVIDIA_TF32_OVERRIDE"]  = "0"

    critic_type = str(cfg.network.critic_type)
    per_agent   = (critic_type == "agent_centric")
    actor_memory = bool(cfg.network.get("actor_memory", False))
    critic_memory = bool(cfg.network.get("critic_memory", False))
    recurrent   = actor_memory or critic_memory

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
    terminate_on_target_found = bool(cfg.env.get("terminate_on_target_found", False))
    autoreset_step   = _make_autoreset_step(env_step, reset, compute_reward, max_steps, hold_chain_for, terminate_on_target_found)
    autoreset_step_v = jax.jit(jax.vmap(autoreset_step))
    obs_fn_v         = jax.jit(jax.vmap(compute_obs))
    reset_v          = jax.jit(jax.vmap(reset))
    reset_s          = jax.jit(reset)
    env_step_jit       = jax.jit(env_step)
    compute_obs_jit    = jax.jit(compute_obs)
    compute_reward_jit = jax.jit(compute_reward)
    diag_decoder = None
    if (
        bool(cfg.logging.get("memory_diagnostic_probe", True))
        and bool(cfg.network.get("memory_comm_enabled", False))
        and bool(cfg.network.get("actor_memory", False))
        and cfg.env.get("map_names")
    ):
        from core.config import MAP_DIR
        from env.maps import MapDefinition
        diag_map = MapDefinition.load(MAP_DIR / f"{cfg.env.map_names[0]}.yaml", cell_size=1.0)
        if diag_map.maze_cell_cols and diag_map.maze_cell_rows:
            diag_decoder = {
                "cols": int(diag_map.maze_cell_cols),
                "rows": int(diag_map.maze_cell_rows),
                "w": float(diag_map.width) / int(diag_map.maze_cell_cols),
                "h": float(diag_map.height) / int(diag_map.maze_cell_rows),
                "W": np.zeros((int(cfg.network.tarmac_sig_dim) + int(cfg.network.tarmac_val_dim), int(diag_map.maze_cell_cols) * int(diag_map.maze_cell_rows)), dtype=np.float32),
                "b": np.zeros((int(diag_map.maze_cell_cols) * int(diag_map.maze_cell_rows),), dtype=np.float32),
            }

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
        memory_comm_every_k_steps = int(cfg.network.get("memory_comm_every_k_steps", 5)),
        tarmac_sig_dim = int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim = int(cfg.network.get("tarmac_val_dim", 128)),
        tarmac_include_self = bool(cfg.network.get("tarmac_include_self", True)),
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
        tarmac_sig_dim = int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim = int(cfg.network.get("tarmac_val_dim", 128)),
        actor_memory  = actor_memory,
        critic_memory = critic_memory,
    )

    _, params = nnx.split(model)
    n_params  = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model params : {n_params:,}")

    # ── Checkpoint Resumption ─────────────────────────────────────────────
    import orbax.checkpoint as ocp
    if cfg.training.checkpoint_path:
        checkpoint_path = Path(str(cfg.training.checkpoint_path).replace("\\", "/")).absolute()
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

    # Auto-disable evaluation target heatmaps if parallel evaluation is disabled
    if not bool(cfg.training.get("eval_parallel", False)):
        if bool(cfg.logging.get("eval_failed_chain_heatmap", False)) or bool(cfg.logging.get("eval_not_delivered_or_visually_found_heatmap", False)):
            print("\n  ⚠️  [WARNING] training.eval_parallel is False. Heatmap generation requires parallel evaluation.")
            print("               Automatically setting logging.eval_failed_chain_heatmap and logging.eval_not_delivered_or_visually_found_heatmap to False.\n")
            cfg.logging.eval_failed_chain_heatmap = False
            cfg.logging.eval_not_delivered_or_visually_found_heatmap = False

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
    eval_offset = int(cfg.logging.get("eval_offset", 0) or 0)
    eval_video_freq = cfg.logging.get("eval_video_freq", None)
    eval_video_every = int(eval_video_freq) if eval_video_freq is not None else eval_every
    eval_video_offset = int(cfg.logging.get("eval_video_offset", 0) or 0)

    checkpoint_freq = int(cfg.logging.get("checkpoint_freq", 50))
    checkpoint_offset = int(cfg.logging.get("checkpoint_offset", 0) or 0)
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
        "finder_return": np.zeros(E, dtype=np.float32),
    }

    # Sliding window for stable logging metrics
    window_ret  = deque(maxlen=E)
    window_len  = deque(maxlen=E)
    window_succ = deque(maxlen=E)
    window_fnd  = deque(maxlen=E)
    window_gap  = deque(maxlen=E)
    window_prog_pct = deque(maxlen=E)
    window_finder_return = deque(maxlen=E)

    window_r_cov   = deque(maxlen=E)
    window_r_gap   = deque(maxlen=E)
    window_r_coll  = deque(maxlen=E)
    window_r_prox  = deque(maxlen=E)
    window_r_found = deque(maxlen=E)
    window_r_succ  = deque(maxlen=E)
    window_cov     = deque(maxlen=E)
    window_diag_acc = deque(maxlen=E)
    window_diag_samples = deque(maxlen=E)

    completed_eps_count = 0

    start_update = 0
    loading_mode = cfg.training.get("ckpt_loading_mode", "branch").lower()
    # Backward compatibility fallback
    if cfg.training.get("resume_update", None) is not None:
        loading_mode = "resume" if bool(cfg.training.resume_update) else "branch"

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
             raw_r_cov, raw_r_gap, raw_r_coll, raw_r_prox, raw_r_found, raw_r_succ,
             raw_cov,
             raw_finder_return, raw_diag_memories, raw_diag_targets, raw_diag_valids) = _collect_rollout_mappo(
                states, model, buf, autoreset_step_v, obs_fn_v, batched_rollout_step_jit,
                collect_key, max_force, T, ep_trackers,
                actor_h, actor_signature, actor_value, critic_h, rollout_last_dones, base_signature, base_value, base_memory_valid,
            )

            # ── Update sliding window from COMPLETED episodes only ───────────────
            n_eps = len(raw_ret)
            diag_acc = None
            diag_samples = 0

            if n_eps > 0:
                completed_eps_count += n_eps
                window_ret.extend(raw_ret)
                window_len.extend(raw_len)
                window_succ.extend(raw_success)
                window_fnd.extend(raw_found)
                window_gap.extend(raw_gap)
                window_prog_pct.extend(raw_prog_pct)
                window_finder_return.extend(raw_finder_return)

                window_r_cov.extend(raw_r_cov)
                window_r_gap.extend(raw_r_gap)
                window_r_coll.extend(raw_r_coll)
                window_r_prox.extend(raw_r_prox)
                window_r_found.extend(raw_r_found)
                window_r_succ.extend(raw_r_succ)
                window_cov.extend(raw_cov)

                if diag_decoder is not None:
                    window_diag_samples.extend(raw_diag_valids)
                    if len(raw_diag_memories) > 0:
                        X = np.asarray(raw_diag_memories, dtype=np.float32)
                        pos = np.asarray(raw_diag_targets, dtype=np.float32)
                        cx = np.clip(np.floor(pos[:, 0] / diag_decoder["w"]).astype(np.int32), 0, diag_decoder["cols"] - 1)
                        cy = np.clip(np.floor(pos[:, 1] / diag_decoder["h"]).astype(np.int32), 0, diag_decoder["rows"] - 1)
                        y = cx * diag_decoder["rows"] + cy
                        logits = X @ diag_decoder["W"] + diag_decoder["b"]
                        pred = np.argmax(logits, axis=-1)
                        diag_acc = float(np.mean(pred == y))
                        diag_samples = int(len(y))
                        window_diag_acc.extend(pred == y)
                        logits = logits - logits.max(axis=-1, keepdims=True)
                        probs = np.exp(logits)
                        probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-8)
                        probs[np.arange(len(y)), y] -= 1.0
                        lr_diag = 1e-3
                        diag_decoder["W"] -= lr_diag * (X.T @ probs) / max(1, len(y))
                        diag_decoder["b"] -= lr_diag * probs.mean(axis=0)

            # -- Curriculum transition check (Training stats) -----------------
            curr_thresh = cfg.curriculum.get("success_threshold", None)
            curr_mode = cfg.curriculum.get("mode", "train")
            if curr_thresh is not None and curr_mode == "train":
                curr_metric = cfg.curriculum.get("metric", "success")
                if curr_metric == "target_found":
                    window_metric = window_fnd
                    metric_label = "Target found rate"
                else:
                    window_metric = window_succ
                    metric_label = "Success rate"

                if len(window_metric) == window_metric.maxlen and float(np.mean(window_metric)) >= float(curr_thresh):
                    print(
                        f"  [curriculum] Training {metric_label} {float(np.mean(window_metric)):.1%} >= "
                        f"threshold {curr_thresh:.1%} -- advancing to next level."
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
                        _save_checkpoint_history(early_ckpt, run_name, update, E, T, prior_history)
                        print(f"  [ckpt-early] saved -> {early_ckpt}")
                        early_ckpt_str = str(early_ckpt)
                    else:
                        print("  [ckpt-early] skipped (save_model=false)")

                    # ── Early-exit eval + video (same logic as final eval) ────────
                    if eval_video:
                        master_key, eval_key = jax.random.split(master_key)
                        _run_eval_with_render(
                            model=model, reset_s=reset_s, env_step_jit=env_step_jit,
                            compute_obs_jit=compute_obs_jit, compute_reward_jit=compute_reward_jit,
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
                _diag = f"{np.mean(window_diag_acc):>5.1%}" if (diag_decoder is not None and len(window_diag_acc) > 0) else " ----"

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
                    f"mem_probe={_diag}  "
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
                    "perf/global_step":         steps_done,
                }
                if len(window_ret) == window_ret.maxlen:
                    logs.update({
                        "train/ep_return":          float(np.mean(window_ret)),
                        "train/ep_length":          float(np.mean(window_len)),
                        "train/success_rate":       float(np.mean(window_succ)),
                        "train/target_found_rate":  float(np.mean(window_fnd)),
                        "train/chain_progress_pct": float(np.mean(window_prog_pct)),
                        "train/finder_return_to_target_after_delivery_rate": float(np.mean(window_finder_return)),
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
                    if len(window_diag_acc) > 0:
                        diag_samples_pct = float(np.mean(window_diag_samples))
                        logs.update({
                            "diagnostics/memory_target_cell_samples": diag_samples_pct,
                        })
                        if diag_samples_pct >= 0.7:
                            logs.update({
                                "diagnostics/memory_target_cell_accuracy": float(np.mean(window_diag_acc)),
                            })
                logs.update(comm_summary)
                wandb.log(logs, step=steps_done)

            is_eval_step = ((update - eval_offset) % eval_every == 0) and update > eval_offset
            is_video_step = (eval_video and ((update - eval_video_offset) % eval_video_every == 0) and update > eval_video_offset)

            # ── Mid-training eval + single video ─────────────────────────────
            if (is_eval_step or is_video_step) and update != n_updates:
                eval_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
                heatmaps_generated = False

                if bool(cfg.training.get("eval_parallel", False)):
                    if is_eval_step:
                        master_key, eval_key = jax.random.split(master_key)
                        (rets, lengths, gaps, progs, succs, fnds,
                         final_state, final_succs, final_delivered, final_visual) = _evaluate_parallel(
                            model, reset, env_step, compute_obs, compute_reward,
                            cfg, eval_key, num_envs=int(cfg.training.eval_parallel_envs),
                        )
                        eval_ret = float(jnp.mean(rets))
                        eval_len = float(jnp.mean(lengths))
                        eval_gap = float(jnp.mean(gaps))
                        eval_prog_pct = float(jnp.mean(progs))
                        eval_success = float(jnp.mean(succs))
                        eval_found = float(jnp.mean(fnds))
                        eval_cov = float(jnp.mean(final_state.coverage_grid.astype(jnp.float32)))

                        eval_prefix = "[EVAL]" + " " * 50
                        _l_eval = f"{eval_len:>6.0f}"
                        _cov_eval = f"{eval_cov:>5.1%}"
                        _f_eval = f"{eval_found:>5.1%}"
                        _prog_eval = f"{eval_prog_pct:>5.1f}%"
                        _s_eval = f"{eval_success:>5.1%}"

                        num_envs = int(cfg.training.eval_parallel_envs)
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

                        if wandb_run:
                            import wandb
                            wandb.log({
                                "eval/ep_return":           eval_ret,
                                "eval/ep_length":           eval_len,
                                "eval/chain_progress_pct":  eval_prog_pct,
                                "eval/ep_length_reduction": (1.0 - (eval_len / max_steps)) * 100.0,
                                "eval/success_rate":        eval_success,
                                "eval/target_found_rate":   eval_found,
                                "eval/map_coverage_pct":    eval_cov * 100.0,
                            }, step=steps_done)

                        # Generate mid-run evaluation heatmaps using parallel evaluation results
                        generate_any_heatmap = bool(
                            cfg.logging.get("eval_failed_chain_heatmap", False) or
                            cfg.logging.get("eval_not_delivered_or_visually_found_heatmap", False)
                        )
                        if generate_any_heatmap:
                            try:
                                from training.evaluate_pipeline import (
                                    load_map_data,
                                    render_and_save_failed_chain_heatmap,
                                    render_and_save_not_found_heatmap,
                                    render_and_save_merged_heatmap,
                                )
                                _, map_data, map_def = load_map_data(cfg)
                                
                                target_pos_arr = np.asarray(final_state.target_pos)
                                target_success_arr = np.asarray(final_succs)
                                target_delivered_arr = np.asarray(final_delivered)
                                target_visually_found_arr = np.asarray(final_visual)
                                
                                n_completed = len(target_success_arr)
                                run_timestamp = f"{eval_timestamp}_update_{update:06d}"
                                
                                # Create subfolders for heatmaps if not exist
                                chain_heatmaps_dir = train_video_dir / "chain_heatmaps"
                                found_heatmaps_dir = train_video_dir / "found_heatmaps"
                                chain_heatmaps_dir.mkdir(parents=True, exist_ok=True)
                                found_heatmaps_dir.mkdir(parents=True, exist_ok=True)

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
                                        video_dir=chain_heatmaps_dir,
                                        run_timestamp=run_timestamp,
                                        save_csv=False,
                                        save_png=True,
                                        total_episodes=n_completed
                                    )
                                    
                                # 2 & 3. Merged or Separate Target Found/Delivered Heatmaps
                                if cfg.logging.get("eval_not_delivered_or_visually_found_heatmap", False):
                                    split_in_two = bool(cfg.logging.get("eval_not_deliv_not_visual_splitt_in_two", False))
                                    if not split_in_two:
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
                                            video_dir=found_heatmaps_dir,
                                            run_timestamp=run_timestamp,
                                            total_episodes=n_completed
                                        )
                                    else:
                                        # 2. Delivered-to-base Heatmap
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
                                            video_dir=found_heatmaps_dir,
                                            run_timestamp=run_timestamp,
                                            filename_prefix="not_delivered_targets_heatmap",
                                            label="Not Delivered",
                                            total_episodes=n_completed
                                        )
                                        
                                        # 3. Visually Found Heatmap
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
                                            video_dir=found_heatmaps_dir,
                                            run_timestamp=run_timestamp,
                                            filename_prefix="not_visually_found_targets_heatmap",
                                            label="Not Visually Found",
                                            total_episodes=n_completed
                                        )
                                heatmaps_generated = True
                            except Exception as heatmap_err:
                                print(f"  [heatmap-error] Failed to render evaluation heatmaps: {heatmap_err}")

                        # ── Free GPU memory from parallel eval ──────────────
                        # The vmapped eval returns full EnvState for all envs
                        # plus the JIT-cached XLA executable.  Both consume
                        # significant GPU memory that the PPO backward pass
                        # needs.  Delete result arrays immediately and clear
                        # the XLA cache to prevent late OOM.
                        del rets, lengths, gaps, progs, succs, fnds, final_state, final_succs, final_delivered, final_visual
                        gc.collect()
                        jax.clear_caches()
                        jax.block_until_ready(jnp.asarray(0, dtype=jnp.int32))
                        time.sleep(1.0)

                        # Check for parallel evaluation early exit
                        eval_success_metric = eval_found if int(cfg.env.get("num_bases", 1)) == 0 else eval_success
                        early_exit_thresh = cfg.training.get("eval_parallel_early_exit_threshold", None)
                        if early_exit_thresh is not None and eval_success_metric >= float(early_exit_thresh):
                            print(
                                f"\n  [eval-early-exit] Evaluation success metric {eval_success_metric:.1%} >= "
                                f"threshold {early_exit_thresh:.1%} -- concluding training early."
                            )
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
                                _save_checkpoint_history(early_ckpt, run_name, update, E, T, prior_history)
                                print(f"  [ckpt-early] saved -> {early_ckpt}")
                                early_ckpt_str = str(early_ckpt)

                            if eval_video:
                                master_key, eval_key = jax.random.split(master_key)
                                _run_eval_with_render(
                                    model=model, reset_s=reset_s, env_step_jit=env_step_jit,
                                    compute_obs_jit=compute_obs_jit, compute_reward_jit=compute_reward_jit,
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
                                time.sleep(1.0)
                            return early_ckpt_str

                        # Check for curriculum transition (Eval stats, parallel eval)
                        curr_thresh = cfg.curriculum.get("success_threshold", None)
                        curr_mode = cfg.curriculum.get("mode", "train")
                        if curr_thresh is not None and curr_mode == "eval":
                            curr_metric = cfg.curriculum.get("metric", "success")
                            eval_val = eval_found if curr_metric == "target_found" else eval_success
                            metric_label = "target_found" if curr_metric == "target_found" else "success"
                            if eval_val >= float(curr_thresh):
                                print(
                                    f"\n  [curriculum-eval-exit] Evaluation metric '{metric_label}' {eval_val:.1%} >= "
                                    f"threshold {curr_thresh:.1%} -- advancing to next level."
                                )
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
                                    _save_checkpoint_history(early_ckpt, run_name, update, E, T, prior_history)
                                    print(f"  [ckpt-early] saved -> {early_ckpt}")
                                    early_ckpt_str = str(early_ckpt)

                                if eval_video:
                                    master_key, eval_key = jax.random.split(master_key)
                                    _run_eval_with_render(
                                        model=model, reset_s=reset_s, env_step_jit=env_step_jit,
                                        compute_obs_jit=compute_obs_jit, compute_reward_jit=compute_reward_jit,
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
                                    time.sleep(1.0)
                                return early_ckpt_str

                    if is_video_step:
                        master_key, video_key = jax.random.split(master_key)
                        (ep_states_list, ep_rewards_list, all_metrics_list,
                         _, _, _, _, _, _) = _evaluate(
                            model, reset_s,
                            env_step_jit, compute_obs_jit, compute_reward_jit,
                            cfg, video_key, num_episodes=1,
                        )
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
                        del ep_states_list, ep_rewards_list, all_metrics_list
                        _release_video_eval_trajectory()
                        time.sleep(1.0)
                else:
                    if is_eval_step or is_video_step:
                        master_key, eval_key = jax.random.split(master_key)
                        (ep_states_list, ep_rewards_list, all_metrics_list,
                         eval_ret, eval_len, eval_gap, eval_prog_pct, eval_success, eval_found) = _evaluate(
                            model, reset_s,
                            env_step_jit, compute_obs_jit, compute_reward_jit,
                            cfg, eval_key, num_episodes=1,
                        )

                        if is_video_step:
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
                            del ep_states_list, ep_rewards_list, all_metrics_list
                            _release_video_eval_trajectory()
                            time.sleep(1.0)

                        if is_eval_step:
                            print(
                                f"  [eval-train] update={update}  "
                                f"steps={steps_done:,}  "
                                f"ep_return={eval_ret:.2f}  "
                                f"chain={eval_prog_pct:.1f}%  "
                                f"success={eval_success:.1%}"
                            )
                            time.sleep(1.0)

                            # Check for curriculum transition (Eval stats, non-parallel eval)
                            curr_thresh = cfg.curriculum.get("success_threshold", None)
                            curr_mode = cfg.curriculum.get("mode", "train")
                            if curr_thresh is not None and curr_mode == "eval":
                                curr_metric = cfg.curriculum.get("metric", "success")
                                eval_val = eval_found if curr_metric == "target_found" else eval_success
                                metric_label = "target_found" if curr_metric == "target_found" else "success"
                                if eval_val >= float(curr_thresh):
                                    print(
                                        f"\n  [curriculum-eval-exit] Evaluation metric '{metric_label}' {eval_val:.1%} >= "
                                        f"threshold {curr_thresh:.1%} -- advancing to next level."
                                    )
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
                                        _save_checkpoint_history(early_ckpt, run_name, update, E, T, prior_history)
                                        print(f"  [ckpt-early] saved -> {early_ckpt}")
                                        early_ckpt_str = str(early_ckpt)

                                    if eval_video:
                                        master_key, eval_key = jax.random.split(master_key)
                                        _run_eval_with_render(
                                            model=model, reset_s=reset_s, env_step_jit=env_step_jit,
                                            compute_obs_jit=compute_obs_jit, compute_reward_jit=compute_reward_jit,
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
                                        time.sleep(1.0)
                                    return early_ckpt_str



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
                    model=model, reset_s=reset_s, env_step_jit=env_step_jit,
                    compute_obs_jit=compute_obs_jit, compute_reward_jit=compute_reward_jit,
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
            is_ckpt_step = ((update - ckpt_offset) % ckpt_every == 0)
            if save_model and not is_benchmark and is_ckpt_step:
                import orbax.checkpoint as ocp
                import shutil
                ckpt_path = (ckpt_dir / f"ckpt_{update:06d}").absolute()
                if ckpt_path.exists():
                    shutil.rmtree(ckpt_path)
                _, state_dict = nnx.split(model)
                checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
                checkpointer.save(str(ckpt_path), args=ocp.args.StandardSave(state_dict))
                _save_checkpoint_history(ckpt_path, run_name, update, E, T, prior_history)
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
            _save_checkpoint_history(final_ckpt, run_name, n_updates, E, T, prior_history)
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
    model, reset_s, env_step_jit, compute_obs_jit, compute_reward_jit,
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
            env_step_jit, compute_obs_jit, compute_reward_jit,
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
            env_step_jit, compute_obs_jit, compute_reward_jit,
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
            ep_states_list[idx] = None
            ep_rewards_list[idx] = None
            all_metrics_list[idx] = None
            _release_video_eval_trajectory()
        del ep_states_list, ep_rewards_list, all_metrics_list
        _release_video_eval_trajectory()

    _print_eval_stats(
        label=label,
        num_computed=eval_max_compute if selective else eval_render_videos,
        mean_ret=eval_ret,
        mean_len=eval_len,
        mean_prog=eval_prog_pct,
        success_rate=eval_success,
        found_rate=eval_found,
    )
