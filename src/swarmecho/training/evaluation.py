"""Public parallel metric evaluation and deterministic video-episode collection."""

from __future__ import annotations

import functools
import gc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig

from swarmecho.models.mappo import MAPPOModel


@dataclass(frozen=True)
class ParallelEvaluationResult:
    """Outputs from one deterministic parallel evaluation batch."""

    returns: Any
    lengths: Any
    chain_gaps: Any
    chain_progress: Any
    successes: Any
    target_found: Any
    final_state: Any
    final_successes: Any
    final_delivered: Any
    final_visually_found: Any


def collect_video_episode(
    model: MAPPOModel,
    reset_fn,
    env_step_fn,
    obs_fn,
    reward_fn,
    cfg: DictConfig,
    key: jax.Array,
) -> tuple:
    """Collect exactly one deterministic episode trajectory for video rendering."""
    max_force = float(cfg.env.max_force)
    max_steps = int(cfg.env.max_steps)
    all_states, all_rewards, all_metrics = [], [], []

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

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    actor_h = model.initial_actor_hidden(()) if model.actor_memory else None
    use_memory_comm = model.actor_memory and model.memory_comm_enabled
    actor_signature = model.initial_actor_signature(()) if use_memory_comm else None
    actor_value = model.initial_actor_value(()) if use_memory_comm else None
    base_signature = (
        jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32)
        if use_memory_comm else None
    )
    base_value = (
        jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32)
        if use_memory_comm else None
    )
    base_memory_valid = jnp.bool_(False)
    ep_states, ep_rewards = [], []
    ep_metrics = {
        "r_explor": [],
        "r_gap": [],
        "r_coll": [],
        "chain_pct": [],
        "chain_gap": [],
        "r_total": [],
    }
    for step_index in range(max_steps):
        ep_states.append(jax.device_get(state))
        obs = obs_fn(state)
        physics = state.physics
        communication = state.communication

        if model.actor_memory:
            resets = jnp.logical_not(physics.active)
            if model.memory_comm_enabled:
                num_agents = obs.shape[0]
                share_now = (
                    int(model.memory_comm_every_k_steps) <= 1
                    or step_index % int(model.memory_comm_every_k_steps) == 0
                )
                raw_comm_mask = (
                    communication.adj_matrix[:num_agents, :num_agents]
                    if communication.adj_matrix.shape[-1]
                    else jnp.zeros((num_agents, num_agents), dtype=bool)
                )
                comm_mask = raw_comm_mask & jnp.asarray(share_now)
                base_mask = (
                    (
                        communication.adj_matrix[:num_agents, num_agents]
                        if communication.adj_matrix.shape[-1]
                        else jnp.zeros((num_agents,), dtype=bool)
                    )
                    & ~communication.target_known[:num_agents]
                    & base_memory_valid
                    & jnp.asarray(share_now)
                )
                (
                    actor_h,
                    actor_signature,
                    actor_value,
                    actions,
                    _,
                ) = act_team_fn(
                    obs,
                    actor_h,
                    actor_signature,
                    actor_value,
                    resets,
                    comm_mask,
                    physics.active,
                    base_signature,
                    base_value,
                    base_mask,
                )
                connected_to_base = (
                    communication.adj_matrix[:num_agents, num_agents]
                    if communication.adj_matrix.shape[-1]
                    else jnp.zeros((num_agents,), dtype=bool)
                )
                reporters = (
                    communication.target_known
                    & connected_to_base
                    & physics.active
                )
                first_index = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
                has_reporter = jnp.any(reporters)
                reported_signature = actor_signature[first_index]
                reported_value = actor_value[first_index]
                should_store = has_reporter & ~base_memory_valid
                base_signature = jnp.where(
                    should_store, reported_signature, base_signature
                )
                base_value = jnp.where(should_store, reported_value, base_value)
                base_memory_valid = base_memory_valid | should_store
            else:
                actor_h, actions = vmapped_act(obs, actor_h, resets)
        else:
            actions = vmapped_act(obs)

        actions = jnp.tanh(actions) * max_force
        old_state = state
        state = env_step_fn(state, actions)

        hold_chain_for = int(cfg.env.get("hold_chain_for", 0))
        rewards, info = reward_fn(old_state, state, jnp.bool_(False))
        fully_connected = info["fully_connected"] > 0.5
        held_steps = jnp.where(
            fully_connected,
            state.relay.chain_held_steps + jnp.int32(1),
            jnp.int32(0),
        )
        state = state.replace(
            relay=state.relay.replace(chain_held_steps=held_steps),
        )
        success_achieved = held_steps >= (hold_chain_for + 1)

        if bool(success_achieved):
            success_bonus_per_agent = (
                float(cfg.reward.success_bonus) / rewards.shape[0]
            )
            rewards = rewards + success_bonus_per_agent

        ep_rewards.append(np.array(rewards))
        ep_metrics["r_explor"].append(float(info["r_coverage"]))
        ep_metrics["r_gap"].append(float(info["r_chain_gap"]))
        ep_metrics["r_coll"].append(float(info["r_collision"]))
        ep_metrics["chain_pct"].append(float(info["chain_progress_pct"]))
        ep_metrics["chain_gap"].append(float(info["chain_gap_dist"]))
        ep_metrics["r_total"].append(float(rewards.sum()))
        target_found = bool(info["global_target_found"] > 0.5)
        terminate_on_target_found = bool(
            cfg.env.get("terminate_on_target_found", False)
        )
        episode_ended = success_achieved or (
            terminate_on_target_found and target_found
        )

        if bool(episode_ended):
            ep_states.append(jax.device_get(state))
            ep_metrics["r_explor"].append(ep_metrics["r_explor"][-1])
            ep_metrics["r_gap"].append(ep_metrics["r_gap"][-1])
            ep_metrics["r_coll"].append(ep_metrics["r_coll"][-1])
            ep_metrics["chain_pct"].append(ep_metrics["chain_pct"][-1])
            ep_metrics["chain_gap"].append(ep_metrics["chain_gap"][-1])
            ep_metrics["r_total"].append(float(rewards.sum()))
            break

    all_states.append(ep_states)
    all_rewards.append(ep_rewards)
    all_metrics.append({name: np.array(values) for name, values in ep_metrics.items()})
    return all_states, all_rewards, all_metrics


@functools.partial(
    nnx.jit,
    static_argnames=("reset", "env_step", "obs_fn", "reward_fn", "cfg", "num_envs"),
)
def _run_parallel_evaluation_jit(
    model: MAPPOModel,
    key: jax.Array,
    reset,
    env_step,
    obs_fn,
    reward_fn,
    cfg: DictConfig,
    num_envs: int = 4000,
) -> tuple:
    max_steps = int(cfg.env.max_steps)
    hold_chain_for = int(cfg.env.get("hold_chain_for", 0))
    actor_memory = bool(cfg.network.get("actor_memory", False))
    memory_comm_enabled = bool(cfg.network.get("memory_comm_enabled", False))
    memory_comm_every_k_steps = int(
        cfg.network.get("memory_comm_every_k_steps", 5)
    )
    max_force = float(cfg.env.max_force)
    env_keys = jax.random.split(key, num_envs)

    def run_single_environment(env_key):
        state = reset(env_key)
        actor_h = model.initial_actor_hidden(()) if actor_memory else None
        use_memory_comm = actor_memory and memory_comm_enabled
        actor_signature = (
            model.initial_actor_signature(()) if use_memory_comm else None
        )
        actor_value = model.initial_actor_value(()) if use_memory_comm else None
        base_signature = (
            jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32)
            if use_memory_comm else None
        )
        base_value = (
            jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32)
            if use_memory_comm else None
        )
        base_memory_valid = jnp.bool_(False) if use_memory_comm else None

        initial_carry = (
            state,
            actor_h,
            actor_signature,
            actor_value,
            base_signature,
            base_value,
            base_memory_valid,
            jnp.bool_(False),
            jnp.bool_(False),
            jnp.bool_(False),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.int32(0),
        )

        def evaluation_step(carry, step_index):
            (
                state,
                actor_h,
                actor_signature,
                actor_value,
                base_signature,
                base_value,
                base_memory_valid,
                has_succeeded,
                has_found_delivered,
                has_found_visual,
                max_found,
                returns,
                lengths,
            ) = carry
            obs = obs_fn(state)
            physics = state.physics
            communication = state.communication

            if actor_memory:
                resets = jnp.logical_not(physics.active)
                if memory_comm_enabled:
                    num_agents = obs.shape[0]
                    share_now = (memory_comm_every_k_steps <= 1) | (
                        (step_index % memory_comm_every_k_steps) == 0
                    )
                    raw_comm_mask = (
                        communication.adj_matrix[:num_agents, :num_agents]
                        if communication.adj_matrix.shape[-1]
                        else jnp.zeros((num_agents, num_agents), dtype=bool)
                    )
                    comm_mask = raw_comm_mask & share_now
                    base_mask = (
                        (
                            communication.adj_matrix[:num_agents, num_agents]
                            if communication.adj_matrix.shape[-1]
                            else jnp.zeros((num_agents,), dtype=bool)
                        )
                        & ~communication.target_known[:num_agents]
                        & base_memory_valid
                        & share_now
                    )
                    (
                        actor_h,
                        actor_signature,
                        actor_value,
                        actions,
                        _,
                    ) = model.actor.__call_team__(
                        obs,
                        actor_h,
                        actor_signature,
                        actor_value,
                        resets,
                        comm_mask,
                        physics.active,
                        base_signature,
                        base_value,
                        base_mask,
                    )
                    connected_to_base = (
                        communication.adj_matrix[:num_agents, num_agents]
                        if communication.adj_matrix.shape[-1]
                        else jnp.zeros((num_agents,), dtype=bool)
                    )
                    reporters = (
                        communication.target_known
                        & connected_to_base
                        & physics.active
                    )
                    first_index = jnp.argmax(
                        reporters.astype(jnp.int32), axis=-1
                    )
                    has_reporter = jnp.any(reporters)
                    reported_signature = actor_signature[first_index]
                    reported_value = actor_value[first_index]
                    should_store = has_reporter & ~base_memory_valid
                    base_signature = jnp.where(
                        should_store, reported_signature, base_signature
                    )
                    base_value = jnp.where(
                        should_store, reported_value, base_value
                    )
                    base_memory_valid = base_memory_valid | should_store
                else:
                    def _act_eval(observation, hidden, reset_agent):
                        hidden, mean, _ = model.actor(
                            observation, hidden, reset_agent
                        )
                        return hidden, mean

                    actor_h, actions = jax.vmap(_act_eval)(
                        obs, actor_h, resets
                    )
            else:
                actions = jax.vmap(lambda observation: model.actor(observation)[0])(
                    obs
                )

            actions = jnp.tanh(actions) * max_force
            old_state = state
            state = env_step(state, actions)
            rewards, info = reward_fn(old_state, state, jnp.bool_(False))

            fully_connected = info["fully_connected"] > 0.5
            held_steps = jnp.where(
                fully_connected,
                state.relay.chain_held_steps + jnp.int32(1),
                jnp.int32(0),
            )
            state = state.replace(
                relay=state.relay.replace(chain_held_steps=held_steps),
            )
            success_achieved = held_steps >= (hold_chain_for + 1)

            success_bonus_per_agent = (
                jnp.float32(cfg.reward.success_bonus) / rewards.shape[0]
            )
            rewards = jnp.where(
                success_achieved & ~has_succeeded,
                rewards + success_bonus_per_agent,
                rewards,
            )

            new_has_succeeded = has_succeeded | success_achieved
            new_has_found_delivered = (
                has_found_delivered | state.communication.base_target_known
            )
            new_has_found_visual = (
                has_found_visual
                | jnp.any(state.communication.target_known, axis=-1)
            )
            new_max_found = jnp.maximum(
                max_found, info["global_target_found"]
            )

            terminate_on_target_found = bool(
                cfg.env.get("terminate_on_target_found", False)
            )
            episode_ended_previous = has_succeeded | (
                jnp.bool_(terminate_on_target_found) & has_found_delivered
            )
            new_returns = jnp.where(
                episode_ended_previous, returns, returns + rewards.sum()
            )
            new_lengths = jnp.where(
                episode_ended_previous, lengths, lengths + 1
            )
            new_carry = (
                state,
                actor_h,
                actor_signature,
                actor_value,
                base_signature,
                base_value,
                base_memory_valid,
                new_has_succeeded,
                new_has_found_delivered,
                new_has_found_visual,
                new_max_found,
                new_returns,
                new_lengths,
            )
            return new_carry, (
                info["chain_gap_dist"],
                info["chain_progress_pct"],
            )

        final_carry, scan_outputs = jax.lax.scan(
            evaluation_step,
            initial_carry,
            jnp.arange(max_steps),
        )
        (
            final_state,
            _,
            _,
            _,
            _,
            _,
            _,
            success,
            delivered,
            visually_found,
            found,
            returns,
            lengths,
        ) = final_carry
        gap_distances, progress_percentages = scan_outputs
        return (
            returns,
            lengths,
            gap_distances[-1],
            progress_percentages[-1],
            success,
            found,
            final_state,
            success,
            delivered,
            visually_found,
        )

    return jax.vmap(run_single_environment)(env_keys)


def evaluate_parallel(
    model: MAPPOModel,
    reset,
    env_step,
    obs_fn,
    reward_fn,
    cfg: DictConfig,
    key: jax.Array,
    *,
    num_envs: int | None = None,
) -> ParallelEvaluationResult:
    """Run one deterministic metric batch through the maintained evaluator."""
    batch_size = (
        int(cfg.evaluation.eval_parallel_envs)
        if num_envs is None
        else int(num_envs)
    )
    values = _run_parallel_evaluation_jit(
        model,
        key,
        reset,
        env_step,
        obs_fn,
        reward_fn,
        cfg,
        batch_size,
    )
    return ParallelEvaluationResult(*values)


def release_video_evaluation_trajectory() -> None:
    """Synchronize and promptly release host-side video trajectory garbage."""
    gc.collect()
    jax.block_until_ready(jnp.asarray(0, dtype=jnp.int32))
