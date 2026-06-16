"""
swarmecho/training/mappo_trainer.py
=====================================
MAPPO loss function and trainer.

Supports two critic output shapes via the `per_agent` flag:

  per_agent=False  (GlobalMeanCritic):
    old_values  : (MB,)    — one centralised value per sample
    advantages  : (MB,)    — broadcast to all agents inside loss
    returns     : (MB,)

  per_agent=True   (AgentCentricCritic):
    old_values  : (MB, N)  — one value per agent
    advantages  : (MB, N)  — already per-agent from buffer GAE
    returns     : (MB, N)

In both cases value_loss = mean((new_values - returns)²) and the
jnp.mean() collapses all dimensions to a scalar automatically.
"""

from __future__ import annotations

import functools
from typing import Dict, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from models.mappo import MAPPOModel


# ---------------------------------------------------------------------------
# Update statistics
# ---------------------------------------------------------------------------

class MAPPOStats(NamedTuple):
    policy_loss:   jax.Array
    value_loss:    jax.Array
    entropy:       jax.Array
    total_loss:    jax.Array
    approx_kl:     jax.Array
    clip_fraction: jax.Array


# ---------------------------------------------------------------------------
# Single minibatch MAPPO loss
# ---------------------------------------------------------------------------

def mappo_loss(
    model:         MAPPOModel,
    obs:           jax.Array,   # (MB, N, D)
    actions:       jax.Array,   # (MB, N, A)  — normalised [-1, 1]
    old_log_probs: jax.Array,   # (MB, N)
    old_values:    jax.Array,   # (MB,) or (MB, N)
    advantages:    jax.Array,   # (MB,) or (MB, N)
    returns:       jax.Array,   # (MB,) or (MB, N)
    clip_eps:      float,
    vf_coef:       float,
    ent_coef:      float,
    per_agent:     bool,
) -> tuple[jax.Array, MAPPOStats]:
    """
    MAPPO loss for one minibatch.

    Policy loss : PPO-clip on per-agent log ratios
    Value loss  : MSE on critic output vs GAE returns
    Entropy     : mean across agents for the bonus
    """
    MB, N, D = obs.shape

    # ── Actor path ─────────────────────────────────────────────────────────
    obs_flat     = obs.reshape(MB * N, D)
    actions_flat = actions.reshape(MB * N, -1)
    new_log_probs_flat, entropy_flat = model.actor.evaluate_actions(obs_flat, actions_flat)

    new_log_probs = new_log_probs_flat.reshape(MB, N)   # (MB, N)
    entropy       = entropy_flat.reshape(MB, N)          # (MB, N)

    # ── Policy loss — PPO clip ──────────────────────────────────────────────
    if per_agent:
        # advantages already (MB, N) — use directly
        adv = jax.lax.stop_gradient(advantages)          # (MB, N)
    else:
        # shared advantage (MB,) — broadcast to all agents
        adv = jax.lax.stop_gradient(advantages[:, None]) # (MB, 1)

    log_ratio = new_log_probs - old_log_probs            # (MB, N)
    ratio     = jnp.exp(log_ratio)

    pg_loss1    = -adv * ratio
    pg_loss2    = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # ── Critic path ─────────────────────────────────────────────────────────
    # critic(obs) returns (MB, N) for agent_centric, (MB,) for global_mean
    new_values = model.critic(obs, deterministic=False)
    value_loss = jnp.mean((new_values - returns) ** 2)   # shape-agnostic

    # ── Entropy ─────────────────────────────────────────────────────────────
    mean_entropy = jnp.mean(entropy)

    # ── Total ───────────────────────────────────────────────────────────────
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy

    # ── Diagnostics ──────────────────────────────────────────────────────────
    approx_kl     = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    stats = MAPPOStats(
        policy_loss   = policy_loss,
        value_loss    = value_loss,
        entropy       = mean_entropy,
        total_loss    = total_loss,
        approx_kl     = approx_kl,
        clip_fraction = clip_fraction,
    )
    return total_loss, stats


def recurrent_mappo_loss(
    model:           MAPPOModel,
    obs:             jax.Array,   # (T, B, N, D)
    actions:         jax.Array,   # (T, B, N, A)
    old_log_probs:   jax.Array,   # (T, B, N)
    old_values:      jax.Array,   # (T, B) or (T, B, N)
    advantages:      jax.Array,   # (T, B) or (T, B, N)
    returns:         jax.Array,   # (T, B) or (T, B, N)
    rnn_resets:      jax.Array,   # (T, B, N)
    initial_actor_h: jax.Array,   # (B, N, H)
    initial_critic_h:jax.Array,   # (B, N, H)
    comm_masks:      jax.Array | None = None,
    active_masks:    jax.Array | None = None,
    base_memories:   jax.Array | None = None,
    base_memory_masks: jax.Array | None = None,
    clip_eps:        float = 0.2,
    vf_coef:         float = 0.5,
    ent_coef:        float = 0.01,
    per_agent:       bool = True,
) -> tuple[jax.Array, MAPPOStats]:
    """
    Recurrent MAPPO loss over full rollout sequences.

    Time order is preserved until after actor/critic replay. Loss terms are
    then averaged across time, environment minibatch, and agent dimensions.
    """
    T, B, N, D = obs.shape

    if model.actor_memory:
        if model.memory_comm_enabled:
            def _eval_env(obs_env, act_env, reset_env, init_h_env, comm_env, active_env, base_mem_env, base_mask_env):
                return model.actor.evaluate_actions_sequence(
                    obs_env,
                    act_env,
                    init_h_env,
                    reset_env,
                    comm_env,
                    active_env,
                    base_mem_env,
                    base_mask_env,
                )
            _, log_probs_flat, entropy_flat = jax.vmap(_eval_env, in_axes=(1, 1, 1, 0, 1, 1, 1, 1))(
                obs,
                actions,
                rnn_resets,
                initial_actor_h,
                comm_masks,
                active_masks,
                base_memories,
                base_memory_masks,
            )
            new_log_probs = jnp.swapaxes(log_probs_flat, 0, 1)
            entropy = jnp.swapaxes(entropy_flat, 0, 1)
        else:
            obs_actor = obs.reshape(T, B * N, D)
            actions_actor = actions.reshape(T, B * N, actions.shape[-1])
            resets_actor = rnn_resets.reshape(T, B * N)
            init_actor = initial_actor_h.reshape(B * N, model.hidden_dim)
            _, log_probs_flat, entropy_flat = model.actor.evaluate_actions_sequence(
                obs_actor,
                actions_actor,
                init_actor,
                resets_actor,
            )
            new_log_probs = log_probs_flat.reshape(T, B, N)
            entropy = entropy_flat.reshape(T, B, N)
    else:
        obs_flat = obs.reshape(T * B * N, D)
        actions_flat = actions.reshape(T * B * N, -1)
        lp_flat, ent_flat = model.actor.evaluate_actions(obs_flat, actions_flat)
        new_log_probs = lp_flat.reshape(T, B, N)
        entropy = ent_flat.reshape(T, B, N)

    if per_agent:
        adv = jax.lax.stop_gradient(advantages)
    else:
        adv = jax.lax.stop_gradient(advantages[..., None])

    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    if model.critic_memory:
        _, new_values = model.critic.values_sequence(
            obs,
            initial_critic_h,
            rnn_resets,
            deterministic=False,
        )
    else:
        flat_values = model.critic(obs.reshape(T * B, N, D), deterministic=False)
        new_values = flat_values.reshape(old_values.shape)

    clipped_values = old_values + jnp.clip(new_values - old_values, -clip_eps, clip_eps)
    value_losses = (new_values - returns) ** 2
    value_losses_clipped = (clipped_values - returns) ** 2
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

    mean_entropy = jnp.mean(entropy)
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    stats = MAPPOStats(
        policy_loss   = policy_loss,
        value_loss    = value_loss,
        entropy       = mean_entropy,
        total_loss    = total_loss,
        approx_kl     = approx_kl,
        clip_fraction = clip_fraction,
    )
    return total_loss, stats


# ---------------------------------------------------------------------------
# Module-level JIT-able step
# ---------------------------------------------------------------------------

def _mappo_step(
    model:         MAPPOModel,
    optimizer:     nnx.Optimizer,
    obs:           jax.Array,
    actions:       jax.Array,
    old_log_probs: jax.Array,
    old_values:    jax.Array,
    advantages:    jax.Array,
    returns:       jax.Array,
    *,
    clip_eps:  float,
    vf_coef:   float,
    ent_coef:  float,
    per_agent: bool,
) -> tuple[jax.Array, MAPPOStats]:
    def loss_fn(m):
        return mappo_loss(
            m, obs, actions, old_log_probs, old_values,
            advantages, returns, clip_eps, vf_coef, ent_coef, per_agent,
        )
    (loss, stats), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, stats


def _recurrent_mappo_step(
    model:           MAPPOModel,
    optimizer:       nnx.Optimizer,
    obs:             jax.Array,
    actions:         jax.Array,
    old_log_probs:   jax.Array,
    old_values:      jax.Array,
    advantages:      jax.Array,
    returns:         jax.Array,
    rnn_resets:      jax.Array,
    initial_actor_h: jax.Array,
    initial_critic_h:jax.Array,
    comm_masks:      jax.Array | None = None,
    active_masks:    jax.Array | None = None,
    base_memories:   jax.Array | None = None,
    base_memory_masks: jax.Array | None = None,
    *,
    clip_eps:  float,
    vf_coef:   float,
    ent_coef:  float,
    per_agent: bool,
) -> tuple[jax.Array, MAPPOStats]:
    def loss_fn(m):
        return recurrent_mappo_loss(
            m, obs, actions, old_log_probs, old_values,
            advantages, returns, rnn_resets,
            initial_actor_h, initial_critic_h,
            comm_masks, active_masks, base_memories, base_memory_masks,
            clip_eps, vf_coef, ent_coef, per_agent,
        )
    (loss, stats), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, stats


# ---------------------------------------------------------------------------
# MAPPOTrainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    """
    Manages MAPPO gradient updates over multiple epochs and minibatches.

    Parameters
    ----------
    model         : MAPPOModel
    lr            : Adam learning rate
    max_grad_norm : gradient clipping norm
    clip_eps      : PPO clip ratio ε
    vf_coef       : value loss weight
    ent_coef      : entropy bonus weight
    num_epochs    : PPO gradient epochs per rollout
    per_agent     : True if using AgentCentricCritic (advantages/returns are (MB, N))
    """

    def __init__(
        self,
        model:         MAPPOModel,
        lr:            float = 3e-4,
        max_grad_norm: float = 0.5,
        clip_eps:      float = 0.2,
        vf_coef:       float = 0.5,
        ent_coef:      float = 0.01,
        num_epochs:    int   = 4,
        per_agent:     bool  = True,
        actor_memory:  bool  = False,
        critic_memory: bool  = False,
    ) -> None:
        self.model      = model
        self.num_epochs = num_epochs
        self.per_agent  = per_agent
        self.recurrent  = actor_memory or critic_memory

        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        step_impl = _recurrent_mappo_step if self.recurrent else _mappo_step
        step_fn = functools.partial(
            step_impl,
            clip_eps  = clip_eps,
            vf_coef   = vf_coef,
            ent_coef  = ent_coef,
            per_agent = per_agent,
        )
        self._jit_step = nnx.jit(step_fn)

    def update(self, minibatches: list[Dict[str, jax.Array]]) -> Dict[str, float]:
        """
        Run num_epochs × len(minibatches) gradient steps.

        Feed-forward minibatches contain flat samples. Recurrent minibatches
        preserve time order and additionally include reset masks plus rollout
        initial actor/critic hidden states.
        """
        stats_sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_stat_steps = 0

        for _epoch in range(self.num_epochs):
            for mb in minibatches:
                _loss, stats = self._jit_step(
                    self.model,
                    self.optimizer,
                    mb["obs"],
                    mb["actions"],
                    mb["old_log_probs"],
                    mb["old_values"],
                    mb["advantages"],
                    mb["returns"],
                    *((
                        mb["rnn_resets"],
                        mb["initial_actor_h"],
                        mb["initial_critic_h"],
                        mb["comm_masks"],
                        mb["active_masks"],
                        mb["base_memories"],
                        mb["base_memory_masks"],
                    ) if self.recurrent else ()),
                )
                # Materialise the scalar diagnostics immediately instead of
                # keeping one device-resident stats tuple per PPO minibatch.
                # On large recurrent runs the queued update work can otherwise
                # accumulate until the final jnp.array([...]) conversion below,
                # making the stats readback the first place that trips a large
                # XLA allocation/OOM even though the diagnostics themselves are
                # tiny.
                stats_host = jax.device_get(stats)
                stats_sums["policy_loss"] += float(stats_host.policy_loss)
                stats_sums["value_loss"] += float(stats_host.value_loss)
                stats_sums["entropy"] += float(stats_host.entropy)
                stats_sums["total_loss"] += float(stats_host.total_loss)
                stats_sums["approx_kl"] += float(stats_host.approx_kl)
                stats_sums["clip_fraction"] += float(stats_host.clip_fraction)
                num_stat_steps += 1
                del _loss, stats, stats_host

        denom = max(1, num_stat_steps)
        return {
            "policy_loss":   stats_sums["policy_loss"] / denom,
            "value_loss":    stats_sums["value_loss"] / denom,
            "entropy":       stats_sums["entropy"] / denom,
            "total_loss":    stats_sums["total_loss"] / denom,
            "approx_kl":     stats_sums["approx_kl"] / denom,
            "clip_fraction": stats_sums["clip_fraction"] / denom,
        }
