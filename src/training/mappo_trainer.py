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
    diversity_loss:        jax.Array
    diversity_shared_loss: jax.Array
    diversity_role_loss:   jax.Array
    role_l1:               jax.Array


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
    role_ids:      jax.Array | None = None,
    histories:     jax.Array | None = None,
    next_obs:      jax.Array | None = None,
    diversity_masks: jax.Array | None = None,
    diversity_enabled: bool = False,
    diversity_aux_coef: float = 0.0,
    diversity_l1_coef: float = 0.0,
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
    role_ids_flat = role_ids.reshape(MB * N) if diversity_enabled else None
    new_log_probs_flat, entropy_flat = model.actor.evaluate_actions(obs_flat, actions_flat, role_ids_flat)

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
    diversity_loss = jnp.array(0.0)
    diversity_shared_loss = jnp.array(0.0)
    diversity_role_loss = jnp.array(0.0)
    role_l1 = jnp.array(0.0)
    if diversity_enabled:
        diversity_loss, diversity_shared_loss, diversity_role_loss = model.diversity_loss(
            histories, next_obs, role_ids, diversity_masks,
        )
        role_l1 = model.role_adapter_l1(obs_flat, role_ids_flat)

    total_loss = (
        policy_loss
        + vf_coef * value_loss
        - ent_coef * mean_entropy
        + diversity_aux_coef * diversity_loss
        + diversity_l1_coef * role_l1
    )

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
        diversity_loss        = diversity_loss,
        diversity_shared_loss = diversity_shared_loss,
        diversity_role_loss   = diversity_role_loss,
        role_l1               = role_l1,
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
    role_ids:      jax.Array,
    histories:     jax.Array,
    next_obs:      jax.Array,
    diversity_masks: jax.Array,
    *,
    clip_eps:  float,
    vf_coef:   float,
    ent_coef:  float,
    per_agent: bool,
    diversity_enabled: bool,
    diversity_aux_coef: float,
    diversity_l1_coef: float,
) -> tuple[jax.Array, MAPPOStats]:
    def loss_fn(m):
        return mappo_loss(
            m, obs, actions, old_log_probs, old_values,
            advantages, returns, clip_eps, vf_coef, ent_coef, per_agent,
            role_ids, histories, next_obs, diversity_masks,
            diversity_enabled, diversity_aux_coef, diversity_l1_coef,
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
        diversity_enabled: bool = False,
        diversity_aux_coef: float = 0.0,
        diversity_l1_coef: float = 0.0,
    ) -> None:
        self.model      = model
        self.num_epochs = num_epochs
        self.per_agent  = per_agent
        self.diversity_enabled = diversity_enabled

        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        step_fn = functools.partial(
            _mappo_step,
            clip_eps  = clip_eps,
            vf_coef   = vf_coef,
            ent_coef  = ent_coef,
            per_agent = per_agent,
            diversity_enabled = diversity_enabled,
            diversity_aux_coef = diversity_aux_coef,
            diversity_l1_coef = diversity_l1_coef,
        )
        self._jit_step = nnx.jit(step_fn)

    def update(self, minibatches: list[Dict[str, jax.Array]]) -> Dict[str, float]:
        """
        Run num_epochs × len(minibatches) gradient steps.

        Expects each minibatch dict to contain:
            obs, actions, old_log_probs, old_values, advantages, returns
        """
        all_stats: list[MAPPOStats] = []

        for _epoch in range(self.num_epochs):
            for mb in minibatches:
                if self.diversity_enabled:
                    role_ids = mb["role_ids"]
                    histories = mb["histories"]
                    next_obs = mb["next_obs"]
                    diversity_masks = mb["diversity_masks"]
                else:
                    role_ids = jnp.zeros((1,), dtype=jnp.int32)
                    histories = jnp.zeros((1,), dtype=jnp.float32)
                    next_obs = jnp.zeros((1,), dtype=jnp.float32)
                    diversity_masks = jnp.zeros((1,), dtype=jnp.float32)
                _loss, stats = self._jit_step(
                    self.model,
                    self.optimizer,
                    mb["obs"],
                    mb["actions"],
                    mb["old_log_probs"],
                    mb["old_values"],
                    mb["advantages"],
                    mb["returns"],
                    role_ids,
                    histories,
                    next_obs,
                    diversity_masks,
                )
                all_stats.append(stats)

        return {
            "policy_loss":   float(jnp.mean(jnp.array([s.policy_loss   for s in all_stats]))),
            "value_loss":    float(jnp.mean(jnp.array([s.value_loss    for s in all_stats]))),
            "entropy":       float(jnp.mean(jnp.array([s.entropy       for s in all_stats]))),
            "total_loss":    float(jnp.mean(jnp.array([s.total_loss    for s in all_stats]))),
            "approx_kl":     float(jnp.mean(jnp.array([s.approx_kl    for s in all_stats]))),
            "clip_fraction": float(jnp.mean(jnp.array([s.clip_fraction for s in all_stats]))),
            "diversity_loss": float(jnp.mean(jnp.array([s.diversity_loss for s in all_stats]))),
            "diversity_shared_loss": float(jnp.mean(jnp.array([s.diversity_shared_loss for s in all_stats]))),
            "diversity_role_loss": float(jnp.mean(jnp.array([s.diversity_role_loss for s in all_stats]))),
            "role_l1": float(jnp.mean(jnp.array([s.role_l1 for s in all_stats]))),
        }
