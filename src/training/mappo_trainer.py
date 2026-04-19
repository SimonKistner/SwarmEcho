"""
swarmecho/training/mappo_trainer.py
=====================================
MAPPO loss function and trainer for 

Key differences from IPPO (ppo.py)
-----------------------------------
  - Actor loss   : uses per-agent log ratios, but SHARED team advantage
  - Value loss   : centralised value V(global_obs), one per env (not per agent)
  - PPOStats     : exposes per-agent clip_fraction for diagnostics

All JAX / Flax NNX patterns compatible with JIT.
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
    model:          MAPPOModel,
    obs:            jax.Array,   # (MB, N, D)         — local observations
    actions:        jax.Array,   # (MB, N, A)
    old_log_probs:  jax.Array,   # (MB, N)
    old_values:     jax.Array,   # (MB,)               — centralised
    advantages:     jax.Array,   # (MB,)               — same for all agents
    returns:        jax.Array,   # (MB,)               — centralised
    clip_eps:       float,
    vf_coef:        float,
    ent_coef:       float,
) -> tuple[jax.Array, MAPPOStats]:
    """
    MAPPO loss for one minibatch.

    Policy loss : PPO-clip on per-agent log ratios with SHARED advantage
    Value loss  : MSE on centralised critic
    Entropy     : mean across agents for the entropy bonus
    """
    MB, N, D = obs.shape

    # ── Actor path ─────────────────────────────────────────────────────────
    obs_flat     = obs.reshape(MB * N, D)
    actions_flat = actions.reshape(MB * N, -1)
    new_log_probs_flat, entropy_flat = model.actor.evaluate_actions(obs_flat, actions_flat)

    new_log_probs = new_log_probs_flat.reshape(MB, N)   # (MB, N)
    entropy       = entropy_flat.reshape(MB, N)          # (MB, N)

    # PPO clip — broadcast shared advantage to all agents: (MB,) → (MB, 1)
    adv = jax.lax.stop_gradient(advantages[:, None])
    log_ratio = new_log_probs - old_log_probs            # (MB, N)
    ratio     = jnp.exp(log_ratio)

    pg_loss1    = -adv * ratio
    pg_loss2    = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # ── Critic path ─────────────────────────────────────────────────────────
    # The Attentive Critic takes (MB, N, D) directly
    new_values = model.critic(obs, deterministic=False)   # (MB,)
    value_loss = jnp.mean((new_values - returns) ** 2)

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


# ---------------------------------------------------------------------------
# Module-level JIT-able step (avoids self-in-JIT issue)
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
    clip_eps: float,
    vf_coef:  float,
    ent_coef: float,
) -> tuple[jax.Array, MAPPOStats]:
    def loss_fn(m):
        return mappo_loss(
            m, obs, actions, old_log_probs, old_values,
            advantages, returns, clip_eps, vf_coef, ent_coef,
        )
    (loss, stats), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, stats


# ---------------------------------------------------------------------------
# MAPPO Trainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    """
    Manages MAPPO gradient updates over multiple epochs and minibatches.

    Parameters
    ----------
    model         : MAPPOModel
    lr            : Adam learning rate
    max_grad_norm : gradient clipping norm
    clip_eps      : PPO clip ratio
    vf_coef       : value loss weight
    ent_coef      : entropy bonus weight
    num_epochs    : PPO gradient epochs per rollout
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
    ) -> None:
        self.model      = model
        self.num_epochs = num_epochs

        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        step_fn = functools.partial(
            _mappo_step, clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef
        )
        self._jit_step = nnx.jit(step_fn)

    def update(self, minibatches: list[Dict[str, jax.Array]]) -> Dict[str, float]:
        """
        Run num_epochs × len(minibatches) gradient steps.

        Expects each minibatch dict to contain keys:
          obs, actions, old_log_probs, old_values, advantages, returns
        """
        all_stats: list[MAPPOStats] = []

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
                )
                all_stats.append(stats)

        return {
            "policy_loss":   float(jnp.mean(jnp.array([s.policy_loss   for s in all_stats]))),
            "value_loss":    float(jnp.mean(jnp.array([s.value_loss    for s in all_stats]))),
            "entropy":       float(jnp.mean(jnp.array([s.entropy       for s in all_stats]))),
            "total_loss":    float(jnp.mean(jnp.array([s.total_loss    for s in all_stats]))),
            "approx_kl":     float(jnp.mean(jnp.array([s.approx_kl    for s in all_stats]))),
            "clip_fraction": float(jnp.mean(jnp.array([s.clip_fraction for s in all_stats]))),
        }


