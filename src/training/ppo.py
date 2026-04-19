"""
swarmecho/training/ppo.py
==========================
PPO (clipped surrogate) update function for 

Operates on data from RolloutBuffer. The model's parameters are updated
using optax for gradient clipping + Adam.

All JAX / Flax NNX patterns used here are compatible with JIT.

Loss components
---------------
    L_total = L_policy + vf_coef × L_value - ent_coef × L_entropy

    L_policy  : PPO clipped-surrogate objective (minimised)
    L_value   : MSE between predicted value and GAE return targets
    L_entropy : Mean policy entropy (maximised → subtracted from total loss)
"""

from __future__ import annotations

import functools
from typing import Dict, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from models.actor_critic import ActorCritic


# ---------------------------------------------------------------------------
# Update statistics (returned per minibatch for logging)
# ---------------------------------------------------------------------------

class PPOStats(NamedTuple):
    policy_loss:   jax.Array
    value_loss:    jax.Array
    entropy:       jax.Array
    total_loss:    jax.Array
    approx_kl:     jax.Array   # approx KL between old and new policy
    clip_fraction: jax.Array   # fraction of samples that triggered the PPO clip


# ---------------------------------------------------------------------------
# Single minibatch loss
# ---------------------------------------------------------------------------

def ppo_loss(
    model:       ActorCritic,
    obs:         jax.Array,   # (MB, N, D)
    actions:     jax.Array,   # (MB, N, A)
    old_log_probs: jax.Array, # (MB, N)
    old_values:  jax.Array,   # (MB, N)
    advantages:  jax.Array,   # (MB, N)
    returns:     jax.Array,   # (MB, N)
    clip_eps:    float,
    vf_coef:     float,
    ent_coef:    float,
) -> tuple[jax.Array, PPOStats]:
    """
    Compute PPO loss on a single minibatch.

    Shapes: MB = minibatch size, N = num_agents, D = obs_dim, A = act_dim
    """
    MB, N, D = obs.shape

    # Flatten (MB, N, ...) → (MB*N, ...) so the model sees a flat batch
    obs_flat     = obs.reshape(MB * N, D)
    actions_flat = actions.reshape(MB * N, -1)

    log_probs_flat, values_flat, entropy_flat = model.evaluate_actions(
        obs_flat, actions_flat
    )

    # Reshape back to (MB, N)
    new_log_probs = log_probs_flat.reshape(MB, N)
    new_values    = values_flat.reshape(MB, N)
    entropy       = entropy_flat.reshape(MB, N)

    # ---- Policy loss (PPO-clip) ------------------------------------------
    log_ratio   = new_log_probs - old_log_probs          # (MB, N)
    ratio       = jnp.exp(log_ratio)                     # (MB, N)

    pg_loss1    = -advantages * ratio
    pg_loss2    = -advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # ---- Value loss (MSE with optional value clipping) -------------------
    value_loss  = jnp.mean((new_values - returns) ** 2)

    # ---- Entropy -----------------------------------------------------------
    mean_entropy = jnp.mean(entropy)

    # ---- Total loss --------------------------------------------------------
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy

    # ---- Diagnostics -------------------------------------------------------
    approx_kl     = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    stats = PPOStats(
        policy_loss   = policy_loss,
        value_loss    = value_loss,
        entropy       = mean_entropy,
        total_loss    = total_loss,
        approx_kl     = approx_kl,
        clip_fraction = clip_fraction,
    )

    return total_loss, stats


# ---------------------------------------------------------------------------
# Module-level JIT-able step function
# (module-level so nnx.jit sees model + optimizer as NNX modules, not self)
# ---------------------------------------------------------------------------

def _ppo_step(
    model:          ActorCritic,
    optimizer:      nnx.Optimizer,
    obs:            jax.Array,
    actions:        jax.Array,
    old_log_probs:  jax.Array,
    old_values:     jax.Array,
    advantages:     jax.Array,
    returns:        jax.Array,
    *,
    clip_eps: float,
    vf_coef:  float,
    ent_coef: float,
) -> tuple[jax.Array, PPOStats]:
    """Pure step function: computes loss, gradients, and applies optimizer."""
    def loss_fn(m):
        return ppo_loss(m, obs, actions, old_log_probs, old_values,
                        advantages, returns, clip_eps, vf_coef, ent_coef)
    (loss, stats), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, stats


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Handles PPO gradient updates over multiple epochs and minibatches.

    Parameters
    ----------
    model       : ActorCritic (Flax NNX) — will be mutated in-place
    lr          : Adam learning rate
    max_grad_norm : gradient clipping norm
    clip_eps    : PPO surrogate clip ratio
    vf_coef     : value loss weight
    ent_coef    : entropy bonus weight
    num_epochs  : PPO gradient epochs per rollout
    """

    def __init__(
        self,
        model:         ActorCritic,
        lr:            float = 3e-4,
        max_grad_norm: float = 0.5,
        clip_eps:      float = 0.2,
        vf_coef:       float = 0.5,
        ent_coef:      float = 0.01,
        num_epochs:    int   = 4,
    ) -> None:
        self.model      = model
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.num_epochs = num_epochs

        # Build optax transform
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        # JIT-compile the step: bake scalar hyperparams via partial so nnx.jit
        # only sees (model, optimizer, arrays) — all safely traceable.
        step_with_hparams = functools.partial(
            _ppo_step, clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef
        )
        self._jit_step = nnx.jit(step_with_hparams)

    def update(
        self,
        minibatches: list[Dict[str, jax.Array]],
    ) -> Dict[str, float]:
        """
        Run num_epochs × len(minibatches) gradient steps.

        Returns
        -------
        Mean statistics across all updates.
        """
        all_stats: list[PPOStats] = []

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

        # Average across all updates
        return {
            "policy_loss":   float(jnp.mean(jnp.array([s.policy_loss   for s in all_stats]))),
            "value_loss":    float(jnp.mean(jnp.array([s.value_loss    for s in all_stats]))),
            "entropy":       float(jnp.mean(jnp.array([s.entropy       for s in all_stats]))),
            "total_loss":    float(jnp.mean(jnp.array([s.total_loss    for s in all_stats]))),
            "approx_kl":     float(jnp.mean(jnp.array([s.approx_kl    for s in all_stats]))),
            "clip_fraction": float(jnp.mean(jnp.array([s.clip_fraction for s in all_stats]))),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import numpy as np
    from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
    from training.buffer import RolloutBuffer, Transition

    print("── PPO Self-Test ────────────────────────────────────────────")

    cfg     = load_config(cli_overrides=False)
    validate_config(cfg)
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N       = cfg.env.num_agents
    E       = 8       # small number of envs for test
    T       = cfg.training.num_steps
    MB_N    = cfg.training.num_minibatches

    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  N={N}  E={E}  T={T}  MB={MB_N}")

    # Build model
    rngs  = nnx.Rngs(0)
    model = ActorCritic(obs_dim, act_dim, cfg.network.hidden_dim, cfg.network.num_layers, rngs)

    # Build trainer
    trainer = PPOTrainer(
        model       = model,
        lr          = cfg.training.lr,
        max_grad_norm = cfg.training.max_grad_norm,
        clip_eps    = cfg.training.clip_eps,
        vf_coef     = cfg.training.vf_coef,
        ent_coef    = cfg.training.ent_coef,
        num_epochs  = cfg.training.num_epochs,
    )

    # Fill a buffer with random data
    buf = RolloutBuffer(T, E, N, obs_dim, act_dim, cfg.training.gamma, cfg.training.gae_lambda)
    key = jax.random.PRNGKey(42)
    for _ in range(T):
        key, k1, k2 = jax.random.split(key, 3)
        tr = Transition(
            obs       = jax.random.normal(k1, (E, N, obs_dim)),
            actions   = jax.random.uniform(k2, (E, N, act_dim), minval=-1, maxval=1),
            log_probs = jax.random.normal(k1, (E, N)),
            values    = jax.random.normal(k1, (E, N)),
            rewards   = jax.random.normal(k1, (E,)),
            dones     = jnp.zeros((E,)),
        )
        buf.add(tr)

    last_vals  = jnp.zeros((E, N))
    last_dones = jnp.zeros((E,))
    advs, rets = buf.compute_gae(last_vals, last_dones)
    print(f"  GAE advantages shape: {advs.shape}  ✓")

    mbs = buf.get_minibatches(advs, rets, MB_N, key)
    print(f"  Minibatches: {len(mbs)} × {mbs[0]['obs'].shape}  ✓")

    # One PPO update
    stats = trainer.update(mbs)
    print(f"  PPO update stats:")
    for k, v in stats.items():
        print(f"    {k}: {v:.4f}")

    print("\nPPO self-test passed ✓")


