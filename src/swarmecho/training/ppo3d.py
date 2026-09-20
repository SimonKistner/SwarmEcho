"""Recurrent 3D PPO: explicit value units, active agents, and fixed-target diagnostics.

Rewards and buffer values/returns are raw. Only critic regression uses normalized
units. Diagnostic RNGs never consume rollout or optimizer entropy RNGs.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from swarmecho.env.critic3d import assemble_privileged_observations_3d


class RunningValueNormalizer(nnx.Module):
    """Pooled running target moments (parallel variance merge), checkpointed by NNX."""

    def __init__(self):
        self.mean = nnx.Variable(jnp.float32(0.0))
        self.variance = nnx.Variable(jnp.float32(1.0))
        self.count = nnx.Variable(jnp.float32(0.0))

    def normalize(self, value):
        return (value - self.mean[...]) / jnp.sqrt(jnp.maximum(self.variance[...], 1e-4))

    def denormalize(self, value):
        return value * jnp.sqrt(jnp.maximum(self.variance[...], 1e-4)) + self.mean[...]

    def update(self, count, mean, variance):
        old_count = self.count[...]
        total = old_count + count
        weight = count / jnp.maximum(total, 1.0)
        delta = mean - self.mean[...]
        merged_var = ((1.0 - weight) * self.variance[...] + weight * variance
                      + weight * (1.0 - weight) * delta ** 2)
        self.mean[...] = jnp.where(count > 0, self.mean[...] + weight * delta, self.mean[...])
        self.variance[...] = jnp.where(count > 0, jnp.maximum(merged_var, 0.0), self.variance[...])
        self.count[...] = total


def masked_mean(values, active):
    return jnp.sum(jnp.where(active, values, 0.0)) / jnp.maximum(jnp.sum(active), 1)


def replay(model, mb, key):
    """Replay complete sequences with exactly the stored communication masks."""
    def one_env(obs, resets, active, comm, base_sig, base_val, base_mask, h, sig, val):
        def step(carry, inputs):
            obs_t, reset_t, active_t, comm_t, bs, bv, bm = inputs
            h_t, sig_t, val_t, mu, log_std = model.actor.__call_team__(
                obs_t, *carry, reset=reset_t | ~active_t, comm_mask=comm_t,
                active=active_t, base_signature=bs, base_value=bv, base_memory_mask=bm,
            )
            return (h_t, sig_t, val_t), (mu, log_std)
        _, distribution = jax.lax.scan(
            step, (h, sig, val), (obs, resets, active, comm, base_sig, base_val, base_mask)
        )
        return distribution

    mu, log_std = jax.vmap(one_env, in_axes=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0))(
        mb["obs"], mb["rnn_resets"], mb["active_masks"], mb["comm_masks"],
        mb["base_signatures"], mb["base_values"], mb["base_memory_masks"],
        mb["initial_actor_h"], mb["initial_actor_signature"], mb["initial_actor_value"],
    )
    mu, log_std = jnp.swapaxes(mu, 0, 1), jnp.swapaxes(log_std, 0, 1)
    std = jnp.exp(log_std)
    actions = jnp.where(mb["active_masks"][..., None], mb["actions"], 0.0)
    log_probs = -0.5 * jnp.sum(
        ((actions - mu) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )
    gaussian = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)

    def correction(u):
        return jnp.sum(2 * (jnp.log(2.0) - u - jax.nn.softplus(-2 * u)), axis=-1)

    legacy = gaussian + correction(actions)
    # Fresh reparameterized samples: gradients include mean/std and tanh compression.
    sampled = mu + std * jax.random.normal(key, mu.shape)
    squashed = gaussian + correction(sampled)
    active = mb["active_masks"]
    critic_obs = mb["critic_obs"]
    if getattr(model, "privileged_3d", False):
        critic_obs = assemble_privileged_observations_3d(
            mb["obs"], mb["critic_agent_features"], mb["critic_global_features"], active,
            include_base_vector=model.privileged_3d_include_base_vector,
        )
    _, values = model.critic.values_sequence(
        critic_obs, mb["initial_critic_h"], mb["rnn_resets"] | ~active,
        deterministic=False, active=active,
    )
    return values, log_probs, legacy, gaussian, squashed


def loss(model, mb, key, *, actor_clip_eps, value_clip_eps, entropy_mode, vf_coef, ent_coef):
    values, log_probs, legacy, _, squashed = replay(model, mb, key)
    active = mb["active_masks"]
    targets, old = mb["returns"], mb["old_values"]
    if hasattr(model, "value_normalizer"):
        targets = model.value_normalizer.normalize(targets)
        old = model.value_normalizer.normalize(old)
    targets = jnp.where(active, targets, 0.0)
    old = jnp.where(active, old, 0.0)
    values = jnp.where(active, values, 0.0)
    log_ratio = jnp.where(active, log_probs - mb["old_log_probs"], 0.0)
    ratio = jnp.exp(log_ratio)
    advantages = jax.lax.stop_gradient(jnp.where(active, mb["advantages"], 0.0))
    policy = masked_mean(jnp.maximum(
        -advantages * ratio,
        -advantages * jnp.clip(ratio, 1 - actor_clip_eps, 1 + actor_clip_eps),
    ), active)
    errors = (values - targets) ** 2
    blocked = jnp.zeros_like(active)
    if value_clip_eps is not None:
        clipped = old + jnp.clip(values - old, -value_clip_eps, value_clip_eps)
        clipped_errors = (clipped - targets) ** 2
        blocked = (jnp.abs(values - old) > value_clip_eps) & (clipped_errors > errors)
        errors = jnp.maximum(errors, clipped_errors)
    value = 0.5 * masked_mean(errors, active)
    entropy = masked_mean(squashed if entropy_mode == "squashed" else legacy, active)
    total = policy + vf_coef * value - ent_coef * entropy
    return total, {
        "policy_loss": policy, "value_loss": value, "entropy": entropy, "total_loss": total,
        "approx_kl": masked_mean(ratio - 1 - log_ratio, active),
        "clip_fraction": masked_mean(jnp.abs(ratio - 1) > actor_clip_eps, active),
        "critic/value_clip_blocked_fraction": masked_mean(blocked, active),
        "count": jnp.sum(active),
    }


def step(model, optimizer, mb, key, **settings):
    def objective(m):
        return loss(m, mb, key, **settings)
    (_, stats), grads = nnx.value_and_grad(objective, has_aux=True)(model)
    optimizer.update(model, grads)
    return stats


def diagnostics(model, mb, key, target_mean):
    values, _, _, gaussian, squashed = replay(model, mb, key)
    if hasattr(model, "value_normalizer"):
        values = model.value_normalizer.denormalize(values)
    active = mb["active_masks"]
    target = mb["returns"]
    return {
        "count": jnp.sum(active),
        "error_sq": jnp.sum(jnp.where(active, (values - target) ** 2, 0.0)),
        "target_centered_sq": jnp.sum(jnp.where(active, (target - target_mean) ** 2, 0.0)),
        "gaussian": jnp.sum(jnp.where(active, gaussian, 0.0)),
        "squashed": jnp.sum(jnp.where(active, squashed, 0.0)),
        "saturation": jnp.sum(jnp.where(active[..., None],
            (jnp.abs(jnp.tanh(mb["actions"])) > 0.99).astype(jnp.float32), 0.0), axis=(0, 1, 2)),
    }


class PPO3DTrainer:
    def __init__(self, model, training):
        self.model = model
        self.num_epochs = training.num_epochs
        self.diagnostics_every = training.diagnostics_every
        self.seed = training.seed
        self.updates = 0
        self.optimizer = nnx.Optimizer(model, optax.chain(
            optax.clip_by_global_norm(training.max_grad_norm), optax.adam(training.lr)
        ), wrt=nnx.Param)
        self._step = nnx.jit(functools.partial(
            step, actor_clip_eps=training.actor_clip_eps, value_clip_eps=training.value_clip_eps,
            entropy_mode=training.entropy_mode, vf_coef=training.vf_coef, ent_coef=training.ent_coef,
        ))
        self._diagnostics = nnx.jit(diagnostics)

    def _measure(self, minibatches, target_mean):
        summed = None
        # Same diagnostic samples before/after; a separate namespace from training.
        root = jax.random.fold_in(jax.random.PRNGKey(self.seed), 0xD1A6)
        root = jax.random.fold_in(root, self.updates)
        for index, mb in enumerate(minibatches):
            stats = self._diagnostics(self.model, mb, jax.random.fold_in(root, index), target_mean)
            summed = stats if summed is None else jax.tree.map(jnp.add, summed, stats)
        return jax.device_get(summed)

    def update(self, minibatches):
        self.updates += 1
        # One compact transfer; skip empty batches without advancing Adam momentum.
        counts = jax.device_get(jnp.stack([jnp.sum(mb["active_masks"]) for mb in minibatches]))
        minibatches = [mb for mb, count in zip(minibatches, counts) if count > 0]
        if not minibatches:
            return {name: 0.0 for name in ("policy_loss", "value_loss", "entropy", "total_loss",
                    "approx_kl", "clip_fraction", "critic/value_clip_blocked_fraction")}
        count = sum(jnp.sum(mb["active_masks"]) for mb in minibatches)
        mean = sum(jnp.sum(jnp.where(mb["active_masks"], mb["returns"], 0.0))
                   for mb in minibatches) / jnp.maximum(count, 1)
        if hasattr(self.model, "value_normalizer"):
            # Merge batch moments before changing the normalizer. No inactive targets.
            variance = sum(jnp.sum(jnp.where(mb["active_masks"], (mb["returns"] - mean) ** 2, 0.0))
                           for mb in minibatches) / jnp.maximum(count, 1)
            self.model.value_normalizer.update(count, mean, variance)
        measure = (self.updates - 1) % self.diagnostics_every == 0
        before = self._measure(minibatches, mean) if measure else None
        summed = None
        root = jax.random.fold_in(jax.random.PRNGKey(self.seed), 0xE170)
        root = jax.random.fold_in(root, self.updates)
        for epoch in range(self.num_epochs):
            for index, mb in enumerate(minibatches):
                key = jax.random.fold_in(root, epoch * len(minibatches) + index)
                stats = self._step(self.model, self.optimizer, mb, key)
                count = stats["count"]
                weighted = {k: v if k == "count" else v * count for k, v in stats.items()}
                summed = weighted if summed is None else jax.tree.map(jnp.add, summed, weighted)
        host = jax.device_get(summed)
        denominator = max(float(host["count"]), 1.0)
        result = {k: float(v) / denominator for k, v in host.items() if k != "count"}
        if measure:
            after = self._measure(minibatches, mean)
            count = max(float(after["count"]), 1.0)
            mse = float(after["error_sq"]) / count
            target_variance = float(after["target_centered_sq"]) / count
            result.update({
                "critic/rmse_before": math.sqrt(float(before["error_sq"]) / count),
                "critic/rmse_after": math.sqrt(mse),
                "critic/relative_mse_after": mse / target_variance if target_variance > 1e-8 else float("nan"),
                "policy/gaussian_entropy": float(after["gaussian"]) / count,
                "policy/squashed_entropy": float(after["squashed"]) / count,
                **{f"policy/action_saturation_{axis}": float(after["saturation"][i]) / count
                   for i, axis in enumerate("xyz")},
            })
        return result
