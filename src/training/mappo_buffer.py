"""
swarmecho/training/mappo_buffer.py
=====================================
Rollout buffer for MAPPO (Centralised Critic variant).

Key difference from the IPPO buffer
-------------------------------------
  - Stores `global_obs` (N*D per env) alongside agent-local `obs`
  - `values` shape is (T, E) — ONE centralised value per env, not (T, E, N)
  - GAE is computed on the centralised value stream
  - Advantages are shape (T, E) — broadcast to all N agents inside the loss

All storage is numpy (CPU). JAX arrays are converted on `add()`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MAPPOTransition(NamedTuple):
    obs:          np.ndarray   # (E, N, D)       local observations
    actions:      np.ndarray   # (E, N, A)
    log_probs:    np.ndarray   # (E, N)
    values:       np.ndarray   # (E,)             centralised value
    rewards:      np.ndarray   # (E,)
    dones:        np.ndarray   # (E,)


class MAPPORolloutBuffer:
    """
    Fixed-length numpy ring buffer for MAPPO transitions.

    Parameters
    ----------
    num_steps  : rollout horizon T
    num_envs   : parallel environments E
    num_agents : agents per environment N
    obs_dim    : per-agent observation dimension D
    act_dim    : action dimension A
    gamma      : discount factor
    gae_lambda : GAE λ
    """

    def __init__(
        self,
        num_steps:  int,
        num_envs:   int,
        num_agents: int,
        obs_dim:    int,
        act_dim:    int,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.T   = num_steps
        self.E   = num_envs
        self.N   = num_agents
        self.D   = obs_dim
        self.A   = act_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda

        self._obs         = np.zeros((self.T, self.E, self.N, self.D), dtype=np.float32)
        self._actions     = np.zeros((self.T, self.E, self.N, self.A), dtype=np.float32)
        self._log_probs   = np.zeros((self.T, self.E, self.N),          dtype=np.float32)
        self._values      = np.zeros((self.T, self.E),                  dtype=np.float32)
        self._rewards     = np.zeros((self.T, self.E),                  dtype=np.float32)
        self._dones       = np.zeros((self.T, self.E),                  dtype=np.float32)
        self._ptr = 0

    def reset(self) -> None:
        self._ptr = 0

    def add(self, tr: MAPPOTransition) -> None:
        assert self._ptr < self.T, "Buffer full — call reset() first."
        self._obs[self._ptr]        = np.asarray(tr.obs)
        self._actions[self._ptr]    = np.asarray(tr.actions)
        self._log_probs[self._ptr]  = np.asarray(tr.log_probs)
        self._values[self._ptr]     = np.asarray(tr.values)
        self._rewards[self._ptr]    = np.asarray(tr.rewards)
        self._dones[self._ptr]      = np.asarray(tr.dones)
        self._ptr += 1

    # ── GAE ─────────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_value: jax.Array,   # (E,) — centralised bootstrap value
        last_done:  jax.Array,   # (E,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.

        Returns
        -------
        advantages : (T, E)
        returns    : (T, E)
        """
        last_value = np.asarray(last_value)
        last_done  = np.asarray(last_done)

        advantages = np.zeros((self.T, self.E), dtype=np.float32)
        gae = np.zeros(self.E, dtype=np.float32)

        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self._values[t + 1]
            next_done  = last_done  if t == self.T - 1 else self._dones[t + 1]

            delta = (
                self._rewards[t]
                + self.gamma * next_value * (1.0 - next_done)
                - self._values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self._dones[t]) * gae
            advantages[t] = gae

        returns = advantages + self._values
        return advantages, returns

    # ── Minibatch sampling ───────────────────────────────────────────────────

    def get_minibatches(
        self,
        advantages: np.ndarray,   # (T, E)
        returns:    np.ndarray,   # (T, E)
        n_minibatches: int,
        key:           jax.Array,
    ) -> list[dict]:
        """
        Flatten (T, E) → (T*E,), shuffle, split into n_minibatches.

        Returns a list of dicts with JAX arrays ready for the update step.
        """
        T, E = self.T, self.E
        total = T * E

        # Normalise advantages over the whole batch
        adv_flat = advantages.reshape(total)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Shuffle
        perm = np.array(jax.random.permutation(key, total))
        mb_size = total // n_minibatches

        def _flat(arr):
            # arr shape: (T, E, ...) → (T*E, ...)
            return arr.reshape(total, *arr.shape[2:])

        obs_f        = _flat(self._obs)
        actions_f    = _flat(self._actions)
        log_probs_f  = _flat(self._log_probs)
        values_f     = self._values.reshape(total)
        returns_f    = returns.reshape(total)

        minibatches = []
        for i in range(n_minibatches):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            minibatches.append({
                "obs":           jnp.array(obs_f[idx]),
                "actions":       jnp.array(actions_f[idx]),
                "old_log_probs": jnp.array(log_probs_f[idx]),
                "old_values":    jnp.array(values_f[idx]),
                "advantages":    jnp.array(adv_flat[idx]),
                "returns":       jnp.array(returns_f[idx]),
            })
        return minibatches


