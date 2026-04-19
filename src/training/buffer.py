"""
swarmecho/training/buffer.py
=============================
Rollout buffer and GAE (Generalised Advantage Estimation) for PPO.

All data is stored as plain JAX arrays — no Python lists.
The buffer is filled step-by-step in pure Python during rollout collection,
then converted to a flat minibatch dictionary for PPO updates.

Shapes
------
T   = num_steps  (rollout horizon)
E   = num_envs
N   = num_agents
D   = obs_dim
A   = act_dim

Stored per transition:
    obs       : (T, E, N, D)
    actions   : (T, E, N, A)
    log_probs : (T, E, N)      — per-agent log-prob of action taken
    values    : (T, E, N)      — per-agent value estimate
    rewards   : (T, E)         — shared team reward (same for all agents)
    dones     : (T, E)         — episode-terminal flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Transition storage (one step)
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    obs:       jax.Array   # (E, N, D)
    actions:   jax.Array   # (E, N, A)
    log_probs: jax.Array   # (E, N)
    values:    jax.Array   # (E, N)
    rewards:   jax.Array   # (E,)
    dones:     jax.Array   # (E,)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-size rollout buffer for IPPO training.

    Parameters
    ----------
    num_steps     : rollout horizon T
    num_envs      : number of parallel environments E
    num_agents    : swarm size N
    obs_dim       : observation vector length D
    act_dim       : action vector length A
    gamma         : discount factor
    gae_lambda    : GAE smoothing λ
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

        # Pre-allocate NumPy arrays (faster indexing than JAX during collection)
        self._obs       = np.zeros((num_steps, num_envs, num_agents, obs_dim),  dtype=np.float32)
        self._actions   = np.zeros((num_steps, num_envs, num_agents, act_dim),  dtype=np.float32)
        self._log_probs = np.zeros((num_steps, num_envs, num_agents),           dtype=np.float32)
        self._values    = np.zeros((num_steps, num_envs, num_agents),           dtype=np.float32)
        self._rewards   = np.zeros((num_steps, num_envs),                       dtype=np.float32)
        self._dones     = np.zeros((num_steps, num_envs),                       dtype=np.float32)

        self._ptr = 0   # write pointer

    # -----------------------------------------------------------------------
    # Fill API
    # -----------------------------------------------------------------------

    def add(self, t: Transition) -> None:
        """Store one transition for all envs at step `_ptr`."""
        assert self._ptr < self.T, "Buffer is full! Call compute_gae() first."
        i = self._ptr
        self._obs[i]       = np.array(t.obs)
        self._actions[i]   = np.array(t.actions)
        self._log_probs[i] = np.array(t.log_probs)
        self._values[i]    = np.array(t.values)
        self._rewards[i]   = np.array(t.rewards)
        self._dones[i]     = np.array(t.dones)
        self._ptr += 1

    def reset(self) -> None:
        """Reset write pointer; data is overwritten on next fill."""
        self._ptr = 0

    # -----------------------------------------------------------------------
    # GAE computation
    # -----------------------------------------------------------------------

    def compute_gae(
        self,
        last_values: jax.Array,   # (E, N) — value of the state AFTER the rollout
        last_dones:  jax.Array,   # (E,)   — done flag of the state after the rollout
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.

        The team reward is broadcast over agents: each agent gets the same
        shared scalar reward, but has its own value estimate.

        Returns
        -------
        advantages : (T, E, N) float32
        returns    : (T, E, N) float32  (used as value targets)
        """
        T, E, N = self._obs.shape[:3]

        last_vals_np  = np.array(last_values)    # (E, N)
        last_dones_np = np.array(last_dones)      # (E,)

        advantages = np.zeros_like(self._values)  # (T, E, N)
        gae        = np.zeros((E, N), dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values  = last_vals_np
                next_nonterminal = 1.0 - last_dones_np[:, None]   # (E, 1) broadcast
            else:
                next_values  = self._values[t + 1]          # (E, N)
                next_nonterminal = 1.0 - self._dones[t + 1][:, None]

            # Expand reward (E,) -> (E, N) — shared reward for all agents
            reward_t = self._rewards[t][:, None]              # (E, 1)

            delta = reward_t + self.gamma * next_values * next_nonterminal - self._values[t]
            gae   = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + self._values   # value targets
        return advantages.astype(np.float32), returns.astype(np.float32)

    # -----------------------------------------------------------------------
    # Minibatch generation
    # -----------------------------------------------------------------------

    def get_minibatches(
        self,
        advantages:    np.ndarray,          # (T, E, N)
        returns:       np.ndarray,          # (T, E, N)
        num_minibatches: int,
        key:           jax.Array,
    ) -> list[Dict[str, jax.Array]]:
        """
        Flatten buffer and shuffle into `num_minibatches` minibatches.

        Each minibatch is a dict with keys:
            obs, actions, old_log_probs, old_values, advantages, returns
        All arrays have a leading batch dimension of size batch_size // num_minibatches.

        Advantages are normalised (zero mean, unit std) across the whole batch
        before splitting — standard PPO practice.
        """
        # Flatten T × E → B  (keep N and D)
        B  = self.T * self.E
        def _flat(x):
            return x.reshape(B, *x.shape[2:])

        obs_f         = _flat(self._obs)          # (B, N, D)
        actions_f     = _flat(self._actions)      # (B, N, A)
        log_probs_f   = _flat(self._log_probs)    # (B, N)
        values_f      = _flat(self._values)       # (B, N)
        advantages_f  = _flat(advantages)          # (B, N)
        returns_f     = _flat(returns)             # (B, N)

        # Normalise advantages
        adv_mean = advantages_f.mean()
        adv_std  = advantages_f.std() + 1e-8
        advantages_f = (advantages_f - adv_mean) / adv_std

        # Shuffle
        indices = np.random.permutation(B)
        mb_size = B // num_minibatches

        minibatches = []
        for start in range(0, B, mb_size):
            idx = indices[start : start + mb_size]
            mb  = {
                "obs":          jnp.array(obs_f[idx]),
                "actions":      jnp.array(actions_f[idx]),
                "old_log_probs":jnp.array(log_probs_f[idx]),
                "old_values":   jnp.array(values_f[idx]),
                "advantages":   jnp.array(advantages_f[idx]),
                "returns":      jnp.array(returns_f[idx]),
            }
            minibatches.append(mb)
        return minibatches


