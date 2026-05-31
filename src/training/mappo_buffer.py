"""
swarmecho/training/mappo_buffer.py
=====================================
Rollout buffer for MAPPO — supports both critic variants.

per_agent=False  (GlobalMeanCritic)
  values shape : (T, E)         — one centralised value per env
  GAE output   : advantages/returns (T, E)

per_agent=True   (AgentCentricCritic)
  values shape : (T, E, N)      — one value per agent per env
  GAE output   : advantages/returns (T, E, N)
  Shared reward (E,) is broadcast to (E, 1) → each agent sees the same
  team reward but has its own value baseline → learns from its own
  geometric perspective on the global signal.

All storage is numpy (CPU). JAX arrays are converted on add().
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MAPPOTransition(NamedTuple):
    obs:       np.ndarray   # (E, N, D)       local observations
    actions:   np.ndarray   # (E, N, A)       normalised actions in [-1, 1]
    log_probs: np.ndarray   # (E, N)
    values:    np.ndarray   # (E,) or (E, N)  depends on critic type
    rewards:   np.ndarray   # (E,)
    dones:     np.ndarray   # (E,)
    next_obs:        np.ndarray | None = None  # (E, N, D)
    histories:       np.ndarray | None = None  # (E, N, H, D+A)
    role_ids:        np.ndarray | None = None  # (E, N)
    diversity_masks: np.ndarray | None = None  # (E, N), 0 across episode reset boundaries


class MAPPORolloutBuffer:
    """
    Fixed-length numpy rollout buffer for MAPPO.

    Parameters
    ----------
    num_steps  : rollout horizon T
    num_envs   : parallel environments E
    num_agents : agents per environment N
    obs_dim    : per-agent observation dimension D
    act_dim    : action dimension A
    gamma      : discount factor
    gae_lambda : GAE λ
    per_agent  : True → values shape (T,E,N); False → (T,E)
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
        per_agent:  bool  = True,
        diversity_enabled:     bool = False,
        history_len:           int  = 1,
        history_feature_dim:   int  = 0,
    ) -> None:
        self.T          = num_steps
        self.E          = num_envs
        self.N          = num_agents
        self.D          = obs_dim
        self.A          = act_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.per_agent  = per_agent
        self.diversity_enabled = diversity_enabled
        self.history_len = history_len
        self.history_feature_dim = history_feature_dim

        self._obs       = np.zeros((self.T, self.E, self.N, self.D), dtype=np.float32)
        self._actions   = np.zeros((self.T, self.E, self.N, self.A), dtype=np.float32)
        self._log_probs = np.zeros((self.T, self.E, self.N),          dtype=np.float32)
        self._dones     = np.zeros((self.T, self.E),                   dtype=np.float32)

        if per_agent:
            self._rewards = np.zeros((self.T, self.E, self.N), dtype=np.float32)
            self._values = np.zeros((self.T, self.E, self.N), dtype=np.float32)
        else:
            self._rewards = np.zeros((self.T, self.E), dtype=np.float32)
            self._values = np.zeros((self.T, self.E),  dtype=np.float32)

        if diversity_enabled:
            if history_len < 1 or history_feature_dim < 1:
                raise ValueError("history_len and history_feature_dim must be positive when diversity is enabled.")
            self._next_obs = np.zeros((self.T, self.E, self.N, self.D), dtype=np.float32)
            self._histories = np.zeros(
                (self.T, self.E, self.N, self.history_len, self.history_feature_dim),
                dtype=np.float32,
            )
            self._role_ids = np.zeros((self.T, self.E, self.N), dtype=np.int32)
            self._diversity_masks = np.zeros((self.T, self.E, self.N), dtype=np.float32)

        self._ptr = 0

    def reset(self) -> None:
        self._ptr = 0

    def add(self, tr: MAPPOTransition) -> None:
        assert self._ptr < self.T, "Buffer full — call reset() first."
        self._obs[self._ptr]       = np.asarray(tr.obs)
        self._actions[self._ptr]   = np.asarray(tr.actions)
        self._log_probs[self._ptr] = np.asarray(tr.log_probs)
        self._values[self._ptr]    = np.asarray(tr.values)
        if self.per_agent:
            rew_arr = np.asarray(tr.rewards)
            if rew_arr.ndim == 1:
                # Received global rewards (E,) -> broadcast to (E, N)
                self._rewards[self._ptr] = np.broadcast_to(rew_arr[:, None], (self.E, self.N))
            else:
                # Received per-agent rewards (E, N) -> store directly
                self._rewards[self._ptr] = rew_arr
        else:
            rew_arr = np.asarray(tr.rewards)
            if rew_arr.ndim == 2:
                # Received per-agent rewards -> sum for global critic
                self._rewards[self._ptr] = rew_arr.sum(axis=1)
            else:
                self._rewards[self._ptr] = rew_arr
        
        self._dones[self._ptr]     = np.asarray(tr.dones)
        if self.diversity_enabled:
            if tr.next_obs is None or tr.histories is None or tr.role_ids is None or tr.diversity_masks is None:
                raise ValueError("Diversity transitions must include next_obs, histories, role_ids, and diversity_masks.")
            self._next_obs[self._ptr] = np.asarray(tr.next_obs)
            self._histories[self._ptr] = np.asarray(tr.histories)
            self._role_ids[self._ptr] = np.asarray(tr.role_ids)
            self._diversity_masks[self._ptr] = np.asarray(tr.diversity_masks)
        self._ptr += 1

    # ── GAE ─────────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_value: jax.Array,   # (E,) or (E, N) matching per_agent
        last_done:  jax.Array,   # (E,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and value-target returns.

        Returns
        -------
        advantages : (T, E) or (T, E, N)
        returns    : (T, E) or (T, E, N)
        """
        last_value_np = np.asarray(last_value)   # (E,) or (E, N)
        last_done_np  = np.asarray(last_done)    # (E,)

        advantages = np.zeros_like(self._values)

        if self.per_agent:
            # gae: (E, N)
            gae = np.zeros((self.E, self.N), dtype=np.float32)
            # done mask needs to broadcast over N agents: (E,) → (E, 1)
            last_nonterminal = (1.0 - last_done_np)[:, None]   # (E, 1)

            for t in reversed(range(self.T)):
                if t == self.T - 1:
                    next_values      = last_value_np             # (E, N)
                    next_nonterminal = last_nonterminal          # (E, 1)
                else:
                    next_values      = self._values[t + 1]                        # (E, N)
                    next_nonterminal = (1.0 - self._dones[t + 1])[:, None]        # (E, 1)

                reward_t = self._rewards[t]                       # (E, N)

                delta = (
                    reward_t
                    + self.gamma * next_values * next_nonterminal
                    - self._values[t]
                )
                done_mask = (1.0 - self._dones[t])[:, None]      # (E, 1)
                gae       = delta + self.gamma * self.gae_lambda * next_nonterminal * gae * done_mask
                advantages[t] = gae

        else:
            # Scalar centralised value — original behaviour
            gae = np.zeros(self.E, dtype=np.float32)
            for t in reversed(range(self.T)):
                next_value = last_value_np if t == self.T - 1 else self._values[t + 1]
                next_done  = last_done_np  if t == self.T - 1 else self._dones[t + 1]

                delta = (
                    self._rewards[t]
                    + self.gamma * next_value * (1.0 - next_done)
                    - self._values[t]
                )
                gae = delta + self.gamma * self.gae_lambda * (1.0 - self._dones[t]) * gae
                advantages[t] = gae

        returns = advantages + self._values
        return advantages.astype(np.float32), returns.astype(np.float32)

    # ── Minibatch sampling ───────────────────────────────────────────────────

    def get_minibatches(
        self,
        advantages:    np.ndarray,   # (T, E) or (T, E, N)
        returns:       np.ndarray,   # (T, E) or (T, E, N)
        n_minibatches: int,
        key:           jax.Array,
    ) -> list[dict]:
        """
        Flatten (T, E) → (T*E,), shuffle, split into n_minibatches.

        Minibatch dict keys:
            obs, actions, old_log_probs, old_values, advantages, returns

        Shapes (per_agent=True):
            obs           : (MB, N, D)
            actions       : (MB, N, A)
            old_log_probs : (MB, N)
            old_values    : (MB, N)
            advantages    : (MB, N)
            returns       : (MB, N)

        Shapes (per_agent=False):
            old_values    : (MB,)
            advantages    : (MB,)
            returns       : (MB,)
        """
        total = self.T * self.E

        def _flat(arr):
            # (T, E, ...) → (T*E, ...)
            return arr.reshape(total, *arr.shape[2:])

        obs_f       = _flat(self._obs)        # (B, N, D)
        actions_f   = _flat(self._actions)    # (B, N, A)
        lp_f        = _flat(self._log_probs)  # (B, N)
        values_f    = _flat(self._values)     # (B, N) or (B,)
        adv_f       = _flat(advantages)        # (B, N) or (B,)
        returns_f   = _flat(returns)           # (B, N) or (B,)

        # Normalise advantages over the whole batch
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        perm    = np.array(jax.random.permutation(key, total))
        mb_size = total // n_minibatches

        minibatches = []
        for i in range(n_minibatches):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            minibatches.append({
                "obs":           jnp.array(obs_f[idx]),
                "actions":       jnp.array(actions_f[idx]),
                "old_log_probs": jnp.array(lp_f[idx]),
                "old_values":    jnp.array(values_f[idx]),
                "advantages":    jnp.array(adv_f[idx]),
                "returns":       jnp.array(returns_f[idx]),
            })
            if self.diversity_enabled:
                minibatches[-1].update({
                    "next_obs":        jnp.array(_flat(self._next_obs)[idx]),
                    "histories":       jnp.array(_flat(self._histories)[idx]),
                    "role_ids":        jnp.array(_flat(self._role_ids)[idx]),
                    "diversity_masks": jnp.array(_flat(self._diversity_masks)[idx]),
                })
        return minibatches
