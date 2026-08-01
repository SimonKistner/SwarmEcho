"""
swarmecho/training/mappo_buffer.py
=====================================
Rollout buffer for agent-centric MAPPO.

Values, advantages, and returns have shape (T, E, N). Shared reward (E,) is
broadcast to every agent, so each agent sees the same team reward but has its
own value baseline.

All storage is numpy (CPU). JAX arrays are converted on add().
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import numpy as np


class MAPPOTransition(NamedTuple):
    obs:       np.ndarray   # (E, N, D)       local observations
    actions:   np.ndarray   # (E, N, A)       normalised actions in [-1, 1]
    log_probs: np.ndarray   # (E, N)
    values:    np.ndarray   # (E, N)
    rewards:   np.ndarray   # (E,)
    dones:     np.ndarray   # (E,)
    rnn_resets: Optional[np.ndarray] = None  # (E, N), recurrent path only
    comm_masks: Optional[np.ndarray] = None  # (E, N, N), TarMAC agent-agent mask
    active_masks: Optional[np.ndarray] = None  # (E, N), TarMAC active-agent mask
    base_signatures: Optional[np.ndarray] = None  # (E, S), saved base TarMAC signature
    base_values: Optional[np.ndarray] = None  # (E, V), saved base TarMAC value
    base_memory_masks: Optional[np.ndarray] = None  # (E, N), receivers that may hear base replay
    critic_obs: Optional[np.ndarray] = None  # (E, N, P), privileged critic only
    critic_map: Optional[np.ndarray] = None  # (E, ceil(H*W/8)), packed coverage


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
    recurrent  : True → minibatches preserve rollout time order
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
        recurrent:  bool  = False,
        hidden_dim: int   = 0,
        tarmac_sig_dim: int = 64,
        tarmac_val_dim: int = 128,
        actor_memory:  bool = False,
        critic_memory: bool = False,
        critic_obs_dim: int = 0,
        critic_map_shape: tuple[int, int] | None = None,
    ) -> None:
        self.T          = num_steps
        self.E          = num_envs
        self.N          = num_agents
        self.D          = obs_dim
        self.A          = act_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.recurrent  = recurrent
        self.hidden_dim = hidden_dim
        self.tarmac_sig_dim = tarmac_sig_dim
        self.tarmac_val_dim = tarmac_val_dim
        self.actor_memory = actor_memory
        self.critic_memory = critic_memory
        self.critic_obs_dim = critic_obs_dim
        self.critic_map_shape = critic_map_shape

        self._obs       = np.zeros((self.T, self.E, self.N, self.D), dtype=np.float32)
        self._critic_obs = (
            np.zeros((self.T, self.E, self.N, critic_obs_dim), dtype=np.float32)
            if critic_obs_dim else None
        )
        self._critic_maps = (
            np.zeros(
                (self.T, self.E, (critic_map_shape[0] * critic_map_shape[1] + 7) // 8),
                dtype=np.uint8,
            )
            if critic_map_shape else None
        )
        self._actions   = np.zeros((self.T, self.E, self.N, self.A), dtype=np.float32)
        self._log_probs = np.zeros((self.T, self.E, self.N),          dtype=np.float32)
        self._dones     = np.zeros((self.T, self.E),                   dtype=np.float32)

        self._rewards = np.zeros((self.T, self.E, self.N), dtype=np.float32)
        self._values = np.zeros((self.T, self.E, self.N), dtype=np.float32)

        self._rnn_resets = np.zeros((self.T, self.E, self.N), dtype=bool)
        self._comm_masks = np.zeros((self.T, self.E, self.N, self.N), dtype=bool)
        self._active_masks = np.zeros((self.T, self.E, self.N), dtype=bool)
        self._base_signatures = np.zeros((self.T, self.E, self.tarmac_sig_dim), dtype=np.float32)
        self._base_values = np.zeros((self.T, self.E, self.tarmac_val_dim), dtype=np.float32)
        self._base_memory_masks = np.zeros((self.T, self.E, self.N), dtype=bool)
        self._initial_actor_h = None
        self._initial_actor_signature = None
        self._initial_actor_value = None
        self._initial_critic_h = None

        self._ptr = 0

    def reset(self, actor_h=None, critic_h=None, actor_signature=None, actor_value=None) -> None:
        self._ptr = 0
        if self.recurrent:
            if self.actor_memory:
                self._initial_actor_h = np.asarray(actor_h, dtype=np.float32)
                if actor_signature is not None:
                    self._initial_actor_signature = np.asarray(actor_signature, dtype=np.float32)
                if actor_value is not None:
                    self._initial_actor_value = np.asarray(actor_value, dtype=np.float32)
            if self.critic_memory:
                self._initial_critic_h = np.asarray(critic_h, dtype=np.float32)

    def add(self, tr: MAPPOTransition) -> None:
        assert self._ptr < self.T, "Buffer full — call reset() first."
        self._obs[self._ptr]       = np.asarray(tr.obs)
        if self._critic_obs is not None:
            assert tr.critic_obs is not None
            self._critic_obs[self._ptr] = np.asarray(tr.critic_obs)
            assert tr.critic_map is not None
            self._critic_maps[self._ptr] = np.asarray(tr.critic_map, dtype=np.uint8)
        self._actions[self._ptr]   = np.asarray(tr.actions)
        self._log_probs[self._ptr] = np.asarray(tr.log_probs)
        self._values[self._ptr]    = np.asarray(tr.values)
        rew_arr = np.asarray(tr.rewards)
        if rew_arr.ndim == 1:
            # Received global rewards (E,) -> broadcast to (E, N)
            self._rewards[self._ptr] = np.broadcast_to(rew_arr[:, None], (self.E, self.N))
        else:
            # Received per-agent rewards (E, N) -> store directly
            self._rewards[self._ptr] = rew_arr
        
        self._dones[self._ptr]     = np.asarray(tr.dones)
        if self.recurrent and tr.rnn_resets is not None:
            self._rnn_resets[self._ptr] = np.asarray(tr.rnn_resets).astype(bool)
        if self.recurrent and tr.comm_masks is not None:
            self._comm_masks[self._ptr] = np.asarray(tr.comm_masks).astype(bool)
        if self.recurrent and tr.active_masks is not None:
            self._active_masks[self._ptr] = np.asarray(tr.active_masks).astype(bool)
        if self.recurrent and tr.base_signatures is not None:
            self._base_signatures[self._ptr] = np.asarray(tr.base_signatures, dtype=np.float32)
        if self.recurrent and tr.base_values is not None:
            self._base_values[self._ptr] = np.asarray(tr.base_values, dtype=np.float32)
        if self.recurrent and tr.base_memory_masks is not None:
            self._base_memory_masks[self._ptr] = np.asarray(tr.base_memory_masks).astype(bool)
        self._ptr += 1

    # ── GAE ─────────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_value: jax.Array,   # (E, N)
        last_done:  jax.Array,   # (E,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and value-target returns.

        Returns
        -------
        advantages : (T, E, N)
        returns    : (T, E, N)
        """
        last_value_np = np.asarray(last_value)   # (E, N)
        # ``last_done`` is retained for API compatibility.  This buffer stores
        # done *after* each transition, so the authoritative bootstrap mask for
        # every timestep (including the final one) is self._dones[t].
        _ = last_done

        advantages = np.zeros_like(self._values)

        gae = np.zeros((self.E, self.N), dtype=np.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_values      = last_value_np
            else:
                next_values      = self._values[t + 1]

            # dones[t] describes the transition from state_t to state_{t+1}.
            # Masking with dones[t + 1] leaks values and advantages across an
            # auto-reset boundary and suppresses the valid bootstrap one step
            # before that boundary.
            next_nonterminal = (1.0 - self._dones[t])[:, None]

            delta = (
                self._rewards[t]
                + self.gamma * next_values * next_nonterminal
                - self._values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + self._values
        return advantages.astype(np.float32), returns.astype(np.float32)

    # ── Minibatch sampling ───────────────────────────────────────────────────

    def get_minibatches(
        self,
        advantages:    np.ndarray,   # (T, E, N)
        returns:       np.ndarray,   # (T, E, N)
        n_minibatches: int,
        key:           jax.Array,
    ) -> list[dict]:
        """
        Flatten (T, E) → (T*E,), shuffle, split into n_minibatches.

        Minibatch dict keys:
            obs, actions, old_log_probs, old_values, advantages, returns

        Shapes:
            obs           : (MB, N, D)
            actions       : (MB, N, A)
            old_log_probs : (MB, N)
            old_values    : (MB, N)
            advantages    : (MB, N)
            returns       : (MB, N)
        """
        if self.recurrent:
            return self._get_sequence_minibatches(advantages, returns, n_minibatches, key)

        total = self.T * self.E

        def _unpack_maps(packed):
            if packed is None:
                return None
            size = self.critic_map_shape[0] * self.critic_map_shape[1]
            return np.unpackbits(packed, axis=-1, count=size).reshape(
                *packed.shape[:-1], *self.critic_map_shape
            ).astype(np.float32)

        def _flat(arr):
            # (T, E, ...) → (T*E, ...)
            return arr.reshape(total, *arr.shape[2:])

        obs_f       = _flat(self._obs)        # (B, N, D)
        critic_obs_f = _flat(self._critic_obs) if self._critic_obs is not None else obs_f
        critic_maps_f = _flat(self._critic_maps) if self._critic_maps is not None else None
        actions_f   = _flat(self._actions)    # (B, N, A)
        lp_f        = _flat(self._log_probs)  # (B, N)
        values_f    = _flat(self._values)     # (B, N)
        adv_f       = _flat(advantages)        # (B, N)
        returns_f   = _flat(returns)           # (B, N)

        # Normalise advantages over the whole batch
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        perm    = np.array(jax.random.permutation(key, total))
        mb_size = total // n_minibatches

        minibatches = []
        for i in range(n_minibatches):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            minibatches.append({
                # Keep minibatches host-resident. The trainer stages only the
                # minibatch currently being updated, which is essential for
                # semantic maps that are much larger than vector observations.
                "obs":           obs_f[idx],
                "critic_obs":    critic_obs_f[idx],
                "critic_map":    (
                    _unpack_maps(critic_maps_f[idx])
                    if critic_maps_f is not None else None
                ),
                "actions":       actions_f[idx],
                "old_log_probs": lp_f[idx],
                "old_values":    values_f[idx],
                "advantages":    adv_f[idx],
                "returns":       returns_f[idx],
            })
        return minibatches

    def _get_sequence_minibatches(
        self,
        advantages:    np.ndarray,
        returns:       np.ndarray,
        n_minibatches: int,
        key:           jax.Array,
    ) -> list[dict]:
        """
        Recurrent minibatches preserve the time axis and keep all agents in an
        environment together so the ACC can attend over the full team.
        """
        assert self.E % n_minibatches == 0, "num_envs must divide num_minibatches for recurrent MAPPO."

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        perm = np.array(jax.random.permutation(key, self.E))
        mb_envs = self.E // n_minibatches

        actor_h = self._initial_actor_h
        actor_signature = self._initial_actor_signature
        actor_value = self._initial_actor_value
        critic_h = self._initial_critic_h
        if actor_h is None:
            actor_h = np.zeros((self.E, self.N, self.hidden_dim), dtype=np.float32)
        if actor_signature is None:
            actor_signature = np.zeros((self.E, self.N, self.tarmac_sig_dim), dtype=np.float32)
        if actor_value is None:
            actor_value = np.zeros((self.E, self.N, self.tarmac_val_dim), dtype=np.float32)
        if critic_h is None:
            critic_h = np.zeros((self.E, self.N, self.hidden_dim), dtype=np.float32)

        minibatches = []
        for i in range(n_minibatches):
            idx = perm[i * mb_envs : (i + 1) * mb_envs]
            critic_maps = None
            if self._critic_maps is not None:
                size = self.critic_map_shape[0] * self.critic_map_shape[1]
                critic_maps = np.unpackbits(
                    self._critic_maps[:, idx], axis=-1, count=size
                ).reshape(self.T, len(idx), *self.critic_map_shape).astype(np.float32)
            minibatches.append({
                "obs":             self._obs[:, idx],
                "critic_obs":      (
                    self._critic_obs[:, idx] if self._critic_obs is not None else self._obs[:, idx]
                ),
                "critic_map":       critic_maps,
                "actions":          self._actions[:, idx],
                "old_log_probs":    self._log_probs[:, idx],
                "old_values":       self._values[:, idx],
                "advantages":       adv[:, idx],
                "returns":          returns[:, idx],
                "rnn_resets":       self._rnn_resets[:, idx],
                "initial_actor_h":   actor_h[idx],
                "initial_actor_signature": actor_signature[idx],
                "initial_actor_value": actor_value[idx],
                "initial_critic_h":  critic_h[idx],
                "comm_masks":        self._comm_masks[:, idx],
                "active_masks":      self._active_masks[:, idx],
                "base_signatures":   self._base_signatures[:, idx],
                "base_values":       self._base_values[:, idx],
                "base_memory_masks": self._base_memory_masks[:, idx],
            })
        return minibatches
