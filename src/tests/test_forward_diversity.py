"""
Shape smoke test for the optional forward-history diversity path.

Run from the project root:
    uv run python src/tests/test_forward_diversity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from models.mappo import MAPPOModel
from training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from training.mappo_trainer import mappo_loss


cfg = load_config(cli_overrides=False)
validate_config(cfg)

OBS_DIM = compute_obs_dim(cfg)
ACT_DIM = compute_action_dim(cfg)
N = int(cfg.env.num_agents)
E = 2
T = 4
HISTORY_LEN = 3
HISTORY_FEATURE_DIM = OBS_DIM + ACT_DIM
NUM_ROLES = min(3, N)

rngs = nnx.Rngs(0)
model = MAPPOModel(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    num_agents=N,
    hidden_dim=32,
    num_layers=1,
    actor_num_layers=1,
    critic_type="agent_centric",
    rngs=rngs,
    diversity_enabled=True,
    num_roles=NUM_ROLES,
    history_len=HISTORY_LEN,
    history_feature_dim=HISTORY_FEATURE_DIM,
    diversity_hidden_dim=32,
    diversity_num_layers=1,
)

role_ids = jnp.arange(N, dtype=jnp.int32) % NUM_ROLES
obs = jnp.zeros((N, OBS_DIM), dtype=jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(0), N)
actions, log_probs, values = model.rollout_step(obs, keys, role_ids=role_ids)

assert actions.shape == (N, ACT_DIM)
assert log_probs.shape == (N,)
assert values.shape == (N,)

histories = jnp.zeros((E, N, HISTORY_LEN, HISTORY_FEATURE_DIM), dtype=jnp.float32)
next_obs = jnp.zeros((E, N, OBS_DIM), dtype=jnp.float32)
role_ids_b = jnp.broadcast_to(role_ids[None, :], (E, N))
masks = jnp.ones((E, N), dtype=jnp.float32)
intrinsic = model.diversity_intrinsic_reward(histories, next_obs, role_ids_b, masks)
div_loss, shared_loss, role_loss = model.diversity_loss(histories, next_obs, role_ids_b, masks)

assert intrinsic.shape == (E, N)
assert div_loss.shape == ()
assert shared_loss.shape == ()
assert role_loss.shape == ()

buf = MAPPORolloutBuffer(
    num_steps=T,
    num_envs=E,
    num_agents=N,
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    per_agent=True,
    diversity_enabled=True,
    history_len=HISTORY_LEN,
    history_feature_dim=HISTORY_FEATURE_DIM,
)

for _ in range(T):
    buf.add(MAPPOTransition(
        obs=np.zeros((E, N, OBS_DIM), dtype=np.float32),
        actions=np.zeros((E, N, ACT_DIM), dtype=np.float32),
        log_probs=np.zeros((E, N), dtype=np.float32),
        values=np.zeros((E, N), dtype=np.float32),
        rewards=np.zeros((E, N), dtype=np.float32),
        dones=np.zeros(E, dtype=np.float32),
        next_obs=np.zeros((E, N, OBS_DIM), dtype=np.float32),
        histories=np.zeros((E, N, HISTORY_LEN, HISTORY_FEATURE_DIM), dtype=np.float32),
        role_ids=np.broadcast_to(np.asarray(role_ids), (E, N)).astype(np.int32),
        diversity_masks=np.ones((E, N), dtype=np.float32),
    ))

advs, rets = buf.compute_gae(jnp.zeros((E, N)), jnp.zeros(E))
mb = buf.get_minibatches(advs, rets, 2, jax.random.PRNGKey(1))[0]

loss, stats = mappo_loss(
    model=model,
    obs=mb["obs"],
    actions=mb["actions"],
    old_log_probs=mb["old_log_probs"],
    old_values=mb["old_values"],
    advantages=mb["advantages"],
    returns=mb["returns"],
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    per_agent=True,
    role_ids=mb["role_ids"],
    histories=mb["histories"],
    next_obs=mb["next_obs"],
    diversity_masks=mb["diversity_masks"],
    diversity_enabled=True,
    diversity_aux_coef=1.0,
    diversity_l1_coef=0.001,
)

assert loss.shape == ()
assert jnp.isfinite(loss)
assert jnp.isfinite(stats.diversity_loss)
assert jnp.isfinite(stats.role_l1)

print("forward-history diversity smoke test passed")
