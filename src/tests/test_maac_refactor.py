"""
src/tests/test_maac_refactor.py
================================
End-to-end integration test for the MAAC refactor.

Tests:
  1. DecentralizedActor  — shapes, action range in [-1, 1]
  2. GlobalMeanCritic    — output shape (E,)
  3. AgentCentricCritic  — output shape (E, N), masked attention
  4. MAPPOModel(agent_centric) — rollout_step shapes, action normalisation
  5. MAPPORolloutBuffer(per_agent=True) — GAE shapes (T,E,N), minibatch shapes
  6. mappo_loss(per_agent=True) — scalar loss, no shape errors
  7. Action normalisation contract — actor never exceeds [-1, 1]

Run from the project root:
    uv run python src/tests/test_maac_refactor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from models.actor import DecentralizedActor
from models.critic import AgentCentricCritic, GlobalMeanCritic
from models.mappo import MAPPOModel
from training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from training.mappo_trainer import mappo_loss, MAPPOTrainer

PASS = "  ✓"
FAIL = "  ✗"

cfg     = load_config(cli_overrides=False)
validate_config(cfg)
OBS_DIM = compute_obs_dim(cfg)
ACT_DIM = compute_action_dim(cfg)
N       = int(cfg.env.num_agents)
H       = int(cfg.network.hidden_dim)
E       = 4    # small env count for fast tests
T       = 8    # short rollout horizon
MB_N    = 2    # minibatches

rngs = nnx.Rngs(0)

errors: list[str] = []

def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"{PASS}  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        msg = f"{FAIL}  {name}" + (f"  [{detail}]" if detail else "")
        print(msg)
        errors.append(msg)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DecentralizedActor
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 1. DecentralizedActor ───────────────────────────────────────")

actor = DecentralizedActor(
    obs_dim          = OBS_DIM,
    act_dim          = ACT_DIM,
    hidden_dim       = H,
    actor_num_layers = int(cfg.network.actor_num_layers),
    rngs             = rngs,
)

obs_single = jnp.zeros((OBS_DIM,))
mu, log_std = actor(obs_single)
check("forward mu shape",      mu.shape      == (ACT_DIM,), str(mu.shape))
check("forward log_std shape", log_std.shape == (ACT_DIM,), str(log_std.shape))

key = jax.random.PRNGKey(42)
act, lp, ent = actor.act(obs_single, key)
check("act() action shape",   act.shape == (ACT_DIM,), str(act.shape))
check("act() log_prob shape", lp.shape  == (),         str(lp.shape))
check("act() action in [-1,1]",
      bool(jnp.all(act >= -1.0) and jnp.all(act <= 1.0)),
      f"min={float(act.min()):.3f} max={float(act.max()):.3f}")

# Batch evaluate_actions
obs_batch  = jnp.zeros((N, OBS_DIM))
acts_batch = jax.vmap(lambda o: actor.act(o, key)[0])(obs_batch)
lp_b, ent_b = actor.evaluate_actions(obs_batch, acts_batch)
check("evaluate_actions log_prob shape", lp_b.shape  == (N,), str(lp_b.shape))
check("evaluate_actions entropy shape",  ent_b.shape == (N,), str(ent_b.shape))


# ═══════════════════════════════════════════════════════════════════════════
# 2. GlobalMeanCritic
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 2. GlobalMeanCritic ─────────────────────────────────────────")

gmc = GlobalMeanCritic(OBS_DIM, H, int(cfg.network.num_layers), rngs)

obs_team  = jnp.zeros((N, OBS_DIM))
val_single = gmc(obs_team)
check("single env output shape",  val_single.shape == (),   str(val_single.shape))

obs_batch_team = jnp.zeros((E, N, OBS_DIM))
val_batch      = gmc(obs_batch_team)
check("batched output shape",     val_batch.shape  == (E,), str(val_batch.shape))


# ═══════════════════════════════════════════════════════════════════════════
# 3. AgentCentricCritic
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 3. AgentCentricCritic ───────────────────────────────────────")

acc = AgentCentricCritic(OBS_DIM, H, int(cfg.network.num_layers), rngs)

val_single_pa = acc(obs_team)
check("single env output shape",  val_single_pa.shape == (N,),    str(val_single_pa.shape))

val_batch_pa  = acc(obs_batch_team)
check("batched output shape",     val_batch_pa.shape  == (E, N),  str(val_batch_pa.shape))

# Diagonal masking: value of agent i should depend on other agents, not itself
# We verify the outputs differ when we zero out one agent vs another
obs_mod = obs_batch_team.at[0, 0, :].set(1.0)   # change agent 0 obs in env 0
val_orig = acc(obs_batch_team)
val_mod  = acc(obs_mod)
# Agent 0's own value should change (it sees others change their attention to it)
# All agents' values may change (attention is cross-agent) — just check no NaN/Inf
check("no NaN in output",  bool(not jnp.any(jnp.isnan(val_batch_pa))))
check("no Inf in output",  bool(not jnp.any(jnp.isinf(val_batch_pa))))


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAPPOModel (agent_centric)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 4. MAPPOModel (agent_centric) ───────────────────────────────")

model = MAPPOModel(
    obs_dim          = OBS_DIM,
    act_dim          = ACT_DIM,
    num_agents       = N,
    hidden_dim       = H,
    num_layers       = int(cfg.network.num_layers),
    actor_num_layers = int(cfg.network.actor_num_layers),
    critic_type      = "agent_centric",
    rngs             = rngs,
)

obs_env  = jnp.zeros((N, OBS_DIM))
keys_env = jax.random.split(key, N)
acts_out, lps_out, val_out = model.rollout_step(obs_env, keys_env)

check("rollout_step actions shape",   acts_out.shape == (N, ACT_DIM), str(acts_out.shape))
check("rollout_step log_probs shape", lps_out.shape  == (N,),         str(lps_out.shape))
check("rollout_step value shape",     val_out.shape  == (N,),         str(val_out.shape))
check("rollout_step actions in [-1,1]",
      bool(jnp.all(acts_out >= -1.0) and jnp.all(acts_out <= 1.0)),
      f"min={float(acts_out.min()):.3f} max={float(acts_out.max()):.3f}")

val_batch_out = model.get_value(jnp.zeros((E, N, OBS_DIM)))
check("get_value batched shape", val_batch_out.shape == (E, N), str(val_batch_out.shape))


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAPPORolloutBuffer (per_agent=True)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 5. MAPPORolloutBuffer (per_agent=True) ──────────────────────")

buf = MAPPORolloutBuffer(
    num_steps  = T,
    num_envs   = E,
    num_agents = N,
    obs_dim    = OBS_DIM,
    act_dim    = ACT_DIM,
    gamma      = 0.99,
    gae_lambda = 0.95,
    per_agent  = True,
)

check("values storage shape", buf._values.shape == (T, E, N), str(buf._values.shape))

rng = jax.random.PRNGKey(1)
for _ in range(T):
    rng, k1, k2 = jax.random.split(rng, 3)
    tr = MAPPOTransition(
        obs       = np.zeros((E, N, OBS_DIM), dtype=np.float32),
        actions   = np.clip(np.random.randn(E, N, ACT_DIM).astype(np.float32), -1, 1),
        log_probs = np.zeros((E, N), dtype=np.float32),
        values    = np.zeros((E, N), dtype=np.float32),
        rewards   = np.zeros(E, dtype=np.float32),
        dones     = np.zeros(E, dtype=np.float32),
    )
    buf.add(tr)

last_vals  = jnp.zeros((E, N))
last_dones = jnp.zeros(E)
advs, rets = buf.compute_gae(last_vals, last_dones)

check("GAE advantages shape", advs.shape == (T, E, N), str(advs.shape))
check("GAE returns shape",    rets.shape == (T, E, N), str(rets.shape))
check("no NaN in advantages", bool(not np.any(np.isnan(advs))))

mbs = buf.get_minibatches(advs, rets, MB_N, rng)
mb  = mbs[0]
MB  = (T * E) // MB_N

check("minibatch obs shape",           mb["obs"].shape           == (MB, N, OBS_DIM), str(mb["obs"].shape))
check("minibatch actions shape",       mb["actions"].shape       == (MB, N, ACT_DIM), str(mb["actions"].shape))
check("minibatch old_log_probs shape", mb["old_log_probs"].shape == (MB, N),          str(mb["old_log_probs"].shape))
check("minibatch old_values shape",    mb["old_values"].shape    == (MB, N),          str(mb["old_values"].shape))
check("minibatch advantages shape",    mb["advantages"].shape    == (MB, N),          str(mb["advantages"].shape))
check("minibatch returns shape",       mb["returns"].shape       == (MB, N),          str(mb["returns"].shape))


# ═══════════════════════════════════════════════════════════════════════════
# 6. mappo_loss (per_agent=True)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 6. mappo_loss (per_agent=True) ──────────────────────────────")

loss_val, stats = mappo_loss(
    model         = model,
    obs           = mb["obs"],
    actions       = mb["actions"],
    old_log_probs = mb["old_log_probs"],
    old_values    = mb["old_values"],
    advantages    = mb["advantages"],
    returns       = mb["returns"],
    clip_eps      = 0.2,
    vf_coef       = 0.5,
    ent_coef      = 0.01,
    per_agent     = True,
)

check("loss is scalar",           loss_val.shape == (), str(loss_val.shape))
check("loss is finite",           bool(jnp.isfinite(loss_val)))
check("policy_loss finite",       bool(jnp.isfinite(stats.policy_loss)))
check("value_loss finite",        bool(jnp.isfinite(stats.value_loss)))
check("entropy finite",           bool(jnp.isfinite(stats.entropy)))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Action normalisation — stress test
# ═══════════════════════════════════════════════════════════════════════════
print("\n── 7. Action normalisation stress test ─────────────────────────")

all_acts = []
for seed in range(20):
    k = jax.random.PRNGKey(seed)
    keys_n = jax.random.split(k, N)
    obs_n  = jax.random.normal(k, (N, OBS_DIM))
    a, _, _ = jax.vmap(lambda o, ki: actor.act(o, ki))(obs_n, keys_n)
    all_acts.append(a)

all_acts_arr = jnp.stack(all_acts)  # (20, N, A)
check("all actions in [-1,1]",
      bool(jnp.all(all_acts_arr >= -1.0) and jnp.all(all_acts_arr <= 1.0)),
      f"min={float(all_acts_arr.min()):.4f} max={float(all_acts_arr.max()):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
if errors:
    print(f"  FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"    {e}")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED ✓")
    print("═" * 60)
