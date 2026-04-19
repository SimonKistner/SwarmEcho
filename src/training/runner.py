"""
swarmecho/training/runner.py
==============================
Main MAPPO/IPPO training loop for 

Auto-reset design
-----------------
Rather than a gymnasium-style wrapper, we integrate auto-reset directly into
the rollout step (_make_autoreset_step).  When an episode terminates (time
limit *or* success), the state is transparently replaced with a fresh reset
state before being returned.  From the buffer's perspective each transition is
stored as (obs, action, reward, done, ...) where done=True marks a boundary.
GAE correctly bootstraps at episode boundaries via the done flag.

Episode tracking
----------------
Inside the rollout we maintain per-env accumulators:
  ep_ret_accum, ep_len_accum, ep_success_accum, ep_found_accum, ep_gap_accum

When done fires for env e we append the completed episode's stats to lists.
After the rollout we average over all completed episodes; if none completed in
this window (long episodes) we fall back to the partial accumulators so we
always have *something* to log.

W&B x-axis
-----------
All wandb.log() calls use step=steps_done (environment timesteps), not the PPO
update index.

Run organisation
-----------------
    outputs/{run_name}_{timestamp}/
        checkpoints/ckpt_{update:06d}/
        videos/eval_update_{update:06d}_{timestamp}.mp4
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from core.config import compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from models.actor_critic import ActorCritic
from training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from training.mappo_trainer import MAPPOTrainer
from training.buffer import RolloutBuffer, Transition
from training.ppo import PPOTrainer
from training.video_worker import VideoRenderWorker


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig, run_name: str) -> Optional[object]:
    mode = cfg.logging.get("wandb_mode", "disabled")
    if mode == "disabled":
        return None

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    import wandb
    run = wandb.init(
        project = cfg.logging.get("wandb_project", "swarmecho"),
        entity  = cfg.logging.get("wandb_entity", None),
        name    = run_name,
        mode    = mode,
        config  = OmegaConf.to_container(cfg, resolve=True),
    )
    return run


# ---------------------------------------------------------------------------
# Auto-reset step factory
# ---------------------------------------------------------------------------

def _make_autoreset_step(env_step_fn, reset_fn, reward_fn, max_steps: int):
    """
    Wrap env_step to auto-reset on episode termination.

    Done conditions (either triggers reset):
      • time_up        : new_state.step >= max_steps
      • fully_connected: the chain is closed (success)

    The success bonus (+success_bonus) fires when done=True AND fully_connected,
    because we pass `done` as the `is_done` flag to compute_reward.

    Approach to avoid double reward computation:
      1. call reward_fn(is_done=False) → get fully_connected from info
      2. derive done = time_up | fully_connected
      3. if success without time_up, add the terminal bonus separately
         (extracted from the info dict that already has fully_connected)

    Returns
    -------
    (next_state, reward, done, info)
      next_state : already reset if done; otherwise new_state
      done       : (,) bool scalar
    """
    max_steps_jnp = jnp.int32(max_steps)

    def step(state, actions):
        new_state = env_step_fn(state, actions)

        time_up = new_state.step >= max_steps_jnp

        # Compute reward with is_done=time_up to get all info including
        # fully_connected; success bonus fires when time_up & connected
        reward, info = reward_fn(state, new_state, time_up)

        # Actual done: time limit OR chain closed
        fully_connected = info["fully_connected"] > jnp.float32(0.5)
        done = time_up | fully_connected

        # If chain closed mid-episode (not at time limit), we didn't
        # include the success bonus yet — add it now.
        # success_bonus comes from info["r_success"] only when time_up;
        # we compute it manually here for early termination.
        # Extract success_bonus from reward components already present:
        #   r_success was computed as: time_up * fully_connected * w_success
        # For early success (fully_connected & ~time_up), replicate the bonus
        # by computing the delta.
        # Simplest: recompute for early_success using same weight embedded in info.
        # Since we can't easily extract the weight, we call reward_fn once more
        # ONLY for the early-success case where we need the correct terminal bonus.
        # XLA will constant-fold identical branches.
        _, info_terminal = reward_fn(state, new_state, done)
        reward_terminal = info_terminal["r_success"]
        # If time_up, reward_terminal == info["r_success"] (already included).
        # If early success (~time_up & fully_connected), add the difference.
        extra_bonus = jnp.where(
            fully_connected & ~time_up,
            reward_terminal,   # the bonus that wasn't added before
            jnp.float32(0.0),
        )
        reward = reward + extra_bonus

        # Auto-reset: swap in a fresh state when done
        reset_state = reset_fn(new_state.key)
        next_state = jax.tree_util.tree_map(
            lambda r, c: jnp.where(done, r, c),
            reset_state, new_state,
        )

        return next_state, reward, done, info

    return step


# ---------------------------------------------------------------------------
# Rollout collection — MAPPO
# ---------------------------------------------------------------------------

def _collect_rollout_mappo(
    states,
    model:            MAPPOModel,
    buf:              MAPPORolloutBuffer,
    autoreset_step_v,   # jit+vmap of _make_autoreset_step output
    obs_fn_v,
    key:              jax.Array,
    max_force:        float,
    T:                int,
    ep_trackers:      dict,
) -> tuple:
    """
    Collect T steps, tracking episode boundaries for correct ep metrics.
    """
    buf.reset()
    E, N = buf.E, buf.N

    # Load persistent accumulators
    ep_ret_accum     = ep_trackers["ret"]
    ep_len_accum     = ep_trackers["len"]
    ep_success_accum = ep_trackers["success"]
    ep_found_accum   = ep_trackers["found"]
    ep_gap_accum     = ep_trackers["gap"]

    # Completed-episode aggregates
    completed_returns  = []
    completed_lengths  = []
    completed_success  = []
    completed_found    = []
    completed_gaps     = []

    E_ = N_ = D_ = None  # set on first loop iteration

    for t in range(T):
        key, act_key = jax.random.split(key)

        obs_batch = obs_fn_v(states)          # (E, N, D)
        E_, N_, D_ = obs_batch.shape
        act_keys = jax.random.split(act_key, E_ * N_).reshape(E_, N_, 2)

        def _rollout_one_env(obs_n, keys_n):
            actions, log_probs, value = model.rollout_step(obs_n, keys_n, max_force)
            return actions, log_probs, value

        actions_b, log_probs_b, values_b = jax.vmap(_rollout_one_env)(obs_batch, act_keys)

        # Auto-resetting step — returns (next_states, rewards, dones, info)
        states, rewards_b, dones_b, info = autoreset_step_v(states, actions_b)

        rewards_np = np.array(rewards_b)          # (E,)
        dones_np   = np.array(dones_b).astype(bool)  # (E,)

        ep_ret_accum     += rewards_np
        ep_len_accum     += 1
        ep_success_accum  = np.maximum(ep_success_accum, np.array(info["fully_connected"]))
        ep_found_accum    = np.maximum(ep_found_accum, np.array(info["global_target_found"]))
        ep_gap_accum      = np.array(info["chain_gap_dist"])

        # Record completed episodes
        done_envs = np.where(dones_np)[0]
        for e in done_envs:
            completed_returns.append(float(ep_ret_accum[e]))
            completed_lengths.append(int(ep_len_accum[e]))
            completed_success.append(float(ep_success_accum[e]))
            completed_found.append(float(ep_found_accum[e]))
            completed_gaps.append(float(ep_gap_accum[e]))

        # Reset accumulators for finished envs
        ep_ret_accum     = np.where(dones_np, 0.0, ep_ret_accum)
        ep_len_accum     = np.where(dones_np, 0,   ep_len_accum)
        ep_success_accum = np.where(dones_np, 0.0, ep_success_accum)
        ep_found_accum   = np.where(dones_np, 0.0, ep_found_accum)

        buf.add(MAPPOTransition(
            obs        = np.array(obs_batch),
            actions    = np.array(actions_b),
            log_probs  = np.array(log_probs_b),
            values     = np.array(values_b),
            rewards    = rewards_np,
            dones      = dones_np.astype(np.float32),
        ))

    # Bootstrap value for last state
    last_obs    = obs_fn_v(states)
    last_values = model.get_value(last_obs)          # (E,)
    last_dones  = jnp.zeros(E, dtype=jnp.float32)

    # If no episodes completed in this window, use partial accumulators
    if completed_returns:
        mean_ret     = float(np.mean(completed_returns))
        mean_len     = float(np.mean(completed_lengths))
        mean_success = float(np.mean(completed_success))
        mean_found   = float(np.mean(completed_found))
        mean_gap     = float(np.mean(completed_gaps))
    else:
        mean_ret     = float(ep_ret_accum.mean())
        mean_len     = float(ep_len_accum.mean())
        mean_success = float(ep_success_accum.mean())
        mean_found   = float(ep_found_accum.mean())
        mean_gap     = float(ep_gap_accum.mean())

    n_eps = len(completed_returns)

    # Save persistent accumulators back
    ep_trackers["ret"]     = ep_ret_accum
    ep_trackers["len"]     = ep_len_accum
    ep_trackers["success"] = ep_success_accum
    ep_trackers["found"]   = ep_found_accum
    ep_trackers["gap"]     = ep_gap_accum

    return (
        states, key, last_values, last_dones,
        mean_ret, mean_len, mean_success, mean_found, mean_gap, n_eps,
    )


# ---------------------------------------------------------------------------
# Rollout collection — IPPO (baseline)
# ---------------------------------------------------------------------------

def _collect_rollout_ippo(
    states,
    model:            ActorCritic,
    buf:              RolloutBuffer,
    autoreset_step_v,
    obs_fn_v,
    key:              jax.Array,
    max_force:        float,
    T:                int,
    ep_trackers:      dict,
) -> tuple:
    buf.reset()
    E, N = buf.E, buf.N

    ep_ret_accum     = ep_trackers["ret"]
    ep_len_accum     = ep_trackers["len"]
    ep_success_accum = ep_trackers["success"]
    ep_found_accum   = ep_trackers["found"]
    ep_gap_accum     = ep_trackers["gap"]

    completed_returns, completed_lengths = [], []
    completed_success, completed_found, completed_gaps = [], [], []

    E_ = N_ = D_ = None

    for t in range(T):
        key, act_key = jax.random.split(key)
        obs_batch = obs_fn_v(states)
        E_, N_, D_ = obs_batch.shape
        act_keys = jax.random.split(act_key, E_ * N_).reshape(E_, N_, 2)

        def _act_env(obs_n, keys_n):
            def _act_one(o, k):
                a, lp, v, _ = model.act(o, k, max_force=max_force)
                return a, lp, v
            return jax.vmap(_act_one)(obs_n, keys_n)

        actions_b, log_probs_b, values_b = jax.vmap(_act_env)(obs_batch, act_keys)

        states, rewards_b, dones_b, info = autoreset_step_v(states, actions_b)

        rewards_np = np.array(rewards_b)
        dones_np   = np.array(dones_b).astype(bool)

        ep_ret_accum     += rewards_np
        ep_len_accum     += 1
        ep_success_accum  = np.maximum(ep_success_accum, np.array(info["fully_connected"]))
        ep_found_accum    = np.maximum(ep_found_accum, np.array(info["global_target_found"]))
        ep_gap_accum      = np.array(info["chain_gap_dist"])

        done_envs = np.where(dones_np)[0]
        for e in done_envs:
            completed_returns.append(float(ep_ret_accum[e]))
            completed_lengths.append(int(ep_len_accum[e]))
            completed_success.append(float(ep_success_accum[e]))
            completed_found.append(float(ep_found_accum[e]))
            completed_gaps.append(float(ep_gap_accum[e]))

        ep_ret_accum     = np.where(dones_np, 0.0, ep_ret_accum)
        ep_len_accum     = np.where(dones_np, 0,   ep_len_accum)
        ep_success_accum = np.where(dones_np, 0.0, ep_success_accum)
        ep_found_accum   = np.where(dones_np, 0.0, ep_found_accum)

        # For IPPO buffer, values is (E, N)
        buf.add(Transition(
            obs       = np.array(obs_batch),
            actions   = np.array(actions_b),
            log_probs = np.array(log_probs_b),
            values    = np.array(values_b),
            rewards   = rewards_np,
            dones     = dones_np.astype(np.float32),
        ))

    # Bootstrap for IPPO: (E, N)
    last_obs    = obs_fn_v(states)
    E_, N_, D_  = last_obs.shape
    last_vals   = model(last_obs.reshape(E_ * N_, D_))[2].reshape(E_, N_)
    last_dones  = jnp.zeros(E, dtype=jnp.float32)

    if completed_returns:
        mean_ret     = float(np.mean(completed_returns))
        mean_len     = float(np.mean(completed_lengths))
        mean_success = float(np.mean(completed_success))
        mean_found   = float(np.mean(completed_found))
        mean_gap     = float(np.mean(completed_gaps))
    else:
        mean_ret     = float(ep_ret_accum.mean())
        mean_len     = float(ep_len_accum.mean())
        mean_success = float(ep_success_accum.mean())
        mean_found   = float(ep_found_accum.mean())
        mean_gap     = float(ep_gap_accum.mean())

    # Save persistent accumulators back
    ep_trackers["ret"]     = ep_ret_accum
    ep_trackers["len"]     = ep_len_accum
    ep_trackers["success"] = ep_success_accum
    ep_trackers["found"]   = ep_found_accum
    ep_trackers["gap"]     = ep_gap_accum

    n_eps = len(completed_returns)

    return (
        states, key, last_vals, last_dones,
        mean_ret, mean_len, mean_success, mean_found, mean_gap, n_eps,
    )


# ---------------------------------------------------------------------------
# Deterministic eval rollout (single env, no auto-reset needed)
# ---------------------------------------------------------------------------

def _evaluate(
    model,
    reset_fn,
    env_step_fn,
    obs_fn,
    reward_fn,
    cfg:          DictConfig,
    key:          jax.Array,
    num_episodes: int = 1,
    use_mappo:    bool = True,
) -> tuple:
    max_force = float(cfg.env.max_force)
    max_steps = int(cfg.env.max_steps)
    N         = int(cfg.env.num_agents)

    all_states, all_rewards = [], []
    total_ret = total_gap = total_success = total_found = 0.0

    for _ in range(num_episodes):
        key, rk = jax.random.split(key)
        state   = reset_fn(rk)
        ep_ret  = ep_gap = 0.0
        ep_success = ep_found = False
        ep_states, ep_rewards = [], []

        for t in range(max_steps):
            ep_states.append(state)
            obs = obs_fn(state)

            if use_mappo:
                # Deterministic: take the mean action (no sampling)
                actions = jax.vmap(lambda o: model.actor(o)[0])(obs)
                actions = jnp.clip(actions, -max_force, max_force)
            else:
                actions = jnp.stack([
                    model.act(obs[i], key, max_force=max_force, deterministic=True)[0]
                    for i in range(N)
                ])

            old_state = state
            state     = env_step_fn(state, actions)

            time_up = (t == max_steps - 1)
            rew, info = reward_fn(old_state, state, jnp.bool_(time_up))
            ep_ret   += float(rew)
            ep_gap    = float(info["chain_gap_dist"])
            ep_success = ep_success or bool(info["fully_connected"] > 0.5)
            ep_found   = ep_found   or bool(info["global_target_found"] > 0.5)
            ep_rewards.append(float(rew))

            # Early exit on success
            if bool(info["fully_connected"] > 0.5):
                break

        all_states.append(ep_states)
        all_rewards.append(ep_rewards)
        total_ret     += ep_ret
        total_gap     += ep_gap
        total_success += float(ep_success)
        total_found   += float(ep_found)

    n = num_episodes
    return (
        all_states, all_rewards,
        total_ret / n, total_gap / n,
        total_success / n, total_found / n,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: DictConfig):
    # ── Logging config ────────────────────────────────────────────────────
    if cfg.logging.get("suppress_xla_warnings", True):
        # We set this before JAX does any heavy compilation
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        # Silence some WSL-specific driver noise
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    use_mappo = bool(cfg.training.get("use_mappo", True))

    print("\n══════════════════════════════════════════════════════")
    print(f"  SwarmEcho — {'MAPPO' if use_mappo else 'IPPO'} Training")
    print("══════════════════════════════════════════════════════")

    N         = int(cfg.env.num_agents)
    E         = int(cfg.training.num_envs)
    T         = int(cfg.training.num_steps)
    max_steps = int(cfg.env.max_steps)
    total_ts  = int(cfg.training.total_timesteps)
    n_updates = total_ts // (E * T)
    obs_dim   = compute_obs_dim(cfg)
    act_dim   = compute_action_dim(cfg)
    max_force = float(cfg.env.max_force)

    print(f"  Devices      : {jax.devices()}")
    print(f"  Mode         : {'MAPPO' if use_mappo else 'IPPO'}")
    print(f"  num_agents N : {N}")
    print(f"  num_envs E   : {E}")
    print(f"  rollout T    : {T}")
    print(f"  max_steps    : {max_steps}")
    print(f"  obs_dim      : {obs_dim}")
    print(f"  ppo updates  : {n_updates:,}  ({total_ts:,} total timesteps)")
    print()

    # ── Environment ───────────────────────────────────────────────────────
    env_step, reset, _ = make_env_fns(cfg)
    compute_obs, _     = make_obs_fns(cfg)
    compute_reward     = make_reward_fn(cfg)

    # Single-env auto-reset step
    autoreset_step   = _make_autoreset_step(env_step, reset, compute_reward, max_steps)
    # JIT + vmap for vectorised training
    autoreset_step_v = jax.jit(jax.vmap(autoreset_step))
    obs_fn_v         = jax.jit(jax.vmap(compute_obs))
    reset_v          = jax.jit(jax.vmap(reset))
    reset_s          = jax.jit(reset)

    # ── Model ─────────────────────────────────────────────────────────────
    master_key = jax.random.PRNGKey(int(cfg.training.seed))
    model_key, env_key, master_key = jax.random.split(master_key, 3)
    rngs = nnx.Rngs(int(model_key[0]))

    if use_mappo:
        model = MAPPOModel(
            obs_dim    = obs_dim,
            act_dim    = act_dim,
            num_agents = N,
            hidden_dim = int(cfg.network.hidden_dim),
            num_layers = int(cfg.network.num_layers),
            rngs       = rngs,
        )
        trainer = MAPPOTrainer(
            model         = model,
            lr            = float(cfg.training.lr),
            max_grad_norm = float(cfg.training.max_grad_norm),
            clip_eps      = float(cfg.training.clip_eps),
            vf_coef       = float(cfg.training.vf_coef),
            ent_coef      = float(cfg.training.ent_coef),
            num_epochs    = int(cfg.training.num_epochs),
        )
        buf = MAPPORolloutBuffer(T, E, N, obs_dim, act_dim,
                                  float(cfg.training.gamma),
                                  float(cfg.training.gae_lambda))
        collect_fn = _collect_rollout_mappo
    else:
        model = ActorCritic(obs_dim, act_dim,
                             int(cfg.network.hidden_dim),
                             int(cfg.network.num_layers), rngs)
        trainer = PPOTrainer(
            model         = model,
            lr            = float(cfg.training.lr),
            max_grad_norm = float(cfg.training.max_grad_norm),
            clip_eps      = float(cfg.training.clip_eps),
            vf_coef       = float(cfg.training.vf_coef),
            ent_coef      = float(cfg.training.ent_coef),
            num_epochs    = int(cfg.training.num_epochs),
        )
        buf = RolloutBuffer(T, E, N, obs_dim, act_dim,
                            float(cfg.training.gamma),
                            float(cfg.training.gae_lambda))
        collect_fn = _collect_rollout_ippo

    _, params = nnx.split(model)
    n_params  = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model params : {n_params:,}")

    # ── Checkpoint Resumption ─────────────────────────────────────────────
    import orbax.checkpoint as ocp
    if cfg.training.checkpoint_path:
        checkpoint_path = Path(cfg.training.checkpoint_path).absolute()
        print(f"  Resuming from: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        graphdef, empty_state = nnx.split(model)
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        restored_state = checkpointer.restore(
            str(checkpoint_path),
            args=ocp.args.StandardRestore(empty_state),
        )
        nnx.update(model, restored_state)
        print(f"  Weights restored ✓")

    # ── Environments init ─────────────────────────────────────────────────
    env_keys = jax.random.split(env_key, E)
    states   = reset_v(env_keys)
    print("  Environments initialised ✓")

    # ── Run directory & Name ──────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_run_name = cfg.training.get("run_name", None)
    run_name = f"{cfg_run_name}_{ts}" if cfg_run_name else f"run_{ts}"

    # Target directory structure
    log_root = Path(cfg.logging.get("log_dir", "outputs")).absolute()
    run_dir  = log_root / run_name
    ckpt_dir  = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save active config for local discovery and analysis
    OmegaConf.save(cfg, run_dir / "config.yaml")

    print(f"  Run dir      : {run_dir}")

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_run = _init_wandb(cfg, run_name)
    if wandb_run:
        print(f"  W&B run      : {wandb_run.url}")

    # ── Async render worker ───────────────────────────────────────────────
    render_worker = VideoRenderWorker(cfg, video_dir, wandb_run)
    render_worker.start()

    # ── Training loop ─────────────────────────────────────────────────────
    video_every = int(cfg.logging.get("video_every_n_updates", 50))
    ckpt_every  = int(cfg.logging.get("checkpoint_every_n_updates", 100))
    eval_eps    = int(cfg.logging.get("eval_episodes", 3))
    log_every   = max(1, n_updates // 200)

    # ── Persistent Accumulators ──────────────────────────────────────────
    ep_trackers = {
        "ret":     np.zeros(E, dtype=np.float32),
        "len":     np.zeros(E, dtype=np.int32),
        "success": np.zeros(E, dtype=np.float32),
        "found":   np.zeros(E, dtype=np.float32),
        "gap":     np.zeros(E, dtype=np.float32),
    }

    t_start = time.perf_counter()
    for update in range(1, n_updates + 1):
        master_key, collect_key = jax.random.split(master_key)
        steps_done = update * E * T   # environment timesteps processed so far

        # ── Rollout ───────────────────────────────────────────────────────
        (states, collect_key,
         last_values, last_dones,
         mean_ret, mean_len, mean_success, mean_found, mean_gap,
         n_eps) = collect_fn(
            states, model, buf, autoreset_step_v, obs_fn_v,
            collect_key, max_force, T, ep_trackers,
        )

        # ── GAE + minibatches ─────────────────────────────────────────────
        advs, rets = buf.compute_gae(last_values, last_dones)
        mbs        = buf.get_minibatches(advs, rets, int(cfg.training.num_minibatches), collect_key)

        # ── PPO update ────────────────────────────────────────────────────
        ppo_stats = trainer.update(mbs)

        elapsed = time.perf_counter() - t_start
        sps     = steps_done / elapsed

        # ── Stdout ───────────────────────────────────────────────────────
        if update % log_every == 0 or update == 1:
            print(
                f"  [{update:>6}/{n_updates}]  "
                f"steps={steps_done:>12,}  "
                f"sps={sps:>7,.0f}  "
                f"ep_ret={mean_ret:>7.2f}  "
                f"ep_len={mean_len:>6.0f}  "
                f"gap={mean_gap:>6.1f}m  "
                f"succ={mean_success:>5.1%}  "
                f"found={mean_found:>5.1%}  "
                f"eps={n_eps:>4d}  "
                f"ent={ppo_stats['entropy']:>5.3f}"
            )

        # ── W&B logging (x-axis = env timesteps) ─────────────────────────
        if wandb_run:
            import wandb
            wandb.log({
                "train/ep_return":          mean_ret,
                "train/ep_length":          mean_len,
                "train/success_rate":       mean_success,
                "train/target_found_rate":  mean_found,
                "train/chain_gap_dist":     mean_gap,
                "train/episodes_completed": n_eps,
                "ppo/policy_loss":          ppo_stats["policy_loss"],
                "ppo/value_loss":           ppo_stats["value_loss"],
                "ppo/entropy":              ppo_stats["entropy"],
                "ppo/approx_kl":            ppo_stats["approx_kl"],
                "ppo/clip_fraction":        ppo_stats["clip_fraction"],
                "perf/sps":                 sps,
                "perf/ppo_updates":         update,
            }, step=steps_done)          # ← x-axis is env timesteps

        # ── Eval + async video ────────────────────────────────────────────
        if update % video_every == 0 or update == n_updates:
            master_key, eval_key = jax.random.split(master_key)
            (ep_states_list, ep_rewards_list,
             eval_ret, eval_gap, eval_success, eval_found) = _evaluate(
                model, reset_s,
                jax.jit(env_step), jax.jit(compute_obs), jax.jit(compute_reward),
                cfg, eval_key, eval_eps, use_mappo,
            )

            render_worker.submit(
                ep_states  = ep_states_list[0],
                ep_rewards = ep_rewards_list[0],
                update     = update,
                eval_ret   = eval_ret,
            )

            if wandb_run:
                import wandb
                wandb.log({
                    "eval/ep_return":      eval_ret,
                    "eval/chain_gap_dist": eval_gap,
                    "eval/success_rate":   eval_success,
                    "eval/target_found":   eval_found,
                }, step=steps_done)      # ← same x-axis

            print(
                f"  [eval] update={update}  "
                f"steps={steps_done:,}  "
                f"ep_return={eval_ret:.2f}  "
                f"gap={eval_gap:.1f}m  "
                f"success={eval_success:.1%}  "
                f"→ render queued"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if update % ckpt_every == 0 or update == n_updates:
            import orbax.checkpoint as ocp
            ckpt_path = (ckpt_dir / f"ckpt_{update:06d}").absolute()
            _, state_dict = nnx.split(model)
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
            checkpointer.save(str(ckpt_path), args=ocp.args.StandardSave(state_dict))
            print(f"  [ckpt] saved → {ckpt_path}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Training complete in {total_time:.1f}s  ({total_time/60:.1f} min)")

    render_worker.shutdown(wait=True)

    if wandb_run:
        wandb_run.finish()

    # ── Final Checkpoint (Guaranteed for Curriculum Safety) ───────────
    final_ckpt = (ckpt_dir / f"ckpt_{n_updates:06d}").absolute()
    if not final_ckpt.exists():
        import orbax.checkpoint as ocp
        _, state_dict = nnx.split(model)
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        checkpointer.save(str(final_ckpt), args=ocp.args.StandardSave(state_dict))
        print(f"  [ckpt-final] saved → {final_ckpt}")

    return str(final_ckpt)


