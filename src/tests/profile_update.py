"""
src/tests/profile_update.py
===========================
Expanded diagnostic tool to profile JIT compilation and execution speed of the MAPPO recurrent update step.
Separates profiling of Actor forward, Critic forward, Actor backward, Critic backward, and full step.
Run:
    uv run python src/tests/profile_update.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make src/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from models.mappo import MAPPOModel
from training.mappo_trainer import MAPPOTrainer

def main():
    print("── Loading Configuration ──")
    cfg = load_config(cli_overrides=True)
    from omegaconf import OmegaConf
    OmegaConf.set_readonly(cfg, False)
    validate_config(cfg)
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N = int(cfg.env.num_agents)
    H = int(cfg.network.hidden_dim)

    # Force recurrent and memory communication settings
    cfg.network.actor_memory = True
    cfg.network.critic_memory = True
    cfg.network.memory_comm_enabled = True
    
    print(f"  num_agents N              : {N}")
    print(f"  obs_dim                   : {obs_dim}")
    print(f"  act_dim                   : {act_dim}")
    print(f"  hidden_dim H              : {H}")
    print(f"  tarmac_sig_dim            : {int(cfg.network.get('tarmac_sig_dim', 64))}")
    print(f"  tarmac_val_dim            : {int(cfg.network.get('tarmac_val_dim', 128))}")

    T = int(cfg.training.num_steps) # 50
    E = int(cfg.training.num_envs)  # 4000
    MB_num = int(cfg.training.num_minibatches) # 20
    B = E // MB_num  # 200
    
    print(f"  rollout steps T           : {T}")
    print(f"  envs per minibatch B      : {B}")

    # ── Initialize Model & Trainer ──
    print("\n── Initializing Model & Trainer ──")
    rngs = nnx.Rngs(0)
    model = MAPPOModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_agents=N,
        hidden_dim=H,
        num_layers=int(cfg.network.num_layers),
        actor_num_layers=int(cfg.network.actor_num_layers),
        critic_type=str(cfg.network.critic_type),
        actor_memory=True,
        critic_memory=True,
        rngs=rngs,
        memory_comm_enabled=True,
        memory_comm_every_k_steps=int(cfg.network.get("memory_comm_every_k_steps", 5)),
        tarmac_sig_dim=int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim=int(cfg.network.get("tarmac_val_dim", 128)),
        tarmac_include_self=bool(cfg.network.get("tarmac_include_self", True)),
    )
    
    trainer = MAPPOTrainer(
        model=model,
        lr=3e-4,
        max_grad_norm=0.5,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        num_epochs=1,  # run 1 epoch for profiling
        per_agent=True,
        actor_memory=True,
        critic_memory=True,
    )

    # ── Create Dummy Minibatch ──
    print("\n── Creating Dummy Minibatch ──")
    mb = {
        "obs": jnp.zeros((T, B, N, obs_dim), dtype=jnp.float32),
        "actions": jnp.zeros((T, B, N, act_dim), dtype=jnp.float32),
        "old_log_probs": jnp.zeros((T, B, N), dtype=jnp.float32),
        "old_values": jnp.zeros((T, B, N), dtype=jnp.float32),
        "advantages": jnp.zeros((T, B, N), dtype=jnp.float32),
        "returns": jnp.zeros((T, B, N), dtype=jnp.float32),
        "rnn_resets": jnp.zeros((T, B, N), dtype=jnp.bool_),
        "initial_actor_h": jnp.zeros((B, N, H), dtype=jnp.float32),
        "initial_actor_signature": jnp.zeros((B, N, int(cfg.network.get("tarmac_sig_dim", 64))), dtype=jnp.float32),
        "initial_actor_value": jnp.zeros((B, N, int(cfg.network.get("tarmac_val_dim", 128))), dtype=jnp.float32),
        "initial_critic_h": jnp.zeros((B, N, H), dtype=jnp.float32),
        "comm_masks": jnp.ones((T, B, N, N), dtype=jnp.bool_),
        "active_masks": jnp.ones((T, B, N), dtype=jnp.bool_),
        "base_signatures": jnp.zeros((T, B, int(cfg.network.get("tarmac_sig_dim", 64))), dtype=jnp.float32),
        "base_values": jnp.zeros((T, B, int(cfg.network.get("tarmac_val_dim", 128))), dtype=jnp.float32),
        "base_memory_masks": jnp.zeros((T, B, N), dtype=jnp.bool_),
    }

    # ── Define Profiling Targets ──
    
    # 1. Actor Forward Pass
    @nnx.jit
    def run_actor_forward(m, mb):
        def _eval_env(obs_env, act_env, reset_env, init_h_env, init_sig_env, init_val_env, comm_env, active_env, base_sig_env, base_val_env, base_mask_env):
            return m.actor.evaluate_actions_sequence(
                obs_env, act_env, init_h_env, reset_env, init_sig_env, init_val_env, comm_env, active_env, base_sig_env, base_val_env, base_mask_env
            )
        return jax.vmap(_eval_env, in_axes=(1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1))(
            mb["obs"], mb["actions"], mb["rnn_resets"], mb["initial_actor_h"], mb["initial_actor_signature"], mb["initial_actor_value"],
            mb["comm_masks"], mb["active_masks"], mb["base_signatures"], mb["base_values"], mb["base_memory_masks"]
        )

    # 2. Critic Forward Pass
    @nnx.jit
    def run_critic_forward(m, mb):
        return m.critic.values_sequence(mb["obs"], mb["initial_critic_h"], mb["rnn_resets"], deterministic=False)

    # 3. Actor Backward Pass
    def actor_loss_only(actor_m, mb):
        def _eval_env(obs_env, act_env, reset_env, init_h_env, init_sig_env, init_val_env, comm_env, active_env, base_sig_env, base_val_env, base_mask_env):
            return actor_m.evaluate_actions_sequence(
                obs_env, act_env, init_h_env, reset_env, init_sig_env, init_val_env, comm_env, active_env, base_sig_env, base_val_env, base_mask_env
            )
        _, log_probs, _ = jax.vmap(_eval_env, in_axes=(1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1))(
            mb["obs"], mb["actions"], mb["rnn_resets"], mb["initial_actor_h"], mb["initial_actor_signature"], mb["initial_actor_value"],
            mb["comm_masks"], mb["active_masks"], mb["base_signatures"], mb["base_values"], mb["base_memory_masks"]
        )
        return jnp.mean(log_probs)

    @nnx.jit
    def run_actor_grad(actor_m, mb):
        return nnx.grad(actor_loss_only)(actor_m, mb)

    # 4. Critic Backward Pass
    def critic_loss_only(critic_m, mb):
        _, values = critic_m.values_sequence(mb["obs"], mb["initial_critic_h"], mb["rnn_resets"], deterministic=False)
        return jnp.mean(values ** 2)

    @nnx.jit
    def run_critic_grad(critic_m, mb):
        return nnx.grad(critic_loss_only)(critic_m, mb)

    # ── Profiling Routine ──
    
    def profile_fn(name, fn, *args):
        print(f"\n[Profile] {name}...")
        
        # Warmup / Compilation
        t0 = time.perf_counter()
        res = fn(*args)
        jax.block_until_ready(res)
        t_comp = time.perf_counter() - t0
        print(f"  - Compile + Warmup: {t_comp:.4f} seconds")
        
        # Pure execution
        t1 = time.perf_counter()
        res = fn(*args)
        jax.block_until_ready(res)
        t_exec = time.perf_counter() - t1
        print(f"  - Pure Execution  : {t_exec:.4f} seconds")
        return t_comp, t_exec

    # Run sub-components
    comp_af, exec_af = profile_fn("1. Actor Forward Pass", run_actor_forward, model, mb)
    comp_cf, exec_cf = profile_fn("2. Critic Forward Pass", run_critic_forward, model, mb)
    comp_ab, exec_ab = profile_fn("3. Actor Backward Pass (Grad)", run_actor_grad, model.actor, mb)
    comp_cb, exec_cb = profile_fn("4. Critic Backward Pass (Grad)", run_critic_grad, model.critic, mb)
    
    # Run full trainer step
    print("\n[Profile] 5. Full Trainer Step (Actor + Critic Updates)...")
    t0 = time.perf_counter()
    _ = trainer.update([mb])
    jax.block_until_ready(_)
    t_comp_full = time.perf_counter() - t0
    print(f"  - Compile + Warmup: {t_comp_full:.4f} seconds")
    
    t1 = time.perf_counter()
    _ = trainer.update([mb])
    jax.block_until_ready(_)
    t_exec_full = time.perf_counter() - t1
    print(f"  - Pure Execution  : {t_exec_full:.4f} seconds")
    
    # Summary
    print("\n" + "═"*60)
    print("  PROFILING SUMMARY")
    print("═"*60)
    print(f"  Actor Forward   | Compile: {comp_af:>6.2f}s | Exec: {exec_af:>6.4f}s")
    print(f"  Critic Forward  | Compile: {comp_cf:>6.2f}s | Exec: {exec_cf:>6.4f}s")
    print(f"  Actor Backward  | Compile: {comp_ab:>6.2f}s | Exec: {exec_ab:>6.4f}s")
    print(f"  Critic Backward | Compile: {comp_cb:>6.2f}s | Exec: {exec_cb:>6.4f}s")
    print(f"  Full Update Step| Compile: {t_comp_full:>6.2f}s | Exec: {t_exec_full:>6.4f}s")
    print("═"*60)
    
    steps_updated = B * T
    sps_full = steps_updated / t_exec_full
    print(f"  Full Update SPS : {sps_full:,.0f} steps/second")
    print("═"*60)

if __name__ == "__main__":
    main()
