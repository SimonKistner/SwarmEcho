"""
scripts/test_perfect_chain.py
=============================
A scripted trajectory test for 
Drones are driven by a simple P-controller to form a perfect equidistant
line between the Base and Target. This allows us to inspect the resulting
rewards and ensure the Two-Chain Projection works under ideal conditions.
A video is also generated to cross-reference visual behaviour with rewards.
"""

import os
import sys
from pathlib import Path

# Suppress JAX warning about our large (2.5GB) 1.0m visibility matrix.
# We are intentionally capturing it in closures to avoid VRAM duplication.
os.environ["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "-1"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
from omegaconf import OmegaConf

from core.config import load_config, validate_config
from env.physics import make_env_fns
from env.rewards import make_reward_fn
from visualize.renderer import render_video


def run_scripted_test():
    # Load config and respect CLI overrides (e.g. level=00)
    cfg = load_config(cli_overrides=True)
    
    # We use OmegaConf API to update the config dynamically for the test scene
    validate_config(cfg)
    
    # Force configuration values to match the test assumptions
    OmegaConf.set_readonly(cfg, False)
    cfg.env.num_agents = 10
    cfg.env.map_names = ["open_field"]
    cfg.env.use_random_base_spawn = False
    cfg.env.use_random_drone_spawn = False
    cfg.env.target_spawn_method = "map_defined"
    cfg.env.spawn_delay = 0
    cfg.env.comm_radius = 50.0
    cfg.env.visual_radius = 50.0
    cfg.env.max_speed = 30.0
    cfg.env.max_force = 150.0
    OmegaConf.set_readonly(cfg, True)
    
    is_warehouse = "warehouse" in cfg.env.map_names
    test_y = 110.0 if is_warehouse else 50.0

    N = cfg.env.num_agents
    
    env_step, reset, _, _ = make_env_fns(cfg)
    compute_reward     = make_reward_fn(cfg)

    step_jit   = jax.jit(env_step)
    reward_jit = jax.jit(compute_reward)

    state = reset(jax.random.PRNGKey(42))
    
    # Override positions for the perfect chain test
    # This keeps the test-specific geometry inside the test file
    base_pos   = jnp.array([50.0, test_y])
    target_pos = jnp.array([450.0, test_y])
    pos        = jnp.tile(base_pos[None, :], (N, 1))
    state = state.replace(base_pos=base_pos, target_pos=target_pos, pos=pos)

    # Calculate optimal positions for each agent to form a perfect chain
    # With N=10, gaps become 345/8 = 43.125m (Safe!)
    optimal_x = jnp.concatenate([
        jnp.array([50.0]),
        jnp.linspace(95.0, 440.0, N - 1)
    ])
    optimal_positions = jnp.stack([optimal_x, jnp.full(N, test_y)], axis=-1)

    traj_states = []
    traj_rewards = []
    traj_gap_dist = []
    traj_chain_pct = []
    traj_r_gap = []
    traj_r_explor = []
    traj_r_coll = []
    
    print(f"Running scripted test with {N} agents...")
    print(f"  {'Step':>4} | {'r_cov':>6} | {'r_fnd':>6} | {'r_gap':>8} | {'r_coll':>8} | {'r_succ':>6} || {'Total R':>8} | {'GapDst':>6} | {'TGT':>3} | {'CONN':>4}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} || {'-'*8} | {'-'*6} | {'-'*3} | {'-'*4}")

    # First, everybody flies to the middle (X=250), except Drone 0 (Base) and Drone 8 (Target)
    explore_x = jnp.full(N, 250.0)
    # Give drones some spread in Y to avoid perfect overlap
    explore_y = jnp.linspace(30.0, 70.0, N)
    
    # Drone 0 sits at base
    explore_x = explore_x.at[0].set(50.0)
    explore_y = explore_y.at[0].set(test_y)
    
    # Drone 8 runs to the target instantly to "find" it
    explore_x = explore_x.at[N-1].set(440.0)
    explore_y = explore_y.at[N-1].set(test_y)
    
    explore_positions = jnp.stack([explore_x, explore_y], axis=-1)

    chain_formed_step = None
    target_found_step = None

    hold_chain_for = int(cfg.env.get("hold_chain_for", 10))
    print(f"Required hold chain steps: {hold_chain_for}")
    success_achieved_step = None

    # Run for up to 1500 steps (exits early upon success)
    for t in range(1500):
        # Controller target positions based on target knowledge
        # Drone 0 (base) stays at base (50)
        # Drone 9 (finder) flies to 440 to find target, then flies to base (50) to deliver info,
        # and finally back to 440 once base knows target.
        target_x_9 = jnp.where(
            state.base_target_known,
            optimal_positions[9, 0],
            jnp.where(state.target_known[9], 50.0, 440.0)
        )
        
        # Drones 1-8 stay at explore positions until they receive target knowledge
        target_x_1_8 = jnp.where(
            state.target_known[1:N-1],
            optimal_positions[1:N-1, 0],
            explore_positions[1:N-1, 0]
        )
        
        target_x = jnp.concatenate([
            jnp.array([50.0]),
            target_x_1_8,
            jnp.array([target_x_9])
        ])
        
        target_y = jnp.full(N, test_y)
        target_pos = jnp.stack([target_x, target_y], axis=-1)
            
        kp = 12.0
        kd = 5.0
        
        error = target_pos - state.pos
        actions = kp * error - kd * state.vel
        
        # Clip actions to max_force
        actions = jnp.clip(actions, -cfg.env.max_force, cfg.env.max_force)

        old_state = state
        state = step_jit(old_state, actions)
        
        # Compute rewards with is_done=False so success bonus is not prematurely added
        is_done = jnp.bool_(False)
        rew, info = reward_jit(old_state, state, is_done)
        
        fully_connected = info['fully_connected'] > 0.5
        new_chain_held_steps = jnp.where(
            fully_connected,
            state.chain_held_steps + jnp.int32(1),
            jnp.int32(0)
        )
        state = dataclasses.replace(state, chain_held_steps=new_chain_held_steps)
        success_achieved = new_chain_held_steps >= (hold_chain_for + 1)
        
        # If success is achieved, add the success bonus
        if bool(success_achieved) and success_achieved_step is None:
            success_achieved_step = t
            is_done = jnp.bool_(True)
            rew, info = reward_jit(old_state, state, is_done)

        traj_states.append(state)
        traj_rewards.append(rew)
        traj_gap_dist.append(float(info["chain_gap_dist"]))
        traj_r_gap.append(float(info["r_chain_gap"]))
        traj_r_explor.append(float(info["r_coverage"]))
        traj_r_coll.append(float(info["r_collision"]))
        traj_chain_pct.append(float(info["chain_progress_pct"]))

        if info['just_found'] and target_found_step is None:
            target_found_step = t
        
        if fully_connected and chain_formed_step is None:
            chain_formed_step = t

        if t % 50 == 0 or info['just_found'] or (chain_formed_step == t) or bool(success_achieved):
            r_tot = float(rew.sum())
            rt_cov  = float(info['r_coverage'])
            rt_fnd  = float(info.get('r_target_found', 0.0))
            rt_gap  = float(info['r_chain_gap'])
            rt_coll = float(info['r_collision'])
            rt_succ = float(info['r_success'])
            
            gap_dist = float(info['chain_gap_dist'])
            tgt = "YES" if info['global_target_found'] else "no"
            conn = f"YES({int(new_chain_held_steps)})" if fully_connected else "no"
            
            highlight = ">>> " if info['just_found'] or (chain_formed_step == t) or bool(success_achieved) else "    "
            
            print(f"{highlight}{t:>4} | {rt_cov:>6.3f} | {rt_fnd:>6.1f} | {rt_gap:>8.3f} | {rt_coll:>8.3f} | {rt_succ:>6.1f} || {r_tot:>8.3f} | {gap_dist:>6.1f} | {tgt:>3} | {conn}")

        if success_achieved_step is not None:
            print(f"\nChain successfully held for {hold_chain_for} steps! Ending episode early.")
            break

    # Stack the trajectory history along a new time dimension (T, ...)
    traj_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *traj_states)
    rewards_stacked = jnp.array(traj_rewards)
    
    extra_metrics = {
        "r_gap":    np.array(traj_r_gap),
        "r_explor": np.array(traj_r_explor),
        "r_total":  np.array([float(r.sum()) for r in traj_rewards]),
        "chain_pct": np.array(traj_chain_pct),
    }

    print("\nRendering video to visually confirm behavior...")

    # 1. Slow (Matplotlib) renderer - Beautiful scientific layout
    # render_video(
    #     traj_stacked, 
    #     cfg, 
    #     filename="outputs/videos/perfect_chain_mpl.mp4", 
    #     renderer="slow",
    #     rewards=rewards_stacked,
    #     extra_metrics=extra_metrics
    # )

    # 2. Fast (OpenCV) renderer - Fast layout
    render_video(
        traj_stacked, 
        cfg, 
        filename="outputs/videos/perfect_chain_cv2.mp4", 
        renderer="fast",
        rewards=rewards_stacked,
        extra_metrics=extra_metrics
    )
    
    print("\nDone. Check out:")
    print("  - outputs/videos/perfect_chain_cv2.mp4")

if __name__ == "__main__":
    run_scripted_test()
