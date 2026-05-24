"""
scripts/test_perfect_chain_warehouse.py
=======================================
A scripted trajectory test for the SwarmEcho environment inside the Warehouse map.
Drones are driven by a waypoint-based controller to form a perfect line around obstacles 
between the Base and Target. This showcases:
1) The new visual first-link logic (base and target require direct line-of-sight within vis_r).
2) The new collision indicators (drones turning red + glowing).
3) Waypoint-based obstacle avoidance.
4) One drone spawned near target to find it early.

A video is generated to cross-reference visual behaviour with rewards.
"""

import os
import sys
import dataclasses
from pathlib import Path

# Suppress JAX warning about our large (2.5GB) 1.0m visibility matrix.
os.environ["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "-1"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from core.config import load_config, validate_config
from env.physics import make_env_fns
from env.rewards import make_reward_fn
from visualize.renderer import render_video


def run_scripted_test():
    # Force warehouse map by simulating CLI arguments
    sys.argv = [sys.argv[0], "env.map_names=[warehouse]"]
    cfg = load_config(cli_overrides=True)
    
    OmegaConf.set_readonly(cfg, False)
    cfg.env.num_agents = 13
    cfg.env.spawn_delay = 0      # Start all together for the test
    cfg.env.dt = 0.1
    cfg.env.max_steps = 750     
    cfg.env.max_speed = 10.0 
    cfg.reward.collision_penalty = 1.0
    OmegaConf.set_readonly(cfg, True)
    
    validate_config(cfg)
    
    N = cfg.env.num_agents
    
    env_step, reset, _, _ = make_env_fns(cfg)
    compute_reward     = make_reward_fn(cfg)

    step_jit   = jax.jit(env_step)
    reward_jit = jax.jit(compute_reward)

    # 1. Setup Initial Positions
    # Most drones start near base (20, 60)
    start_pos = np.full((N, 2), [20.0, 60.0])
    
    # Drone 12 starts near target (480, 60)
    start_pos[12] = [475.0, 65.0]
    
    start_pos = jnp.array(start_pos, dtype=jnp.float32)

    state = reset(jax.random.PRNGKey(42))
    # FORCE initial positions AND anchors
    base_override = jnp.array([20.0, 60.0])   # Match D0
    target_override = jnp.array([475.0, 60.0]) # Match D12
    state = dataclasses.replace(state, pos=start_pos, base_pos=base_override, target_pos=target_override)

    # 2. Define Optimal Chain Positions (around obstacles)
    # Move D1 from (60,90) to (65,90) and D10 from (420,90) to (415,90) to give 50m links some slack.
    # 13 drones used.
    opt_x = [ 20.0,  60.0, 100.0, 140.0, 180.0, 220.0, 260.0, 300.0, 340.0, 380.0, 415.0, 450.0, 475.0]
    opt_y = [ 60.0,  85.0,  90.0,  90.0, 100.0, 110.0, 110.0, 110.0, 100.0,  90.0,  90.0,  70.0,  60.0]
    optimal_positions = jnp.stack([jnp.array(opt_x), jnp.array(opt_y)], axis=-1)

    # 3. Define Waypoints for each drone to avoid obstacles
    # WP1: (125, 100), WP2: (275, 115), WP3: (415, 95)
    waypoints = [
        [], # D0 connects to base
        [[45, 82]], # D1
        [[60, 90], [90, 90]], # D2
        [[60, 90], [125, 100], [140, 90]], # D3
        [[60, 90], [125, 100], [180, 100]], # D4
        [[60, 90], [125, 100], [220, 110]], # D5
        [[60, 90], [125, 100], [275, 115], [260, 110]], # D6
        [[60, 90], [125, 100], [275, 115], [300, 110]], # D7
        [[60, 90], [125, 100], [275, 115], [330, 115], [340, 100]], # D8
        [[60, 90], [125, 100], [275, 115], [330, 115], [380, 90]], # D9
        [[60, 90], [125, 100], [275, 115], [330, 115], [400, 105], [410, 100]], # D10
        [[60, 90], [125, 100], [275, 115], [330, 115], [415, 95], [425, 85]], # D11
        [], # D12 starts at target
    ]

    current_wp_idx = np.zeros(N, dtype=int)
    
    traj_states = []
    traj_rewards = []
    traj_gap_dist = []
    traj_r_gap = []
    traj_r_explor = []
    
    print(f"Running scripted test with {N} agents...")
    print(f"  {'Step':>4} | {'r_cov':>6} | {'r_fnd':>6} | {'r_gap':>8} | {'r_coll':>8} | {'r_succ':>6} || {'Total R':>8} | {'GapDst':>6} | {'TGT':>3} | {'CONN':>4}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} || {'-'*8} | {'-'*6} | {'-'*3} | {'-'*4}")

    chain_formed_step = None
    target_found_step = None

    # Delays for unfurling
    unfurl_delays = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 0])

    for t in range(cfg.env.max_steps):
        
        target_pos_np = np.array(state.pos)
        
        for i in range(N):
            if t < unfurl_delays[i] and i < 12:
                # Stay put until unfurl delay
                target_pos_np[i] = np.array(state.pos[i])
                continue
                
            wps = waypoints[i]
            if current_wp_idx[i] < len(wps):
                target_pos_np[i] = wps[current_wp_idx[i]]
                # If close to current waypoint, advance
                dist = np.linalg.norm(np.array(state.pos[i]) - target_pos_np[i])
                if dist < 3.0:
                    current_wp_idx[i] += 1
            else:
                # Final destination
                target_pos_np[i] = np.array(optimal_positions[i])

        target_pos = jnp.array(target_pos_np)
            
        kp = 8.0
        kd = 3.0
        
        error = target_pos - state.pos
        actions = kp * error - kd * state.vel
        actions = jnp.clip(actions, -cfg.env.max_force, cfg.env.max_force)

        old_state = state
        state = step_jit(old_state, actions)
        
        # Compute rewards
        is_done = jnp.bool_(False)
        rew, info = reward_jit(old_state, state, is_done)
        
        traj_states.append(state)
        traj_rewards.append(rew)
        traj_gap_dist.append(float(info["chain_gap_dist"]))
        traj_r_gap.append(float(info["r_chain_gap"]))
        traj_r_explor.append(float(info["r_coverage"]))
        
        if info['just_found'] and target_found_step is None:
            target_found_step = t
        
        if info['fully_connected'] and chain_formed_step is None:
            chain_formed_step = t
            is_done = jnp.bool_(True)
            rew, info = reward_jit(old_state, state, is_done)
            traj_rewards[-1] = rew

        if t % 50 == 0 or info['just_found'] or (chain_formed_step == t):
            r_tot = float(rew.sum())
            rt_cov  = float(info['r_coverage'])
            rt_fnd  = float(info.get('r_target_found', 0.0))
            rt_gap  = float(info['r_chain_gap'])
            rt_coll = float(info['r_collision'])
            rt_succ = float(info['r_success'])
            
            gap_dist = float(info['chain_gap_dist'])
            tgt = "YES" if info['global_target_found'] else "no"
            conn = "YES!" if info['fully_connected'] else "no"
            
            highlight = ">>> " if info['just_found'] or (chain_formed_step == t) else "    "
            
            print(f"{highlight}{t:>4} | {rt_cov:>6.3f} | {rt_fnd:>6.1f} | {rt_gap:>8.3f} | {rt_coll:>8.3f} | {rt_succ:>6.1f} || {r_tot:>8.3f} | {gap_dist:>6.1f} | {tgt:>3} | {conn:>4}")

        if chain_formed_step is not None and t > chain_formed_step + 50:
            print("\nChain successfully formed! Ending episode after short delay.")
            break

    traj_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *traj_states)
    rewards_stacked = jnp.array(traj_rewards)
    
    extra_metrics = {
        "r_gap":   np.array(traj_r_gap),
        "r_explor": np.array(traj_r_explor),
        "r_coll":  np.array([float(s.collides.sum() * -cfg.reward.collision_penalty) for s in traj_states]),
        "r_total": np.array([float(r.sum()) for r in traj_rewards]),
        "GapDist": np.array(traj_gap_dist),
    }

    print("\nRendering videos...")
    render_video(
        traj_stacked, cfg, "outputs/videos/perfect_chain_warehouse_cv2.mp4", 
        fps=20, rewards=rewards_stacked, extra_metrics=extra_metrics, renderer="fast"
    )
    """
    render_video(
        traj_stacked, cfg, "outputs/videos/perfect_chain_warehouse_mpl.mp4", 
        fps=20, rewards=rewards_stacked, extra_metrics=extra_metrics, renderer="slow"
    )
    """
    print("Done. Check out outputs/videos/!")

if __name__ == "__main__":
    run_scripted_test()
