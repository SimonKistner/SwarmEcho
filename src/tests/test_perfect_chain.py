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
from omegaconf import OmegaConf

from core.config import load_config, validate_config
from env.physics import make_env_fns
from env.rewards import make_reward_fn
from visualize.renderer import render_video


def run_scripted_test():
    # Load config and override specific values for this test
    # We need enough agents to span the 400m gap (400 / 50m = 8 agents minimum)
    # We also disable stagger spawn so they form up quickly.
    cfg = load_config(cli_overrides=False)
    
    # We use OmegaConf API to update the config dynamically
    OmegaConf.set_readonly(cfg, False)
    cfg.env.num_agents = 8
    cfg.env.spawn_delay = 5      # Everyone starts active
    cfg.env.dt = 0.1
    OmegaConf.set_readonly(cfg, True)
    
    validate_config(cfg)
    
    N = cfg.env.num_agents
    
    env_step, reset, _ = make_env_fns(cfg)
    compute_reward     = make_reward_fn(cfg)

    step_jit   = jax.jit(env_step)
    reward_jit = jax.jit(compute_reward)

    state = reset(jax.random.PRNGKey(42))

    # Calculate optimal positions for each agent to form a perfect chain
    # Base is at x=50, Target is at x=450.
    # Drones should space themselves out evenly along the x-axis to exactly cover the 400m gap.
    # With 8 drones, the spacing needed is exactly 50m!
    # Base (50) -> D1 (95) -> D2 (144.2) -> D3 (193.5) -> D4 (242.8) -> D5 (292.1) -> D6 (341.4) -> D7 (390.7) -> D8 (440 - sees target at 450 visually).
    # We use 95.0 instead of 100.0 for D1 to give 5 meters of slack on the Base connection, 
    # ensuring P-controller physics overshoots don't accidentally snap the comm_radius!
    optimal_x = jnp.linspace(95.0, 440.0, N)
    optimal_positions = jnp.stack([optimal_x, jnp.full(N, 50.0)], axis=-1)

    traj_states = []
    traj_rewards = []
    
    print(f"Running scripted test with {N} agents...")
    print(f"  {'Step':>4} | {'r_cov':>6} | {'r_fnd':>6} | {'r_gap':>8} | {'r_coll':>8} | {'r_succ':>6} || {'Total R':>8} | {'GapDst':>6} | {'TGT':>3} | {'CONN':>4}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} || {'-'*8} | {'-'*6} | {'-'*3} | {'-'*4}")

    # First, everybody flies to the middle (X=250), except Drone 7 which flies to the target (X=440)
    explore_x = jnp.full(N, 250.0)
    explore_y = jnp.array([70.0, 30.0, 60.0, 40.0, 50.0, 65.0, 35.0, 55.0])
    
    # Drone 7 runs to the target instantly
    explore_x = explore_x.at[7].set(440.0)
    explore_y = explore_y.at[7].set(50.0)
    explore_positions = jnp.stack([explore_x, explore_y], axis=-1)

    chain_formed_step = None
    target_found_step = None

    # Thresholds for when each drone receives the order to move to optimal position
    # The chain "unfurls" gradually from both ends inwards, ticking every 30 frames (1.5 seconds)
    # AFTER the target is found:
    # Drone 0 (Base end): t_found + 0
    # Drone 6 (Target end): t_found + 30
    # Drone 1: t_found + 60
    # Drone 5: t_found + 90
    # Drone 2: t_found + 120
    # Drone 4: t_found + 150
    # Drone 3: t_found + 180
    # Drone 7: already at target
    unfurl_delays = jnp.array([0, 60, 120, 180, 150, 90, 30, -999])

    # Run for up to 400 steps (exits early upon success)
    for t in range(400):
        
        time_since_found = (t - target_found_step) if target_found_step is not None else -1
        
        target_x = jnp.where(time_since_found >= unfurl_delays, optimal_positions[:, 0], explore_positions[:, 0])
        target_y = jnp.where(time_since_found >= unfurl_delays, optimal_positions[:, 1], explore_positions[:, 1])
        target_pos = jnp.stack([target_x, target_y], axis=-1)
            
        kp = 5.0
        kd = 2.0
        
        error = target_pos - state.pos
        actions = kp * error - kd * state.vel
        
        # Clip actions to max_force
        actions = jnp.clip(actions, -cfg.env.max_force, cfg.env.max_force)

        old_state = state
        state = step_jit(old_state, actions)
        
        traj_states.append(state)
        
        # Compute rewards
        # We manually trigger success on the exact step the chain is first fully connected
        is_connected = bool(info_dry['fully_connected'] > 0.5) if 'info_dry' in locals() else False # Need to test before calling again, but jit expects scalar bool
        # Real logic: RL environments pass is_done=True on the step the episode terminates.
        
        # We'll just define is_done as: "it wasn't connected before, but it is now"
        # Or simply, the first time it fully connects.
        connects_now = bool(info_dry['fully_connected']) if 'info_dry' in locals() else False
        is_done = jnp.bool_(False)
        
        rew, info = reward_jit(old_state, state, is_done)
        info_dry = info
        
        if info['just_found'] and target_found_step is None:
            target_found_step = t
        
        if info['fully_connected'] and chain_formed_step is None:
            chain_formed_step = t
            # Assign the success bonus retroactively for visual effect right on connection
            is_done = jnp.bool_(True)
            rew, info = reward_jit(old_state, state, is_done)

        traj_rewards.append(rew)

        if t % 5 == 0 or info['just_found'] or (chain_formed_step == t):
            r_tot = float(rew)
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

        if chain_formed_step is not None:
            print("\nChain successfully formed! Ending episode early.")
            break

    # Stack the trajectory history along a new time dimension (T, ...)
    traj_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *traj_states)
    rewards_stacked = jnp.stack(traj_rewards)
    
    print("\nRendering video to visually confirm behavior...")
    render_video(traj_stacked, cfg, "outputs/videos/perfect_chain_test.mp4", fps=20, rewards=rewards_stacked, renderer="slow")
    print("Done. Check out outputs/videos/perfect_chain_test.mp4!")

if __name__ == "__main__":
    run_scripted_test()


