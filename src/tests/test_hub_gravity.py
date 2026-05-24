"""
Verification Suite: Hub Gravity Reward
======================================
Tests the individual 'intuition' rewards for Base/Target proximity.
Focuses on native discovery and communication sharing.

Run via: uv run python src/tests/test_hub_gravity.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from core.config import load_config
from env.physics import make_env_fns
from env.rewards import make_reward_fn
from env.state import EnvState

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def run_test():
    # 1. Setup Config
    cfg = load_config()
    
    # Unlock for testing
    OmegaConf.set_readonly(cfg, False)
    
    # Force a large open field for the test
    cfg.env.map_names = ["open_field"]
    cfg.env.num_agents = 2
    
    # Force dimensions (must match the map logic)
    cfg.env.box_width = 500.0
    cfg.env.box_height = 100.0
    cfg.env.visual_radius = 15.0
    cfg.env.comm_radius = 50.0
    
    # Set bonuses to 1.0 for easy normalization checks
    cfg.reward.base_proximity_bonus = 1.0
    cfg.reward.target_proximity_bonus = 1.0
    
    # 2. Setup Env Fns
    env_step, env_reset, _, _ = make_env_fns(cfg)
    compute_reward = make_reward_fn(cfg)
    
    key = jax.random.PRNGKey(42)
    state = env_reset(key)
    
    # Fix Base and Target far apart
    base_pos = jnp.array([0.0, 50.0])
    target_pos = jnp.array([500.0, 50.0])
    
    state = state.replace(
        base_pos=base_pos,
        target_pos=target_pos,
        box_width=jnp.array(500.0),
        box_height=jnp.array(100.0)
    )
    
    diagonal = np.sqrt(500**2 + 100**2)
    
    # --------------------------------------------------------------------------
    print_header("SCENARIO 1: Pure Home Drive (Isolated Unknown)")
    # --------------------------------------------------------------------------
    # Drone at (450, 50). Target is at (500, 50). Blind to target (dist=50 > 15).
    pos_far = jnp.array([[450.0, 50.0], [0.0, 0.0]])
    # Step once to let sensors run (discovery happens in step)
    state_s1 = state.replace(pos=pos_far, target_known=jnp.zeros(2, dtype=bool))
    _, info = compute_reward(state_s1, state_s1, jnp.bool_(False))
    
    r_start = info["r_hub_proximity"] * cfg.env.num_agents # Undo division for raw check
    
    # Move closer to Base
    pos_closer = jnp.array([[400.0, 50.0], [0.0, 0.0]])
    state_s1_closer = state_s1.replace(pos=pos_closer)
    _, info_closer = compute_reward(state_s1_closer, state_s1_closer, jnp.bool_(False))
    r_closer = info_closer["r_hub_proximity"] * cfg.env.num_agents
    
    print(f"  Drone at (450, 50): Hub Reward = {r_start:.4f} (Unknown target)")
    print(f"  Drone at (400, 50): Hub Reward = {r_closer:.4f} (Unknown target)")
    print(f"  Delta: {r_closer - r_start:+.4f} (Expected: positive)")
    
    # --------------------------------------------------------------------------
    print_header("SCENARIO 2: The Decision Switch (Native Knower)")
    # --------------------------------------------------------------------------
    # Drone at (500, 50) discovers target natively.
    pos_on_tgt = jnp.array([[500.0, 50.0], [0.0, 0.0]])
    state_s2 = state.replace(pos=pos_on_tgt, target_known=jnp.array([True, False]))
    
    # Move drone along the line. It should switch hubs at 250m.
    test_xs = [450, 300, 250, 200, 50]
    print(f"  {'X-Pos':>8} | {'Dist-Base':>10} | {'Dist-Tgt':>10} | {'Reward':>10} | {'Hub Selected'}")
    print(f"  {'-'*60}")
    
    for x in test_xs:
        p = jnp.array([[float(x), 50.0], [0.0, 0.0]])
        s = state_s2.replace(pos=p)
        _, info = compute_reward(s, s, jnp.bool_(False))
        r = info["r_hub_proximity"] * cfg.env.num_agents
        
        hub = "Target" if x > 250 else "Base"
        if x == 250: hub = "Equilibrium"
        
        d_b = float(x)
        d_t = 500 - x
        print(f"  {x:>8.0f} | {d_b:>10.1f} | {d_t:>10.1f} | {r:>10.4f} | {hub}")

    # --------------------------------------------------------------------------
    print_header("SCENARIO 3: Individual Knowledge Masking")
    # --------------------------------------------------------------------------
    # Two drones at same X, but different distance to target.
    # Neither knows the target.
    p_mask = jnp.array([[300.0, 20.0], [300.0, 80.0]])
    state_s3 = state.replace(pos=p_mask, target_known=jnp.array([False, False]))
    _, info = compute_reward(state_s3, state_s3, jnp.bool_(False))
    
    # Individual reward calculation for check
    diagonal = jnp.sqrt(500.0**2 + 100.0**2)
    d_base = jnp.linalg.norm(p_mask - base_pos, axis=-1)
    expected_r = jnp.mean(1.0 - (d_base / diagonal))
    
    print(f"  Drone A at (300, 20): Dist-Tgt is large")
    print(f"  Drone B at (300, 80): Dist-Tgt is even larger")
    print(f"  Both unknown. Total Hub Reward = {info['r_hub_proximity']:.4f}")
    print(f"  Expected (Base-only) = {expected_r:.4f}")

    # --------------------------------------------------------------------------
    print_header("SCENARIO 4: Communication Sharing (The Spike)")
    # --------------------------------------------------------------------------
    # Drone A at Target (knows it). Drone B at (400, 50) (doesn't know it).
    p_sharing = jnp.array([[500.0, 50.0], [400.0, 50.0]])
    # B is 100m from A, so B DOES NOT know it yet.
    state_s4 = state.replace(pos=p_sharing, target_known=jnp.array([True, False]))
    _, info_iso = compute_reward(state_s4, state_s4, jnp.bool_(False))
    
    # Now move B to (460, 50) -> dist is 40m (< 50m comm radius)
    # Note: In a real env, target_known is updated in the step logic.
    # Here we simulate the result of that update.
    p_shared = jnp.array([[500.0, 50.0], [460.0, 50.0]])
    state_s4_shared = state_s4.replace(pos=p_shared, target_known=jnp.array([True, True]))
    _, info_shared = compute_reward(state_s4_shared, state_s4_shared, jnp.bool_(False))
    
    print(f"  Drone B at (400, 50) - Isolated Unknown: Hub Reward (team-avg) = {info_iso['r_hub_proximity']:.4f}")
    print(f"  Drone B at (460, 50) - Sharing Contact:   Hub Reward (team-avg) = {info_shared['r_hub_proximity']:.4f}")
    print(f"  Observation: Reward spikes from {info_iso['r_hub_proximity']:.4f} to {info_shared['r_hub_proximity']:.4f} via native sharing.")

if __name__ == "__main__":
    run_test()
