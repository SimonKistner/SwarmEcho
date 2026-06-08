"""
Verification Suite: Chain Gating & Tip Selection
================================================
Tests the 190% projection bug fix, Euclidean tip selection,
and the Min-Plus shortest path gating toggle.

Run via: uv run python src/tests/test_chain_gating.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from core.config import load_config
from env.physics import make_env_fns
from env.rewards import make_reward_fn

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def run_test():
    # 1. Setup Config
    cfg = load_config()
    OmegaConf.set_readonly(cfg, False)
    
    cfg.env.map_names = ["open_field"]
    cfg.env.num_agents = 4
    cfg.env.box_width = 500.0
    cfg.env.box_height = 100.0
    cfg.env.visual_radius = 15.0
    cfg.env.comm_radius = 50.0
    
    # 2. Base Setup Fns
    env_step, env_reset, _, _ = make_env_fns(cfg)
    
    # Base and Target far apart on X axis
    base_pos = jnp.array([0.0, 50.0])
    target_pos = jnp.array([200.0, 50.0])
    
    key = jax.random.PRNGKey(42)
    state = env_reset(key)
    state = state.replace(
        base_pos=base_pos,
        target_pos=target_pos,
        box_width=jnp.array(500.0),
        box_height=jnp.array(100.0)
    )

    # --------------------------------------------------------------------------
    print_header("SCENARIO 1: The 190% Bug (Flying Behind Target)")
    # --------------------------------------------------------------------------
    # Drone 0 connects to Target but flies BEHIND it (away from base).
    # Previous code reported massive >100% progress. Should now cap.
    pos_behind = jnp.array([
        [210.0, 50.0], # Drone 0: Behind target (sees it)
        [100.0, 90.0], # D1: disconnected
        [100.0, 10.0], # D2: disconnected
        [450.0, 0.0],  # D3: dummy hidden
    ])
    state_s1 = state.replace(pos=pos_behind)
    
    compute_reward_baseline = make_reward_fn(cfg)
    state_s1_next = env_step(state_s1, jnp.zeros((4, 2)))
    _, info_s1 = compute_reward_baseline(state_s1, state_s1_next, jnp.bool_(False))
    
    prog = info_s1["chain_progress_pct"]
    print(f"  Drone located at X=210 (Target at X=200).")
    print(f"  Reported Progress: {prog:.2f}% (Expected: <= 100%)")
    if prog > 110.0:
        print("  [FAIL] Projection is still leaking / unclipped!")
    else:
        print("  [PASS] Progress is bounded.")

    # --------------------------------------------------------------------------
    print_header("SCENARIO 2: Euclidean Tip Selection vs 1D Projection")
    # --------------------------------------------------------------------------
    # Target is at (200, 50). Base at (0, 50).
    # Drone 0: (50, 50) -> Connected to base, perfectly on line. Dist to Tgt = 150. 
    # Drone 1: (51, 80) -> Connected to base, slightly better X, but terrible Y. Dist to Tgt = ~152.
    pos_lateral = jnp.array([
        [40.0, 50.0], # D0: On line, connects via D2
        [41.0, 70.0], # D1: Offset, connects via D2
        [10.0, 50.0],  # D2: Link to base (Dist 10 < 15)
        [450.0, 0.0],  # D3: dummy hidden
    ])
    state_s2 = state.replace(pos=pos_lateral)
    
    state_s2_next = env_step(state_s2, jnp.zeros((4, 2)))
    _, info_s2 = compute_reward_baseline(state_s2, state_s2_next, jnp.bool_(False))
    
    # If using Euclidean selection, Drone 0 is the tip (dist 150). Gap = 150.
    # If using 1D selection, Drone 1 is the tip. Gap = 152.
    print(f"  Drone 0: (40, 50) [On line, Dist to Tgt=160.0]")
    print(f"  Drone 1: (41, 70) [Lateral offset, Dist to Tgt~160.2]")
    print(f"  Calculated Chain Gap Dist: {info_s2['chain_gap_dist']:.2f}")
    if info_s2['chain_gap_dist'] < 160.1:
        print("  [PASS] Selected Drone 0 (Euclidean logic).")
    else:
        print("  [FAIL] Selected Drone 1 (1D Projection logic).")

    # --------------------------------------------------------------------------
    print_header("SCENARIO 3: Strict Path Gating Toggle (Target Found)")
    # --------------------------------------------------------------------------
    # Base (0,50). Target (200,50). Max gap = 200.
    # D0 (10, 50)    <- Connected to Base.
    # D1 (50, 50)    <- Connected to D0. (Tip of the Base Chain).
    # D2 (-10, 50)   <- Connected to Base, but going backwards. (Redundant).
    # D3 (190, 50)   <- Connected to Target. Sets target_known = True. (Target Tip).
    
    pos_path = jnp.array([
        [10.0, 50.0],  # D0: Hero link
        [50.0, 50.0],  # D1: Hero Tip
        [-10.0, 50.0], # D2: Redundant / Dead end
        [190.0, 50.0], # D3: Target finder / tip
    ])
    
    # CRITICAL: We must explicitly tell the state the target is known 
    # to unlock the dynamic chain gap reward!
    state_s3 = state.replace(
        pos=pos_path,
        target_known=jnp.array([False, False, False, True]) 
    )

    state_s3_next = env_step(state_s3, jnp.zeros((4, 2)))

    # 1. Baseline (Toggle OFF)
    cfg.reward.only_shortest_path_chain_reward = False
    reward_fn_off = make_reward_fn(cfg)
    r_off, info_off = reward_fn_off(state_s3, state_s3_next, jnp.bool_(False))
    
    # 2. Strict (Toggle ON)
    cfg.reward.only_shortest_path_chain_reward = True
    reward_fn_on = make_reward_fn(cfg)
    r_on, info_on = reward_fn_on(state_s3, state_s3_next, jnp.bool_(False))
    
    print(f"  Setup: Base -> D0 -> D1 ... [140m gap] ... D3 -> Target")
    print(f"  D2 is at (-10, 50) acting as a useless dead-end.")
    print(f"  Target is DISCOVERED (Unlocking gap rewards).")
    
    print(f"\n  --- Strict Gating OFF ---")
    print(f"  Active Chain Drones: {info_off.get('active_chain_drones', 4)}")
    print(f"  D0 (Hero) Reward    : {r_off[0]:.4f}")
    print(f"  D2 (Useless) Reward : {r_off[2]:.4f}  <-- Farmed the hero's work!")
    
    print(f"\n  --- Strict Gating ON ---")
    print(f"  Active Chain Drones: {info_on.get('active_chain_drones', 'N/A')} (D0, D1, D3)")
    print(f"  D0 (Hero) Reward    : {r_on[0]:.4f}  <-- Reduced penalty (Gap is 140m)")
    print(f"  D2 (Useless) Reward : {r_on[2]:.4f}  <-- Slammed with max penalty")
    
    if r_on[2] < r_on[0]:
        print("\n  [PASS] Redundant drone correctly penalized!")
    else:
        print("\n  [FAIL] Redundant drone still receiving chain rewards.")

if __name__ == "__main__":
    run_test()