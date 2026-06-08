"""
src/tests/test_connectivity_mismatch.py
=========================================
Deliberately tests and verifies the difference in connectivity calculations
between the SwarmEcho physics/rewards engine (which uses Euclidean distance only)
and the visual renderer (which uses DDA raycasting to check wall occlusions).

Scenario configuration (M02 grid maze):
- Base: (45.0, 45.0)
- Target: (85.0, 15.0)
- Drones:
  0: (45.0, 45.0) -> At base
  1: (45.0, 35.0) -> Near base
  2: (55.0, 35.0) -> In corridor (blocked from 3 by wall at y=30)
  3: (58.0, 22.0) -> Blocked from 5 by wall at y=20
  4: (85.0, 15.0) -> At target
  5: (75.0, 15.0) -> Left of target
  6, 7: (0.0, 0.0) -> Inactive
"""

import os
import sys
from pathlib import Path

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
from visualize.renderer_mpl import _build_adjacency, _bfs


def run_mismatch_test():
    print("════════════════════════════════════════════════════════════")
    print(" SwarmEcho Connectivity Mismatch Regression Test")
    print("════════════════════════════════════════════════════════════")

    # 1. Load config for M02 small grid maze
    level_path = Path("src/curriculum_config/levels/M02_small_grid_maze.yaml")
    cfg = load_config(
        config_path=level_path,
        cli_overrides=True,
        overrides=[
            "training.warn_vram_limit=false",
            "training.abort_on_vram_limit=false",
            "training.num_minibatches=16"
        ]
    )
    OmegaConf.set_readonly(cfg, False)
    cfg.env.num_agents = 8
    cfg.env.map_names = ["M02_small_grid_maze"]
    cfg.env.use_random_base_spawn = False
    cfg.env.use_random_drone_spawn = False
    cfg.env.target_spawn_method = "map_defined"
    cfg.env.spawn_delay = 0
    cfg.training.warn_vram_limit = False
    cfg.training.abort_on_vram_limit = False
    OmegaConf.set_readonly(cfg, True)

    validate_config(cfg)

    # 2. Initialize env functions
    env_step, reset, _, (W, H, occ_grid) = make_env_fns(cfg)
    compute_reward = make_reward_fn(cfg)

    state = reset(jax.random.PRNGKey(42))

    # 3. Setup positions and active flags
    # Coordinates picked to be within Euclidean comm/vis ranges but physically separated by walls
    base_pos = jnp.array([45.0, 45.0], dtype=jnp.float32)
    target_pos = jnp.array([85.0, 15.0], dtype=jnp.float32)

    pos_np = np.zeros((8, 2), dtype=np.float32)
    pos_np[0] = [45.0, 45.0]  # Drone 0: Base
    pos_np[1] = [45.0, 35.0]  # Drone 1: South of base
    pos_np[2] = [55.0, 35.0]  # Drone 2: East corridor (above y=30 wall segment)
    pos_np[3] = [58.0, 22.0]  # Drone 3: Between y=20 and y=30 wall segments
    pos_np[4] = [85.0, 15.0]  # Drone 4: Target position
    pos_np[5] = [75.0, 15.0]  # Drone 5: Left of target (below y=20 wall segment)
    pos_np[6] = [0.0, 0.0]    # Drone 6: Inactive
    pos_np[7] = [0.0, 0.0]    # Drone 7: Inactive

    active_np = np.array([True, True, True, True, True, True, False, False], dtype=bool)

    state = state.replace(
        base_pos=base_pos,
        target_pos=target_pos,
        pos=jnp.array(pos_np),
        active=jnp.array(active_np),
    )

    print("\n--- Running Physics/Rewards Engine Evaluation ---")
    state_after = env_step(state, jnp.zeros((8, 2)))
    _, info = compute_reward(state, state_after, is_done=jnp.bool_(False))

    fully_connected_physics = bool(info["fully_connected"] > 0.5)
    progress_pct_physics = float(info["chain_progress_pct"])

    print(f"Physics Engine 'fully_connected' : {fully_connected_physics}")
    print(f"Physics Engine 'progress_pct'    : {progress_pct_physics:.1f}%")

    print("\n--- Running Renderer Adjacency Matrix Evaluation ---")
    comm_r = float(cfg.env.comm_radius)
    vis_r = float(cfg.env.visual_radius)
    comm_r_base = float(cfg.env.get("comm_radius_base", cfg.env.comm_radius))
    
    # Renderers expect numpy inputs
    pos_cpu = np.array(state.pos)
    base_pos_cpu = np.array(state.base_pos)
    target_pos_cpu = np.array(state.target_pos)
    occ_grid_cpu = np.array(occ_grid)

    adj, base_idx, target_idx, target_indices, drone_start, ents, dists = _build_adjacency(
        pos_cpu,
        base_pos_cpu,
        target_pos_cpu,
        comm_r,
        vis_r,
        comm_r_base,
        occ_grid_cpu,
        (float(W), float(H)),
        cfg,
    )

    # Compute base and target components via BFS over adjacency matrix (which respects walls)
    base_comp = _bfs(adj, source=base_idx) if base_idx >= 0 else set()
    target_comp = set()
    for ti in target_indices:
        target_comp |= _bfs(adj, source=ti)

    # In rendering/visuals, target is connected to base iff the target is in the base component
    fully_connected_visual = bool(target_idx in base_comp)

    print(f"Renderer Base-connected Component Drones: {[idx - drone_start for idx in base_comp if idx >= drone_start]}")
    print(f"Renderer Target-connected Component Drones: {[idx - drone_start for idx in target_comp if idx >= drone_start]}")
    print(f"Renderer Visual 'fully_connected'        : {fully_connected_visual}")

    # Check pairwise render links
    print("\nRenderer wall-occlusion verification details:")
    print(f"  - Link Drone 2 <-> Drone 3 (across y=30 wall): {'RENDERED' if adj[2+drone_start, 3+drone_start] else 'BLOCKED'}")
    print(f"  - Link Drone 3 <-> Drone 5 (across y=20 wall): {'RENDERED' if adj[3+drone_start, 5+drone_start] else 'BLOCKED'}")

    print("\n════════════════════════════════════════════════════════════")
    print(" RESULTS SUMMARY")
    print("════════════════════════════════════════════════════════════")
    if fully_connected_physics and not fully_connected_visual:
        print("🔴 MISMATCH DETECTED (REPRODUCED SYSTEM BUG):")
        print("   The physics engine declared the chain 100% connected/solved,")
        print("   but the visual renderer correctly detected wall occlusions")
        print("   and rendered no base-to-target path link!")
        print("════════════════════════════════════════════════════════════")
        sys.exit(0)
    else:
        print("🟢 NO MISMATCH DETECTED (Logic matches or scenario parameters different):")
        print(f"   Physics fully connected: {fully_connected_physics}")
        print(f"   Visual fully connected:  {fully_connected_visual}")
        print("════════════════════════════════════════════════════════════")
        sys.exit(1)


if __name__ == "__main__":
    run_mismatch_test()
