"""
scripts/test_perfect_chain_maze.py
==================================
An advanced scripted trajectory test for the SwarmEcho environment inside the M02 small grid maze.
It uses 12 agents, a corner target, reduced comm range (cell_size + 1 = 11.0m), and detours.

Once target knowledge is delivered by the finder:
- Main path drones form the optimal chain from Base to Target.
- Detour drones are placed on separate non-optimal detour cells.
- All drones target their assigned centers plus random noise.
- Due to the low comm range and noise, links on the main path will frequently break and heal,
  causing the shortest-path visual selection to dynamically shift between the optimal path and
  the non-optimal detours.
"""

import os
import sys
import dataclasses
from pathlib import Path
from collections import deque

# Suppress JAX warning about our large (2.5GB) 1.0m visibility matrix.
os.environ["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "-1"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from core.config import load_config, validate_config, MAP_DIR
from env.physics import make_env_fns
from env.rewards import make_reward_fn
from env.raycast import dda_raycast_np
from env.maps import MapDefinition
from visualize.renderer import render_video


def find_shortest_path(start, end, occ_grid, cols, rows, cell_w, cell_h, avoid_cells=None):
    if avoid_cells is None:
        avoid_cells = set()
        
    def get_neighbors(cell):
        c, r = cell
        neighbors = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows:
                p1 = np.array([(c + 0.5) * cell_w, (r + 0.5) * cell_h], dtype=np.float32)
                p2 = np.array([(nc + 0.5) * cell_w, (nr + 0.5) * cell_h], dtype=np.float32)
                if dda_raycast_np(p1, p2, occ_grid):
                    neighbors.append((nc, nr))
        return neighbors

    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == end:
            return path
        for n in get_neighbors(curr):
            if n not in visited and n not in avoid_cells:
                visited.add(n)
                queue.append(path + [n])
    return None


def search_scenario(occ_grid, map_def):
    cols = int(map_def.maze_cell_cols)
    rows = int(map_def.maze_cell_rows)
    W = float(map_def.width)
    H = float(map_def.height)
    cell_w = W / cols
    cell_h = H / rows
    base_cell = (4, 4)

    # Prioritize corners, then boundary cells
    target_candidates = [(0, 8), (0, 0), (8, 8), (8, 0)]
    for c in range(cols):
        target_candidates.extend([(c, 0), (c, rows - 1)])
    for r in range(rows):
        target_candidates.extend([(0, r), (cols - 1, r)])

    unique_candidates = []
    for c in target_candidates:
        if c not in unique_candidates and c != base_cell:
            unique_candidates.append(c)

    for tc in unique_candidates:
        # Check target exclude zones
        x = (tc[0] + 0.5) * cell_w
        y = (tc[1] + 0.5) * cell_h
        in_exclude = False
        for zone in map_def.target_exclude_zones:
            if zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]:
                in_exclude = True
                break
        if in_exclude:
            continue

        if occ_grid[int(tc[0] * cell_w), int(tc[1] * cell_h)]:
            continue

        p = find_shortest_path(base_cell, tc, occ_grid, cols, rows, cell_w, cell_h)
        if p is not None:
            # Find a detour for some intermediate cell in the path
            for i in range(1, len(p) - 1):
                d = find_shortest_path(
                    p[i-1], p[i+1], occ_grid, cols, rows, cell_w, cell_h,
                    avoid_cells={p[i]}
                )
                # Budget check: len(p) + len(d) - 3 <= 12
                if d is not None and (len(p) + len(d) - 3) <= 12:
                    return tc, p, d, i

    return None


def run_scripted_test():
    # Load level configuration for M03b
    sys.argv = [sys.argv[0]]
    cfg = load_config(
        config_path=Path("src/curriculum_config/levels/M03b_small_grid_maze.yaml"),
        cli_overrides=True
    )
    
    # Configure parameters for the test with 12 agents and low comm range
    OmegaConf.set_readonly(cfg, False)
    cfg.env.num_agents = 12
    cfg.env.spawn_delay = 0
    cfg.env.use_random_base_spawn = False
    cfg.env.use_random_drone_spawn = False
    cfg.env.target_spawn_method = "map_defined"
    cfg.reward.chain_reward_system = "discrete_finders_path"
    cfg.reward.reward_single_shortest_path = True
    cfg.visualize.render_finders_path_debug = True
    cfg.visualize.render_conn_matrix = True
    cfg.env.comm_radius = 11.0
    cfg.env.comm_radius_base = 11.0
    OmegaConf.set_readonly(cfg, True)
    
    validate_config(cfg)
    
    N = cfg.env.num_agents
    
    # Load map definition
    active_map_name = cfg.env.map_names[0]
    map_def = MapDefinition.load(MAP_DIR / f"{active_map_name}.yaml", cell_size=1.0)
    occ_grid = map_def.occupancy_grid
    
    cols = int(map_def.maze_cell_cols)
    rows = int(map_def.maze_cell_rows)
    cell_w = float(map_def.width) / cols
    cell_h = float(map_def.height) / rows
    max_finders_path_len = cols * rows
    
    base_cell = (4, 4)
    base_pos = np.array([45.0, 45.0], dtype=np.float32)
    
    search_result = search_scenario(occ_grid, map_def)
    if search_result is None:
        print("ERROR: Could not find a suitable target cell with a detour within agent budget.")
        sys.exit(1)
        
    tgt_cell, path, detour, detour_parent_idx = search_result
    target_pos = np.array([(tgt_cell[0] + 0.5) * cell_w, (tgt_cell[1] + 0.5) * cell_h], dtype=np.float32)
    
    # Path lengths
    k = len(path) - 1
    m = len(detour)
    num_main_int = k - 1  # number of main path intermediate cells
    num_detour_int = m - 2 # number of detour intermediate cells
    
    # Drones mapping:
    # - Main path intermediate drones: D0 to D_{num_main_int - 1}
    # - Detour intermediate drones: D_{num_main_int} to D_{num_main_int + num_detour_int - 1}
    # - Finder drone: D_{N - 1} (Drone 11)
    # - Unused drones stay at Base
    
    print("=" * 60)
    print("ADVANCED SCENARIO CONFIGURATION")
    print(f"Base Cell     : {base_cell} at {base_pos.tolist()}")
    print(f"Target Cell   : {tgt_cell} at {target_pos.tolist()}")
    print(f"Optimal Path  : {path}")
    print(f"Detour Path   : {detour} around {path[detour_parent_idx]}")
    print(f"Main path drones: D0..D{num_main_int-1}")
    print(f"Detour drones   : D{num_main_int}..D{num_main_int+num_detour_int-1}")
    print(f"Finder Drone    : D11")
    print(f"Comm Radius   : {cfg.env.comm_radius} m")
    print("=" * 60)

    # Initialize environment functions
    env_step, reset, _, _ = make_env_fns(cfg)
    compute_reward = make_reward_fn(cfg)

    step_jit = jax.jit(env_step)
    reward_jit = jax.jit(compute_reward)

    state = reset(jax.random.PRNGKey(42))
    
    # Force initial positions and target override
    pos = jnp.tile(jnp.array(base_pos), (N, 1))
    state = state.replace(
        base_pos=jnp.array(base_pos),
        target_pos=jnp.array(target_pos),
        pos=pos
    )

    def center(cell):
        return np.array([(cell[0] + 0.5) * cell_w, (cell[1] + 0.5) * cell_h], dtype=np.float32)

    # Waypoints queues for each drone
    drone_wps = [[] for _ in range(N)]
    
    # Drone 11 (Finder) discovery waypoints to target and back to base
    discover_wps = [center(c) for c in path[1:]]
    return_wps = [center(c) for c in path[::-1][1:]]
    drone_wps[11] = discover_wps + return_wps

    traj_states = []
    traj_rewards = []
    traj_gap_dist = []
    traj_r_gap = []
    traj_r_explor = []
    traj_r_coll = []
    traj_chain_pct = []

    phase = 1  # 1: Discovery, 2: Run with detours and noise
    
    print("\nRunning scripted test...")
    print(f"  {'Step':>4} | {'Phase':>5} | {'r_cov':>6} | {'r_fnd':>6} | {'r_gap':>8} | {'r_coll':>8} | {'r_succ':>6} || {'Total R':>8} | {'GapDst':>6} | {'TGT':>3} | {'CONN':>4}")
    print(f"  {'-'*4} | {'-'*5} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} || {'-'*8} | {'-'*6} | {'-'*3} | {'-'*4}")

    max_steps = 1200
    np.random.seed(42)

    for t in range(max_steps):
        target_pos_np = np.array(state.pos)
        
        # Phase transition from Discovery to Build with Noise
        if phase == 1:
            dist_to_base = np.linalg.norm(np.array(state.pos[11]) - base_pos)
            if len(drone_wps[11]) == 0 and dist_to_base < 2.0:
                phase = 2
                print(f"\n>>> Step {t}: Transition to PHASE 2 (Build & Run with Noise). Base knows target.")
                
                # Populate waypoint queues for all drones to reach their links
                # Main path intermediate drones
                for i in range(num_main_int):
                    drone_wps[i] = [center(c) for c in path[1 : i + 2]]
                
                # Detour intermediate drones
                # Walk along main path to detour start, then detour intermediate cells
                detour_start_idx = detour_parent_idx - 1
                base_to_detour_start = path[1 : detour_start_idx + 1]
                for j in range(num_detour_int):
                    drone_wps[num_main_int + j] = [center(c) for c in base_to_detour_start] + [center(c) for c in detour[1 : j + 2]]
                
                # Finder drone D11 goes back to target cell
                drone_wps[11] = [center(c) for c in path[1:]]

        # Target positions from waypoints, adding noise when arrived at link destination
        for i in range(N):
            wps = drone_wps[i]
            if len(wps) > 0:
                target_pos_np[i] = wps[0]
                dist = np.linalg.norm(np.array(state.pos[i]) - target_pos_np[i])
                if dist < 1.5:
                    drone_wps[i].pop(0)
            else:
                if phase == 1:
                    target_pos_np[i] = base_pos
                elif phase == 2:
                    # Link assignment
                    if i < num_main_int:
                        cell_dest = path[i + 1]
                    elif i < num_main_int + num_detour_int:
                        j = i - num_main_int
                        cell_dest = detour[j + 1]
                    elif i == 11:
                        cell_dest = path[-1]
                    else:
                        cell_dest = base_cell
                    
                    # Target center + noise (within 3.8 meters to avoid hitting walls)
                    noise = np.random.uniform(-3.8, 3.8, size=2)
                    target_pos_np[i] = center(cell_dest) + noise

        # PD Controller
        kp = 8.0
        kd = 3.0
        
        target_pos_jnp = jnp.array(target_pos_np)
        error = target_pos_jnp - state.pos
        actions = kp * error - kd * state.vel
        actions = jnp.clip(actions, -cfg.env.max_force, cfg.env.max_force)

        old_state = state
        state = step_jit(old_state, actions)
        
        # Force-freeze target/finders_path fields during Phase 2
        if phase >= 2:
            grid = np.full((cols, rows), -1, dtype=np.int16)
            for idx, cell in enumerate(path):
                grid[cell[0], cell[1]] = idx
                
            padded_path = np.zeros((max_finders_path_len, 2), dtype=np.int16)
            padded_path[:len(path)] = path
            
            state = state.replace(
                base_target_known=jnp.bool_(True),
                target_known=state.target_known.at[:].set(True),
                finders_path=jnp.array(padded_path),
                finders_path_len=jnp.int16(len(path)),
                finders_path_valid=jnp.bool_(True),
                finders_path_index_grid=jnp.array(grid),
            )

        # Compute rewards
        is_done = jnp.bool_(False)
        rew, info = reward_jit(old_state, state, is_done)

        traj_states.append(state)
        traj_rewards.append(rew)
        traj_gap_dist.append(float(info["chain_gap_dist"]))
        traj_r_gap.append(float(info["r_chain_gap"]))
        traj_r_explor.append(float(info["r_coverage"]))
        traj_r_coll.append(float(info["r_collision"]))
        traj_chain_pct.append(float(info["chain_progress_pct"]))

        if t % 50 == 0 or info['just_found']:
            r_tot = float(rew.sum())
            rt_cov  = float(info['r_coverage'])
            rt_fnd  = float(info.get('r_target_found', 0.0))
            rt_gap  = float(info['r_chain_gap'])
            rt_coll = float(info['r_collision'])
            rt_succ = float(info['r_success'])
            
            gap_dist = float(info['chain_gap_dist'])
            tgt = "YES" if info['global_target_found'] else "no"
            conn = "YES" if info['fully_connected'] > 0.5 else "no"
            
            highlight = ">>> " if info['just_found'] else "    "
            
            print(f"{highlight}{t:>4} | {phase:>5} | {rt_cov:>6.3f} | {rt_fnd:>6.1f} | {rt_gap:>8.3f} | {rt_coll:>8.3f} | {rt_succ:>6.1f} || {r_tot:>8.3f} | {gap_dist:>6.1f} | {tgt:>3} | {conn:>4}")

    # Stack trajectory history
    traj_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *traj_states)
    rewards_stacked = jnp.array(traj_rewards)
    
    extra_metrics = {
        "r_gap":    np.array(traj_r_gap),
        "r_explor": np.array(traj_r_explor),
        "r_total":  np.array([float(r.sum()) for r in traj_rewards]),
        "chain_pct": np.array(traj_chain_pct),
    }

    print("\nRendering video using the fast (CV2) renderer...")
    video_path = render_video(
        traj_stacked, 
        cfg, 
        filename="outputs/videos/perfect_chain_maze_cv2.mp4", 
        renderer="fast",
        rewards=rewards_stacked,
        extra_metrics=extra_metrics
    )
    
    print("\nDone. Check out:")
    print(f"  - {video_path}")


if __name__ == "__main__":
    run_scripted_test()
