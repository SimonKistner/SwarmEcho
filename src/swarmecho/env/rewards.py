"""
swarmecho/env/rewards.py
========================
Reward function for the SwarmEcho relay-chain task.

All functions are pure JAX — JIT/vmap/scan compatible.
Use `make_reward_fn(cfg)` so config scalars become XLA compile-time constants.

Design
------
The reward structure uses a mix of SHARED team components and INDIVIDUAL agent
components to balance cooperative behaviour with precise credit assignment.
The function returns an array of shape (N,) containing the scalar reward for each agent.

Reward components
-----------------

    r_i = r_target_found + r_coverage + r_chain_gap + r_collision + r_success

    r_target_found  [sparse]
        Granted when the global target is first discovered.
        Contains a shared team bonus (target_found_bonus / N) and a local individual bonus
        (finder_bonus) given only to the drone(s) that physically spotted the target.

    r_coverage      [dense]
        Individual reward proportional to the area of NEW grid cells covered this step.
        Value: delta_cells × cell_size² × exploration_bonus.
        Stops applying for individual drones once they know the target position.

    r_chain_gap     [dense, direct-distance]
        Provides a smooth gradient for closing the gap between the base chain and the target chain.
        Selection: Identifies the "tips" (the base-connected drone closest to the target,
                   and the target-connected drone closest to the base).
        Measurement: The actual Euclidean distance between the two tips.
        Condition: The gap penalty is dynamic. Before the target is found, it uses the maximum
                   distance. Once the target is found, it scales with the actual chain gap.
        Individual aspect: Only drones on one deterministic shortest route receive
                           the dynamic penalty. Non-contributing drones receive the
                           maximum gap penalty.

    r_collision     [dense, penalty]
        Individual penalty for agents hitting walls or obstacles.
        Value: -1.0 × collision_penalty.

    r_success       [sparse, optional]
        One-shot shared terminal bonus when the episode ends with a fully connected chain.
        Value: is_done × fully_connected × (success_bonus / N).

"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from swarmecho.env.state import EnvState


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def make_reward_fn(cfg: DictConfig):
    """
    Build a pure JAX team reward function closed over config scalars.

    Returns
    -------
    compute_reward : (old_state, new_state, is_done) -> (scalar, info_dict)
    """

    # ---- Extract scalars (XLA compile-time constants) --------------------
    N           = int(cfg.env.num_agents)
    cell_s = 1.0

    # Reward weights
    w_exp        = float(cfg.reward.exploration_bonus)
    w_found      = float(cfg.reward.target_found_bonus)
    w_finder     = float(cfg.reward.finder_bonus)
    p_gap_max = float(cfg.reward.max_gap_penalty)
    p_coll  = float(cfg.reward.collision_penalty)
    w_success    = float(cfg.reward.success_bonus)
    target_found_requires_delivery = bool(cfg.reward.get("target_found_requires_delivery", True))
    chain_reward_system = str(cfg.reward.get("chain_reward_system", "euclidean"))
    use_finders_path_reward = chain_reward_system == "discrete_finders_path"
    maze_cols = 1
    maze_rows = 1
    maze_cell_w = 1.0
    maze_cell_h = 1.0
    if use_finders_path_reward and cfg.env.get("map_names") and len(cfg.env.map_names) > 0:
        from swarmecho.core.config import MAP_DIR
        from swarmecho.env.maps import MapDefinition
        map_def = MapDefinition.load(MAP_DIR / f"{cfg.env.map_names[0]}.yaml", cell_size=1.0)
        maze_cols = int(map_def.maze_cell_cols or 1)
        maze_rows = int(map_def.maze_cell_rows or 1)
        maze_cell_w = float(map_def.width) / maze_cols
        maze_cell_h = float(map_def.height) / maze_rows

    # ---- Internal: shortest path calculation -----------------------------

    def _compute_shortest_paths(state: EnvState, idx_b, idx_t, has_b_chain, has_t_chain):
        """
        Calculates shortest path hop counts using Min-Plus matrix squaring.
        Returns is_contributing (N,) mask.
        """
        V = N + 2
        H = jnp.full((V, V), 9999, dtype=jnp.int32)
        H = H.at[jnp.arange(V), jnp.arange(V)].set(0) # Diagonal is 0

        # Physics owns all range, activity, and wall checks. Its direct graph
        # uses nodes [drones..., base]; append the target endpoint from the
        # compact direct-visibility vector.
        direct_graph = state.adj_matrix
        H = H.at[:N + 1, :N + 1].set(
            jnp.where(
                direct_graph,
                1,
                H[:N + 1, :N + 1],
            )
        )
        adj_dt = state.directly_sees_target
        H = H.at[:N, N+1].set(jnp.where(adj_dt, 1, H[:N, N+1]))
        H = H.at[N+1, :N].set(jnp.where(adj_dt, 1, H[N+1, :N]))

        def _min_plus_step(M, _):
            M_next = jnp.min(M[:, :, None] + M[None, :, :], axis=1)
            return jnp.clip(M_next, 0, 9999), None # Prevent overflow

        n_steps = math.ceil(math.log2(V))
        H_final, _ = jax.lax.scan(_min_plus_step, H, None, length=n_steps)

        def _trace_single_path(start_node, dest_node):
            def scan_fn(u, _):
                is_valid = (H[u, :] == 1) & (H_final[:, dest_node] == H_final[u, dest_node] - 1)
                any_valid = jnp.any(is_valid)
                next_node = jnp.where(
                    u == dest_node,
                    dest_node,
                    jnp.where(any_valid, jnp.argmax(is_valid), dest_node)
                )
                return next_node, u

            _, path_nodes = jax.lax.scan(scan_fn, start_node, None, length=V)
            agents = jnp.arange(N)
            return jnp.any(path_nodes == agents[:, None], axis=-1)

        # Safe gating. Only check path equality if the chain actually exists.
        is_on_base_path = jnp.where(
            has_b_chain,
            _trace_single_path(N, idx_b),
            False
        )
        is_on_tgt_path = jnp.where(
            has_t_chain,
            _trace_single_path(N+1, idx_t),
            False
        )

        return is_on_base_path | is_on_tgt_path

    def _maze_cell_from_pos(pos):
        cx = jnp.clip(jnp.floor(pos[0] / maze_cell_w).astype(jnp.int32), 0, maze_cols - 1)
        cy = jnp.clip(jnp.floor(pos[1] / maze_cell_h).astype(jnp.int32), 0, maze_rows - 1)
        return cx, cy

    def _path_cell_center(cell):
        return jnp.array([
            (cell[0].astype(jnp.float32) + 0.5) * maze_cell_w,
            (cell[1].astype(jnp.float32) + 0.5) * maze_cell_h,
        ], dtype=jnp.float32)

    def _compute_finders_path_chain(new_state: EnvState, fully_connected):
        cells_x, cells_y = jax.vmap(_maze_cell_from_pos)(new_state.pos)
        path_idx = new_state.finders_path_index_grid[cells_x, cells_y].astype(jnp.int32)
        on_path = path_idx >= 0
        valid_len = jnp.maximum(new_state.finders_path_len.astype(jnp.int32), 1)

        is_base_chain = new_state.is_conn_base & new_state.active & on_path
        is_tgt_chain = new_state.is_conn_target & new_state.active & on_path
        any_base_chain = jnp.any(is_base_chain)
        any_tgt_chain = jnp.any(is_tgt_chain)

        base_rank = jnp.where(is_base_chain, path_idx, -1)
        base_best = jnp.max(base_rank)
        base_tied = is_base_chain & (path_idx == base_best)
        next_idx = jnp.minimum(base_best + 1, valid_len - 1)
        next_cell = new_state.finders_path[next_idx]
        next_center = _path_cell_center(next_cell)

        target_pos_agents = jnp.tile(new_state.target_pos[None, :], (N, 1))
        # If we reached the final path cell (target), tiebreak toward the actual target position
        base_target_center = jnp.where(
            base_best == valid_len - 1,
            target_pos_agents,
            next_center[None, :]
        )
        base_tie_dist = jnp.linalg.norm(new_state.pos - base_target_center, axis=-1)
        idx_b = jnp.argmin(jnp.where(base_tied, base_tie_dist, 1e9))

        tgt_rank = jnp.where(is_tgt_chain, path_idx, valid_len + 1)
        tgt_best = jnp.min(tgt_rank)
        tgt_tied = is_tgt_chain & (path_idx == tgt_best)
        prev_idx = jnp.maximum(tgt_best - 1, 0)
        prev_cell = new_state.finders_path[prev_idx]
        prev_center = _path_cell_center(prev_cell)
        # If we reached the start cell (base), tiebreak toward the actual base position
        tgt_target_center = jnp.where(
            tgt_best == 0,
            new_state.base_pos[None, :],
            prev_center[None, :]
        )
        tgt_tie_dist = jnp.linalg.norm(new_state.pos - tgt_target_center, axis=-1)
        idx_t = jnp.argmin(jnp.where(tgt_tied, tgt_tie_dist, 1e9))

        base_cells = jnp.where(any_base_chain, base_best + 1, 0)
        tgt_cells = jnp.where(any_tgt_chain, valid_len - tgt_best, 0)
        progress_cells = jnp.minimum(base_cells + tgt_cells, valid_len)
        progress_cells = jnp.where(fully_connected, valid_len, progress_cells)
        progress_frac = jnp.where(new_state.finders_path_valid, progress_cells.astype(jnp.float32) / valid_len.astype(jnp.float32), 0.0)
        chain_progress_pct = 100.0 * progress_frac
        chain_gap_dist = jnp.float32(0.0)
        base_gap_penalty = -p_gap_max * (1.0 - progress_frac)

        return chain_gap_dist, chain_progress_pct, base_gap_penalty, idx_b, idx_t, any_base_chain, any_tgt_chain

    # ---- Public: compute_reward ------------------------------------------

    def compute_reward(
        old_state: EnvState,
        new_state: EnvState,
        is_done:   jax.Array = jnp.bool_(False),
    ) -> tuple[jax.Array, dict]:

        target_pos_agents = jnp.tile(new_state.target_pos[None, :], (N, 1))
        primary_target_pos = new_state.target_pos

        # Dynamic world geometry from state
        bt_dist = jnp.linalg.norm(primary_target_pos - new_state.base_pos)

        # Normalized gap weight
        w_gap = jnp.where(bt_dist > 0, p_gap_max / bt_dist, 0.0)

        # ---- 2. Global Target Found --------------------------------------
        # Direct delivery/visibility facts come from the same wall-aware graph
        # that drives communication and observations.
        adj_db = new_state.adj_matrix[:N, N]
        is_visible = new_state.directly_sees_target

        knew_or_sees_target = old_state.target_known | is_visible
        actual_deliverers = adj_db & knew_or_sees_target

        if target_found_requires_delivery:
            was_target_found = old_state.base_target_known
            global_target_found = new_state.base_target_known
            target_found_local_receivers = actual_deliverers
        else:
            was_target_found = jnp.any(old_state.target_known)
            global_target_found = jnp.any(new_state.target_known)
            target_found_local_receivers = is_visible & ~old_state.target_known
        just_found = global_target_found & ~was_target_found
        r_target_found_shared = jnp.where(just_found, w_found, 0.0)
        r_target_found_local = jnp.where(just_found & target_found_local_receivers, w_finder, 0.0)

        # Shared component is divided by N to be agent-invariant
        r_target_found = (r_target_found_shared / N) + r_target_found_local

        delta_cells = new_state.last_cov_delta.astype(jnp.float32)
        # Stop exploration reward for drones that know the target position
        r_coverage = (delta_cells * (cell_s**2) * w_exp) * (~new_state.target_known)

        # ---- 4. Chain Gap Distance ---------------------------------------
        is_conn_base, is_conn_target = new_state.is_conn_base, new_state.is_conn_target
        fully_connected = jnp.any(is_conn_base & is_conn_target)

        # Base Chain "Tip" (Drone closest to Target)
        dist_to_target = jnp.linalg.norm(
            new_state.pos - target_pos_agents,
            axis=-1,
        )
        is_base_chain = is_conn_base & new_state.active
        any_base_chain = jnp.any(is_base_chain)
        idx_b = jnp.argmin(jnp.where(is_base_chain, dist_to_target, 1e9))
        pos_b = jnp.where(any_base_chain, new_state.pos[idx_b], new_state.base_pos)

        # Target Chain "Tip" (Drone closest to Base)
        is_tgt_chain = is_conn_target & new_state.active
        any_tgt_chain = jnp.any(is_tgt_chain)
        dist_to_base = jnp.linalg.norm(
            new_state.pos - new_state.base_pos[None, :],
            axis=-1,
        )
        idx_t = jnp.argmin(jnp.where(is_tgt_chain, dist_to_base, 1e9))
        pos_t = jnp.where(any_tgt_chain, new_state.pos[idx_t], primary_target_pos)

        # Gap/progress calculation selected once by the run config.
        if use_finders_path_reward:
            (chain_gap_dist, chain_progress_pct, base_gap_penalty,
             idx_b, idx_t, any_base_chain, any_tgt_chain) = _compute_finders_path_chain(new_state, fully_connected)
        else:
            raw_gap_dist = jnp.linalg.norm(pos_b - pos_t)
            chain_gap_dist = jnp.where(fully_connected, 0.0, raw_gap_dist)
            chain_progress_pct = jnp.where(
                fully_connected,
                100.0,
                jnp.clip(100.0 * (1.0 - chain_gap_dist / (bt_dist + 1e-6)), 0.0, 100.0)
            )
            use_dynamic_gap = global_target_found
            active_gap = jnp.where(use_dynamic_gap, chain_gap_dist, bt_dist)
            base_gap_penalty = -active_gap * w_gap

        # Apply the dynamic penalty only to one deterministic shortest route.
        is_contributing = _compute_shortest_paths(
            new_state, idx_b, idx_t, any_base_chain, any_tgt_chain
        )
        r_chain_gap = jnp.where(
            is_contributing,
            base_gap_penalty / N,
            -p_gap_max / N
        )

        # ---- 5. Collision penalty ----------------------------------------
        # Penalty is per-agent hitting a wall/obstacle
        r_collision = -new_state.collides.astype(jnp.float32) * p_coll

        # ---- 6. Terminal success bonus -----------------------------------
        # NOTE: Both `is_done` and `fully_connected` checks are intentional and NOT redundant.
        # `is_done` can be True for two reasons: (a) time_up=True with fully_connected=False
        # (timeout without chain), or (b) fully_connected=True which also sets done=True.
        # The `fully_connected` guard prevents awarding the bonus in case (a).
        r_success_scalar = jnp.float32(is_done) * jnp.float32(fully_connected) * w_success
        r_success = r_success_scalar / N # Divide by N to be agent-invariant

        # ---- Total -------------------------------------------------------
        reward = r_coverage + r_target_found + r_chain_gap + r_collision + r_success

        info = {
            "r_coverage":     jnp.sum(r_coverage),
            "r_target_found": jnp.sum(r_target_found),
            "r_chain_gap":    jnp.sum(r_chain_gap),
            "r_collision":    jnp.sum(r_collision),
            "r_success":      jnp.sum(jnp.full(N, r_success_scalar)) / N,
            "chain_gap_dist": chain_gap_dist,
            "chain_progress_pct": chain_progress_pct,
            "is_contributing": is_contributing,
            "active_chain_drones": jnp.sum(is_contributing),
            "fully_connected": fully_connected.astype(jnp.float32),
            "global_target_found": global_target_found.astype(jnp.float32),
            "just_found":     just_found.astype(jnp.float32),
            "global_coverage": jnp.mean(new_state.coverage_grid.astype(jnp.float32)),
        }

        return reward, info

    return compute_reward


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from swarmecho.core.config import load_config, validate_config
    from swarmecho.env.physics import make_env_fns

    print("── Reward Self-Test ─────────────────────────────────────────")

    cfg = load_config(cli_overrides=True)
    OmegaConf.set_readonly(cfg, False)
    cfg.env.box_width = 100.0
    cfg.env.box_height = 100.0
    cfg.env.map_names = ["M01_small_maze"]
    validate_config(cfg)

    N = cfg.env.num_agents

    env_step, reset, _, _  = make_env_fns(cfg)
    compute_reward       = make_reward_fn(cfg)

    reward_jit = jax.jit(compute_reward)
    step_jit   = jax.jit(env_step)
    reset_jit  = jax.jit(reset)

    key   = jax.random.PRNGKey(42)
    state = reset_jit(key)

    total_reward = 0.0
    print(f"\n  Running {cfg.env.max_steps} steps with zero actions...")
    print(f"  {'Step':>5}  {'reward':>8}  {'gap_dst':>7}  {'connected':>10}  {'tgt_fnd':>10}  {'coverage':>9}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*9}")

    actions = jnp.zeros((N, 2))

    for t in range(cfg.env.max_steps):
        old_state = state
        state     = step_jit(state, actions)
        is_done   = jnp.bool_(t == cfg.env.max_steps - 1)
        rew, info = reward_jit(old_state, state, is_done)
        total_reward += float(rew.sum())

        if t < 5 or t % 100 == 99 or t == cfg.env.max_steps - 1:
            cov_pct = 100.0 * int(state.coverage_grid.sum()) / state.coverage_grid.size
            print(
                f"  {t+1:>5}  {float(rew.sum()):>8.4f}  "
                f"{float(info['chain_gap_dist']):>7.1f}  "
                f"{bool(info['fully_connected'] > 0.5):>10}  "
                f"{bool(info['global_target_found'] > 0.5):>10}  "
                f"{cov_pct:>8.1f}%"
            )

    print(f"\n  Total reward over episode : {total_reward:.2f}")
    print(f"  max_steps                 : {cfg.env.max_steps}")
    print(f"  Reward per step (avg)     : {total_reward / cfg.env.max_steps:.4f}")

    # Shape check: reward must be (N,) array
    r, _ = reward_jit(state, state, jnp.bool_(False))
    assert r.shape == (N,), f"Reward must be shape (N,), got {r.shape}"
    assert not jnp.isnan(r).any(), "NaN reward!"
    assert not jnp.isinf(r).any(), "Inf reward!"

    # Batch vmap check
    print(f"\n  Batch vmap (256 envs) ...")
    batch_keys   = jax.random.split(key, 256)
    batch_states = jax.jit(jax.vmap(reset))(batch_keys)
    batch_acts   = jnp.zeros((256, N, 2))
    batch_next   = jax.jit(jax.vmap(env_step))(batch_states, batch_acts)
    batch_dones  = jnp.zeros(256, dtype=jnp.bool_)
    batch_r, batch_info = jax.jit(jax.vmap(compute_reward))(batch_states, batch_next, batch_dones)
    batch_r.block_until_ready()
    assert batch_r.shape == (256, N), f"Batch reward shape mismatch: {batch_r.shape}"
    print(f"  Batch reward shape : {batch_r.shape}  ✓")
    print(f"  Mean batch reward  : {float(batch_r.mean()):.4f}")

    print("\nReward self-test passed ✓")

    # ── Boundary Leak Regression Test ──────────────────────────────────
    import dataclasses
    print(f"\n  Testing Boundary Leak Regression...")
    # Place drone 0 at the extreme corner (0.1, 0.1)
    corner_pos = state.pos.at[0].set(jnp.array([0.1, 0.1]))
    state = dataclasses.replace(state, pos=corner_pos)

    # Step once to "clear" initial coverage
    state = step_jit(state, actions)
    # Step twice to check for leaks
    old_state = state
    state = step_jit(state, actions)
    _, info = reward_jit(old_state, state, jnp.bool_(False))

    leak = float(info['r_coverage'])
    if leak == 0.0:
        print("  Boundary Leak Test: PASSED (Zero reward at wall) ✓")
    else:
        print(f"  Boundary Leak Test: FAILED (Leak detected: {leak}) ✗")
        assert leak == 0.0, "Exploration reward leaked at boundary!"
