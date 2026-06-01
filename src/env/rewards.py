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

    r_i = r_target_found + r_coverage + r_chain_gap + r_proximity + r_collision + r_success + r_hub_proximity

    r_target_found  [sparse]
        Granted when the global target is first discovered.
        Contains a shared team bonus (target_found_bonus / N) and a local individual bonus
        (finder_bonus) given only to the drone(s) that physically spotted the target.
        MEM_T8-only diagnostic path: when target_pos is (N, 2), each agent can
        receive the local bonus for its own target, and paired anti-target
        discovery gives that same local amount as a one-shot negative reward.

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
        Individual aspect: Only "contributing" drones (part of the connected chains or shortest path)
                           receive the dynamic penalty. Non-contributing drones receive
                           the maximum gap penalty.

    r_proximity     [dense, penalty]
        Individual penalty for drones that are too close to each other (collision avoidance).
        Value: -number_of_nearby_drones × proximity_penalty.

    r_collision     [dense, penalty]
        Individual penalty for agents hitting walls or obstacles.
        Value: -1.0 × collision_penalty.

    r_success       [sparse, optional]
        One-shot shared terminal bonus when the episode ends with a fully connected chain.
        Value: is_done × fully_connected × (success_bonus / N).

    r_hub_proximity [dense]
        Encourages drones to stay near points of interest (base or target).
        If the target is known, drones are rewarded for proximity to either the base or the target.
        Otherwise, only proximity to the base is rewarded.
        Normalized by the map diagonal to remain map-invariant.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from env.state import EnvState


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
    N      = int(cfg.env.num_agents)
    comm_r = float(cfg.env.comm_radius)
    # Base station uses its own comm radius for the first hop.
    # Defaults to comm_radius if not set, so old configs are fully backward-compatible.
    comm_r_base = float(cfg.env.get("comm_radius_base", cfg.env.comm_radius))
    vis_r  = float(cfg.env.visual_radius)
    cell_s = 1.0

    # Reward weights
    w_exp        = float(cfg.reward.exploration_bonus)
    w_found      = float(cfg.reward.target_found_bonus)
    w_finder     = float(cfg.reward.finder_bonus)
    p_gap_max = float(cfg.reward.max_gap_penalty)
    p_coll  = float(cfg.reward.collision_penalty)
    p_prox  = float(cfg.reward.proximity_penalty)
    w_success    = float(cfg.reward.success_bonus)
    w_base_prox  = float(cfg.reward.get("base_proximity_bonus", 0.0))
    w_target_prox = float(cfg.reward.get("target_proximity_bonus", 0.0))
    use_task     = bool(int(cfg.env.num_targets) > 0) or (int(cfg.env.num_bases) > 0)
    target_found_requires_delivery = bool(cfg.reward.get("target_found_requires_delivery", True))
    every_reward_global = bool(cfg.reward.get("every_reward_global", False))
    only_explor_individual = bool(cfg.reward.get("only_explor_individual", False)) and not every_reward_global
    only_shortest_path_chain_reward = (
        bool(cfg.reward.get("only_shortest_path_chain_reward", False))
        and not only_explor_individual
        and not every_reward_global
    )
    chain_rewards_global = only_explor_individual or every_reward_global

    # Reachability matrix squarings

    _n_reach = max(1, math.ceil(math.log2(N + 2)))
    _eye_Np1 = jnp.eye(N + 1, dtype=jnp.float32)

    # ---- Internal: graph connectivity ------------------------------------

    def _compute_connectivity(state: EnvState):
        """
        Returns
        -------
        is_conn_base   : (N,) bool
        is_conn_target : (N,) bool

        Rule: the FIRST hop to base or target requires vis_r (visual contact).
              Subsequent drone↔drone hops use comm_r.
        """
        diff_pp        = state.pos[:, None, :] - state.pos[None, :, :]
        pairwise_dists = jnp.linalg.norm(diff_pp, axis=-1)

        # Drone↔drone edges: comm_r
        adj_dd = (
            (pairwise_dists <= comm_r)
            & ~jnp.eye(N, dtype=bool)
            & state.active[:, None]
            & state.active[None, :]
        ).astype(jnp.float32)

        # Base first-hop: comm_r_base  (configurable, defaults to comm_r)
        base_dists = jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
        adj_db     = ((base_dists <= comm_r_base) & state.active).astype(jnp.float32)

        A_top  = jnp.concatenate([adj_dd, adj_db[:, None]], axis=-1)
        A_bot  = jnp.concatenate([adj_db[None, :], jnp.zeros((1, 1))], axis=-1)
        A_full = jnp.concatenate([A_top, A_bot], axis=0) + _eye_Np1

        def _square(R, _):
            return jnp.clip(R @ R, 0.0, 1.0), None

        R, _ = jax.lax.scan(_square, jnp.clip(A_full, 0.0, 1.0), None, length=_n_reach)

        is_conn_base = R[:N, N] > 0.5

        # Target first-hop: vis_r  (unchanged)
        target_pos_agents = state.target_pos if state.target_pos.ndim == 2 else jnp.tile(state.target_pos[None, :], (N, 1))
        target_dists    = jnp.linalg.norm(state.pos - target_pos_agents, axis=-1)
        directly_sees   = (target_dists <= vis_r) & state.active
        is_conn_target  = jnp.any(
            (R[:N, :N] > 0.5) & directly_sees[None, :], axis=-1
        )

        return is_conn_base, is_conn_target

    def _compute_shortest_paths(state: EnvState, idx_b, idx_t, has_b_chain, has_t_chain):
        """
        Calculates shortest path hop counts using Min-Plus matrix squaring.
        Returns is_contributing (N,) mask.
        """
        V = N + 2
        H = jnp.full((V, V), 9999, dtype=jnp.int32)
        H = H.at[jnp.arange(V), jnp.arange(V)].set(0) # Diagonal is 0

        diff_pp = state.pos[:, None, :] - state.pos[None, :, :]
        pairwise_dists = jnp.linalg.norm(diff_pp, axis=-1)

        # FIX: Added ~jnp.eye(N, dtype=bool) to prevent self-loops overriding the 0 diagonal
        adj_dd = (pairwise_dists <= comm_r) & ~jnp.eye(N, dtype=bool) & state.active[:, None] & state.active[None, :]
        H = H.at[:N, :N].set(jnp.where(adj_dd, 1, H[:N, :N]))

        base_dists = jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
        adj_db = (base_dists <= comm_r_base) & state.active  # base uses comm_r_base
        H = H.at[:N, N].set(jnp.where(adj_db, 1, H[:N, N]))
        H = H.at[N, :N].set(jnp.where(adj_db, 1, H[N, :N]))

        target_pos_agents = state.target_pos if state.target_pos.ndim == 2 else jnp.tile(state.target_pos[None, :], (N, 1))
        target_dists = jnp.linalg.norm(state.pos - target_pos_agents, axis=-1)
        adj_dt = (target_dists <= vis_r) & state.active
        H = H.at[:N, N+1].set(jnp.where(adj_dt, 1, H[:N, N+1]))
        H = H.at[N+1, :N].set(jnp.where(adj_dt, 1, H[N+1, :N]))

        def _min_plus_step(M, _):
            M_next = jnp.min(M[:, :, None] + M[None, :, :], axis=1)
            return jnp.clip(M_next, 0, 9999), None # Prevent overflow

        n_steps = math.ceil(math.log2(V))
        H_final, _ = jax.lax.scan(_min_plus_step, H, None, length=n_steps)

        # FIX: Safe gating. Only check path equality if the chain actually exists.
        is_on_base_path = jnp.where(
            has_b_chain,
            (H_final[N, :N] + H_final[:N, idx_b] == H_final[N, idx_b]),
            False
        )
        is_on_tgt_path = jnp.where(
            has_t_chain,
            (H_final[N+1, :N] + H_final[:N, idx_t] == H_final[N+1, idx_t]),
            False
        )

        return is_on_base_path | is_on_tgt_path

    # ---- Public: compute_reward ------------------------------------------

    def compute_reward(
        old_state: EnvState,
        new_state: EnvState,
        is_done:   jax.Array = jnp.bool_(False),
    ) -> tuple[jax.Array, dict]:

        per_agent_targets = (new_state.target_pos.ndim == 2)
        target_pos_agents = (
            new_state.target_pos
            if per_agent_targets
            else jnp.tile(new_state.target_pos[None, :], (N, 1))
        )
        primary_target_pos = target_pos_agents[0]

        # Dynamic world geometry from state
        bt_vec = primary_target_pos - new_state.base_pos
        bt_dist = jnp.linalg.norm(bt_vec)
        # Avoid division by zero if base == target
        bt_unit = jnp.where(bt_dist > 0, bt_vec / bt_dist, jnp.array([1.0, 0.0], dtype=jnp.float32))

        # Normalized gap weight
        w_gap = jnp.where(bt_dist > 0, p_gap_max / bt_dist, 0.0)

        # ---- 2. Global Target Found --------------------------------------
        dist_to_base = jnp.linalg.norm(new_state.pos - new_state.base_pos[None, :], axis=-1)
        adj_db = (dist_to_base <= comm_r_base) & new_state.active

        dist_to_target = jnp.linalg.norm(new_state.pos - target_pos_agents, axis=-1)
        is_visible = (dist_to_target <= vis_r) & new_state.active

        knew_or_sees_target = old_state.target_known | is_visible
        actual_deliverers = adj_db & knew_or_sees_target

        finder_bonus_value = w_finder if (use_task and not chain_rewards_global) else 0.0
        if per_agent_targets:
            # MEM_T8-only diagnostic path: one fixed target slot per agent.
            if target_found_requires_delivery:
                newly_found_agents = actual_deliverers & ~old_state.target_known
            else:
                newly_found_agents = new_state.target_known & ~old_state.target_known
            was_target_found = jnp.any(old_state.target_known)
            global_target_found = jnp.any(new_state.target_known)
            just_found = global_target_found & ~was_target_found
            r_target_found_shared = jnp.where(use_task & just_found, w_found, 0.0)
            r_target_found_local = jnp.where(newly_found_agents, finder_bonus_value, 0.0)
        else:
            if target_found_requires_delivery:
                was_target_found = old_state.base_target_known
                global_target_found = new_state.base_target_known
                target_found_local_receivers = actual_deliverers
            else:
                was_target_found = jnp.any(old_state.target_known)
                global_target_found = jnp.any(new_state.target_known)
                target_found_local_receivers = is_visible & ~old_state.target_known
            just_found = global_target_found & ~was_target_found
            r_target_found_shared = jnp.where(use_task & just_found, w_found, 0.0)
            r_target_found_local = jnp.where(just_found & target_found_local_receivers, finder_bonus_value, 0.0)

        # MEM_T8-only diagnostic path: paired wrong-branch decoys.
        if per_agent_targets and new_state.anti_target_known.size:
            newly_found_anti = new_state.anti_target_known & ~old_state.anti_target_known
            r_anti_target = jnp.where(newly_found_anti, -finder_bonus_value, 0.0)
        else:
            newly_found_anti = jnp.zeros(N, dtype=jnp.bool_)
            r_anti_target = jnp.zeros(N, dtype=jnp.float32)

        # Shared component is divided by N to be agent-invariant
        r_target_found = (r_target_found_shared / N) + r_target_found_local + r_anti_target

        delta_cells = new_state.last_cov_delta.astype(jnp.float32)
        # Stop exploration reward for drones that know the target position
        r_coverage = (delta_cells * (cell_s**2) * w_exp) * (~new_state.target_known)

        # ---- 4. Chain Gap Distance ---------------------------------------
        is_conn_base, is_conn_target = _compute_connectivity(new_state)
        fully_connected = jnp.any(is_conn_base & is_conn_target)

        bt_vec = primary_target_pos - new_state.base_pos
        bt_dist = jnp.linalg.norm(bt_vec)
        # Avoid division by zero if base == target
        bt_unit = jnp.where(bt_dist > 0, bt_vec / bt_dist, jnp.array([1.0, 0.0], dtype=jnp.float32))

        # Normalized gap weight
        w_gap = jnp.where(bt_dist > 0, p_gap_max / bt_dist, 0.0)

        # Base Chain "Tip" (Drone closest to Target)
        is_base_chain = is_conn_base & new_state.active
        any_base_chain = jnp.any(is_base_chain)
        dist_to_target = jnp.linalg.norm(new_state.pos - target_pos_agents, axis=-1)
        idx_b = jnp.argmin(jnp.where(is_base_chain, dist_to_target, 1e9))
        pos_b = jnp.where(any_base_chain, new_state.pos[idx_b], new_state.base_pos)

        # Target Chain "Tip" (Drone closest to Base)
        is_tgt_chain = is_conn_target & new_state.active
        any_tgt_chain = jnp.any(is_tgt_chain)
        dist_to_base = jnp.linalg.norm(new_state.pos - new_state.base_pos[None, :], axis=-1)
        idx_t = jnp.argmin(jnp.where(is_tgt_chain, dist_to_base, 1e9))
        pos_t = jnp.where(any_tgt_chain, new_state.pos[idx_t], primary_target_pos)

        # Projections for tracking (Clipped to [0, bt_dist] to fix 190% bug)
        rel_b  = new_state.pos - new_state.base_pos[None, :]
        proj_b = jnp.clip(jnp.sum(rel_b * bt_unit[None, :], axis=-1), 0.0, bt_dist)

        rel_t = new_state.pos - target_pos_agents
        proj_t = jnp.clip(jnp.sum(rel_t * (-bt_unit[None, :]), axis=-1), 0.0, bt_dist)

        # Gap calculation: direct Euclidean distance between the two tips
        # If fully connected, we force gap to 0 to avoid the "flip" where tips jump to opposite ends
        raw_gap_dist = jnp.linalg.norm(pos_b - pos_t)
        chain_gap_dist = jnp.where(fully_connected, 0.0, raw_gap_dist)

        # Progress tracking (Euclidean based)
        chain_progress_pct = jnp.where(
            fully_connected,
            100.0,
            jnp.clip(100.0 * (1.0 - chain_gap_dist / (bt_dist + 1e-6)), 0.0, 100.0)
        )



        # Chain gap is the primary timer/incentive.
        use_dynamic_gap = jnp.logical_or(global_target_found, jnp.logical_not(use_task))
        active_gap = jnp.where(use_dynamic_gap, chain_gap_dist, bt_dist)
        base_gap_penalty = -active_gap * w_gap

        # Only apply dynamic gap penalty to contributing drones unless chain
        # rewards are deliberately shared as a team signal.
        if chain_rewards_global:
            is_contributing = jnp.ones(N, dtype=jnp.bool_)
            r_chain_gap = jnp.full((N,), base_gap_penalty / N, dtype=jnp.float32)
        else:
            if only_shortest_path_chain_reward:
                is_contributing = _compute_shortest_paths(new_state, idx_b, idx_t, any_base_chain, any_tgt_chain)
            else:
                is_contributing = is_conn_base | is_conn_target

            r_chain_gap = jnp.where(
                is_contributing,
                base_gap_penalty / N,
                -p_gap_max / N
            )

        # ---- 5. Proximity penalty ----------------------------------------
        pairwise_dists = jnp.linalg.norm(new_state.pos[:, None, :] - new_state.pos[None, :, :], axis=-1)
        too_close = (pairwise_dists <= vis_r) & ~jnp.eye(N, dtype=jnp.bool_) & new_state.active[:, None] & new_state.active[None, :]
        r_proximity = -jnp.sum(too_close, axis=-1).astype(jnp.float32) * p_prox

        # ---- 6. Collision penalty ----------------------------------------
        # Penalty is per-agent hitting a wall/obstacle
        r_collision = -new_state.collides.astype(jnp.float32) * p_coll

        # ---- 7. Terminal success bonus -----------------------------------
        # NOTE: Both `is_done` and `fully_connected` checks are intentional and NOT redundant.
        # `is_done` can be True for two reasons: (a) time_up=True with fully_connected=False
        # (timeout without chain), or (b) fully_connected=True which also sets done=True.
        # The `fully_connected` guard prevents awarding the bonus in case (a).
        r_success_scalar = jnp.where(use_task, jnp.float32(is_done) * jnp.float32(fully_connected) * w_success, 0.0)
        r_success = r_success_scalar / N # Divide by N to be agent-invariant

        # ---- 8. Hub Proximity Reward (Intuition) -------------------------
        # Normalized by diagonal to be map-invariant
        # Gated by individual knowledge (new_state.target_known accounts for comms)

        diagonal = jnp.sqrt(new_state.box_width**2 + new_state.box_height**2 + 1e-6)

        dist_to_base   = jnp.linalg.norm(new_state.pos - new_state.base_pos[None, :], axis=-1)
        dist_to_target = jnp.linalg.norm(new_state.pos - target_pos_agents, axis=-1)

        p_base   = jnp.clip(1.0 - (dist_to_base / diagonal), 0.0, 1.0)
        p_target = jnp.clip(1.0 - (dist_to_target / diagonal), 0.0, 1.0)

        # Selection: Closest hub if know target, else only base
        hub_prox_reward = jnp.where(
            new_state.target_known,
            jnp.maximum(w_base_prox * p_base, w_target_prox * p_target),
            w_base_prox * p_base
        )
        r_hub_proximity = hub_prox_reward / N  # Agent-invariant division

        # ---- Total -------------------------------------------------------
        reward = r_coverage + r_target_found + r_chain_gap + r_proximity + r_collision + r_success + r_hub_proximity
        if every_reward_global:
            reward = jnp.full((N,), jnp.sum(reward) / N, dtype=jnp.float32)

        # MEM_T8-only diagnostic reporting: fractional target-found progress
        # for eight independent cue/choice tasks. Normal levels stay binary.
        target_found_fraction = jnp.where(
            per_agent_targets,
            jnp.mean(new_state.target_known.astype(jnp.float32)),
            global_target_found.astype(jnp.float32),
        )

        info = {
            "r_coverage":     jnp.sum(r_coverage),
            "r_target_found": jnp.sum(r_target_found),
            # MEM_T8-only diagnostic component; zero for normal one-target levels.
            "r_anti_target":  jnp.sum(r_anti_target),
            "r_chain_gap":    jnp.sum(r_chain_gap),
            "r_proximity":    jnp.sum(r_proximity),
            "r_collision":    jnp.sum(r_collision),
            "r_success":      jnp.sum(jnp.full(N, r_success_scalar)) / N,
            "r_hub_proximity": jnp.sum(r_hub_proximity),
            "chain_gap_dist": chain_gap_dist,
            "chain_progress_pct": chain_progress_pct,
            "is_contributing": is_contributing,
            "active_chain_drones": jnp.sum(is_contributing),
            "fully_connected": fully_connected.astype(jnp.float32),
            "global_target_found": global_target_found.astype(jnp.float32),
            # MEM_T8-only diagnostic metric; same as global_target_found normally.
            "target_found_fraction": target_found_fraction,
            "just_found":     just_found.astype(jnp.float32),
            # MEM_T8-only diagnostic metric; zero for normal one-target levels.
            "anti_target_found": jnp.sum(newly_found_anti.astype(jnp.float32)),
            "global_coverage": jnp.mean(new_state.coverage_grid.astype(jnp.float32)),
        }

        return reward, info

    return compute_reward


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import jax
    from core.config import load_config, validate_config
    from env.physics import make_env_fns

    print("── Reward Self-Test ─────────────────────────────────────────")

    cfg = load_config(cli_overrides=True)
    OmegaConf.set_readonly(cfg, False)
    cfg.env.box_width = 100.0
    cfg.env.box_height = 100.0
    cfg.env.map_names = ["open_field"]
    validate_config(cfg)

    N = cfg.env.num_agents
    # G = cfg.env.grid_resolution  # REMOVED: using rectangular grid

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
