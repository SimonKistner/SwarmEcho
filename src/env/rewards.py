"""
swarmecho/env/rewards.py
========================
Team reward function for the SwarmEcho relay-chain task.

All functions are pure JAX — JIT/vmap/scan compatible.
Use `make_reward_fn(cfg)` so config scalars become XLA compile-time constants.

Design
------
The reward is SHARED across all agents (cooperative MARL). Every agent in the
team receives the same scalar r_t each step. This avoids credit-assignment
ambiguity and encourages fully cooperative behaviour.

Reward components
-----------------

    r_t = r_time + r_coverage + r_chain_gap [+ r_success]

    r_time
        A small negative constant every step → discourages dawdling.
        Value: time_penalty (e.g. -0.01).

    r_coverage   [dense]
        Reward proportional to the number of NEW grid cells covered this step.
        Value: delta_cells × exploration_bonus.
        If `deactivate_coverage_on_find` is True, this becomes 0.0 once the
        target has been found globally.

    r_chain_gap  [dense, two-chain projection]
        Provides a smooth gradient for closing the gap between the base chain
        and the target chain.
        Condition: Exactly 0.0 UNLESS global_target_found is True.
        Base Chain Proj: How far active, base-connected drones project along the Base->Target vector.
        Target Chain Proj: How far active, target-connected drones project along the Target->Base vector.
        chain_gap_distance = Target_Dist - (Base_Chain_Proj + Target_Chain_Proj) (clipped at 0)
        Value: -chain_gap_distance × chain_gap_penalty.

    r_success  [sparse, optional]
        One-shot terminal bonus when the episode ends with a closed chain.
        Controlled via `is_done` flag; only applied at episode termination.
        Value: is_done × is_fully_connected × success_bonus.
        Pass is_done=False during mid-episode steps to suppress it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

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
    vis_r  = float(cfg.env.visual_radius)
    cell_s = float(cfg.env.grid_cell_size)

    # Reward weights
    w_exp        = float(cfg.reward.exploration_bonus)
    w_found      = float(cfg.reward.target_found_bonus)
    p_gap_max    = float(cfg.reward.max_gap_penalty)
    p_coll       = float(cfg.reward.collision_penalty)
    w_success    = float(cfg.reward.success_bonus)
    use_task     = bool(int(cfg.env.num_targets) > 0) or (int(cfg.env.num_bases) > 0)

    # Fixed world geometry
    BASE_POS   = jnp.array([cfg.env.base_x,   cfg.env.base_y],   dtype=jnp.float32)
    TARGET_POS = jnp.array([cfg.env.target_x, cfg.env.target_y], dtype=jnp.float32)

    bt_vec  = TARGET_POS - BASE_POS
    bt_dist = float(jnp.linalg.norm(bt_vec))
    bt_unit = bt_vec / bt_dist

    # Normalized gap weight (ensures max penalty is p_gap_max at max distance)
    w_gap = p_gap_max / bt_dist

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
        """
        diff_pp        = state.pos[:, None, :] - state.pos[None, :, :]
        pairwise_dists = jnp.linalg.norm(diff_pp, axis=-1)

        adj_dd = (
            (pairwise_dists <= comm_r)
            & ~jnp.eye(N, dtype=bool)
            & state.active[:, None]
            & state.active[None, :]
        ).astype(jnp.float32)

        base_dists = jnp.linalg.norm(state.pos - BASE_POS[None, :], axis=-1)
        adj_db     = ((base_dists <= comm_r) & state.active).astype(jnp.float32)

        A_top  = jnp.concatenate([adj_dd, adj_db[:, None]], axis=-1)
        A_bot  = jnp.concatenate([adj_db[None, :], jnp.zeros((1, 1))], axis=-1)
        A_full = jnp.concatenate([A_top, A_bot], axis=0) + _eye_Np1

        def _square(R, _):
            return jnp.clip(R @ R, 0.0, 1.0), None

        R, _ = jax.lax.scan(_square, jnp.clip(A_full, 0.0, 1.0), None, length=_n_reach)

        is_conn_base = R[:N, N] > 0.5

        target_dists    = jnp.linalg.norm(state.pos - TARGET_POS[None, :], axis=-1)
        directly_sees   = (target_dists <= vis_r) & state.active
        is_conn_target  = jnp.any(
            (R[:N, :N] > 0.5) & directly_sees[None, :], axis=-1
        )

        return is_conn_base, is_conn_target

    # ---- Public: compute_reward ------------------------------------------

    def compute_reward(
        old_state: EnvState,
        new_state: EnvState,
        is_done:   jax.Array = jnp.bool_(False),
    ) -> tuple[jax.Array, dict]:

        # ---- 2. Global Target Found --------------------------------------
        was_target_found    = jnp.any(old_state.target_known)
        global_target_found = jnp.any(new_state.target_known)
        just_found          = global_target_found & ~was_target_found
        
        r_target_found = jnp.where(use_task & just_found, w_found, 0.0)

        # ---- 3. Coverage delta -------------------------------------------
        delta_cells = (
            new_state.coverage_grid.sum() - old_state.coverage_grid.sum()
        ).astype(jnp.float32)
        
        # Exploration reward is normalized to Area (m^2) discovered this step.
        # This keeps the bonus constant regardless of grid_cell_size.
        r_coverage = delta_cells * (cell_s**2) * w_exp

        # ---- 4. Graph connectivity ---------------------------------------
        is_conn_base, is_conn_target = _compute_connectivity(new_state)

        # ---- 5. Chain gap (Two-Chain Projection) -------------------------
        
        # Base Chain Projection (along Base -> Target)
        is_base_chain = is_conn_base & new_state.active
        rel_b = new_state.pos - BASE_POS[None, :]                     # (N, 2)
        proj_b = jnp.sum(rel_b * bt_unit[None, :], axis=-1)          # (N,)
        base_chain_proj = jnp.max(jnp.where(is_base_chain, proj_b, 0.0))

        # Target Chain Projection (along Target -> Base)
        is_tgt_chain = is_conn_target & new_state.active
        rel_t = new_state.pos - TARGET_POS[None, :]                   # (N, 2)
        tb_unit = -bt_unit
        proj_t = jnp.sum(rel_t * tb_unit[None, :], axis=-1)          # (N,)
        target_chain_proj = jnp.max(jnp.where(is_tgt_chain, proj_t, 0.0))

        # Gap calculation
        chain_gap_dist = bt_dist - (base_chain_proj + target_chain_proj)
        chain_gap_dist = jnp.maximum(0.0, chain_gap_dist)
        
        # Chain gap is the primary timer/incentive. 
        # For exploration levels (use_task=False), it stays constant.
        # For task levels (use_task=True), it stays at MAXIMUM until target is found.
        use_dynamic_gap = jnp.logical_or(global_target_found, jnp.logical_not(use_task))
        active_gap = jnp.where(use_dynamic_gap, chain_gap_dist, bt_dist)
        r_chain_gap = -active_gap * w_gap
        # Ensure chain_gap_dist is still exported in info for tracking

        # ---- 6. Collision penalty ----------------------------------------
        # Penalty is per-agent hitting a wall/obstacle
        r_collision = -jnp.sum(new_state.collides).astype(jnp.float32) * p_coll

        # ---- 7. Terminal success bonus -----------------------------------
        fully_connected = jnp.any(is_conn_base & is_conn_target)
        r_success = jnp.where(use_task, jnp.float32(is_done) * jnp.float32(fully_connected) * w_success, 0.0)

        # ---- Total -------------------------------------------------------
        reward = r_coverage + r_target_found + r_chain_gap + r_collision + r_success

        info = {
            "r_coverage":     r_coverage,
            "r_target_found": r_target_found,
            "r_chain_gap":    r_chain_gap,
            "r_collision":    r_collision,
            "r_success":      r_success,
            "chain_gap_dist": chain_gap_dist,
            "fully_connected": fully_connected.astype(jnp.float32),
            "global_target_found": global_target_found.astype(jnp.float32),
            "just_found":     just_found.astype(jnp.float32),
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

    cfg = load_config(cli_overrides=False)
    validate_config(cfg)

    N = cfg.env.num_agents
    # G = cfg.env.grid_resolution  # REMOVED: using rectangular grid

    env_step, reset, _  = make_env_fns(cfg)
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
        total_reward += float(rew)

        if t < 5 or t % 100 == 99 or t == cfg.env.max_steps - 1:
            cov_pct = 100.0 * int(state.coverage_grid.sum()) / state.coverage_grid.size
            print(
                f"  {t+1:>5}  {float(rew):>8.4f}  "
                f"{float(info['chain_gap_dist']):>7.1f}  "
                f"{bool(info['fully_connected'] > 0.5):>10}  "
                f"{bool(info['global_target_found'] > 0.5):>10}  "
                f"{cov_pct:>8.1f}%"
            )

    print(f"\n  Total reward over episode : {total_reward:.2f}")
    print(f"  max_steps                 : {cfg.env.max_steps}")
    print(f"  Reward per step (avg)     : {total_reward / cfg.env.max_steps:.4f}")

    # Shape check: reward must be scalar
    r, _ = reward_jit(state, state, jnp.bool_(False))
    assert r.shape == (), f"Reward must be scalar, got shape {r.shape}"
    assert not jnp.isnan(r), "NaN reward!"
    assert not jnp.isinf(r), "Inf reward!"

    # Batch vmap check
    print(f"\n  Batch vmap (256 envs) ...")
    batch_keys   = jax.random.split(key, 256)
    batch_states = jax.jit(jax.vmap(reset))(batch_keys)
    batch_acts   = jnp.zeros((256, N, 2))
    batch_next   = jax.jit(jax.vmap(env_step))(batch_states, batch_acts)

    batch_r, _ = jax.jit(jax.vmap(compute_reward))(batch_states, batch_next)
    batch_r.block_until_ready()
    assert batch_r.shape == (256,), f"Batch reward shape wrong: {batch_r.shape}"
    print(f"  Batch reward shape : {batch_r.shape}  ✓")
    print(f"  Mean batch reward  : {float(batch_r.mean()):.4f}")

    print("\nReward self-test passed ✓")


