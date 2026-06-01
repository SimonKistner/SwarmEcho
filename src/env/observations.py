"""
swarmecho/env/observations.py
==============================
Ego-centric, permutation-invariant Radar + Graph observation model.

All functions are pure JAX — JIT/vmap/scan compatible.
Use `make_obs_fns(cfg)` so config scalars become XLA compile-time constants.

Observation structure per agent i  (obs_dim = 9 + 16 + B*4)
---------------------------------------------------------

┌──────────────────────────── Self state (9) ───────────────────────────────┐
│  vel_i / v_max                            (2)  ego velocity               │
│  (base_pos - pos_i) / max_dim            (2)  odometry to base           │
│  is_connected_to_base                    (1)  multi-hop graph flag        │
│  is_connected_to_target                  (1)  multi-hop graph flag        │
│  target_known_flag                       (1)  1.0 if drone knows target   │
│  (target_pos - pos_i) / max_dim × mask   (2)  masked until target_known  │
├────────────────────── Local Coverage Probes (16) ─────────────────────────┤
│  16 radial probes (evenly spaced circle) at sampling_radius:               │
│  returns 1.0 if covered, 0.0 otherwise                                    │
├─────────────────────── 360° Radar  (B bins × 4) ──────────────────────────┤
│  For each of B angular bins:                                               │
│    inv_dist_wall                         (1)  1 - d/vis_r  (ray-cast)    │
│    inv_dist_drone                        (1)  1 - d/comm_r (any drone)   │
│    inv_dist_target_conn_drone            (1)  1 - d/comm_r (tgt-chain)   │
│    inv_dist_base_conn_drone              (1)  1 - d/comm_r (base-chain)  │
└────────────────────────────────────────────────────────────────────────────┘

Design principles
-----------------
Permutation invariance
    No agent IDs or sorted-k neighbour slots. The radar aggregates all
    other agents into spatial bins — the result is independent of agent
    ordering or swarm size.

Symmetry breaking via staggered spawn
    Drones spawn at different timesteps (spawn_delay), so their positions
    and velocities differ by the time they all become active. No one-hot
    encoding required.

Persistent target knowledge
    The target odometry vector is masked by `state.target_known[i]`, which
    is monotonically set to True once the drone has been informed (directly
    or via the comm chain). This persists across communication blackouts.

Graph connectivity
    We build a (N+1) × (N+1) adjacency matrix (N drones + base as node N).
    Reachability is computed via repeated matrix squaring (n_reach steps =
    ceil(log2(N+2))). This is a static-shape, fully JIT-compiled operation.

Radar implementation
    For each bin b (centre angle θ_b = (b+0.5)×2π/B − π):
      - Wall   : ray-AABB intersection at angle θ_b
      - Drones : each drone gets a bin via arctan2, scatter-max over signals
      - Points : base_station and target treated as point entities
    All signals: max(0, 1 − dist/norm_dist), so 0=absent, 1=touching.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from env.state import EnvState
from env.raycast import dda_raycast, dda_dist
from core.config import compute_obs_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_obs_fns(cfg: DictConfig, resolved_W: float, resolved_H: float, occ_grid: jax.Array):
    """
    Close over config scalars and return pure JAX observation functions.

    Returns
    -------
    compute_obs : EnvState -> jnp.ndarray  shape (N, obs_dim)
    obs_dim     : int
    """
    N         = int(cfg.env.num_agents)
    cell_size = 1.0
    B         = int(cfg.env.radar_bins)
    W, H      = float(resolved_W), float(resolved_H)
    max_dim   = math.sqrt(W ** 2 + H ** 2)
    use_task        = (int(cfg.env.num_targets) > 0) or (int(cfg.env.num_bases) > 0)
    sampling_radius = float(cfg.env.exploration_sampling_radius)
    vis_r           = float(cfg.env.visual_radius)
    comm_r          = float(cfg.env.comm_radius)
    comm_r_base     = float(cfg.env.get("comm_radius_base", cfg.env.comm_radius))
    v_max           = float(cfg.env.max_speed)
    # MEM_T8-only diagnostic flag. Normal SwarmEcho levels keep the full
    # observation; the memory test zeros non-local channels that reveal which
    # fixed T-corridor an agent occupies.
    mem_test_mask_nonlocal_obs = bool(cfg.env.get("mem_test_mask_nonlocal_obs", False))

    obs_dim: int = compute_obs_dim(cfg)
    GW, GH = occ_grid.shape
    is_unobstructed = (not jnp.any(occ_grid))

    # Pre-compute bin centre angles: θ_b ∈ (−π, π]
    _bin_angles  = (jnp.arange(B, dtype=jnp.float32) + 0.5) * (2.0 * jnp.pi / B) - jnp.pi
    _bin_range   = jnp.arange(B, dtype=jnp.int32)          # [0, 1, ..., B-1]
    _bin_width   = 2.0 * jnp.pi / B

    # How many matrix-squaring steps for full-graph reachability
    _n_reach     = max(1, math.ceil(math.log2(N + 2)))

    # Identity for self-loops (float for matmul)
    _eye_N       = jnp.eye(N, dtype=jnp.float32)            # (N, N)
    _eye_Np1     = jnp.eye(N + 1, dtype=jnp.float32)        # (N+1, N+1)

    # ---------------------------------------------------------------------------
    # Internal: reachability via repeated matrix squaring
    # ---------------------------------------------------------------------------

    def _reachability(A: jax.Array) -> jax.Array:
        """
        A : (M, M) float32 adjacency with self-loops already added.
        Returns (M, M) float32 reachability matrix (values in [0,1]).
        """
        def _square(R, _):
            return jnp.clip(R @ R, 0.0, 1.0), None
        R, _ = jax.lax.scan(_square, A, None, length=_n_reach)
        return R

    # ---------------------------------------------------------------------------
    # Internal: ray-AABB wall distance for a single angle
    # ---------------------------------------------------------------------------

    def _ray_wall_dist(x: jax.Array, y: jax.Array, theta: jax.Array) -> jax.Array:
        """
        Combined distance to world boundaries AND internal grid obstacles.
        """
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)

        # 1. Mathematical Ray-AABB (outer boundary)
        t_r = jnp.where(cos_t >  1e-7, (W - x) / cos_t, jnp.inf)
        t_l = jnp.where(cos_t < -1e-7, -x / cos_t,       jnp.inf)
        t_t = jnp.where(sin_t >  1e-7, (H - y) / sin_t,  jnp.inf)
        t_b = jnp.where(sin_t < -1e-7, -y / sin_t,        jnp.inf)
        d_box = jnp.minimum(jnp.minimum(t_r, t_l), jnp.minimum(t_t, t_b))

        # 2. DDA Grid Raycast (internal obstacles)
        # We only check up to vis_r to keep it efficient.
        p1 = jnp.array([x, y]) / cell_size
        p2 = (jnp.array([x, y]) + jnp.array([cos_t, sin_t]) * vis_r) / cell_size

        t_hit = dda_dist(p1, p2, occ_grid)
        d_maze = t_hit * vis_r

        # Return the closest of the two
        return jnp.minimum(d_box, d_maze)

    # ---------------------------------------------------------------------------
    # Internal: scatter point signals into B bins  (for drone and point entities)
    # ---------------------------------------------------------------------------

    def _scatter_max(
        signals:    jax.Array,   # (M,) float32 — signal for each entity
        bin_idx:    jax.Array,   # (M,) int32   — which bin each entity falls in
    ) -> jax.Array:              # (B,) float32
        """
        For each bin b: max over all entities whose bin_idx == b of signal.
        Implemented as a static-shape (M, B) one-hot expansion then max over M.
        M entities × B bins — fully JIT-compilable, no dynamic shapes.
        """
        bins_oh   = (bin_idx[:, None] == _bin_range[None, :])   # (M, B) bool
        per_bin   = jnp.where(bins_oh, signals[:, None], 0.0)   # (M, B)
        return per_bin.max(axis=0)                               # (B,)

    def _angle_to_bin(angle: jax.Array) -> jax.Array:
        """Map angle in [−π, π] to integer bin index in [0, B)."""
        return (jnp.floor((angle + jnp.pi) / _bin_width).astype(jnp.int32)) % B

    # ---------------------------------------------------------------------------
    # Main: compute_obs
    # ---------------------------------------------------------------------------

    def compute_obs(state: EnvState) -> jnp.ndarray:
        """
        Compute observations for ALL N agents simultaneously.

        Parameters
        ----------
        state : EnvState — single environment state (not batched)

        Returns
        -------
        jnp.ndarray  shape (N, obs_dim)
        """

        # ── Step 1: Shared precomputation (once for all agents) ──────────────

        # Pairwise drone distances: (N, N)
        diff_pp        = state.pos[:, None, :] - state.pos[None, :, :]
        pairwise_dists = jnp.linalg.norm(diff_pp, axis=-1)

        # 2. Target/Base pings
        # We only ping if they exist. If count is 0, dist becomes huge to avoid phantom hits.
        num_bases = int(cfg.env.num_bases)
        num_targets = int(cfg.env.num_targets)

        # Base pings
        base_dists = jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
        base_dists = jnp.where(num_bases > 0, base_dists, 1e6)

        per_agent_targets = (state.target_pos.ndim == 2)
        target_pos_agents = state.target_pos if per_agent_targets else jnp.tile(state.target_pos[None, :], (N, 1))

        # Target pings
        target_dists  = jnp.linalg.norm(state.pos - target_pos_agents, axis=-1)
        target_dists  = jnp.where(num_targets > 0, target_dists, 1e6)

        p_base = jnp.where((base_dists <= comm_r_base) & state.active, 1.0, 0.0)  # comm_r_base: matches first-hop rule
        p_tgt  = jnp.where((target_dists <= vis_r) & state.active, 1.0, 0.0)

        def _get_cell_idx(p):
            # p / cell_size -> index
            gx = jnp.clip((p[0] / cell_size), 0, GW-1).astype(jnp.int32)
            gy = jnp.clip((p[1] / cell_size), 0, GH-1).astype(jnp.int32)
            return gx * GH + gy

        idx_all = jax.vmap(_get_cell_idx)(state.pos) # (N,)
        base_idx = _get_cell_idx(state.base_pos)
        target_idx = _get_cell_idx(target_pos_agents[0])

        # Direct target visibility for active drones using DDA
        if is_unobstructed:
            target_los = jnp.ones(N, dtype=jnp.bool_)
        else:
            def _lo_target(i):
                target_i = target_pos_agents[i]
                return jnp.where(
                    (target_dists[i] <= vis_r) & state.active[i],
                    dda_raycast(state.pos[i]/cell_size, target_i/cell_size, occ_grid),
                    False
                )
            target_los = jax.vmap(_lo_target)(jnp.arange(N))

        directly_sees_target = (target_dists <= vis_r) & state.active & target_los  # (N,)

        # ── Step 2: Graph connectivity ────────────────────────────────────────

        # Drone-drone adjacency (N, N)
        # 1. Distance & Activity check
        near_dd = (
            (pairwise_dists <= comm_r)
            & ~jnp.eye(N, dtype=bool)
            & state.active[:, None]
            & state.active[None, :]
        )
        # 2. Visibility lookup using DDA Raycasting
        # Drone-base adjacency (N,)  — first hop uses comm_r_base
        near_db = (base_dists <= comm_r_base) & state.active

        if is_unobstructed:
            los_dd = near_dd
            los_db = near_db
        else:
            # Vectorized DDA check for all pairs
            def _lo_dd(i, j):
                # Only raycast if within range
                return jnp.where(
                    near_dd[i, j],
                    dda_raycast(state.pos[i]/cell_size, state.pos[j]/cell_size, occ_grid),
                    False
                )

            def _lo_db(i):
                return jnp.where(
                    near_db[i],
                    dda_raycast(state.pos[i]/cell_size, state.base_pos/cell_size, occ_grid),
                    False
                )

            los_dd = jax.vmap(jax.vmap(_lo_dd, (None, 0)), (0, None))(jnp.arange(N), jnp.arange(N))
            los_db = jax.vmap(_lo_db)(jnp.arange(N))

        adj_dd = los_dd.astype(jnp.float32)
        adj_db = los_db.astype(jnp.float32)

        # Build (N+1, N+1) matrix
        A_top    = jnp.concatenate([adj_dd, adj_db[:, None]], axis=-1)  # (N, N+1)
        A_bot    = jnp.concatenate([adj_db[None, :], jnp.zeros((1, 1))], axis=-1)  # (1, N+1)
        A_full   = jnp.concatenate([A_top, A_bot], axis=0)  # (N+1, N+1)
        A_full   = A_full + _eye_Np1  # add self-loops

        R_full   = _reachability(jnp.clip(A_full, 0.0, 1.0))  # (N+1, N+1)

        # is_connected_to_base[i] = can drone i reach the base node?
        is_conn_base = R_full[:N, N] > 0.5   # (N,) bool

        # is_connected_to_target[i] = can drone i reach any direct target-seer?
        is_conn_target = jnp.any(
            (R_full[:N, :N] > 0.5) & directly_sees_target[None, :], axis=-1
        )  # (N,) bool

        # ── Step 3: Per-agent observation (vmapped over N) ────────────────────

        def single_obs(i: jnp.ndarray) -> jnp.ndarray:

            pos_i = state.pos[i]   # (2,)
            vel_i = state.vel[i]   # (2,)

            # ── Self block ────────────────────────────────────────────────────

            # Target mask: use persistent knowledge (state.target_known), NOT
            # current live connectivity. Once known, always known.
            target_mask     = state.target_known[i].astype(jnp.float32)
            rel_target      = (target_pos_agents[i] - pos_i) / max_dim
            rel_target_m    = rel_target * target_mask # masked by knowledge
            rel_base        = (state.base_pos - pos_i) / max_dim

            # Task Masking for self-block
            rel_base_f      = jnp.where(use_task, rel_base, 0.0)
            is_conn_base_f  = jnp.where(use_task, is_conn_base[i].astype(jnp.float32), 0.0)
            is_conn_target_f= jnp.where(use_task, is_conn_target[i].astype(jnp.float32), 0.0)
            target_mask_f   = jnp.where(use_task, target_mask, 0.0)
            rel_target_f    = jnp.where(use_task, rel_target_m, 0.0)

            if mem_test_mask_nonlocal_obs:
                rel_base_f = jnp.zeros_like(rel_base_f)
                is_conn_base_f = jnp.float32(0.0)
                is_conn_target_f = jnp.float32(0.0)
                target_mask_f = jnp.float32(0.0)
                rel_target_f = jnp.zeros_like(rel_target_f)

            self_block_final = jnp.concatenate([
                vel_i / v_max,                                              # (2,)
                rel_base_f,                                                 # (2,)
                jnp.array([                                                 # (3,)
                    is_conn_base_f,
                    is_conn_target_f,
                    target_mask_f,
                ], dtype=jnp.float32),
                rel_target_f,                                               # (2,)
            ])

            x_i, y_i = pos_i[0], pos_i[1]

            # ── Local Coverage block (16 cells: circular layout, clockwise starting from North) ──
            # Sensing whether nearby cells are already 'covered' in the global map.
            angles = (jnp.pi / 2.0) - jnp.arange(16, dtype=jnp.float32) * (jnp.pi / 8.0)
            offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1) * sampling_radius

            sample_pts = pos_i[None, :] + offsets

            # Map sample points to grid indices using absolute cell mapping (1m = 1 cell)
            # This MUST match the physics engine mapping in physics.py
            gix = jnp.floor(sample_pts[:, 0] / cell_size).astype(jnp.int32)
            giy = jnp.floor(sample_pts[:, 1] / cell_size).astype(jnp.int32)

            # Bounds check against RESOLVED map dimensions (GW, GH)
            in_bounds = (gix >= 0) & (gix < GW) & \
                        (giy >= 0) & (giy < GH)

            # Sample coverage grid (0 if out of bounds)
            local_cov = jnp.where(in_bounds, state.coverage_grid[gix, giy], False).astype(jnp.float32)
            if mem_test_mask_nonlocal_obs:
                # MEM_T8-only: coverage history can act as an external memory
                # trace, so remove it when testing the recurrent actor itself.
                local_cov = jnp.zeros_like(local_cov)

            # ── Radar block ───────────────────────────────────────────────────

            # -- Wall channel (ray-cast per bin centre) --
            wall_dists = jax.vmap(
                lambda th: _ray_wall_dist(x_i, y_i, th)
            )(_bin_angles)                                                   # (B,)
            inv_wall   = jnp.maximum(0.0, 1.0 - wall_dists / vis_r)        # (B,)

            # -- Drone channels --
            # For each other drone j: compute relative angle + signal
            # We include ALL drones j ≠ i, then mask inactive ones to signal=0

            # Relative vectors from i to every other drone j: (N, 2)
            rel_vecs = state.pos - pos_i[None, :]                           # (N, 2)

            # Angles: (N,)
            angles_j = jnp.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])         # (N,)
            bins_j   = jax.vmap(_angle_to_bin)(angles_j)                    # (N,) int

            # Distances: (N,)
            dists_j  = pairwise_dists[i]                                    # (N,)

            # Self-exclusion: self-distance = 0.0 → signal 1.0, so force to 0
            is_other  = (jnp.arange(N) != i) & state.active                # (N,) bool

            # All-drone signals (any active, non-self drone)
            s_drone   = jnp.maximum(0.0, 1.0 - dists_j / comm_r) * is_other.astype(jnp.float32)

            # Target-connected-drone signals
            s_tgt_conn = s_drone * is_conn_target.astype(jnp.float32)      # zero if not tgt-conn

            # Base-connected-drone signals
            s_base_conn = s_drone * is_conn_base.astype(jnp.float32)

            inv_drone      = _scatter_max(s_drone,    bins_j)               # (B,)
            inv_tgt_conn   = _scatter_max(s_tgt_conn, bins_j)               # (B,)
            inv_base_conn  = _scatter_max(s_base_conn, bins_j)              # (B,)
            if mem_test_mask_nonlocal_obs:
                # MEM_T8-only: keep local geometry and nearby-agent occupancy,
                # but remove graph/topology channels unrelated to the cue task.
                inv_tgt_conn = jnp.zeros_like(inv_tgt_conn)
                inv_base_conn = jnp.zeros_like(inv_base_conn)

            # -- Base station point channel --
            # base_vec   = state.base_pos - pos_i                             # (2,)
            # base_dist  = jnp.linalg.norm(base_vec)
            # base_angle = jnp.arctan2(base_vec[1], base_vec[0])
            # base_bin   = _angle_to_bin(base_angle)                          # scalar int
            # base_sig   = jnp.maximum(0.0, 1.0 - base_dist / vis_r)  # vis_r: matches first-hop connection rule
            # # Scatter into (B,) vector using one-hot expansion
            # inv_base_stn = (
            #     (base_bin == _bin_range).astype(jnp.float32) * base_sig
            # )  # (B,)

            # -- Target point channel (only non-zero if within vis_r) --
            # tgt_vec   = state.target_pos - pos_i                            # (2,)
            # tgt_dist  = jnp.linalg.norm(tgt_vec)
            # tgt_in_range = (tgt_dist <= vis_r).astype(jnp.float32)
            # tgt_angle = jnp.arctan2(tgt_vec[1], tgt_vec[0])
            # tgt_bin   = _angle_to_bin(tgt_angle)
            # tgt_sig   = jnp.maximum(0.0, 1.0 - tgt_dist / vis_r) * tgt_in_range
            # inv_target = (
            #     (tgt_bin == _bin_range).astype(jnp.float32) * tgt_sig
            # )  # (B,)

            # -- Stack radar (B, 4) -> flatten (B*4,) --
            radar = jnp.stack([
                inv_wall,
                inv_drone,
                inv_tgt_conn,
                inv_base_conn,
            ], axis=-1)                                                      # (B, 4)
            radar_block = radar.reshape(-1)                                  # (B*4,)

            return jnp.concatenate([self_block_final, local_cov, radar_block])                # (obs_dim,)

        return jax.vmap(single_obs)(jnp.arange(N, dtype=jnp.int32))         # (N, obs_dim)

    return compute_obs, obs_dim


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from core.config import load_config, validate_config, compute_obs_dim
    from env.physics import make_env_fns

    print("── Observation Self-Test ────────────────────────────────────")

    cfg = load_config(cli_overrides=False)
    # Ensure a map is loaded for the strict-only physics factory
    from omegaconf import OmegaConf
    OmegaConf.set_readonly(cfg, False)
    cfg.env.map_names = ["open_field"]
    OmegaConf.set_readonly(cfg, True)

    validate_config(cfg)

    N   = cfg.env.num_agents
    B   = cfg.env.radar_bins
    expected_obs_dim = 9 + 16 + B * 4

    assert compute_obs_dim(cfg) == expected_obs_dim, \
        f"config.py formula mismatch: {compute_obs_dim(cfg)} ≠ {expected_obs_dim}"

    # Resolve world data from physics engine (Strict Flow)
    env_step, reset, _, (resolved_W, resolved_H, occ_grid) = make_env_fns(cfg)
    compute_obs, obs_dim = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid)

    print(f"  N        : {N}")
    print(f"  B        : {B}")
    print(f"  obs_dim  : {obs_dim}  (= 9 + 16 + {B}×4 = {9 + 16 + B*4})")

    key   = jax.random.PRNGKey(0)
    state = jax.jit(reset)(key)

    obs_fn = jax.jit(compute_obs)
    obs    = obs_fn(state)
    obs.block_until_ready()

    assert obs.shape == (N, obs_dim), \
        f"Shape mismatch: {obs.shape} ≠ ({N}, {obs_dim})"
    assert not jnp.any(jnp.isnan(obs)), "NaN in observations!"
    assert not jnp.any(jnp.isinf(obs)), "Inf in observations!"

    # Radar channels (all of B*4 part) must be in [0, 1]
    radar_part = obs[:, 25:]
    assert jnp.all((radar_part >= 0.0) & (radar_part <= 1.0 + 1e-5)), \
        f"Radar values out of [0,1]: min={radar_part.min():.4f} max={radar_part.max():.4f}"

    # Graph flags must be binary
    graph_flags = obs[:, 4:7]   # now 3 flags: conn_base, conn_target, target_known
    assert jnp.all((graph_flags == 0.0) | (graph_flags == 1.0)), \
        "Graph connectivity / target_known flags are not binary!"

    # Target odometry must be zero for drones that don't know target position
    target_known = jnp.array(state.target_known)
    for i in range(N):
        tgt_odo = obs[i, 7:9]   # shifted by 1 due to new flag
        if not target_known[i]:
            assert jnp.allclose(tgt_odo, 0.0), \
                f"Agent {i}: target odometry should be masked but got {tgt_odo}"

    # ── Coverage Calibration Check (Ground Truth Test) ──────────────────
    print(f"\n  Coverage Calibration Check ...")
    # 1. Paint a 'coverage stripe' at X=10m in the physics grid
    import dataclasses
    stripe_x = 10
    new_grid = state.coverage_grid.at[stripe_x, :].set(True)
    state = dataclasses.replace(state, coverage_grid=new_grid)

    # 2. Place drone 0 exactly on that stripe
    new_pos = state.pos.at[0].set(jnp.array([float(stripe_x), 25.0]))
    state = dataclasses.replace(state, pos=new_pos)

    # 3. Compute observations
    obs = jax.jit(compute_obs)(state)

    # Local coverage block starts at index 9 (8 self-state + 1 target_known_flag)
    # The offsets are circular (16 directions, 0 is North, 8 is South)
    # Drone at (10, 25) sampling North (0) looks at (10, 35).
    # Drone at (10, 25) sampling South (8) looks at (10, 15).

    local_cov_bits = obs[0, 9:25]
    print(f"    Drone at X={stripe_x} sees local coverage bits: {local_cov_bits}")

    assert local_cov_bits[0] == 1.0, "Calibration Failed: Drone should see coverage at its current X-stripe (North)"
    assert local_cov_bits[8] == 1.0, "Calibration Failed: Drone should see coverage at its current X-stripe (South)"
    print("    Calibration passed ✓ (Mapping is 1:1 with Physics)")

    # Batch vmap test
    print(f"\n  Batch vmap (256 envs) ...")
    batch_keys  = jax.random.split(key, 256)
    batch_state = jax.jit(jax.vmap(reset))(batch_keys)
    batch_obs   = jax.jit(jax.vmap(compute_obs))(batch_state)
    batch_obs.block_until_ready()
    assert batch_obs.shape == (256, N, obs_dim), \
        f"Batch shape mismatch: {batch_obs.shape}"
    print(f"  Batch obs shape: {batch_obs.shape}  ✓")

    # Run a few steps and check target_known propagates
    print(f"\n  Running 50 env_step calls ...")
    actions_zero = jnp.zeros((N, 2))
    step_fn = jax.jit(env_step)
    for _ in range(50):
        state = step_fn(state, actions_zero)
    obs2 = obs_fn(state)
    print(f"  step={state.step}  active={state.active}  target_known={state.target_known}")
    print(f"  Coverage: {int(state.coverage_grid.sum())} / {state.coverage_grid.size} cells")

    print("\nObservation self-test passed ✓")
