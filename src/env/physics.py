"""
swarmecho/env/physics.py
========================
Pure JAX physics engine: reset, env_step, coverage update.

Factory pattern
---------------
All public functions are produced by `make_env_fns(cfg)`. This closes over
config scalars so they become XLA compile-time constants — no OmegaConf
objects ever enter a jit-compiled function.

The returned callables are PURE (no internal @jax.jit). Callers apply jit:

    env_step, reset, update_coverage = make_env_fns(cfg)

    # Single-env debug step:
    step_jit = jax.jit(env_step)

    # Vectorised training:
    vmapped_step = jax.jit(jax.vmap(env_step))

    # Inside lax.scan for rollout collection:
    final, traj = jax.lax.scan(lambda s, a: (env_step(s, a), env_step(s, a)), init, actions)

Staggered Spawn
---------------
Drone i activates at timestep `i * spawn_delay`. While inactive, the drone
is held at base_pos with zero velocity, its actions are masked to zero,
and it is invisible to other agents' radar. Set spawn_delay=0 to disable.

Persistent Target Knowledge
----------------------------
`state.target_known[i]` is a monotonically-set boolean. It becomes True the
first time drone i is connected (directly or via the comm chain) to a drone
that visually sees the target. Once True, it never reverts — the drone retains
knowledge of the target's position even if communication is later lost.

MEM_T8-only diagnostic support:
`state.anti_target_known[i]` is set the first time drone i sees its paired
anti-target. This is intentionally not communication-propagated and has no role
in normal one-target SwarmEcho levels.

The update is computed in env_step AFTER physics so new positions are used.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from core.config import MAP_DIR
from env.state import EnvState
from env.maps import MapDefinition
from env.raycast import dda_raycast, get_ray_stencil, compute_local_visibility, PAD_RADIUS


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env_fns(cfg: DictConfig):
    """
    Build pure environment functions closed over config scalars.

    Returns
    -------
    env_step        : (EnvState, actions (N,2)) -> EnvState
    reset           : (PRNGKey) -> EnvState
    update_coverage : (coverage_grid, pos, active) -> coverage_grid
    world_meta      : tuple (W, H, occ_grid, comm_occ_grid) - resolved box width, height,
                      physical occupancy grid, and communication occupancy grid
    """

    # --- Extract config as Python scalars (XLA compile-time constants) -----
    N           = int(cfg.env.num_agents)
    cell_size   = 1.0  # Cell size is universally 1m for physics and coverage grids
    dt          = float(cfg.env.dt)
    drag        = float(cfg.env.drag)
    v_max       = float(cfg.env.max_speed)
    f_max       = float(cfg.env.max_force)
    W           = cfg.env.box_width
    H           = cfg.env.box_height
    vis_r       = float(cfg.env.visual_radius)
    comm_r      = float(cfg.env.comm_radius)
    comm_r_base = float(cfg.env.get("comm_radius_base", cfg.env.comm_radius))
    spawn_delay = int(cfg.env.spawn_delay)
    wall_res    = float(cfg.env.wall_restitution)
    use_task    = (int(cfg.env.num_targets) > 0) or (int(cfg.env.num_bases) > 0)
    # --- Spawn control scalars (B-series curriculum) ----------------------
    use_random_base_spawn   = bool(cfg.env.get("use_random_base_spawn", True))
    use_random_drone_spawn  = bool(cfg.env.get("use_random_drone_spawn", True))
    target_spawn_method     = str(cfg.env.get("target_spawn_method", "map_defined"))
    target_spawn_radius     = float(cfg.env.get("target_spawn_radius", 0.0))
    target_spawn_radius_min = float(cfg.env.get("target_spawn_radius_min", 0.0))
    target_invalid_spawn_base_radius = float(cfg.env.get("target_invalid_spawn_base_radius", 0.0))
    precover_base_comm      = bool(cfg.env.get("precover_base_comm", False))
    log_adj                 = bool(cfg.env.get("log_adjacency_matrix", False)) or bool(cfg.network.get("memory_comm_enabled", False))
    experimental_setup      = bool(cfg.env.get("experimental_setup", False))
    use_finders_path        = (
        str(cfg.reward.get("chain_reward_system", "euclidean")) == "discrete_finders_path"
        and not experimental_setup
    )


    # How many matrix-squaring steps to guarantee full-graph reachability.
    # 2^n_reach steps covers paths up to 2^n_reach hops — always > N.
    n_reach = max(1, math.ceil(math.log2(N + 2)))

    # world geometry will be resolved by map loading
    BASE_POS_FIXED   = None
    TARGET_POS_FIXED = None
    ANTI_TARGET_POINTS = None

    # Map loading
    map_def      = None
    occ_grid     = None
    comm_occ_grid = None
    padded_grid  = None
    if cfg.env.map_names and len(cfg.env.map_names) > 0:
        active_map_name = cfg.env.map_names[0]
        map_path = MAP_DIR / f"{active_map_name}.yaml"

        if map_path.exists():
            map_def = MapDefinition.load(
                map_path,
                cell_size=1.0,
                padding_radius=PAD_RADIUS
            )
            W, H = map_def.width, map_def.height
            if map_def.occupancy_grid is not None:
                occ_grid = jnp.array(map_def.occupancy_grid, dtype=jnp.bool_)
            if map_def.communication_occupancy_grid is not None:
                comm_occ_grid = jnp.array(map_def.communication_occupancy_grid, dtype=jnp.bool_)
            if map_def.padded_occupancy_grid is not None:
                padded_grid = map_def.padded_occupancy_grid

            # Override base and target positions with the center of the map's spawn zones
            if map_def.base_spawn_zone:
                bx = (map_def.base_spawn_zone[0] + map_def.base_spawn_zone[2]) / 2.0
                by = (map_def.base_spawn_zone[1] + map_def.base_spawn_zone[3]) / 2.0
                BASE_POS_FIXED = jnp.array([bx, by], dtype=jnp.float32)

            if map_def.target_spawn_zone:
                tx = (map_def.target_spawn_zone[0] + map_def.target_spawn_zone[2]) / 2.0
                ty = (map_def.target_spawn_zone[1] + map_def.target_spawn_zone[3]) / 2.0
                TARGET_POS_FIXED = jnp.array([tx, ty], dtype=jnp.float32)
            # MEM_T8-only diagnostic scaffolding: paired anti-target slots.
            if map_def.anti_target_spawn_points is not None:
                if len(map_def.anti_target_spawn_points) >= N:
                    ANTI_TARGET_POINTS = map_def.anti_target_spawn_points[:N]
                else:
                    scale = N // len(map_def.anti_target_spawn_points)
                    anti_indices = jnp.arange(N) // scale
                    mapped_points = map_def.anti_target_spawn_points[anti_indices]
                    is_receiver = (jnp.arange(N) % 2 == 1)[:, None]
                    ANTI_TARGET_POINTS = jnp.where(is_receiver, mapped_points, 0.0)
        else:
            raise FileNotFoundError(f"Map file not found: {map_path}")

    # Final Grid Dimensions (Snapped to cell_size)
    if W is None or H is None:
        raise ValueError("Environment dimensions (box_width/height) are missing. Map loading failed or dimensions not in config.")

    W, H = float(W), float(H)
    GW, GH = int(W / cell_size), int(H / cell_size)

    if occ_grid is None:
        raise ValueError("Occupancy grid missing. A valid map MUST be loaded for physics.")
    if comm_occ_grid is None:
        comm_occ_grid = occ_grid

    # Padding size in cells
    R_cells = int(PAD_RADIUS / cell_size) + 2
    if padded_grid is None:
        raise ValueError("Padded occupancy grid missing. Map loading or padding logic failed.")

    maze_cols = int(map_def.maze_cell_cols or 1)
    maze_rows = int(map_def.maze_cell_rows or 1)
    maze_cell_w = float(W) / maze_cols
    maze_cell_h = float(H) / maze_rows
    max_finders_path_len = maze_cols * maze_rows if use_finders_path else 1
    base_maze_cell = jnp.array([
        int(max(0, min(maze_cols - 1, math.floor((float(W) / 2.0) / maze_cell_w)))),
        int(max(0, min(maze_rows - 1, math.floor((float(H) / 2.0) / maze_cell_h)))),
    ], dtype=jnp.int16)
    _maze_offsets = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int16)
    spawn_room_cells = jnp.clip(base_maze_cell[None, :] + _maze_offsets, jnp.array([0, 0], dtype=jnp.int16), jnp.array([maze_cols - 1, maze_rows - 1], dtype=jnp.int16))

    # 1. Pre-compute Ray Stencil for Visibility (Stencil + Cummax approach)
    # Circle sampling for visual radius
    vis_r_cells = int(vis_r / cell_size)
    L_size = 2 * vis_r_cells + 1
    ray_coords_np, _ = get_ray_stencil(vis_r_cells)
    RAY_STENCIL = jnp.array(ray_coords_np) # (NumRays, MaxLen, 2)

    # Local circular mask (to keep visibility round)
    local_dx, local_dy = jnp.meshgrid(jnp.arange(-vis_r_cells, vis_r_cells+1), jnp.arange(-vis_r_cells, vis_r_cells+1), indexing='ij')
    dist_sq = local_dx**2 + local_dy**2
    CIRCULAR_MASK = (dist_sq <= vis_r_cells**2)

    # Pre-compute grid cell centres (used by some logic/reward fns)
    xs = (jnp.arange(GW, dtype=jnp.float32) + 0.5) * cell_size
    ys = (jnp.arange(GH, dtype=jnp.float32) + 0.5) * cell_size
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    CELL_CENTRES = jnp.stack([gx, gy], axis=-1)   # (GW, GH, 2)

    _drone_indices = jnp.arange(N, dtype=jnp.int32)
    _eye = jnp.eye(N, dtype=jnp.float32)

    # ── Static Base Pre-Coverage Precomputation ──────────────────
    STATIC_PRECOVERED = None
    if precover_base_comm and not use_random_base_spawn:
        fixed_base_pos = (
            BASE_POS_FIXED
            if BASE_POS_FIXED is not None
            else jnp.array([float(W) / 2.0, float(H) / 2.0], dtype=jnp.float32)
        )
        dists = jnp.linalg.norm(CELL_CENTRES - fixed_base_pos[None, None, :], axis=-1)
        in_range = dists <= comm_r_base

        def _check_cell(c_pos, ir):
            return jax.lax.cond(
                ir,
                lambda op: dda_raycast(op[0] / cell_size, op[1] / cell_size, occ_grid),
                lambda _: jnp.bool_(False),
                operand=(c_pos, fixed_base_pos)
            )

        STATIC_PRECOVERED = jax.vmap(jax.vmap(_check_cell))(CELL_CENTRES, in_range)

    # ------------------------------------------------------------------
    # update_coverage
    # ------------------------------------------------------------------

    def update_coverage(
        coverage_grid: jax.Array,   # (MGW, MGH) bool
        pos:           jax.Array,   # (N, 2) float32
        active:        jax.Array,   # (N,)   bool
    ) -> tuple[jax.Array, jax.Array]: # ((MGW, MGH) bool, (N,) int32)
        """
        OR the coverage_grid using Local Spotlight Raycasting.
        """
        # Continuous Pos -> Grid Coords (int32)
        g_indices = (pos / cell_size).astype(jnp.int32)

        # We use a padded version of the coverage grid for accumulation to handle edges gracefully
        PAD = R_cells
        # Initialize with True so that any visibility "spilling" into the padding
        # is treated as already covered (delta = 0).
        cov_padded = jnp.ones((GW + 2*PAD, GH + 2*PAD), dtype=jnp.bool_)
        cov_padded = cov_padded.at[PAD:PAD+GW, PAD:PAD+GH].set(coverage_grid[:GW, :GH])

        def _update_one_drone(i, acc):
            acc_grid, cov_deltas = acc
            p_idx = g_indices[i] # world-relative grid index (0..GW, 0..GH)

            # 1. Slice patch from MAP (padded_grid)
            # World(0,0) is at PAD in padded_grid.
            # Drone at p_idx in world -> p_idx + PAD in grid.
            # Spotlight radius is vis_r_cells. Start = center - radius.
            slice_start = (p_idx[0] + PAD - vis_r_cells, p_idx[1] + PAD - vis_r_cells)
            patch = jax.lax.dynamic_slice(padded_grid, slice_start, (L_size, L_size))

            # 2. Compute Visibility
            vis_mask = compute_local_visibility(patch, RAY_STENCIL)
            final_mask = vis_mask & CIRCULAR_MASK & active[i]

            # 3. Accumulate into the PADDED coverage grid
            # Same coordinate system as padded_grid (start = center - radius)
            old_region = jax.lax.dynamic_slice(acc_grid, slice_start, (L_size, L_size))
            merged = jnp.maximum(old_region, final_mask)

            # Calculate how many new cells were covered
            delta = jnp.sum(merged) - jnp.sum(old_region)
            cov_deltas = cov_deltas.at[i].set(delta)

            return jax.lax.dynamic_update_slice(acc_grid, merged, slice_start), cov_deltas

        # Iterate over drones
        cov_deltas_init = jnp.zeros(N, dtype=jnp.int32)
        final_padded, final_deltas = jax.lax.fori_loop(0, N, _update_one_drone, (cov_padded, cov_deltas_init))

        # Slice back to world dimensions
        new_world = final_padded[PAD:PAD+GW, PAD:PAD+GH]
        return coverage_grid.at[:GW, :GH].set(new_world), final_deltas

    # ------------------------------------------------------------------
    # _update_target_known  (internal utility)
    # ------------------------------------------------------------------

    def _update_target_known(state: EnvState) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute the updated target knowledge and connectivity status.

        Returns
        -------
        new_target_known      : (N,) bool
        new_base_target_known : () bool
        is_conn_base          : (N,) bool
        is_conn_target        : (N,) bool
        adj_matrix            : (N+1, N+1) bool
        """
        # Convert continuous positions to grid indices
        def _get_cell_idx(p):
            # pos / cell_size -> index
            gx = jnp.clip((p[0] / cell_size), 0, GW-1).astype(jnp.int32)
            gy = jnp.clip((p[1] / cell_size), 0, GH-1).astype(jnp.int32)
            return gx * GH + gy

        idx_all = jax.vmap(_get_cell_idx)(state.pos) # (N,)

        # 1. Comm graph logic using DDA Raycasting
        def _check_comm(i, j):
            dist = jnp.linalg.norm(state.pos[i] - state.pos[j])
            in_range = (dist <= comm_r) & state.active[i] & state.active[j] & (i != j)
            # DDA raycast against the communication grid: mesh walls are transparent only for comm.
            can_see = dda_raycast(state.pos[i]/cell_size, state.pos[j]/cell_size, comm_occ_grid)
            return in_range & can_see

        adj_dd = jax.vmap(jax.vmap(_check_comm, (None, 0)), (0, None))(
            jnp.arange(N), jnp.arange(N)
        ).astype(jnp.float32)

        # Base connectivity
        def _check_base_comm(i):
            dist = jnp.linalg.norm(state.pos[i] - state.base_pos)
            in_range = (dist <= comm_r_base) & state.active[i]
            # DDA raycast against the communication grid: mesh walls are transparent only for comm.
            can_see = dda_raycast(state.pos[i]/cell_size, state.base_pos/cell_size, comm_occ_grid)
            return in_range & can_see

        adj_db = jax.vmap(_check_base_comm)(jnp.arange(N)).astype(jnp.float32)

        # Experimental memory diagnostics may use one target per agent. Normal
        # levels always use one shared target and should not execute diagnostic
        # target-shape behavior.
        per_agent_targets = experimental_setup and (state.target_pos.ndim == 2)
        target_pos_agents = state.target_pos if per_agent_targets else jnp.tile(state.target_pos[None, :], (N, 1))

        # 2. Drone-to-target LoS
        def _check_target_los(i):
            target_i = target_pos_agents[i]
            dist = jnp.linalg.norm(state.pos[i] - target_i)
            in_range = (dist <= vis_r) & state.active[i]
            can_see = dda_raycast(state.pos[i]/cell_size, target_i/cell_size, occ_grid)
            return in_range & can_see

        directly_sees = jax.vmap(_check_target_los)(jnp.arange(N))

        # 3. Graph reachability matrix squaring
        _eye_Np1 = jnp.eye(N + 1, dtype=jnp.float32)
        A_top = jnp.concatenate([adj_dd, adj_db[:, None]], axis=-1)
        A_bot = jnp.concatenate([adj_db[None, :], jnp.zeros((1, 1))], axis=-1)
        A_full = jnp.concatenate([A_top, A_bot], axis=0) + _eye_Np1

        def _square(R, _):
            return jnp.clip(R @ R, 0.0, 1.0), None

        R, _ = jax.lax.scan(_square, jnp.clip(A_full, 0.0, 1.0), None, length=n_reach)

        is_conn_base = R[:N, N] > 0.5
        is_conn_target = jnp.any((R[:N, :N] > 0.5) & directly_sees[None, :], axis=-1)

        # Persistent target knowledge propagation
        # Nodes that know the target before propagation in the current step
        knows_before = jnp.concatenate([
            state.target_known | directly_sees,
            jnp.array([state.base_target_known], dtype=jnp.bool_)
        ], axis=0)

        # Transitive propagation over the communication graph R (which includes base at index N)
        knows_after = jnp.any((R > 0.5) & knows_before[None, :], axis=-1)

        # Direct connection adjacency matrix of size (N+1, N+1) (excluding self-loops)
        if log_adj or use_finders_path:
            adj_matrix = jnp.concatenate([A_top, A_bot], axis=0) > 0.5
        else:
            adj_matrix = jnp.zeros((0, 0), dtype=jnp.bool_)

        new_target_known = knows_after[:N]
        new_base_target_known = knows_after[N]

        return new_target_known, new_base_target_known, is_conn_base, is_conn_target, adj_matrix, directly_sees

    def _maze_cell_from_pos(pos):
        cx = jnp.clip(jnp.floor(pos[0] / maze_cell_w).astype(jnp.int16), 0, maze_cols - 1)
        cy = jnp.clip(jnp.floor(pos[1] / maze_cell_h).astype(jnp.int16), 0, maze_rows - 1)
        return jnp.stack([cx, cy]).astype(jnp.int16)

    def _cell_center(cell):
        return jnp.array([
            (cell[0].astype(jnp.float32) + 0.5) * maze_cell_w,
            (cell[1].astype(jnp.float32) + 0.5) * maze_cell_h,
        ], dtype=jnp.float32)

    def _is_spawn_room_cell(cell):
        return jnp.any(jnp.all(spawn_room_cells == cell[None, :], axis=-1))

    def _append_loop_erased(path, length, cell):
        valid = jnp.arange(max_finders_path_len) < length.astype(jnp.int32)
        matches = valid & jnp.all(path == cell[None, :], axis=-1)
        already = jnp.any(matches)
        match_idx = jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int16)
        last_idx = jnp.maximum(length.astype(jnp.int32) - 1, 0)
        last_cell = path[last_idx]
        same_last = (length > 0) & jnp.all(last_cell == cell)
        can_append = (length.astype(jnp.int32) < max_finders_path_len) & ~already & ~same_last
        next_len = jnp.where(already, match_idx + jnp.int16(1), jnp.where(can_append, length + jnp.int16(1), length))
        path = jnp.where(can_append, path.at[length.astype(jnp.int32)].set(cell), path)
        return path, next_len

    def _update_finders_path_state(
        state: EnvState,
        mid_state: EnvState,
        directly_sees: jax.Array,
        new_target_known: jax.Array,
        new_base_target_known: jax.Array,
        is_conn_base: jax.Array,
    ) -> dict:
        if not use_finders_path:
            return {}

        cells = jax.vmap(_maze_cell_from_pos)(mid_state.pos)
        in_spawn = jax.vmap(_is_spawn_room_cell)(cells)

        def _update_one(i, carry):
            paths, lens, active = carry
            path = paths[i]
            length = lens[i]
            # Path buffers are fixed-size JAX arrays; this flag is only a
            # feature-state marker, not a memory-saving mechanism.  A path starts
            # once an active agent leaves the start area. If an agent returns to
            # the start area before discovering the target itself, erase that
            # candidate route so it can start over from the next exit.
            was_active = length > 0
            now_active = was_active | (~in_spawn[i] & mid_state.active[i])
            reset_path = in_spawn[i] & was_active & (~state.target_known[i]) & (~directly_sees[i])

            # Seed with base, previous start-area/exit cell, and first non-start cell.
            prev_cell = _maze_cell_from_pos(state.pos[i])
            seed = path.at[0].set(base_maze_cell).at[1].set(prev_cell).at[2].set(cells[i])
            seed_len = jnp.int16(3)
            path = jnp.where((~was_active) & now_active, seed, path)
            length = jnp.where((~was_active) & now_active, seed_len, length)

            path, length = jax.lax.cond(
                was_active & now_active & ~reset_path & mid_state.active[i],
                lambda op: _append_loop_erased(op[0], op[1], cells[i]),
                lambda op: op,
                (path, length),
            )
            path = jnp.where(reset_path, jnp.zeros_like(path), path)
            length = jnp.where(reset_path, jnp.int16(0), length)
            now_active = length > 0
            paths = paths.at[i].set(path)
            lens = lens.at[i].set(length)
            active = active.at[i].set(now_active)
            return paths, lens, active

        paths, lens, active = jax.lax.fori_loop(
            0, N, _update_one,
            (state.finder_path_cells, state.finder_path_lens, state.finder_path_active)
        )

        # Per-agent target-knowledge paths: direct observers lock their own path;
        # agents that learn through communication copy the lowest-index known path
        # in their connected component. The global finder path is frozen from this
        # per-agent memory, not directly from the first target observation when
        # delivery is required.
        path_ready = lens > 0
        per_agent_targets = experimental_setup and (state.target_pos.ndim == 2)
        target_pos_agents = state.target_pos if per_agent_targets else jnp.tile(state.target_pos[None, :], (N, 1))

        def _append_target(path_len_tgt):
            path, length, tgt_pos = path_len_tgt
            tgt_cell = _maze_cell_from_pos(tgt_pos)
            last_cell_idx = jnp.maximum(length - 1, 0)
            last_cell = path[last_cell_idx]
            is_same = jnp.all(last_cell == tgt_cell)
            dx = jnp.abs(last_cell[0] - tgt_cell[0])
            dy = jnp.abs(last_cell[1] - tgt_cell[1])
            is_neighbor = (dx <= 1) & (dy <= 1)
            should_append = (~is_same) & is_neighbor & (length < max_finders_path_len)
            path = jnp.where(should_append, path.at[length.astype(jnp.int32)].set(tgt_cell), path)
            length = jnp.where(should_append, length + jnp.int16(1), length)
            return path, length

        direct_paths, direct_lens = jax.vmap(_append_target)((paths, lens, target_pos_agents))
        direct_valid = directly_sees & path_ready & (~state.target_known_path_valid)

        source_paths = jnp.where(direct_valid[:, None, None], direct_paths, state.target_known_path_cells)
        source_lens = jnp.where(direct_valid, direct_lens, state.target_known_path_lens)
        source_valid = state.target_known_path_valid | direct_valid

        # Recompute reachability over the direct communication graph stored by
        # _update_target_known so path memory follows the same target-known spread.
        A = mid_state.adj_matrix[:N, :N] | jnp.eye(N, dtype=jnp.bool_)
        def _square_bool(R, _):
            return (R.astype(jnp.float32) @ R.astype(jnp.float32) > 0.5), None
        R_agents, _ = jax.lax.scan(_square_bool, A, None, length=n_reach)
        source_matrix = R_agents & source_valid[None, :]
        source_idx = jnp.argmax(source_matrix.astype(jnp.int32), axis=1)
        has_source = jnp.any(source_matrix, axis=1)
        should_set_known_path = new_target_known & (~state.target_known_path_valid) & has_source
        known_paths = jnp.where(should_set_known_path[:, None, None], source_paths[source_idx], state.target_known_path_cells)
        known_lens = jnp.where(should_set_known_path, source_lens[source_idx], state.target_known_path_lens)
        known_valid = state.target_known_path_valid | should_set_known_path

        delivery_freeze = bool(cfg.reward.get("target_found_requires_delivery", True))
        delivered_now = (~state.base_target_known) & new_base_target_known
        freeze_candidates = jnp.where(
            delivery_freeze,
            known_valid & is_conn_base & new_target_known & delivered_now,
            direct_valid,
        )
        first_find_now = (~state.finders_path_valid) & jnp.any(freeze_candidates)
        finder_idx = jnp.argmax(freeze_candidates.astype(jnp.int32))
        selected_path = jnp.where(delivery_freeze, known_paths[finder_idx], direct_paths[finder_idx])
        selected_len = jnp.where(delivery_freeze, known_lens[finder_idx], direct_lens[finder_idx])

        # Sanity check: warning if not same and not neighbor
        # def _warn_fn(idx, slen, lc, tc, drone_pos, tgt_pos, direct, ready, cond):
        #     jax.debug.print(
        #         "WARNING: Target discovered but final finder-path cell {c1} is not a neighbor of target cell {c2}!\n"
        #         "  finder_idx={idx} path_len={slen} drone_pos={dp} target_pos={tp} directly_sees={d} path_ready={r}",
        #         c1=lc, c2=tc, idx=idx, slen=slen, dp=drone_pos, tp=tgt_pos, d=direct, r=ready,
        #         when=cond
        #     )
        #     return None
        # 
        # _warn_fn(
        #     finder_idx, selected_len, last_cell, tgt_cell,
        #     state.pos[finder_idx], target_pos_agents[finder_idx],
        #     directly_sees[finder_idx], path_ready[finder_idx],
        #     first_find_now & (~is_same) & (~is_neighbor)
        # )

        def _build_index_grid(path_len_path):
            path_len, path = path_len_path
            grid = jnp.full((maze_cols, maze_rows), jnp.int16(-1), dtype=jnp.int16)

            def _set_idx(k, g):
                cell = path[k]
                return jax.lax.cond(
                    k < path_len.astype(jnp.int32),
                    lambda gg: gg.at[cell[0].astype(jnp.int32), cell[1].astype(jnp.int32)].set(k.astype(jnp.int16)),
                    lambda gg: gg,
                    g,
                )

            return jax.lax.fori_loop(0, max_finders_path_len, _set_idx, grid)

        new_finders_path = jnp.where(first_find_now, selected_path, state.finders_path)
        new_finders_len = jnp.where(first_find_now, selected_len, state.finders_path_len)
        new_index_grid = jnp.where(
            first_find_now,
            _build_index_grid((selected_len, selected_path)),
            state.finders_path_index_grid,
        )
        keep_tracking_global = ~(state.finders_path_valid | first_find_now)
        keep_tracking_agents = keep_tracking_global & (~known_valid)
        return {
            "finder_path_cells": jnp.where(keep_tracking_agents[:, None, None], paths, state.finder_path_cells),
            "finder_path_lens": jnp.where(keep_tracking_agents, lens, state.finder_path_lens),
            "finder_path_active": jnp.where(keep_tracking_agents, active, state.finder_path_active),
            "target_known_path_cells": known_paths,
            "target_known_path_lens": known_lens,
            "target_known_path_valid": known_valid,
            "finders_path": new_finders_path,
            "finders_path_len": new_finders_len,
            "finders_path_valid": state.finders_path_valid | first_find_now,
            "finders_path_index_grid": new_index_grid,
        }

    def _target_revisit_candidates(state: EnvState, target_known: jax.Array) -> jax.Array:
        """Agents that know and visually observe the target after base delivery."""
        per_agent_targets = experimental_setup and (state.target_pos.ndim == 2)
        target_pos_agents = state.target_pos if per_agent_targets else jnp.tile(state.target_pos[None, :], (N, 1))

        def _sees_target(i):
            target_i = target_pos_agents[i]
            dist = jnp.linalg.norm(state.pos[i] - target_i)
            in_range = (dist <= vis_r) & state.active[i]
            can_see = dda_raycast(state.pos[i] / cell_size, target_i / cell_size, occ_grid)
            return in_range & can_see

        sees_target = jax.vmap(_sees_target)(jnp.arange(N))
        return state.base_target_known & target_known & sees_target

    def _update_anti_target_known(state: EnvState) -> jax.Array:
        if ANTI_TARGET_POINTS is None or state.anti_target_known.size == 0:
            return jnp.zeros(N, dtype=jnp.bool_)

        def _sees_anti(i):
            anti_pos = ANTI_TARGET_POINTS[i]
            dist = jnp.linalg.norm(state.pos[i] - anti_pos)
            in_range = (dist <= vis_r) & state.active[i]
            can_see = dda_raycast(state.pos[i] / cell_size, anti_pos / cell_size, occ_grid)
            return in_range & can_see

        sees_now = jax.vmap(_sees_anti)(jnp.arange(N))
        return state.anti_target_known | sees_now

    def reset(key: jax.Array) -> EnvState:
        key, k1, k2, k2b, k3 = jax.random.split(key, 5)

        if map_def is not None:
            # ── Base position ──────────────────────────────────────────────
            if use_random_base_spawn and use_task:
                base_pos = map_def.sample_base(k1)
            else:
                # Fixed: exact centre of the loaded map
                base_pos = jnp.array([float(W) / 2.0, float(H) / 2.0], dtype=jnp.float32)

            # ── Target position ────────────────────────────────────────────
            if target_spawn_method == "ring":
                # Uniform random point inside a ring (min to max radius)
                # around the base position (using rejection-free square-root trick)
                angle = jax.random.uniform(k2, shape=(), minval=0.0, maxval=2.0 * math.pi)
                r_sq = jax.random.uniform(
                    k2b, shape=(),
                    minval=target_spawn_radius_min**2,
                    maxval=target_spawn_radius**2
                )
                r = jnp.sqrt(r_sq)
                offset = jnp.array([r * jnp.cos(angle), r * jnp.sin(angle)], dtype=jnp.float32)
                target_pos = jnp.clip(
                    base_pos + offset,
                    jnp.array([1.0, 1.0]),
                    jnp.array([float(W) - 1.0, float(H) - 1.0]),
                )
            elif target_spawn_method == "outside_base":
                # Generate candidate points within the 2m boundary, then choose
                # the first candidate that is both outside the base exclusion
                # radius and in a free occupancy-grid cell.
                bounds_min = jnp.array([2.0, 2.0], dtype=jnp.float32)
                bounds_max = jnp.array([float(W) - 2.0, float(H) - 2.0], dtype=jnp.float32)
                candidates = jax.random.uniform(k2, shape=(64, 2), minval=bounds_min, maxval=bounds_max)

                # Compute distance to base for all candidates
                dists_to_base = jnp.linalg.norm(candidates - base_pos[None, :], axis=-1)

                cand_idx = jnp.floor(candidates / cell_size).astype(jnp.int32)
                cand_in_bounds = (
                    (cand_idx[:, 0] >= 0) & (cand_idx[:, 0] < GW)
                    & (cand_idx[:, 1] >= 0) & (cand_idx[:, 1] < GH)
                )
                cand_idx_safe = jnp.stack([
                    jnp.clip(cand_idx[:, 0], 0, GW - 1),
                    jnp.clip(cand_idx[:, 1], 0, GH - 1),
                ], axis=-1)
                cand_free = cand_in_bounds & ~occ_grid[cand_idx_safe[:, 0], cand_idx_safe[:, 1]]

                valid_mask = cand_free & (dists_to_base > target_invalid_spawn_base_radius)

                # Pick the first valid candidate. If every candidate is too
                # close to base, fall back to the farthest free candidate.
                fallback_idx = jnp.argmax(jnp.where(cand_free, dists_to_base, -1.0))
                valid_idx = jnp.where(jnp.any(valid_mask), jnp.argmax(valid_mask), fallback_idx)
                target_pos = candidates[valid_idx]
            elif use_task:  # fallback to "map_defined"
                # MEM_T8-only diagnostic path: fixed per-agent target slots.
                if map_def.target_spawn_points is not None and int(cfg.env.num_targets) > 1:
                    if len(map_def.target_spawn_points) >= N:
                        target_pos = map_def.target_spawn_points[:N]
                    else:
                        scale = N // len(map_def.target_spawn_points)
                        target_indices = jnp.arange(N) // scale
                        mapped_pos = map_def.target_spawn_points[target_indices]
                        is_receiver = (jnp.arange(N) % 2 == 1)[:, None]
                        target_pos = jnp.where(is_receiver, mapped_pos, 0.0)
                else:
                    target_pos = map_def.sample_target(k2)
            else:
                target_pos = TARGET_POS_FIXED if TARGET_POS_FIXED is not None else jnp.zeros(2, dtype=jnp.float32)

            # ── Drone positions ────────────────────────────────────────────
            if use_random_drone_spawn:
                pos = map_def.sample_drones(k3, N)
            else:
                # All drones start stacked at the base position
                pos = jnp.tile(base_pos[None, :], (N, 1))
        else:
            raise ValueError("map_def is None. Map MUST be loaded to sample reset positions.")

        if base_pos is None or target_pos is None:
            raise ValueError("Reset failed: base_pos or target_pos could not be resolved from map.")

        vel           = jnp.zeros((N, 2), dtype=jnp.float32)
        coverage_grid = jnp.zeros((GW, GH), dtype=jnp.bool_)

        if precover_base_comm:
            if STATIC_PRECOVERED is not None:
                coverage_grid = coverage_grid | STATIC_PRECOVERED
            else:
                dists = jnp.linalg.norm(CELL_CENTRES - base_pos[None, None, :], axis=-1)
                in_range = dists <= comm_r_base

                def _check_cell(c_pos, ir):
                    return jax.lax.cond(
                        ir,
                        lambda op: dda_raycast(op[0] / cell_size, op[1] / cell_size, occ_grid),
                        lambda _: jnp.bool_(False),
                        operand=(c_pos, base_pos)
                    )

                precovered = jax.vmap(jax.vmap(_check_cell))(CELL_CENTRES, in_range)
                coverage_grid = coverage_grid | precovered


        if spawn_delay > 0:
            active = (_drone_indices == 0)
        else:
            active = jnp.ones(N, dtype=jnp.bool_)

        target_known = jnp.zeros(N, dtype=jnp.bool_)
        # MEM_T8-only diagnostic state; ignored by normal one-target levels.
        anti_target_known = jnp.zeros(N, dtype=jnp.bool_)
        collides     = jnp.zeros(N, dtype=jnp.bool_)

        return EnvState(
            pos           = pos,
            vel           = vel,
            base_pos      = base_pos,
            target_pos    = target_pos,
            coverage_grid = coverage_grid,
            step          = jnp.int32(0),
            key           = key,
            active        = active,
            target_known  = target_known,
            anti_target_known = anti_target_known,
            collides      = collides,
            last_cov_delta= jnp.zeros(N, dtype=jnp.int32),
            box_width     = jnp.float32(W),
            box_height    = jnp.float32(H),
            base_target_known = jnp.bool_(False),
            target_revisit_reward_claimed = jnp.bool_(False),
            finder_returned_to_target = jnp.bool_(False),
            chain_held_steps = jnp.int32(0),
            is_conn_base      = jnp.zeros(N, dtype=jnp.bool_),
            is_conn_target    = jnp.zeros(N, dtype=jnp.bool_),
            finder_path_cells = jnp.zeros((N, max_finders_path_len, 2), dtype=jnp.int16),
            finder_path_lens = jnp.zeros(N, dtype=jnp.int16),
            finder_path_active = jnp.zeros(N, dtype=jnp.bool_),
            target_known_path_cells = jnp.zeros((N, max_finders_path_len, 2), dtype=jnp.int16),
            target_known_path_lens = jnp.zeros(N, dtype=jnp.int16),
            target_known_path_valid = jnp.zeros(N, dtype=jnp.bool_),
            finders_path = jnp.zeros((max_finders_path_len, 2), dtype=jnp.int16),
            finders_path_len = jnp.int16(0),
            finders_path_valid = jnp.bool_(False),
            finders_path_index_grid = jnp.full((maze_cols, maze_rows), jnp.int16(-1), dtype=jnp.int16),
            adj_matrix        = jnp.zeros((N + 1, N + 1), dtype=jnp.bool_) if log_adj else jnp.zeros((0, 0), dtype=jnp.bool_),
        )

    # ------------------------------------------------------------------
    # _physics_step  (internal — pure Newtonian integration)
    # ------------------------------------------------------------------

    def _physics_step(
        state:   EnvState,
        actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        a_norm    = jnp.linalg.norm(actions, axis=-1, keepdims=True)
        safe_norm = jnp.where(a_norm > 0, a_norm, 1.0)
        actions   = jnp.where(a_norm > f_max, actions / safe_norm * f_max, actions)

        vel = state.vel * drag + actions * dt

        speed      = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        safe_speed = jnp.where(speed > 0, speed, 1.0)
        vel        = jnp.where(speed > v_max, vel / safe_speed * v_max, vel)

        # Position integration
        next_pos = state.pos + vel * dt

        # 1. External Box Collisions
        hit_x_lo = next_pos[:, 0] < 0.0
        hit_x_hi = next_pos[:, 0] > state.box_width
        hit_y_lo = next_pos[:, 1] < 0.0
        hit_y_hi = next_pos[:, 1] > state.box_height

        # 2. Internal Grid Collisions (Sub-stepping CCD)
        # Check 8 intermediate points along the path from state.pos to next_pos.
        # This is more robust than DDA for 1m thick walls at drone speeds.
        t_sub = jnp.linspace(0.125, 1.0, 8) # (8,)
        # intermediate positions: (N, 8, 2)
        p_sub = state.pos[:, None, :] + t_sub[None, :, None] * (next_pos - state.pos)[:, None, :]

        # Grid indices for substeps: (N, 8, 2)
        g_idx_sub = jnp.floor(p_sub / cell_size).astype(jnp.int32)

        # In-bounds check using JAX-safe types
        GW_active = (state.box_width / cell_size).astype(jnp.int32)
        GH_active = (state.box_height / cell_size).astype(jnp.int32)
        in_bounds = (g_idx_sub[..., 0] >= 0) & (g_idx_sub[..., 0] < GW_active) & \
                    (g_idx_sub[..., 1] >= 0) & (g_idx_sub[..., 1] < GH_active)

        # Masked lookup: (N, 8)
        # We handle out-of-bounds by treating them as 'no-hit' as the border check handles those
        def _lookup(idx, ib):
            return jnp.where(ib, occ_grid[idx[0], idx[1]], False)

        hit_sub = jax.vmap(jax.vmap(_lookup))(g_idx_sub, in_bounds)
        hit_grid = jnp.any(hit_sub, axis=1) # (N,)

        collided = hit_x_lo | hit_x_hi | hit_y_lo | hit_y_hi | hit_grid

        vx = jnp.where(hit_x_lo | hit_x_hi | hit_grid, -vel[:, 0] * wall_res, vel[:, 0])
        vy = jnp.where(hit_y_lo | hit_y_hi | hit_grid, -vel[:, 1] * wall_res, vel[:, 1])
        vel = jnp.stack([vx, vy], axis=-1)

        pos = jnp.where(collided[:, None], state.pos, next_pos)
        pos = jnp.stack([
            jnp.clip(pos[:, 0], 0.0, state.box_width),
            jnp.clip(pos[:, 1], 0.0, state.box_height),
        ], axis=-1)

        return pos, vel, collided

    # ------------------------------------------------------------------
    # env_step  (public — full environment step)
    # ------------------------------------------------------------------

    def env_step(
        state:   EnvState,
        actions: jax.Array,   # (N, 2) continuous forces
    ) -> EnvState:
        """
        One full environment step. Returns a new EnvState.

        Pipeline
        --------
        1.  Activate any drone whose spawn step has arrived
        2.  Mask actions for inactive drones (force = 0)
        3.  Euler integration + wall bounce  (_physics_step)
        4.  Hold inactive drones at base_pos with zero velocity
        5.  Update coverage grid (active drones only)
        6.  Update target_known (persistent comm-propagated knowledge)
        7.  Advance PRNG key and step counter
        """
        chex.assert_shape(actions, (N, 2))

        # 1. Spawn activation: drone i wakes up at step == i * spawn_delay
        if spawn_delay > 0:
            should_activate = (_drone_indices * spawn_delay) == state.step   # (N,) bool
            new_active      = state.active | should_activate
        else:
            new_active = jnp.ones(N, dtype=jnp.bool_)

        # 2. Mask actions for inactive drones
        actions_masked = jnp.where(new_active[:, None], actions, 0.0)

        # 3. Physics step
        new_pos, new_vel, collided = _physics_step(state, actions_masked)

        # 4. Inactive enforcement
        # If not active, pos = base_pos, vel = 0, collided = False
        new_pos = jnp.where(new_active[:, None], new_pos, state.base_pos[None, :])
        new_vel = jnp.where(new_active[:, None], new_vel, 0.0)
        collided = collided & new_active

        # 5. Coverage (active drones only)
        new_coverage, new_cov_deltas = update_coverage(state.coverage_grid, new_pos, new_active)

        # 6. Advance PRNG key
        new_key, _ = jax.random.split(state.key)

        # Build intermediate state with updated positions (for target_known calc)
        mid_state = dataclasses.replace(
            state,
            pos           = new_pos,
            vel           = new_vel,
            coverage_grid = new_coverage,
            step          = state.step + jnp.int32(1),
            key           = new_key,
            active        = new_active,
            collides      = collided,
            last_cov_delta= new_cov_deltas,
        )

        # 7. Update persistent target knowledge using new positions
        new_target_known, new_base_target_known, is_conn_base, is_conn_target, new_adj_matrix, directly_sees = _update_target_known(mid_state)
        # MEM_T8-only diagnostic state update; no-op for normal levels.
        new_anti_target_known = _update_anti_target_known(mid_state)
        target_revisit_candidates = _target_revisit_candidates(mid_state, new_target_known)
        new_target_revisit_reward_claimed = (
            state.target_revisit_reward_claimed | jnp.any(target_revisit_candidates)
        )
        path_mid_state = dataclasses.replace(mid_state, adj_matrix=new_adj_matrix)
        path_updates = _update_finders_path_state(
            state, path_mid_state, directly_sees, new_target_known, new_base_target_known, is_conn_base
        )

        return dataclasses.replace(
            mid_state,
            target_known=new_target_known,
            base_target_known=new_base_target_known,
            target_revisit_reward_claimed=new_target_revisit_reward_claimed,
            finder_returned_to_target=state.finder_returned_to_target | (
                state.base_target_known & ~state.target_revisit_reward_claimed & jnp.any(target_revisit_candidates)
            ),
            anti_target_known=new_anti_target_known,
            is_conn_base=is_conn_base,
            is_conn_target=is_conn_target,
            adj_matrix=new_adj_matrix,
            **path_updates,
        )

    return env_step, reset, update_coverage, (W, H, occ_grid, comm_occ_grid)


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    from core.config import load_config, validate_config
    from env.state import assert_env_state

    cfg = load_config(cli_overrides=False)
    validate_config(cfg)

    env_step, reset, update_coverage, (W, H, occ_grid, comm_occ_grid) = make_env_fns(cfg)

    reset_jit = jax.jit(reset)
    step_jit  = jax.jit(env_step)

    key   = jax.random.PRNGKey(0)
    state = reset_jit(key)

    GW, GH = occ_grid.shape
    N   = int(cfg.env.num_agents)

    assert_env_state(state, N, GW, GH)
    print(f"Reset OK  | pos shape: {state.pos.shape} | step: {state.step}")
    print(f"          | active: {state.active} | target_known: {state.target_known}")

    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (N, 2),
        minval=-cfg.env.max_force, maxval=cfg.env.max_force,
    )
    state2 = step_jit(state, actions)
    assert_env_state(state2, N, GW, GH)
    print(f"Step  OK  | pos[0]: {state2.pos[0]} | step: {state2.step}")
    print(f"          | active: {state2.active}")
    print(f"Coverage  | cells covered: {state2.coverage_grid.sum()}")
    print("\nPhysics engine self-test passed ✓")
