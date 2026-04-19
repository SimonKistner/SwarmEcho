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
    """

    # --- Extract config as Python scalars (XLA compile-time constants) -----
    N           = int(cfg.env.num_agents)
    cell_size   = float(cfg.env.grid_cell_size)
    MGW         = int(cfg.env.max_grid_width)
    MGH         = int(cfg.env.max_grid_height)
    dt          = float(cfg.env.dt)
    drag        = float(cfg.env.drag)
    v_max       = float(cfg.env.max_speed)
    f_max       = float(cfg.env.max_force)
    W           = float(cfg.env.box_width)
    H           = float(cfg.env.box_height)
    vis_r       = float(cfg.env.visual_radius)
    comm_r      = float(cfg.env.comm_radius)
    spawn_delay = int(cfg.env.spawn_delay)
    wall_res    = float(cfg.env.wall_restitution)
    use_task    = (int(cfg.env.num_targets) > 0) or (int(cfg.env.num_bases) > 0)

    # How many matrix-squaring steps to guarantee full-graph reachability.
    # 2^n_reach steps covers paths up to 2^n_reach hops — always > N.
    n_reach = max(1, math.ceil(math.log2(N + 2)))

    # Fixed world geometry (may be overridden by map)
    BASE_POS_FIXED   = jnp.array([cfg.env.base_x,   cfg.env.base_y],   dtype=jnp.float32)
    TARGET_POS_FIXED = jnp.array([cfg.env.target_x, cfg.env.target_y], dtype=jnp.float32)

    # Map loading
    map_def      = None
    occ_grid     = None
    padded_grid  = None
    if cfg.env.map_names and len(cfg.env.map_names) > 0:
        active_map_name = cfg.env.map_names[0]
        map_path = MAP_DIR / f"{active_map_name}.yaml"
        
        if map_path.exists():
            map_def = MapDefinition.load(
                map_path, 
                cell_size=float(cfg.env.grid_cell_size),
                padding_radius=PAD_RADIUS
            )
            W, H = map_def.width, map_def.height
            if map_def.occupancy_grid is not None:
                occ_grid = jnp.array(map_def.occupancy_grid, dtype=jnp.bool_)
            if map_def.padded_occupancy_grid is not None:
                padded_grid = map_def.padded_occupancy_grid
        else:
            raise FileNotFoundError(f"Map file not found: {map_path}")

    # Final Grid Dimensions (Snapped to cell_size)
    GW = int(W / cell_size)
    GH = int(H / cell_size)
    
    # Fallback empty grid if no map
    if occ_grid is None:
        occ_grid = jnp.zeros((GW, GH), dtype=jnp.bool_)
    
    # Padding size in cells
    R_cells = int(PAD_RADIUS / cell_size) + 2
    if padded_grid is None:
        padded_grid = jnp.ones((MGW + 2*R_cells, MGH + 2*R_cells), dtype=jnp.bool_)
        # Use config dimensions for initial fallback box if map is missing
        padded_grid = padded_grid.at[R_cells:R_cells+int(W/cell_size), R_cells:R_cells+int(H/cell_size)].set(occ_grid)

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

    # ------------------------------------------------------------------
    # update_coverage
    # ------------------------------------------------------------------

    def update_coverage(
        coverage_grid: jax.Array,   # (MGW, MGH) bool
        pos:           jax.Array,   # (N, 2) float32
        active:        jax.Array,   # (N,)   bool
    ) -> jax.Array:                 # (MGW, MGH) bool
        """
        OR the coverage_grid using Local Spotlight Raycasting.
        """
        # Continuous Pos -> Grid Coords (int32)
        g_indices = (pos / cell_size).astype(jnp.int32)
        
        # We use a padded version of the coverage grid for accumulation to handle edges gracefully
        PAD = R_cells
        cov_padded = jnp.zeros((GW + 2*PAD, GH + 2*PAD), dtype=jnp.bool_)
        cov_padded = cov_padded.at[PAD:PAD+GW, PAD:PAD+GH].set(coverage_grid[:GW, :GH])
        
        def _update_one_drone(acc_grid, i):
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
            return jax.lax.dynamic_update_slice(acc_grid, merged, slice_start)

        # Iterate over drones
        def _body(i, acc):
            return _update_one_drone(acc, i)
        
        final_padded = jax.lax.fori_loop(0, N, _body, cov_padded)
        
        # Slice back to world dimensions
        new_world = final_padded[PAD:PAD+GW, PAD:PAD+GH]
        return coverage_grid.at[:GW, :GH].set(new_world)

    # ------------------------------------------------------------------
    # _update_target_known  (internal utility)
    # ------------------------------------------------------------------

    def _update_target_known(state: EnvState) -> jax.Array:
        """
        Compute the updated (N,) bool target_known array using pre-computed visibility.
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
            # DDA Raycast (structural)
            can_see = dda_raycast(state.pos[i]/cell_size, state.pos[j]/cell_size, occ_grid)
            return in_range & can_see

        adj_dd = jax.vmap(jax.vmap(_check_comm, (None, 0)), (0, None))(
            jnp.arange(N), jnp.arange(N)
        ).astype(jnp.float32)

        # 2. Drone-to-target LoS
        def _check_target_los(i):
            dist = jnp.linalg.norm(state.pos[i] - state.target_pos)
            in_range = (dist <= vis_r) & state.active[i]
            # DDA Raycast (structural)
            can_see = dda_raycast(state.pos[i]/cell_size, state.target_pos/cell_size, occ_grid)
            return in_range & can_see
        
        is_visible = jax.vmap(_check_target_los)(jnp.arange(N))

        # 3. Reachability via repeated matrix squaring (n_reach steps)
        A = adj_dd + _eye   # add self-loops

        def _mat_square(R: jax.Array, _) -> tuple[jax.Array, None]:
            return jnp.clip(R @ R, 0.0, 1.0), None

        R, _ = jax.lax.scan(_mat_square, A, None, length=n_reach)

        # 4. Propagation: Any drone reachable from drone i that directly sees OR knows target
        currently_informed = jnp.any(
            (R > 0.5) & (is_visible | state.target_known)[None, :], axis=-1
        )  # (N,) bool

        # If no target logic is enabled (curriculum levels 0-3), everyone knows nothing
        currently_informed = jnp.where(int(cfg.env.num_targets) > 0, currently_informed, False)

        # 5. Persistent OR: once known, never forgotten
        return state.target_known | currently_informed

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(key: jax.Array) -> EnvState:
        key, k1, k2, k3 = jax.random.split(key, 4)

        if map_def is not None:
            # Only sample Task related positions if using task logic
            base_pos   = map_def.sample_base(k1) if use_task else BASE_POS_FIXED
            target_pos = map_def.sample_target(k2) if use_task else TARGET_POS_FIXED
            pos        = map_def.sample_drones(k3, N)
        else:
            base_pos   = BASE_POS_FIXED
            target_pos = TARGET_POS_FIXED
            pos        = jnp.tile(base_pos, (N, 1)).astype(jnp.float32)

        vel           = jnp.zeros((N, 2), dtype=jnp.float32)
        coverage_grid = jnp.zeros((MGW, MGH), dtype=jnp.bool_)

        if spawn_delay > 0:
            active = (_drone_indices == 0)
        else:
            active = jnp.ones(N, dtype=jnp.bool_)

        target_known = jnp.zeros(N, dtype=jnp.bool_)
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
            collides      = collides,
            box_width     = jnp.float32(W),
            box_height    = jnp.float32(H),
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
        new_coverage = update_coverage(state.coverage_grid, new_pos, new_active)

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
        )

        # 7. Update persistent target knowledge using new positions
        new_target_known = _update_target_known(mid_state)

        return dataclasses.replace(mid_state, target_known=new_target_known)

    return env_step, reset, update_coverage


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

    env_step, reset, update_coverage = make_env_fns(cfg)

    reset_jit = jax.jit(reset)
    step_jit  = jax.jit(env_step)

    key   = jax.random.PRNGKey(0)
    state = reset_jit(key)

    MGW = int(cfg.env.max_grid_width)
    MGH = int(cfg.env.max_grid_height)

    assert_env_state(state, N, MGW, MGH)
    print(f"Reset OK  | pos shape: {state.pos.shape} | step: {state.step}")
    print(f"          | active: {state.active} | target_known: {state.target_known}")

    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (N, 2),
        minval=-cfg.env.max_force, maxval=cfg.env.max_force,
    )
    state2 = step_jit(state, actions)
    assert_env_state(state2, N, MGW, MGH)
    print(f"Step  OK  | pos[0]: {state2.pos[0]} | step: {state2.step}")
    print(f"          | active: {state2.active}")
    print(f"Coverage  | cells covered: {state2.coverage_grid.sum()}")
    print("\nPhysics engine self-test passed ✓")


