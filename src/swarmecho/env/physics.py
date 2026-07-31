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

    env_step, reset, update_coverage, world_metadata = make_env_fns(cfg)

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
`state.communication.target_known[i]` is monotonically set. It becomes True the
first time drone i is connected (directly or via the comm chain) to a drone
that visually sees the target. Once True, it never reverts — the drone retains
knowledge of the target's position even if communication is later lost.

The update is computed in env_step AFTER physics so new positions are used.
"""

from __future__ import annotations

import math

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from swarmecho.core.config import MAP_DIR
from swarmecho.env.grid_utils import has_inner_obstacles
from swarmecho.env.maps import MapDefinition
from swarmecho.env.raycast import (
    PAD_RADIUS,
    compute_local_visibility,
    dda_raycast,
    get_ray_stencil,
)
from swarmecho.env.relay_task import make_relay_task_fns
from swarmecho.env.state import (
    CommunicationState,
    EnvState,
    ExplorationState,
    PhysicsState,
    StepDiagnostics,
)


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
    world_meta      : tuple (W, H, occ_grid) - resolved box width, height,
                      and occupancy grid
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
    # How many matrix-squaring steps to guarantee full-graph reachability.
    # 2^n_reach steps covers paths up to 2^n_reach hops — always > N.
    n_reach = max(1, math.ceil(math.log2(N + 2)))

    # world geometry will be resolved by map loading
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
                cell_size=1.0,
                padding_radius=PAD_RADIUS,
            )
            W, H = map_def.width, map_def.height
            if map_def.occupancy_grid is not None:
                occ_grid = jnp.array(map_def.occupancy_grid, dtype=jnp.bool_)
            if map_def.padded_occupancy_grid is not None:
                padded_grid = map_def.padded_occupancy_grid

        else:
            raise FileNotFoundError(f"Map file not found: {map_path}")

    # Final Grid Dimensions (Snapped to cell_size)
    if W is None or H is None:
        raise ValueError("Environment dimensions (box_width/height) are missing. Map loading failed or dimensions not in config.")

    W, H = float(W), float(H)
    GW, GH = int(W / cell_size), int(H / cell_size)
    if occ_grid is None:
        raise ValueError("Occupancy grid missing. A valid map MUST be loaded for physics.")
    comm_has_inner_obstacles = has_inner_obstacles(occ_grid)

    # Padding size in cells
    R_cells = int(PAD_RADIUS / cell_size) + 2
    if padded_grid is None:
        raise ValueError("Padded occupancy grid missing. Map loading or padding logic failed.")

    initial_relay_state, update_relay_state = make_relay_task_fns(
        cfg,
        map_def,
        N,
        W,
        H,
        n_reach,
    )

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

    _drone_indices = jnp.arange(N, dtype=jnp.int32)
    _eye = jnp.eye(N, dtype=jnp.float32)

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
    # _compute_connectivity  (single graph authority)
    # ------------------------------------------------------------------

    def _compute_connectivity(
        state: EnvState,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """
        Compute direct edges and transient reachability for one state.

        Returns
        -------
        is_conn_base          : (N,) bool
        is_conn_target        : (N,) bool
        adj_matrix            : (N+1, N+1) bool
        directly_sees_target  : (N,) bool
        reachability          : (N+1, N+1) bool, temporary only
        """
        physics = state.physics

        # 1. Comm graph logic using DDA Raycasting, unless the communication
        # grid only contains outer-border walls.
        def _check_comm(i, j):
            dist = jnp.linalg.norm(physics.pos[i] - physics.pos[j])
            in_range = (dist <= comm_r) & physics.active[i] & physics.active[j] & (i != j)
            if comm_has_inner_obstacles:
                can_see = dda_raycast(physics.pos[i]/cell_size, physics.pos[j]/cell_size, occ_grid)
            else:
                can_see = True
            return in_range & can_see

        adj_dd = jax.vmap(jax.vmap(_check_comm, (None, 0)), (0, None))(
            jnp.arange(N), jnp.arange(N)
        ).astype(jnp.float32)

        # Base connectivity
        def _check_base_comm(i):
            dist = jnp.linalg.norm(physics.pos[i] - physics.base_pos)
            in_range = (dist <= comm_r_base) & physics.active[i]
            if comm_has_inner_obstacles:
                can_see = dda_raycast(physics.pos[i]/cell_size, physics.base_pos/cell_size, occ_grid)
            else:
                can_see = True
            return in_range & can_see

        adj_db = jax.vmap(_check_base_comm)(jnp.arange(N)).astype(jnp.float32)

        target_pos_agents = jnp.tile(physics.target_pos[None, :], (N, 1))

        # 2. Drone-to-target LoS
        def _check_target_los(i):
            target_i = target_pos_agents[i]
            dist = jnp.linalg.norm(physics.pos[i] - target_i)
            in_range = (dist <= vis_r) & physics.active[i]
            can_see = dda_raycast(physics.pos[i]/cell_size, target_i/cell_size, occ_grid)
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

        reachability = R > 0.5
        is_conn_base = reachability[:N, N]
        is_conn_target = jnp.any(
            reachability[:N, :N] & directly_sees[None, :],
            axis=-1,
        )
        adj_matrix = jnp.concatenate([A_top, A_bot], axis=0) > 0.5
        return (
            is_conn_base,
            is_conn_target,
            adj_matrix,
            directly_sees,
            reachability,
        )

    def _propagate_target_knowledge(
        state: EnvState,
        directly_sees: jax.Array,
        reachability: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Propagate persistent knowledge through the current physics graph."""
        communication = state.communication
        knows_before = jnp.concatenate(
            [
                communication.target_known | directly_sees,
                jnp.array(
                    [communication.base_target_known],
                    dtype=jnp.bool_,
                ),
            ],
            axis=0,
        )
        knows_after = jnp.any(
            reachability & knows_before[None, :],
            axis=-1,
        )
        return knows_after[:N], knows_after[N]


    def reset(key: jax.Array) -> EnvState:
        key, k1, k2, k3 = jax.random.split(key, 4)

        if map_def is not None:
            # ── Base position ──────────────────────────────────────────────
            base_pos = map_def.sample_base(k1)

            # ── Target position ────────────────────────────────────────────
            target_pos = map_def.sample_target(k2)

            # ── Drone positions ────────────────────────────────────────────
            pos = map_def.sample_drones(k3, N)
        else:
            raise ValueError("map_def is None. Map MUST be loaded to sample reset positions.")

        if base_pos is None or target_pos is None:
            raise ValueError("Reset failed: base_pos or target_pos could not be resolved from map.")

        vel           = jnp.zeros((N, 2), dtype=jnp.float32)
        coverage_grid = jnp.zeros((GW, GH), dtype=jnp.bool_)

        if spawn_delay > 0:
            active = (_drone_indices == 0)
        else:
            active = jnp.ones(N, dtype=jnp.bool_)

        target_known = jnp.zeros(N, dtype=jnp.bool_)
        collides     = jnp.zeros(N, dtype=jnp.bool_)

        initial_state = EnvState(
            physics=PhysicsState(
                pos=pos,
                vel=vel,
                base_pos=base_pos,
                target_pos=target_pos,
                step=jnp.int32(0),
                key=key,
                active=active,
                box_width=jnp.float32(W),
                box_height=jnp.float32(H),
            ),
            communication=CommunicationState(
                target_known=target_known,
                base_target_known=jnp.bool_(False),
                is_conn_base=jnp.zeros(N, dtype=jnp.bool_),
                is_conn_target=jnp.zeros(N, dtype=jnp.bool_),
                directly_sees_target=jnp.zeros(N, dtype=jnp.bool_),
                adj_matrix=jnp.zeros((N + 1, N + 1), dtype=jnp.bool_),
            ),
            exploration=ExplorationState(coverage_grid=coverage_grid),
            relay=initial_relay_state(),
            diagnostics=StepDiagnostics(
                collides=collides,
                last_cov_delta=jnp.zeros(N, dtype=jnp.int32),
            ),
        )
        (
            is_conn_base,
            is_conn_target,
            adj_matrix,
            directly_sees_target,
            _,
        ) = _compute_connectivity(initial_state)
        return initial_state.replace(
            communication=initial_state.communication.replace(
                is_conn_base=is_conn_base,
                is_conn_target=is_conn_target,
                directly_sees_target=directly_sees_target,
                adj_matrix=adj_matrix,
            ),
        )

    # ------------------------------------------------------------------
    # _physics_step  (internal — pure Newtonian integration)
    # ------------------------------------------------------------------

    def _physics_step(
        state:   EnvState,
        actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        physics = state.physics
        a_norm    = jnp.linalg.norm(actions, axis=-1, keepdims=True)
        safe_norm = jnp.where(a_norm > 0, a_norm, 1.0)
        actions   = jnp.where(a_norm > f_max, actions / safe_norm * f_max, actions)

        vel = physics.vel * drag + actions * dt

        speed      = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        safe_speed = jnp.where(speed > 0, speed, 1.0)
        vel        = jnp.where(speed > v_max, vel / safe_speed * v_max, vel)

        # Position integration
        next_pos = physics.pos + vel * dt

        # 1. External Box Collisions
        hit_x_lo = next_pos[:, 0] < 0.0
        hit_x_hi = next_pos[:, 0] > physics.box_width
        hit_y_lo = next_pos[:, 1] < 0.0
        hit_y_hi = next_pos[:, 1] > physics.box_height

        # 2. Internal Grid Collisions (Sub-stepping CCD)
        # Check 8 intermediate points along the path from the old to next position.
        # This is more robust than DDA for 1m thick walls at drone speeds.
        t_sub = jnp.linspace(0.125, 1.0, 8) # (8,)
        # intermediate positions: (N, 8, 2)
        p_sub = physics.pos[:, None, :] + t_sub[None, :, None] * (next_pos - physics.pos)[:, None, :]

        # Grid indices for substeps: (N, 8, 2)
        g_idx_sub = jnp.floor(p_sub / cell_size).astype(jnp.int32)

        # In-bounds check using JAX-safe types
        GW_active = (physics.box_width / cell_size).astype(jnp.int32)
        GH_active = (physics.box_height / cell_size).astype(jnp.int32)
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

        pos = jnp.where(collided[:, None], physics.pos, next_pos)
        pos = jnp.stack([
            jnp.clip(pos[:, 0], 0.0, physics.box_width),
            jnp.clip(pos[:, 1], 0.0, physics.box_height),
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
        6.  Advance PRNG key and step counter
        7.  Compute communication and persistent target knowledge
        8.  Let the relay-task layer update its task-specific state
        """
        chex.assert_shape(actions, (N, 2))
        physics = state.physics

        # 1. Spawn activation: drone i wakes up at step == i * spawn_delay
        if spawn_delay > 0:
            should_activate = (_drone_indices * spawn_delay) == physics.step   # (N,) bool
            new_active      = physics.active | should_activate
        else:
            new_active = jnp.ones(N, dtype=jnp.bool_)

        # 2. Mask actions for inactive drones
        actions_masked = jnp.where(new_active[:, None], actions, 0.0)

        # 3. Physics step
        new_pos, new_vel, collided = _physics_step(state, actions_masked)

        # 4. Inactive enforcement
        # If not active, pos = base_pos, vel = 0, collided = False
        new_pos = jnp.where(new_active[:, None], new_pos, physics.base_pos[None, :])
        new_vel = jnp.where(new_active[:, None], new_vel, 0.0)
        collided = collided & new_active

        # 5. Coverage (active drones only)
        new_coverage, new_cov_deltas = update_coverage(
            state.exploration.coverage_grid,
            new_pos,
            new_active,
        )

        # 6. Advance PRNG key
        new_key, _ = jax.random.split(physics.key)

        # Build intermediate state with updated positions (for target_known calc)
        mid_state = state.replace(
            physics=physics.replace(
                pos=new_pos,
                vel=new_vel,
                step=physics.step + jnp.int32(1),
                key=new_key,
                active=new_active,
            ),
            exploration=state.exploration.replace(
                coverage_grid=new_coverage,
            ),
            diagnostics=state.diagnostics.replace(
                collides=collided,
                last_cov_delta=new_cov_deltas,
            ),
        )

        # 7. Compute connectivity once, then propagate persistent target
        # knowledge from the same transient reachability result.
        (
            is_conn_base,
            is_conn_target,
            new_adj_matrix,
            directly_sees,
            reachability,
        ) = _compute_connectivity(mid_state)
        new_target_known, new_base_target_known = (
            _propagate_target_knowledge(
                mid_state,
                directly_sees,
                reachability,
            )
        )
        path_mid_state = mid_state.replace(
            communication=mid_state.communication.replace(
                is_conn_base=is_conn_base,
                is_conn_target=is_conn_target,
                directly_sees_target=directly_sees,
                adj_matrix=new_adj_matrix,
            ),
        )
        # The relay task consumes physical and communication results, but owns
        # its maze-cell/path transition logic and state.
        new_relay_state = update_relay_state(
            state,
            path_mid_state,
            directly_sees,
            new_target_known,
            new_base_target_known,
            is_conn_base,
        )

        return mid_state.replace(
            communication=mid_state.communication.replace(
                target_known=new_target_known,
                base_target_known=new_base_target_known,
                is_conn_base=is_conn_base,
                is_conn_target=is_conn_target,
                directly_sees_target=directly_sees,
                adj_matrix=new_adj_matrix,
            ),
            relay=new_relay_state,
        )

    return env_step, reset, update_coverage, (W, H, occ_grid)


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from swarmecho.core.config import load_config, validate_config
    from swarmecho.env.state import assert_env_state

    cfg = load_config(cli_overrides=False)
    validate_config(cfg)

    env_step, reset, update_coverage, (W, H, occ_grid) = make_env_fns(cfg)

    reset_jit = jax.jit(reset)
    step_jit  = jax.jit(env_step)

    key   = jax.random.PRNGKey(0)
    state = reset_jit(key)

    GW, GH = occ_grid.shape
    N   = int(cfg.env.num_agents)

    assert_env_state(state, N, GW, GH)
    print(
        f"Reset OK  | pos shape: {state.physics.pos.shape} "
        f"| step: {state.physics.step}"
    )
    print(
        f"          | active: {state.physics.active} "
        f"| target_known: {state.communication.target_known}"
    )

    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (N, 2),
        minval=-cfg.env.max_force, maxval=cfg.env.max_force,
    )
    state2 = step_jit(state, actions)
    assert_env_state(state2, N, GW, GH)
    print(
        f"Step  OK  | pos[0]: {state2.physics.pos[0]} "
        f"| step: {state2.physics.step}"
    )
    print(f"          | active: {state2.physics.active}")
    print(
        "Coverage  | cells covered: "
        f"{state2.exploration.coverage_grid.sum()}"
    )
    print("\nPhysics engine self-test passed ✓")
