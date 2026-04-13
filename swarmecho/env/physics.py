"""
swarmecho/env/physics.py
========================
Pure JAX physics engine: reset, physics_step, coverage update.

Factory pattern
---------------
All public functions are produced by `make_env_fns(cfg)`. This closes over
config scalars so they become XLA compile-time constants — no OmegaConf
objects ever enter a jit-compiled function.

The returned callables are PURE (no internal @jax.jit). Callers apply jit:

    physics_step, reset, update_coverage = make_env_fns(cfg)

    # For testing one step at a time:
    step_jit = jax.jit(physics_step)

    # For vectorised training (jit wraps vmap wraps the pure function):
    vmapped_step = jax.jit(jax.vmap(physics_step))

    # Inside lax.scan for rollout collection (no manual jit needed):
    states, _ = jax.lax.scan(lambda s, a: (physics_step(s, a), s), init, actions)
"""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from swarmecho.env.state import EnvState


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env_fns(cfg: DictConfig):
    """
    Build pure physics functions closed over config scalars.

    Parameters
    ----------
    cfg : OmegaConf DictConfig loaded from configs/default.yaml

    Returns
    -------
    physics_step     : (EnvState, actions) -> EnvState
    reset            : (PRNGKey) -> EnvState
    update_coverage  : (coverage_grid, pos) -> coverage_grid
        Exposed separately so rewards.py can import and reuse it.
    """

    # --- Extract config as Python scalars (XLA compile-time constants) ----
    N      = int(cfg.env.num_agents)
    G      = int(cfg.env.grid_resolution)
    dt     = float(cfg.env.dt)
    drag   = float(cfg.env.drag)
    v_max  = float(cfg.env.max_speed)
    f_max  = float(cfg.env.max_force)
    W      = float(cfg.env.box_width)
    H      = float(cfg.env.box_height)
    vis_r  = float(cfg.env.visual_radius)
    cell_w = W / G
    cell_h = H / G

    # Fixed world geometry — created once as constants
    BASE_POS   = jnp.array([cfg.env.base_x,   cfg.env.base_y],   dtype=jnp.float32)
    TARGET_POS = jnp.array([cfg.env.target_x, cfg.env.target_y], dtype=jnp.float32)

    # Pre-compute grid cell centres once — reused every step in coverage update
    # Shape: (G, G, 2)
    xs = (jnp.arange(G, dtype=jnp.float32) + 0.5) * cell_w   # (G,)
    ys = (jnp.arange(G, dtype=jnp.float32) + 0.5) * cell_h   # (G,)
    # meshgrid with indexing="ij": grid[i, j] = (xs[i], ys[j])
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    CELL_CENTRES = jnp.stack([gx, gy], axis=-1)               # (G, G, 2)

    # ------------------------------------------------------------------
    # update_coverage
    # ------------------------------------------------------------------

    def update_coverage(
        coverage_grid: jax.Array,   # (G, G) bool
        pos: jax.Array,             # (N, 2) float32
    ) -> jax.Array:                 # (G, G) bool
        """
        OR the existing coverage_grid with any cells newly visited.

        A cell (i, j) is newly visited if any agent is within visual_radius
        of its centre. Fully vectorised — no Python loops.

        Complexity: O(G² × N) arithmetic ops on GPU — ~20k for G=50, N=8.
        """
        # CELL_CENTRES: (G, G, 2)
        # pos:          (N, 2)  → broadcast to (G, G, N, 2)
        delta = CELL_CENTRES[:, :, None, :] - pos[None, None, :, :]  # (G, G, N, 2)
        dists = jnp.linalg.norm(delta, axis=-1)                       # (G, G, N)
        newly_seen = jnp.any(dists <= vis_r, axis=-1)                 # (G, G)
        return coverage_grid | newly_seen

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(key: jax.Array) -> EnvState:
        """
        Initialise a fresh episode.

        Agents start at uniformly random positions within the bounding box,
        with zero initial velocity. Coverage grid is empty.
        """
        key, subkey = jax.random.split(key)
        pos = jax.random.uniform(
            subkey, shape=(N, 2), dtype=jnp.float32,
            minval=0.0,
            maxval=jnp.array([W, H], dtype=jnp.float32),
        )
        vel           = jnp.zeros((N, 2), dtype=jnp.float32)
        coverage_grid = jnp.zeros((G, G), dtype=jnp.bool_)

        return EnvState(
            pos=pos,
            vel=vel,
            base_pos=BASE_POS,
            target_pos=TARGET_POS,
            coverage_grid=coverage_grid,
            step=jnp.int32(0),
            key=key,
        )

    # ------------------------------------------------------------------
    # physics_step
    # ------------------------------------------------------------------

    def physics_step(
        state:   EnvState,
        actions: jax.Array,   # (N, 2) continuous forces
    ) -> EnvState:
        """
        One physics timestep. Returns a new EnvState.

        Pipeline
        --------
        1. Clip actions to max force magnitude (norm clipping, not per-axis)
        2. Euler velocity integration with multiplicative drag
        3. Clip velocity to max speed (norm clipping)
        4. Euler position integration
        5. Elastic wall bounce — invert the offending velocity component,
           clamp position to [0, W] × [0, H]
        6. Update coverage grid
        7. Advance PRNG key
        """
        chex.assert_shape(actions, (N, 2))

        # 1. Force magnitude clipping (preserves direction)
        a_norm   = jnp.linalg.norm(actions, axis=-1, keepdims=True)  # (N, 1)
        safe_norm = jnp.where(a_norm > 0, a_norm, 1.0)               # avoid /0
        actions  = jnp.where(a_norm > f_max, actions / safe_norm * f_max, actions)

        # 2. Velocity update
        vel = state.vel * drag + actions * dt

        # 3. Speed clipping (preserves direction)
        speed     = jnp.linalg.norm(vel, axis=-1, keepdims=True)     # (N, 1)
        safe_speed = jnp.where(speed > 0, speed, 1.0)
        vel       = jnp.where(speed > v_max, vel / safe_speed * v_max, vel)

        # 4. Position integration
        pos = state.pos + vel * dt

        # 5. Elastic wall bounce
        #    X-axis walls (x=0 and x=W)
        hit_x_lo = pos[:, 0] < 0.0
        hit_x_hi = pos[:, 0] > W
        vx = jnp.where(hit_x_lo | hit_x_hi, -vel[:, 0], vel[:, 0])

        #    Y-axis walls (y=0 and y=H)
        hit_y_lo = pos[:, 1] < 0.0
        hit_y_hi = pos[:, 1] > H
        vy = jnp.where(hit_y_lo | hit_y_hi, -vel[:, 1], vel[:, 1])

        vel = jnp.stack([vx, vy], axis=-1)

        # Clamp position to bounding box (handles edge case of large dt)
        pos = jnp.stack([
            jnp.clip(pos[:, 0], 0.0, W),
            jnp.clip(pos[:, 1], 0.0, H),
        ], axis=-1)

        # 6. Coverage grid update
        new_coverage = update_coverage(state.coverage_grid, pos)

        # 7. Advance PRNG key (deterministic split — needed for any stochastic
        #    extensions, e.g. observation noise, spawning events)
        new_key, _ = jax.random.split(state.key)

        return dataclasses.replace(
            state,
            pos           = pos,
            vel           = vel,
            coverage_grid = new_coverage,
            step          = state.step + jnp.int32(1),
            key           = new_key,
        )

    return physics_step, reset, update_coverage


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    from swarmecho.config import load_config, validate_config
    from swarmecho.env.state import assert_env_state

    cfg = load_config(cli_overrides=False)
    validate_config(cfg)

    physics_step, reset, update_coverage = make_env_fns(cfg)

    # JIT-compile for the test
    reset_jit = jax.jit(reset)
    step_jit  = jax.jit(physics_step)

    key   = jax.random.PRNGKey(0)
    state = reset_jit(key)

    N = cfg.env.num_agents
    G = cfg.env.grid_resolution

    assert_env_state(state, N, G)
    print(f"Reset OK  | pos shape: {state.pos.shape} | step: {state.step}")

    # One step with random actions
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (N, 2),
        minval=-cfg.env.max_force, maxval=cfg.env.max_force,
    )
    state2 = step_jit(state, actions)
    assert_env_state(state2, N, G)
    print(f"Step  OK  | pos[0]: {state2.pos[0]} | step: {state2.step}")
    print(f"Coverage  | cells covered: {state2.coverage_grid.sum()} / {G*G}")
    print("\nPhysics engine self-test passed ✓")
