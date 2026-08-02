"""Privileged state features for the optional centralized critic."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from swarmecho.core.config import MAP_DIR
from swarmecho.env.state import EnvState


MAX_MAZE_CELL_COLS = 11
MAX_MAZE_CELL_ROWS = 11
MAX_MAZE_CELLS = MAX_MAZE_CELL_COLS * MAX_MAZE_CELL_ROWS


def privileged_critic_dim(num_agents: int, radar_bins: int = 8) -> int:
    """Feature width emitted for each agent token."""
    # geometry(9), task/communication(8), adjacency(N+1), diagnostics(2),
    # free-space-normalized logical-cell coverage(11*11), actor radar(4B)
    return 141 + int(num_agents) + 4 * int(radar_bins)


def _logical_grid_shape(cfg, width: float, height: float) -> tuple[int, int]:
    """Read the active map's declared logical grid, with a 10 m fallback."""
    map_names = cfg.env.get("map_names", [])
    if map_names:
        path = Path(MAP_DIR) / f"{map_names[0]}.yaml"
        with path.open() as stream:
            logical_grid = (yaml.safe_load(stream) or {}).get("maze_cell_grid")
        if logical_grid:
            return int(logical_grid["cols"]), int(logical_grid["rows"])
    return max(1, round(width / 10.0)), max(1, round(height / 10.0))


def make_privileged_critic_state_fn(cfg, width: float, height: float, occupancy_grid):
    """Build a JIT-friendly privileged encoder for one active map.

    Radar is taken verbatim from the tail of each actor observation. Coverage
    is pooled over the logical maze cells declared by the map and normalized
    by the number of non-wall physics cells in each logical cell.
    """
    n = int(cfg.env.num_agents)
    radar_bins = int(cfg.env.radar_bins)
    radar_width = 4 * radar_bins
    max_steps = max(int(cfg.env.max_steps), 1)
    hold_steps = max(int(cfg.env.get("hold_chain_for", 0)), 1)
    max_speed = max(float(cfg.env.max_speed), 1e-6)
    occ = jnp.asarray(occupancy_grid, dtype=jnp.bool_)
    gw, gh = occ.shape
    logical_cols, logical_rows = _logical_grid_shape(cfg, width, height)
    if logical_cols > MAX_MAZE_CELL_COLS or logical_rows > MAX_MAZE_CELL_ROWS:
        raise ValueError(
            f"Logical maze grid {logical_cols}x{logical_rows} exceeds privileged "
            f"critic capacity {MAX_MAZE_CELL_COLS}x{MAX_MAZE_CELL_ROWS}."
        )

    # Assign every 1 m physics cell to its declared logical maze cell once.
    logical_x = jnp.minimum(jnp.arange(gw) * logical_cols // gw, logical_cols - 1)
    logical_y = jnp.minimum(jnp.arange(gh) * logical_rows // gh, logical_rows - 1)
    logical_index = logical_x[:, None] * logical_rows + logical_y[None, :]
    logical_one_hot = jax.nn.one_hot(
        logical_index.reshape(-1), MAX_MAZE_CELLS, dtype=jnp.float32
    )
    free = (~occ).reshape(-1).astype(jnp.float32)
    free_cells_per_logical_cell = jnp.maximum(logical_one_hot.T @ free, 1.0)
    world_scale = jnp.array([max(float(width), 1e-6), max(float(height), 1e-6)])

    def compute(state: EnvState, actor_obs: jax.Array) -> jax.Array:
        physics = state.physics
        communication = state.communication
        coverage = state.exploration.coverage_grid.reshape(-1).astype(jnp.float32)

        positions = physics.pos / world_scale
        base = jnp.broadcast_to(physics.base_pos / world_scale, (n, 2))
        target = jnp.broadcast_to(physics.target_pos / world_scale, (n, 2))
        step = jnp.full((n, 1), physics.step.astype(jnp.float32) / max_steps)
        geometry = jnp.concatenate([
            positions, physics.vel / max_speed, base - positions,
            target - positions, step,
        ], axis=-1)

        task = jnp.stack([
            physics.active,
            communication.target_known,
            jnp.broadcast_to(communication.base_target_known, (n,)),
            communication.is_conn_base,
            communication.is_conn_target,
            communication.directly_sees_target,
            jnp.broadcast_to(state.relay.chain_held_steps / hold_steps, (n,)),
            jnp.broadcast_to(
                jnp.any(communication.is_conn_base & communication.is_conn_target), (n,)
            ),
        ], axis=-1).astype(jnp.float32)

        adjacency = communication.adj_matrix[:n, :n + 1].astype(jnp.float32)
        adjacency = jnp.where(
            communication.adj_matrix.size > 0,
            adjacency,
            jnp.zeros((n, n + 1), dtype=jnp.float32),
        )
        diagnostics = jnp.stack([
            state.diagnostics.collides.astype(jnp.float32),
            state.diagnostics.last_cov_delta.astype(jnp.float32),
        ], axis=-1)

        covered_free = coverage * free
        coverage_by_cell = (
            logical_one_hot.T @ covered_free / free_cells_per_logical_cell
        )
        coverage_by_cell = jnp.broadcast_to(coverage_by_cell, (n, MAX_MAZE_CELLS))
        radar = actor_obs[:, -radar_width:]

        result = jnp.concatenate([
            geometry, task, adjacency, diagnostics, coverage_by_cell, radar,
        ], axis=-1)
        assert result.shape == (n, privileged_critic_dim(n, radar_bins))
        return result

    return compute, privileged_critic_dim(n, radar_bins)
