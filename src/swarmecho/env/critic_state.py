"""Compact privileged state features for the optional centralized critic."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from swarmecho.env.state import EnvState


def privileged_critic_dim(num_agents: int) -> int:
    """Feature width emitted for each agent token."""
    # geometry(9), task/communication(8), adjacency(N+1), diagnostics(2),
    # coverage(global + four fixed map quadrants), local occupancy(9)
    return 34 + int(num_agents)


def make_privileged_critic_state_fn(cfg, width: float, height: float, occupancy_grid):
    """Build a JIT-friendly state encoder, preprocessing the one static map once.

    The returned tensor is still a set of per-agent tokens.  Exact simulator
    state and graph rows replace lossy radar reconstruction, while small shared
    summaries avoid copying the full coverage/map raster into every rollout.
    """
    n = int(cfg.env.num_agents)
    max_steps = max(int(cfg.env.max_steps), 1)
    hold_steps = max(int(cfg.env.get("hold_chain_for", 0)), 1)
    max_speed = max(float(cfg.env.max_speed), 1e-6)
    cell_size = float(cfg.env.get("cell_size", 1.0))
    occ = jnp.asarray(occupancy_grid, dtype=jnp.float32)
    gw, gh = occ.shape

    # Static map preprocessing: fixed quadrant masks and cell offsets are
    # created once, outside rollouts/JIT traces.
    gx = jnp.arange(gw)[:, None]
    gy = jnp.arange(gh)[None, :]
    quadrant_masks = jnp.stack([
        (gx < (gw + 1) // 2) & (gy < (gh + 1) // 2),
        (gx >= gw // 2) & (gy < (gh + 1) // 2),
        (gx < (gw + 1) // 2) & (gy >= gh // 2),
        (gx >= gw // 2) & (gy >= gh // 2),
    ]).astype(jnp.float32)
    quadrant_sizes = jnp.maximum(quadrant_masks.sum(axis=(1, 2)), 1.0)
    local_offsets = jnp.stack(jnp.meshgrid(
        jnp.arange(-1, 2), jnp.arange(-1, 2), indexing="ij"
    ), axis=-1).reshape(9, 2)
    world_scale = jnp.array([max(float(width), 1e-6), max(float(height), 1e-6)])

    def compute(state: EnvState) -> jax.Array:
        physics = state.physics
        communication = state.communication
        coverage = state.exploration.coverage_grid.astype(jnp.float32)

        positions = physics.pos / world_scale
        base = jnp.broadcast_to(physics.base_pos / world_scale, (n, 2))
        target = jnp.broadcast_to(physics.target_pos / world_scale, (n, 2))
        step = jnp.full((n, 1), physics.step.astype(jnp.float32) / max_steps)
        geometry = jnp.concatenate([
            positions,
            physics.vel / max_speed,
            base - positions,
            target - positions,
            step,
        ], axis=-1)

        task = jnp.stack([
            physics.active,
            communication.target_known,
            jnp.broadcast_to(communication.base_target_known, (n,)),
            communication.is_conn_base,
            communication.is_conn_target,
            communication.directly_sees_target,
            jnp.broadcast_to(state.relay.chain_held_steps / hold_steps, (n,)),
            jnp.broadcast_to(jnp.any(communication.is_conn_base & communication.is_conn_target), (n,)),
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

        coverage_summary = jnp.concatenate([
            jnp.mean(coverage)[None],
            (quadrant_masks * coverage[None]).sum(axis=(1, 2)) / quadrant_sizes,
        ])
        coverage_summary = jnp.broadcast_to(coverage_summary, (n, 5))

        cells = jnp.floor(physics.pos / cell_size).astype(jnp.int32)
        sample_cells = cells[:, None, :] + local_offsets[None, :, :]
        sx = jnp.clip(sample_cells[..., 0], 0, gw - 1)
        sy = jnp.clip(sample_cells[..., 1], 0, gh - 1)
        in_bounds = (
            (sample_cells[..., 0] >= 0) & (sample_cells[..., 0] < gw)
            & (sample_cells[..., 1] >= 0) & (sample_cells[..., 1] < gh)
        )
        local_occupancy = jnp.where(in_bounds, occ[sx, sy], 1.0)

        result = jnp.concatenate([
            geometry, task, adjacency, diagnostics, coverage_summary,
            local_occupancy,
        ], axis=-1)
        assert result.shape == (n, privileged_critic_dim(n))
        return result

    return compute, privileged_critic_dim(n)
