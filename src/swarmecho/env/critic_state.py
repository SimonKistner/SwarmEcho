"""Privileged dynamic tokens and semantic-map inputs for the critic."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from swarmecho.env.state import EnvState


def privileged_critic_dim(num_agents: int) -> int:
    """Width of one exact per-agent dynamic token."""
    # geometry(9), task/communication(8), adjacency(N+1), diagnostics(2)
    return 20 + int(num_agents)


def _resize_binary(grid: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Resize a semantic grid without blending channel meanings."""
    return jax.image.resize(
        grid.astype(jnp.float32), shape, method="nearest"
    )


def _resize_walls(grid: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Conservatively retain thin walls when reducing map resolution."""
    if tuple(grid.shape) == shape:
        return grid.astype(jnp.float32)
    occupancy_fraction = jax.image.resize(
        grid.astype(jnp.float32), shape, method="linear", antialias=True
    )
    return (occupancy_fraction > 0.01).astype(jnp.float32)


def make_privileged_critic_state_fn(cfg, width: float, height: float, occupancy_grid):
    """Preprocess the static map and build compact dynamic critic inputs.

    Full-map semantics are retained at a bounded training resolution. Rollouts
    store only the changing coverage plane; wall geometry is preprocessed once,
    while agent/base/target planes are reconstructed cheaply from exact tokens.
    """
    n = int(cfg.env.num_agents)
    max_steps = max(int(cfg.env.max_steps), 1)
    hold_steps = max(int(cfg.env.get("hold_chain_for", 0)), 1)
    max_speed = max(float(cfg.env.max_speed), 1e-6)
    max_resolution = int(cfg.network.get("critic_map_resolution", 64))
    occupancy = jnp.asarray(occupancy_grid, dtype=jnp.float32)
    map_shape = (
        min(int(occupancy.shape[0]), max_resolution),
        min(int(occupancy.shape[1]), max_resolution),
    )
    wall_map = _resize_walls(occupancy, map_shape)
    world_scale = jnp.array([max(float(width), 1e-6), max(float(height), 1e-6)])

    def compute(state: EnvState) -> tuple[jax.Array, jax.Array]:
        physics = state.physics
        communication = state.communication
        positions = physics.pos / world_scale
        base = jnp.broadcast_to(physics.base_pos / world_scale, (n, 2))
        target = jnp.broadcast_to(physics.target_pos / world_scale, (n, 2))

        geometry = jnp.concatenate([
            positions,
            physics.vel / max_speed,
            base - positions,
            target - positions,
            jnp.full((n, 1), physics.step.astype(jnp.float32) / max_steps),
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
                jnp.any(communication.is_conn_base & communication.is_conn_target),
                (n,),
            ),
        ], axis=-1).astype(jnp.float32)
        adjacency = communication.adj_matrix[:n, :n + 1].astype(jnp.float32)
        diagnostics = jnp.stack([
            state.diagnostics.collides.astype(jnp.float32),
            state.diagnostics.last_cov_delta.astype(jnp.float32),
        ], axis=-1)
        tokens = jnp.concatenate(
            [geometry, task, adjacency, diagnostics], axis=-1
        )
        coverage_map = _resize_binary(
            state.exploration.coverage_grid, map_shape
        )
        assert tokens.shape == (n, privileged_critic_dim(n))
        return tokens, coverage_map

    return compute, privileged_critic_dim(n), wall_map
