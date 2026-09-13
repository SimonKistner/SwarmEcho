"""Compact, pre-action privileged inputs for the 3D critic only."""
from __future__ import annotations

import jax.numpy as jnp

from swarmecho.env.baseline3d import coverage_grid_geometry, observation_dim_3d


def privileged_global_dim_3d(cfg):
    # Base xyz, target xyz, holding progress, base knowledge, optional clock.
    return 8 + int(not cfg.observe_current_timestep)


def privileged_input_dim_3d(cfg):
    return (observation_dim_3d(cfg) + 9 + 3
            + 3 * int(not cfg.observe_base_vector)
            + privileged_global_dim_3d(cfg) - 6)


def make_privileged_features_3d(building, cfg):
    """Build a pure device-side collector; no rays, paths, or host callbacks.

    Agent packet: normalized position xyz and six adjacent coverage cells.
    Shared packet: normalized base/target xyz and task counters/flags. Shared
    data is stored once per environment, never once per agent. Boundary probes
    clamp to the grid, matching the existing actor coverage-probe convention.
    """
    voxel_size, grid_shape = coverage_grid_geometry(building, cfg)
    scale = max(float(max(building.world_size_m)), 1e-6)
    offsets = jnp.asarray(((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)), dtype=jnp.int32)
    upper = jnp.asarray(grid_shape, dtype=jnp.int32) - 1

    def collect(state):
        cells = jnp.floor(state.pos / voxel_size).astype(jnp.int32)
        probes = jnp.clip(cells[:, None, :] + offsets, 0, upper)
        coverage = state.coverage[probes[..., 0], probes[..., 1], probes[..., 2]]
        agent = jnp.concatenate((state.pos / scale, coverage.astype(jnp.float32)), axis=-1)
        agent = jnp.where(state.active[:, None], agent, 0.0)
        shared = [state.base_pos / scale, state.target_pos / scale,
                  jnp.asarray([state.chain_held_steps / max(cfg.hold_chain_for, 1),
                               state.base_target_known], dtype=jnp.float32)]
        if not cfg.observe_current_timestep:
            shared.append(jnp.asarray([state.step / cfg.max_steps], dtype=jnp.float32))
        return agent, jnp.concatenate(shared)

    return collect


def assemble_privileged_observations_3d(obs, agent, shared, active, *, include_base_vector):
    """Assemble per-agent tokens before encoding/GRU/attention, for any batch axes."""
    position = agent[..., :3]
    features = [obs, agent]
    if include_base_vector:
        features.append(shared[..., None, :3] - position)
    # Always unmasked: an actor target vector, when enabled, hides unknown targets.
    features.append(shared[..., None, 3:6] - position)
    features.append(jnp.broadcast_to(shared[..., None, 6:],
                                    (*obs.shape[:-1], shared.shape[-1] - 6)))
    return jnp.where(active[..., None], jnp.concatenate(features, axis=-1), 0.0)
