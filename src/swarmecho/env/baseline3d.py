"""Pure-JAX minimum 3D cuboid environment.

The module is intentionally self-contained while the legacy 2D environment is
still present.  It is the executable migration seam: 3D state, action,
coverage, communication, target discovery, spherical radar, and collision can
be validated before replacing the training entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.env.buildings import BuildingArrays


class Baseline3DState(NamedTuple):
    pos: jax.Array
    vel: jax.Array
    base_pos: jax.Array
    target_pos: jax.Array
    active: jax.Array
    coverage: jax.Array
    step: jax.Array
    key: jax.Array
    directly_sees_target: jax.Array
    is_conn_base: jax.Array
    is_conn_target: jax.Array
    target_known: jax.Array
    success: jax.Array


@dataclass(frozen=True)
class Baseline3DConfig:
    num_agents: int = 5
    dt: float = 0.1
    max_force: float = 20.0
    max_speed: float = 5.0
    drag: float = 0.9
    drone_radius: float = 0.25
    base_comm_radius: float = 6.0
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    target_spawn_buffer: float = 0.5
    radar_bins: int = 8
    spawn_delay: int = 0


def spherical_directions(count: int) -> np.ndarray:
    """Return deterministic unit directions, with exact octants for count 8."""
    if count < 4:
        raise ValueError("radar_bins must be at least 4.")
    if count == 8:
        directions = np.asarray(list(product((-1.0, 1.0), repeat=3)), dtype=np.float32)
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    indices = np.arange(count, dtype=np.float32)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = indices * golden_angle
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1).astype(
        np.float32
    )


def minimum_target_distance(cfg: Baseline3DConfig) -> float:
    """Distance that prevents an immediate base/first-drone discovery."""
    return cfg.base_comm_radius + 0.5 * cfg.comm_radius + cfg.target_spawn_buffer


def maximum_five_drone_chain_distance(cfg: Baseline3DConfig) -> float:
    """Unobstructed base→D1→…→D5→target reach for five mobile drones."""
    return cfg.base_comm_radius + 4.0 * cfg.comm_radius + cfg.visual_radius


def _target_candidates(building: BuildingArrays, cfg: Baseline3DConfig) -> jax.Array:
    dims = building.target_exclusion.shape
    cells = np.stack(np.meshgrid(*[np.arange(size) for size in dims], indexing="ij"), axis=-1)
    cells = cells.reshape(-1, 3)
    positions = (cells.astype(np.float32) + 0.5) * building.cell_size_m
    distances = np.linalg.norm(positions - building.base_position_m[None, :], axis=-1)
    excluded = building.target_exclusion[tuple(cells.T)]
    valid = ~excluded & (distances > minimum_target_distance(cfg))
    if not np.any(valid):
        raise ValueError(
            "Building has no target cell beyond base_comm_radius + "
            "0.5 * comm_radius + target_spawn_buffer."
        )
    return jnp.asarray(positions[valid], dtype=jnp.float32)


def make_baseline_3d_fns(building: BuildingArrays, cfg: Baseline3DConfig):
    """Create pure reset, step, observation, and metric functions."""
    if cfg.num_agents < 1:
        raise ValueError("num_agents must be positive.")
    if cfg.spawn_delay < 0:
        raise ValueError("spawn_delay must be non-negative.")
    if cfg.drone_radius <= 0:
        raise ValueError("drone_radius must be positive.")
    directions = jnp.asarray(spherical_directions(cfg.radar_bins))
    candidates = _target_candidates(building, cfg)
    n = cfg.num_agents
    world_size = jnp.asarray(building.world_size_m)
    lower = jnp.asarray(
        [
            building.wall_thickness_m / 2 + cfg.drone_radius,
            building.wall_thickness_m / 2 + cfg.drone_radius,
            building.tile_thickness_m / 2 + cfg.drone_radius,
        ],
        dtype=jnp.float32,
    )
    upper = world_size - lower
    if bool(jnp.any(lower >= upper)):
        raise ValueError("Building is too small for the configured drone radius.")
    coverage_indices = jnp.stack(
        jnp.meshgrid(
            *[jnp.arange(size) for size in building.target_exclusion.shape],
            indexing="ij",
        ),
        axis=-1,
    )
    coverage_centres = (coverage_indices.astype(jnp.float32) + 0.5) * building.cell_size_m

    def connectivity(pos, active, base_pos, target_pos):
        delta = pos[:, None, :] - pos[None, :, :]
        distances = jnp.linalg.norm(delta, axis=-1)
        agent_adj = (
            (distances <= cfg.comm_radius)
            & active[:, None]
            & active[None, :]
            & ~jnp.eye(n, dtype=jnp.bool_)
        )
        base_edges = (jnp.linalg.norm(pos - base_pos, axis=-1) <= cfg.base_comm_radius) & active
        sees = (jnp.linalg.norm(pos - target_pos, axis=-1) <= cfg.visual_radius) & active

        reach = agent_adj | jnp.eye(n, dtype=jnp.bool_)
        for _ in range(n):
            reach = (reach.astype(jnp.int32) @ reach.astype(jnp.int32)) > 0
        conn_base = jnp.any(reach & base_edges[None, :], axis=1) & active
        conn_target = jnp.any(reach & sees[None, :], axis=1) & active
        return sees, conn_base, conn_target, agent_adj

    def update_coverage(coverage, pos, active):
        delta = coverage_centres[None, ...] - pos[:, None, None, None, :]
        visible = jnp.linalg.norm(delta, axis=-1) <= cfg.visual_radius
        visible &= active[:, None, None, None]
        return coverage | jnp.any(visible, axis=0)

    def reset(key):
        target_key, next_key = jax.random.split(key)
        target_idx = jax.random.randint(target_key, (), 0, candidates.shape[0])
        base_pos = jnp.asarray(building.base_position_m)
        pos = jnp.broadcast_to(base_pos, (n, 3))
        active = jnp.arange(n) * cfg.spawn_delay <= 0
        coverage = jnp.zeros(building.target_exclusion.shape, dtype=jnp.bool_)
        coverage = update_coverage(coverage, pos, active)
        sees, conn_base, conn_target, _ = connectivity(pos, active, base_pos, candidates[target_idx])
        return Baseline3DState(
            pos=pos,
            vel=jnp.zeros((n, 3), dtype=jnp.float32),
            base_pos=base_pos,
            target_pos=candidates[target_idx],
            active=active,
            coverage=coverage,
            step=jnp.int32(0),
            key=next_key,
            directly_sees_target=sees,
            is_conn_base=conn_base,
            is_conn_target=conn_target,
            target_known=conn_target,
            success=jnp.any(conn_base & conn_target),
        )

    def step(state: Baseline3DState, action: jax.Array):
        next_step = state.step + 1
        active = next_step >= jnp.arange(n) * cfg.spawn_delay
        force = jnp.clip(action, -1.0, 1.0) * cfg.max_force
        force = jnp.where(active[:, None], force, 0.0)
        velocity = state.vel * cfg.drag + force * cfg.dt
        speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= jnp.minimum(1.0, cfg.max_speed / jnp.maximum(speed, 1e-8))
        proposed = state.pos + velocity * cfg.dt
        collided = (proposed < lower) | (proposed > upper)
        pos = jnp.clip(proposed, lower, upper)
        velocity = jnp.where(collided, 0.0, velocity)
        pos = jnp.where(active[:, None], pos, state.base_pos)
        velocity = jnp.where(active[:, None], velocity, 0.0)
        coverage = update_coverage(state.coverage, pos, active)
        sees, conn_base, conn_target, _ = connectivity(pos, active, state.base_pos, state.target_pos)
        known = state.target_known | conn_target
        return Baseline3DState(
            pos=pos,
            vel=velocity,
            base_pos=state.base_pos,
            target_pos=state.target_pos,
            active=active,
            coverage=coverage,
            step=next_step,
            key=state.key,
            directly_sees_target=sees,
            is_conn_base=conn_base,
            is_conn_target=conn_target,
            target_known=known,
            success=jnp.any(conn_base & conn_target),
        )

    def observations(state: Baseline3DState):
        pos = state.pos
        delta = pos[:, None, :] - pos[None, :, :]
        pair_dist = jnp.linalg.norm(delta, axis=-1)
        _, _, _, agent_adj = connectivity(pos, state.active, state.base_pos, state.target_pos)

        def one_agent(i):
            origin = pos[i]
            positive = jnp.where(directions > 0, (upper - origin) / directions, jnp.inf)
            negative = jnp.where(directions < 0, (lower - origin) / directions, jnp.inf)
            wall_distance = jnp.min(jnp.minimum(positive, negative), axis=-1)
            wall_signal = jnp.maximum(0.0, 1.0 - wall_distance / cfg.visual_radius)

            rel = pos - origin
            rel_norm = rel / jnp.maximum(pair_dist[i, :, None], 1e-8)
            bin_index = jnp.argmax(rel_norm @ directions.T, axis=-1)
            other = (jnp.arange(n) != i) & state.active
            signal = jnp.maximum(0.0, 1.0 - pair_dist[i] / cfg.comm_radius) * other

            def scatter(values):
                return jnp.zeros(cfg.radar_bins, dtype=jnp.float32).at[bin_index].max(values)

            radar = jnp.stack(
                [
                    wall_signal,
                    scatter(signal),
                    scatter(signal * state.is_conn_target),
                    scatter(signal * state.is_conn_base),
                ],
                axis=-1,
            ).reshape(-1)
            self_state = jnp.concatenate(
                [
                    state.vel[i] / cfg.max_speed,
                    jnp.asarray(
                        [state.is_conn_base[i], state.is_conn_target[i], state.target_known[i]],
                        dtype=jnp.float32,
                    ),
                ]
            )
            return jnp.concatenate([self_state, radar])

        del agent_adj  # reserved for future LOS-aware radar filtering
        return jax.vmap(one_agent)(jnp.arange(n))

    def metrics(state: Baseline3DState):
        return {
            "coverage_fraction": jnp.mean(state.coverage),
            "active_agents": jnp.sum(state.active),
            "target_seen": jnp.any(state.directly_sees_target),
            "success": state.success,
        }

    return reset, step, observations, metrics
