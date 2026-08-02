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
    fully_connected: jax.Array
    chain_held_steps: jax.Array
    done: jax.Array
    collided: jax.Array
    coverage_credit: jax.Array


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
    hold_chain_for: int = 5
    max_steps: int = 700


@dataclass(frozen=True)
class Baseline3DRewardConfig:
    exploration_bonus: float = 0.25
    collision_penalty: float = 0.5
    finder_bonus: float = 50.0
    target_found_bonus: float = 100.0
    success_bonus: float = 500.0


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
    if cfg.num_agents != 5:
        raise ValueError("maximum_five_drone_chain_distance requires num_agents=5.")
    return maximum_chain_distance(cfg)


def maximum_chain_distance(cfg: Baseline3DConfig) -> float:
    """Ideal straight-line reach for the configured number of mobile drones."""
    return (
        cfg.base_comm_radius
        + max(0, cfg.num_agents - 1) * cfg.comm_radius
        + cfg.visual_radius
    )


def rewards_3d(
    previous: Baseline3DState,
    current: Baseline3DState,
    cfg: Baseline3DRewardConfig = Baseline3DRewardConfig(),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute per-agent rewards while preserving local coverage credit."""
    n = current.pos.shape[0]
    newly_knows = current.target_known & ~previous.target_known
    target_found_event = jnp.any(current.target_known) & ~jnp.any(previous.target_known)
    success_event = current.success & ~previous.success
    terms = {
        "coverage": cfg.exploration_bonus * current.coverage_credit,
        "collision": -cfg.collision_penalty * current.collided.astype(jnp.float32),
        "finder": cfg.finder_bonus
        * (current.directly_sees_target & ~previous.directly_sees_target).astype(jnp.float32),
        "target_found": jnp.full(n, cfg.target_found_bonus / n) * target_found_event,
        "success": jnp.full(n, cfg.success_bonus / n) * success_event,
    }
    # Agents learning through relayed information still receive the shared event;
    # ``newly_knows`` is exposed for diagnostics without double-paying finders.
    terms["newly_knows"] = newly_knows.astype(jnp.float32)
    total = sum(value for name, value in terms.items() if name != "newly_knows")
    return total, terms


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
    if cfg.hold_chain_for < 1:
        raise ValueError("hold_chain_for must be at least 1.")
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
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
        newly_covered = ~coverage & jnp.any(visible, axis=0)
        viewers = jnp.sum(visible, axis=0)
        credit_per_voxel = newly_covered / jnp.maximum(viewers, 1)
        credit = jnp.sum(visible * credit_per_voxel[None, ...], axis=(1, 2, 3))
        return coverage | newly_covered, credit

    def reset(key):
        target_key, next_key = jax.random.split(key)
        target_idx = jax.random.randint(target_key, (), 0, candidates.shape[0])
        base_pos = jnp.asarray(building.base_position_m)
        pos = jnp.broadcast_to(base_pos, (n, 3))
        active = jnp.arange(n) * cfg.spawn_delay <= 0
        coverage = jnp.zeros(building.target_exclusion.shape, dtype=jnp.bool_)
        coverage, _ = update_coverage(coverage, pos, active)
        sees, conn_base, conn_target, _ = connectivity(pos, active, base_pos, candidates[target_idx])
        fully_connected = jnp.any(conn_base & conn_target)
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
            success=jnp.bool_(False),
            fully_connected=fully_connected,
            chain_held_steps=jnp.int32(0),
            done=jnp.bool_(False),
            collided=jnp.zeros(n, dtype=jnp.bool_),
            coverage_credit=jnp.zeros(n, dtype=jnp.float32),
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
        coverage, coverage_credit = update_coverage(state.coverage, pos, active)
        sees, conn_base, conn_target, _ = connectivity(pos, active, state.base_pos, state.target_pos)
        known = state.target_known | conn_target
        fully_connected = jnp.any(conn_base & conn_target)
        chain_held_steps = jnp.where(
            fully_connected,
            state.chain_held_steps + jnp.int32(1),
            jnp.int32(0),
        )
        success = chain_held_steps >= jnp.int32(cfg.hold_chain_for)
        done = success | (next_step >= jnp.int32(cfg.max_steps))
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
            success=success,
            fully_connected=fully_connected,
            chain_held_steps=chain_held_steps,
            done=done,
            collided=jnp.any(collided, axis=-1) & active,
            coverage_credit=coverage_credit,
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
            "fully_connected": state.fully_connected,
            "done": state.done,
        }

    return reset, step, observations, metrics


def make_autoreset_3d_fns(building: BuildingArrays, cfg: Baseline3DConfig):
    """Return reset and terminal-aware step functions for batched training.

    Terminal rewards and diagnostic state describe the completed transition;
    only the returned carry state is replaced by a fresh episode.
    """
    reset, step, observations, metrics = make_baseline_3d_fns(building, cfg)

    def autoreset_step(state: Baseline3DState, action: jax.Array):
        terminal_state = step(state, action)
        reward, reward_terms = rewards_3d(state, terminal_state)
        reset_state = reset(terminal_state.key)
        next_state = jax.tree_util.tree_map(
            lambda fresh, current: jnp.where(terminal_state.done, fresh, current),
            reset_state,
            terminal_state,
        )
        info = {
            **reward_terms,
            "done": terminal_state.done,
            "success": terminal_state.success,
            "fully_connected": terminal_state.fully_connected,
            "terminal_target_pos": terminal_state.target_pos,
            "terminal_coverage_fraction": jnp.mean(terminal_state.coverage),
        }
        return next_state, reward, terminal_state.done, info

    return reset, autoreset_step, observations, metrics
