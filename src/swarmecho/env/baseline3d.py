"""Pure-JAX minimum 3D cuboid environment.

The module is intentionally self-contained while the legacy 2D environment is
still present.  It is the executable migration seam: 3D state, action,
coverage, communication, target discovery, spherical radar, and collision can
be validated before replacing the training entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
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
    stationary_steps: jax.Array = jnp.int32(0)
    idle_terminated: jax.Array = jnp.bool_(False)
    # Persistent equivalent of 2D CommunicationState.base_target_known.  A
    # momentary complete chain delivers the target information to the base;
    # subsequent chain breaks must not make that delivery reward available
    # again.
    base_target_known: jax.Array = jnp.bool_(False)


@dataclass(frozen=True)
class Baseline3DConfig:
    num_agents: int = 5
    dt: float = 0.1
    max_force: float = 15.0
    max_speed: float = 5.0
    drag: float = 0.85
    drone_radius: float = 0.25
    comm_radius_base: float = 6.0
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    target_spawn_buffer: float = 0.5
    radar_bins: int = 8
    spawn_delay: int = 0
    hold_chain_for: int = 50
    max_steps: int = 700
    no_movement_termination_steps: int = 50
    movement_epsilon: float = 1e-3
    observe_target_vector: bool = False
    observe_base_vector: bool = False
    observe_coverage_probe: bool = False
    coverage_voxel_size: float | None = None


@dataclass(frozen=True)
class Baseline3DRewardConfig:
    target_found_requires_delivery: bool = True
    chain_reward_system: str = "euclidean"
    exploration_bonus: float = 0.25
    collision_penalty: float = 0.5
    finder_bonus: float = 50.0
    max_gap_penalty: float = 5.0
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
    return cfg.comm_radius_base + 0.5 * cfg.comm_radius + cfg.target_spawn_buffer


def coverage_grid_geometry(
    building: BuildingArrays, cfg: Baseline3DConfig
) -> tuple[float, tuple[int, int, int]]:
    """Return the coverage voxel edge length and its world-aligned grid shape.

    ``None`` retains the original behaviour: one coverage voxel per building
    cell. A configured size must tile the complete building, so every coverage
    voxel has the advertised cubic edge length rather than leaving a partial
    boundary voxel.
    """
    voxel_size = (
        building.cell_size_m
        if cfg.coverage_voxel_size is None
        else cfg.coverage_voxel_size
    )
    if (
        isinstance(voxel_size, bool)
        or not isinstance(voxel_size, (int, float))
        or voxel_size <= 0
    ):
        raise ValueError("coverage_voxel_size must be a positive number when configured.")
    voxel_size = float(voxel_size)
    counts = np.asarray(building.world_size_m, dtype=np.float64) / voxel_size
    rounded_counts = np.rint(counts)
    if np.any(rounded_counts < 1) or not np.allclose(
        counts, rounded_counts, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(
            "coverage_voxel_size must evenly divide every building world dimension."
        )
    return voxel_size, tuple(int(count) for count in rounded_counts)


def observation_dim_3d(cfg: Baseline3DConfig) -> int:
    """Return the configured 3D actor observation width."""
    return (
        6
        + 3 * int(cfg.observe_base_vector)
        + 3 * int(cfg.observe_target_vector)
        + cfg.radar_bins * 4
        + cfg.radar_bins * int(cfg.observe_coverage_probe)
    )


def maximum_five_drone_chain_distance(cfg: Baseline3DConfig) -> float:
    """Unobstructed base→D1→…→D5→target reach for five mobile drones."""
    if cfg.num_agents != 5:
        raise ValueError("maximum_five_drone_chain_distance requires num_agents=5.")
    return maximum_chain_distance(cfg)


def maximum_chain_distance(cfg: Baseline3DConfig) -> float:
    """Ideal straight-line reach for the configured number of mobile drones."""
    return (
        cfg.comm_radius_base
        + max(0, cfg.num_agents - 1) * cfg.comm_radius
        + cfg.visual_radius
    )


def chain_diagnostics_3d(state: Baseline3DState) -> tuple[jax.Array, jax.Array]:
    """Return the 3D counterparts of the 2D chain-gap diagnostics.

    ``chain_gap_dist`` is the distance between the tips of the base-connected
    and target-connected components (zero for a complete chain).
    ``chain_progress_pct`` maps that gap onto the direct base-to-target
    distance, so it remains meaningful before the target has been delivered.
    """
    active = state.active
    base_distance = jnp.linalg.norm(state.pos - state.base_pos[None], axis=-1)
    target_distance = jnp.linalg.norm(state.pos - state.target_pos[None], axis=-1)
    has_base_chain = jnp.any(state.is_conn_base & active)
    has_target_chain = jnp.any(state.is_conn_target & active)
    base_tip_index = jnp.argmin(
        jnp.where(state.is_conn_base & active, target_distance, jnp.inf)
    )
    target_tip_index = jnp.argmin(
        jnp.where(state.is_conn_target & active, base_distance, jnp.inf)
    )
    base_tip = jnp.where(has_base_chain, state.pos[base_tip_index], state.base_pos)
    target_tip = jnp.where(has_target_chain, state.pos[target_tip_index], state.target_pos)
    base_target_distance = jnp.linalg.norm(state.target_pos - state.base_pos)
    chain_gap_dist = jnp.where(
        state.fully_connected,
        0.0,
        jnp.linalg.norm(base_tip - target_tip),
    )
    chain_progress_pct = jnp.where(
        state.fully_connected,
        100.0,
        jnp.clip(
            100.0 * (1.0 - chain_gap_dist / jnp.maximum(base_target_distance, 1e-6)),
            0.0,
            100.0,
        ),
    )
    return chain_gap_dist, chain_progress_pct


def _contributing_chain_agents_3d(
    state: Baseline3DState,
    cfg: Baseline3DConfig,
) -> jax.Array:
    """Select 3D relay drones using 2D's deterministic shortest-path rule."""
    n = state.pos.shape[0]
    node_count = n + 2
    base_node = n
    target_node = n + 1
    active = state.active

    delta = state.pos[:, None, :] - state.pos[None, :, :]
    distances = jnp.linalg.norm(delta, axis=-1)
    drone_edges = (
        (distances <= cfg.comm_radius)
        & active[:, None]
        & active[None, :]
        & ~jnp.eye(n, dtype=jnp.bool_)
    )
    base_edges = (
        jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
        <= cfg.comm_radius_base
    ) & active
    target_edges = state.directly_sees_target & active

    hops = jnp.full((node_count, node_count), 9999, dtype=jnp.int32)
    hops = hops.at[jnp.arange(node_count), jnp.arange(node_count)].set(0)
    hops = hops.at[:n, :n].set(jnp.where(drone_edges, 1, hops[:n, :n]))
    hops = hops.at[:n, base_node].set(jnp.where(base_edges, 1, hops[:n, base_node]))
    hops = hops.at[base_node, :n].set(jnp.where(base_edges, 1, hops[base_node, :n]))
    hops = hops.at[:n, target_node].set(jnp.where(target_edges, 1, hops[:n, target_node]))
    hops = hops.at[target_node, :n].set(jnp.where(target_edges, 1, hops[target_node, :n]))

    def min_plus_step(matrix, _):
        next_matrix = jnp.min(matrix[:, :, None] + matrix[None, :, :], axis=1)
        return jnp.clip(next_matrix, 0, 9999), None

    shortest, _ = jax.lax.scan(
        min_plus_step,
        hops,
        None,
        length=math.ceil(math.log2(node_count)),
    )

    base_distance = jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
    target_distance = jnp.linalg.norm(state.pos - state.target_pos[None, :], axis=-1)
    is_base_chain = state.is_conn_base & active
    is_target_chain = state.is_conn_target & active
    has_base_chain = jnp.any(is_base_chain)
    has_target_chain = jnp.any(is_target_chain)
    base_tip = jnp.argmin(jnp.where(is_base_chain, target_distance, jnp.inf))
    target_tip = jnp.argmin(jnp.where(is_target_chain, base_distance, jnp.inf))

    def trace_path(start_node, destination_node):
        def trace_step(current_node, _):
            next_valid = (
                (hops[current_node, :] == 1)
                & (
                    shortest[:, destination_node]
                    == shortest[current_node, destination_node] - 1
                )
            )
            has_next = jnp.any(next_valid)
            next_node = jnp.where(
                current_node == destination_node,
                destination_node,
                jnp.where(has_next, jnp.argmax(next_valid), destination_node),
            )
            return next_node, current_node

        _, path_nodes = jax.lax.scan(
            trace_step, start_node, None, length=node_count
        )
        return jnp.any(path_nodes == jnp.arange(n)[:, None], axis=-1)

    base_path = jnp.where(has_base_chain, trace_path(base_node, base_tip), False)
    target_path = jnp.where(
        has_target_chain, trace_path(target_node, target_tip), False
    )
    return base_path | target_path


def rewards_3d(
    previous: Baseline3DState,
    current: Baseline3DState,
    cfg: Baseline3DRewardConfig,
    env_cfg: Baseline3DConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute 3D rewards with the same event semantics as the 2D task.

    In particular, target delivery and the final held-chain success are
    separate one-shot events.  The local finder bonus is paid only to an
    agent that performs the one-time target delivery/discovery event; merely
    crossing the visual-radius boundary again cannot earn it repeatedly.
    """
    n = current.pos.shape[0]
    newly_knows = current.target_known & ~previous.target_known
    success_event = current.success & ~previous.success
    if cfg.target_found_requires_delivery:
        target_found_event = (
            current.base_target_known & ~previous.base_target_known
        )
        dynamic_gap_enabled = current.base_target_known
        # Match 2D's ``adj_db & (old_target_known | directly_sees_target)``:
        # only a direct base neighbour that already holds the information (or
        # sees the target on this step) gets the local finder credit.
        in_base_range = (
            jnp.linalg.norm(current.pos - current.base_pos[None, :], axis=-1)
            <= env_cfg.comm_radius_base
        ) & current.active
        finder_receivers = in_base_range & (
            previous.target_known | current.directly_sees_target
        )
    else:
        target_found_event = (
            jnp.any(current.target_known) & ~jnp.any(previous.target_known)
        )
        dynamic_gap_enabled = jnp.any(current.target_known)
        finder_receivers = (
            current.directly_sees_target & ~previous.target_known
        )

    gap_distance, _ = chain_diagnostics_3d(current)
    full_distance = jnp.linalg.norm(current.target_pos - current.base_pos)
    # 2D starts its dynamic gap shaping only after the base has received the
    # target information, not when a remote drone first sees it.
    active_gap = jnp.where(dynamic_gap_enabled, gap_distance, full_distance)
    gap_penalty = -cfg.max_gap_penalty * active_gap / jnp.maximum(full_distance, 1e-6)
    is_contributing = _contributing_chain_agents_3d(current, env_cfg)
    terms = {
        # As in 2D, exploration ends for a drone once it knows the target.
        "coverage": cfg.exploration_bonus * current.coverage_credit
        * (~current.target_known).astype(jnp.float32),
        "collision": -cfg.collision_penalty * current.collided.astype(jnp.float32),
        "finder": cfg.finder_bonus * (
            target_found_event & finder_receivers
        ).astype(jnp.float32),
        "target_found": jnp.full(n, cfg.target_found_bonus / n) * target_found_event,
        "chain_gap": jnp.where(
            is_contributing,
            gap_penalty / n,
            -cfg.max_gap_penalty / n,
        ),
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
            "Building has no target cell beyond comm_radius_base + "
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
    if cfg.no_movement_termination_steps < 1:
        raise ValueError("no_movement_termination_steps must be at least 1.")
    if cfg.movement_epsilon < 0:
        raise ValueError("movement_epsilon must be non-negative.")
    if cfg.drone_radius <= 0:
        raise ValueError("drone_radius must be positive.")
    coverage_voxel_size, coverage_shape = coverage_grid_geometry(building, cfg)
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
            *[jnp.arange(size) for size in coverage_shape],
            indexing="ij",
        ),
        axis=-1,
    )
    coverage_centres = (coverage_indices.astype(jnp.float32) + 0.5) * coverage_voxel_size

    def connectivity(pos, active, base_pos, target_pos):
        delta = pos[:, None, :] - pos[None, :, :]
        distances = jnp.linalg.norm(delta, axis=-1)
        agent_adj = (
            (distances <= cfg.comm_radius)
            & active[:, None]
            & active[None, :]
            & ~jnp.eye(n, dtype=jnp.bool_)
        )
        base_edges = (jnp.linalg.norm(pos - base_pos, axis=-1) <= cfg.comm_radius_base) & active
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

    def reset(key, target_pos: jax.Array | None = None):
        """Reset an episode, optionally at one validated external target point.

        The optional target is used by checkpoint inspection: it lets a replay
        reproduce one target sampled by a previous parallel evaluation without
        changing the normal keyed target sampler used for training.
        """
        target_key, next_key = jax.random.split(key)
        target_idx = jax.random.randint(target_key, (), 0, candidates.shape[0])
        target = candidates[target_idx] if target_pos is None else jnp.asarray(target_pos)
        base_pos = jnp.asarray(building.base_position_m)
        pos = jnp.broadcast_to(base_pos, (n, 3))
        active = jnp.arange(n) * cfg.spawn_delay <= 0
        coverage = jnp.zeros(coverage_shape, dtype=jnp.bool_)
        coverage, _ = update_coverage(coverage, pos, active)
        sees, conn_base, conn_target, _ = connectivity(pos, active, base_pos, target)
        fully_connected = jnp.any(conn_base & conn_target)
        return Baseline3DState(
            pos=pos,
            vel=jnp.zeros((n, 3), dtype=jnp.float32),
            base_pos=base_pos,
            target_pos=target,
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
            stationary_steps=jnp.int32(0),
            idle_terminated=jnp.bool_(False),
            base_target_known=jnp.bool_(False),
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
        any_agent_moved = jnp.any(
            jnp.linalg.norm(pos - state.pos, axis=-1) > cfg.movement_epsilon
        )
        stationary_steps = jnp.where(
            any_agent_moved,
            jnp.int32(0),
            state.stationary_steps + jnp.int32(1),
        )
        idle_terminated = stationary_steps >= jnp.int32(
            cfg.no_movement_termination_steps
        )
        coverage, coverage_credit = update_coverage(state.coverage, pos, active)
        sees, conn_base, conn_target, _ = connectivity(pos, active, state.base_pos, state.target_pos)
        known = state.target_known | conn_target
        fully_connected = jnp.any(conn_base & conn_target)
        base_target_known = state.base_target_known | fully_connected
        chain_held_steps = jnp.where(
            fully_connected,
            state.chain_held_steps + jnp.int32(1),
            jnp.int32(0),
        )
        success = chain_held_steps >= jnp.int32(cfg.hold_chain_for)
        done = success | idle_terminated | (next_step >= jnp.int32(cfg.max_steps))
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
            stationary_steps=stationary_steps,
            idle_terminated=idle_terminated,
            base_target_known=base_target_known,
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
            optional = []
            scale = jnp.maximum(jnp.max(world_size), 1e-6)
            if cfg.observe_base_vector:
                optional.append((state.base_pos - origin) / scale)
            if cfg.observe_target_vector:
                optional.append(
                    (state.target_pos - origin)
                    / scale
                    * state.target_known[i].astype(jnp.float32)
                )
            if cfg.observe_coverage_probe:
                probe = jnp.clip(
                    jnp.floor(
                        (origin + directions * coverage_voxel_size)
                        / coverage_voxel_size
                    ).astype(jnp.int32),
                    0,
                    jnp.asarray(state.coverage.shape) - 1,
                )
                optional.append(state.coverage[probe[:, 0], probe[:, 1], probe[:, 2]].astype(jnp.float32))
            return jnp.concatenate([self_state, *optional, radar])

        del agent_adj  # reserved for future LOS-aware radar filtering
        return jax.vmap(one_agent)(jnp.arange(n))

    def metrics(state: Baseline3DState):
        return {
            "coverage_fraction": jnp.mean(state.coverage),
            "active_agents": jnp.sum(state.active),
            "target_seen": jnp.any(state.directly_sees_target),
            "success": state.success,
            "fully_connected": state.fully_connected,
            "base_target_known": state.base_target_known,
            "stationary_steps": state.stationary_steps,
            "idle_terminated": state.idle_terminated,
            "done": state.done,
        }

    return reset, step, observations, metrics


def make_autoreset_3d_fns(
    building: BuildingArrays,
    cfg: Baseline3DConfig,
    reward_cfg: Baseline3DRewardConfig = Baseline3DRewardConfig(),
):
    """Return reset and terminal-aware step functions for batched training.

    Terminal rewards and diagnostic state describe the completed transition;
    only the returned carry state is replaced by a fresh episode.
    """
    reset, step, observations, metrics = make_baseline_3d_fns(building, cfg)

    def autoreset_step(state: Baseline3DState, action: jax.Array):
        terminal_state = step(state, action)
        reward, reward_terms = rewards_3d(state, terminal_state, reward_cfg, cfg)
        chain_gap_dist, chain_progress_pct = chain_diagnostics_3d(terminal_state)
        reset_state = reset(terminal_state.key)
        next_state = jax.tree_util.tree_map(
            lambda fresh, current: jnp.where(terminal_state.done, fresh, current),
            reset_state,
            terminal_state,
        )
        info = {
            **reward_terms,
            "done": terminal_state.done,
            "idle_terminated": terminal_state.idle_terminated,
            # Keep the reward component distinct from the terminal-success
            # flag so training telemetry can accumulate the actual bonus.
            "reward_success": reward_terms["success"],
            "success": terminal_state.success,
            "fully_connected": terminal_state.fully_connected,
            # Persistent direct analogue of 2D's ``base_target_known``.
            # Success additionally requires the configured hold duration.
            "global_target_found": terminal_state.base_target_known,
            "chain_gap_dist": chain_gap_dist,
            "chain_progress_pct": chain_progress_pct,
            "global_coverage": jnp.mean(terminal_state.coverage),
            "terminal_target_pos": terminal_state.target_pos,
            "terminal_coverage_fraction": jnp.mean(terminal_state.coverage),
        }
        return next_state, reward, terminal_state.done, info

    return reset, autoreset_step, observations, metrics
