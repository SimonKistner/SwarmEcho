"""Pure-JAX 3D environment, observations, relay rewards, and episode state."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.env.buildings import BuildingArrays
from swarmecho.env.obstacles import (
    _roadmap_attachment,
    _route_costs,
    authored_roadmap_vertices,
    free_space_distance,
    pairwise_free_space_distance,
    generate_obstacles,
    points_inside_aabbs,
    roadmap_distances,
    roadmap_vertices,
    segments_blocked,
    SharedRoadmap,
    shared_roadmap,
)


class EnvState(NamedTuple):
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
    # Persistent record of target delivery to the base. A
    # momentary complete chain delivers the target information to the base;
    # subsequent chain breaks must not make that delivery reward available
    # again.
    base_target_known: jax.Array = jnp.bool_(False)
    obstacle_min: jax.Array = jnp.zeros((0, 3), dtype=jnp.float32)
    obstacle_max: jax.Array = jnp.zeros((0, 3), dtype=jnp.float32)
    roadmap_vertices: jax.Array = jnp.zeros((0, 3), dtype=jnp.float32)
    roadmap_distances: jax.Array = jnp.zeros((0, 0), dtype=jnp.float32)
    shared_geometry: SharedRoadmap | None = None
    spawn_pair_attempts: jax.Array = jnp.int32(1)
    roadmap_corner_distances: jax.Array = jnp.zeros((0, 0), dtype=jnp.float32)

    @property
    def solid_min(self):
        if self.shared_geometry is None:
            return self.obstacle_min
        authored = jnp.broadcast_to(jnp.asarray(self.shared_geometry.solid_min), self.obstacle_min.shape[:-2] + self.shared_geometry.solid_min.shape)
        return jnp.concatenate((authored, self.obstacle_min), axis=-2)

    @property
    def solid_max(self):
        if self.shared_geometry is None:
            return self.obstacle_max
        authored = jnp.broadcast_to(jnp.asarray(self.shared_geometry.solid_max), self.obstacle_max.shape[:-2] + self.shared_geometry.solid_max.shape)
        return jnp.concatenate((authored, self.obstacle_max), axis=-2)

    @property
    def planning_vertices(self):
        if self.shared_geometry is not None and self.obstacle_min.shape[0] == 0:
            return jnp.asarray(self.shared_geometry.vertices)
        return self.roadmap_vertices

    @property
    def planning_distances(self):
        if self.shared_geometry is not None and self.obstacle_min.shape[0] == 0:
            return jnp.asarray(self.shared_geometry.distances)
        return self.roadmap_distances

    @property
    def planning_min(self):
        margin = self.shared_geometry.clearance - 1e-4 if self.shared_geometry is not None else 0.0
        return self.solid_min - margin

    @property
    def planning_max(self):
        margin = self.shared_geometry.clearance - 1e-4 if self.shared_geometry is not None else 0.0
        return self.solid_max + margin


@dataclass(frozen=True)
class EnvConfig:
    num_agents: int = 5
    dt: float = 0.1
    max_force: float = 15.0
    max_speed: float = 5.0
    drag: float = 0.85
    drone_radius: float = 0.25
    comm_radius_base: float = 6.0
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    target_spawn_buffer: float = 0.5  # Legacy config compatibility; no base-distance restriction.
    target_wall_buffer_fraction: float = 0.1
    radar_bins: int = 8
    spawn_delay: int = 0
    hold_chain_for: int = 50
    max_steps: int = 700
    no_movement_termination_steps: int = 50
    success_when_target_found_or_delivered: bool = False
    movement_epsilon: float = 1e-3
    observe_target_vector: bool = False
    observe_base_vector: bool = False
    observe_coverage_probe: bool = False
    observe_chain_contributor: bool = False
    observe_current_timestep: bool = False  # Episode step / max_steps, in [0, 1].
    coverage_voxel_size: float | None = None
    num_obstacles: int = 0
    obstacle_size_min_m: float = 2.0
    obstacle_size_max_m: float = 4.0
    obstacle_spawn_layer_min: int = 2
    obstacle_spawn_layer_max: int = 5
    obstacle_boundary_buffer_m: float = 0.5
    obstacle_target_buffer_m: float = 0.5
    obstacle_planning_clearance_m: float = 0.1
    roadmap_merge_walls: bool = True
    roadmap_corner_bonus_m: float = 8.0
    obstacle_layout_version: str = "three_aabb_v1"


@dataclass(frozen=True)
class RewardConfig:
    target_found_requires_delivery: bool = True
    chain_reward_system: str = "euclidean"
    exploration_bonus: float = 0.25
    collision_penalty: float = 0.5
    finder_bonus: float = 50.0
    max_gap_penalty: float = 5.0
    target_found_bonus: float = 100.0
    success_bonus: float = 500.0
    no_movement_termination_penalty: float = -1000.0
    # Credit every drone on a simple base-target path, including alternate routes.
    allow_redundancy_reward: bool = False
    enable_chain_efficiency_reward: bool = False
    chain_efficiency_bonus: float = 0.5


def build_obstacle_roadmap(obstacle_min, obstacle_max, cfg: EnvConfig):
    """Build the compact roadmap arrays for persisted/replayed obstacle bounds."""
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    vertices = roadmap_vertices(obstacle_min, obstacle_max, clearance)
    return vertices, roadmap_distances(vertices, obstacle_min, obstacle_max, clearance)


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


def coverage_grid_geometry(
    building: BuildingArrays, cfg: EnvConfig
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


def observation_dim(cfg: EnvConfig) -> int:
    """Return the configured 3D actor observation width."""
    return (
        6
        + 3 * int(cfg.observe_base_vector)
        + 3 * int(cfg.observe_target_vector)
        + cfg.radar_bins * 4
        + cfg.radar_bins * int(cfg.observe_coverage_probe)
        + int(cfg.observe_chain_contributor)
        + int(cfg.observe_current_timestep)
    )


def maximum_five_drone_chain_distance(cfg: EnvConfig) -> float:
    """Unobstructed base→D1→…→D5→target reach for five mobile drones."""
    if cfg.num_agents != 5:
        raise ValueError("maximum_five_drone_chain_distance requires num_agents=5.")
    return maximum_chain_distance(cfg)


def maximum_chain_distance(cfg: EnvConfig) -> float:
    """Ideal straight-line reach for the configured number of mobile drones."""
    return (
        cfg.comm_radius_base
        + max(0, cfg.num_agents - 1) * cfg.comm_radius
        + cfg.visual_radius
    )


def chain_diagnostics(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Return physical chain-gap and progress diagnostics.

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


def chain_distance_matrix(state: EnvState):
    """Distances for [base, target, agents], computed once per transition."""
    points = jnp.concatenate((state.base_pos[None], state.target_pos[None], state.pos))
    return pairwise_free_space_distance(points, state.planning_vertices,
                                       state.planning_distances, state.planning_min, state.planning_max)


def obstacle_chain_diagnostics(state: EnvState, distances=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Jointly select compatible partial-chain leaders using free-space distance."""
    active = state.active
    if distances is None:
        distances = chain_distance_matrix(state)
    agents = jnp.arange(state.pos.shape[0]) + 2
    base_indices = jnp.concatenate((jnp.array([0]), agents))
    target_indices = jnp.concatenate((jnp.array([1]), agents))
    base_valid = jnp.concatenate([jnp.ones(1, dtype=bool), state.is_conn_base & active])
    target_valid = jnp.concatenate([jnp.ones(1, dtype=bool), state.is_conn_target & active & state.target_known])
    gaps = distances[base_indices[:, None], target_indices[None, :]]
    mission = distances[0, 1]
    # Route excess is a secondary tie-breaker only; one millimetre of gap always
    # dominates it, preserving the meaning of the primary joint gap objective.
    from_base = distances[0, base_indices]
    to_target = distances[target_indices, 1]
    excess = jnp.maximum(from_base[:, None] + gaps + to_target[None, :] - mission, 0.0)
    score = gaps + jnp.minimum(excess, 1e3) * 1e-6
    score = jnp.where(base_valid[:, None] & target_valid[None, :], score, 1e6)
    flat = jnp.argmin(score)
    idx_b, idx_t = flat // score.shape[1], flat % score.shape[1]
    gap = jnp.where(state.fully_connected, 0.0, gaps[idx_b, idx_t])
    progress = jnp.where(state.fully_connected, 100.0, jnp.clip(100.0 * (1.0 - gap / jnp.maximum(mission, 1e-6)), 0.0, 100.0))
    return gap, progress, jnp.maximum(excess[idx_b, idx_t], 0.0), jnp.asarray([idx_b - 1, idx_t - 1])


def final_chain_length(state: EnvState, cfg: EnvConfig) -> jax.Array:
    """Length of the shortest physical base-to-target chain in the final graph.

    This is zero when no complete chain exists. Unlike base-target Euclidean
    distance, it measures the relay route actually available around obstacles.
    """
    n = state.pos.shape[0]
    nodes = jnp.concatenate(
        [state.pos, state.base_pos[None, :], state.target_pos[None, :]], axis=0
    )
    distances = jnp.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    active = state.active
    drone_edges = (
        (distances[:n, :n] <= cfg.comm_radius)
        & active[:, None]
        & active[None, :]
        & ~jnp.eye(n, dtype=bool)
    )
    if state.solid_min.shape[0] > 0:
        drone_edges &= ~segments_blocked(
            state.pos[:, None, :], state.pos[None, :, :],
            state.solid_min, state.solid_max,
        )
    base_edges = (
        jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
        <= cfg.comm_radius_base
    ) & active
    if state.solid_min.shape[0] > 0:
        base_edges &= ~segments_blocked(
            state.pos, state.base_pos, state.solid_min, state.solid_max
        )
    target_edges = state.directly_sees_target & active
    vertex_count = n + 2
    weighted = jnp.full((vertex_count, vertex_count), 1e6, dtype=jnp.float32)
    weighted = weighted.at[jnp.arange(vertex_count), jnp.arange(vertex_count)].set(0.0)
    weighted = weighted.at[:n, :n].set(jnp.where(drone_edges, distances[:n, :n], 1e6))
    weighted = weighted.at[:n, n].set(jnp.where(base_edges, distances[:n, n], 1e6))
    weighted = weighted.at[n, :n].set(jnp.where(base_edges, distances[n, :n], 1e6))
    weighted = weighted.at[:n, n + 1].set(jnp.where(target_edges, distances[:n, n + 1], 1e6))
    weighted = weighted.at[n + 1, :n].set(jnp.where(target_edges, distances[n + 1, :n], 1e6))
    for pivot in range(vertex_count):
        weighted = jnp.minimum(
            weighted, weighted[:, pivot, None] + weighted[pivot, None, :]
        )
    length = weighted[n, n + 1]
    return jnp.where(state.fully_connected & (length < 1e6), length, 0.0)


def _contributing_chain_agents(
    state: EnvState,
    cfg: EnvConfig,
    *, return_graph: bool = False,
) -> jax.Array:
    """Select relay drones using the deterministic shortest-path rule."""
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
    if state.solid_min.shape[0] > 0:
        drone_edges &= ~segments_blocked(
            state.pos[:, None, :], state.pos[None, :, :],
            state.solid_min, state.solid_max,
        )
        base_edges &= ~segments_blocked(
            state.pos, state.base_pos, state.solid_min, state.solid_max
        )
    target_edges = state.directly_sees_target & active

    hops = jnp.full((node_count, node_count), 9999, dtype=jnp.int32)
    hops = hops.at[jnp.arange(node_count), jnp.arange(node_count)].set(0)
    hops = hops.at[:n, :n].set(jnp.where(drone_edges, 1, hops[:n, :n]))
    hops = hops.at[:n, base_node].set(jnp.where(base_edges, 1, hops[:n, base_node]))
    hops = hops.at[base_node, :n].set(jnp.where(base_edges, 1, hops[base_node, :n]))
    hops = hops.at[:n, target_node].set(jnp.where(target_edges, 1, hops[:n, target_node]))
    hops = hops.at[target_node, :n].set(jnp.where(target_edges, 1, hops[target_node, :n]))
    if return_graph:
        return hops == 1

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


class ChainPaths(NamedTuple):
    # Each subset represents its shortest simple route and number of valid
    # route permutations. That is sufficient for best-per-drone credit.
    lengths: jax.Array
    counts: jax.Array
    members: jax.Array


def simple_chain_paths(state: EnvState, cfg: EnvConfig) -> ChainPaths:
    adjacency = _contributing_chain_agents(state, cfg, return_graph=True)
    n = state.pos.shape[0]
    nodes = jnp.concatenate((state.pos, state.base_pos[None], state.target_pos[None]))
    distance = jnp.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    bits = jnp.left_shift(jnp.int32(1), jnp.arange(n))
    masks = jnp.arange(1 << n)
    members = (masks[:, None] & bits) != 0
    initial_lengths = jnp.full((1 << n, n), jnp.inf)
    initial_counts = jnp.zeros((1 << n, n), dtype=jnp.int32)
    initial_lengths = initial_lengths.at[bits, jnp.arange(n)].set(jnp.where(adjacency[n, :n], distance[n, :n], jnp.inf))
    initial_counts = initial_counts.at[bits, jnp.arange(n)].set(adjacency[n, :n].astype(jnp.int32))

    def extend(mask, carry):
        lengths, counts = carry
        previous = mask ^ bits
        present = members[mask]
        valid = present[:, None] & adjacency[:n, :n].T
        candidates = jnp.where(valid, lengths[previous] + distance[:n, :n].T, jnp.inf)
        row_lengths = jnp.min(candidates, axis=1)
        row_counts = jnp.sum(jnp.where(valid, counts[previous], 0), axis=1)
        singleton = mask == bits
        return (lengths.at[mask].set(jnp.where(singleton, lengths[mask], row_lengths)),
                counts.at[mask].set(jnp.where(singleton, counts[mask], row_counts)))

    lengths, counts = jax.lax.fori_loop(1, 1 << n, extend, (initial_lengths, initial_counts))
    terminal = adjacency[:n, n + 1]
    complete_lengths = jnp.min(jnp.where(terminal, lengths + distance[:n, n + 1], jnp.inf), axis=1)
    complete_counts = jnp.sum(jnp.where(terminal, counts, 0), axis=1)
    return ChainPaths(jnp.where(state.fully_connected, complete_lengths, jnp.inf),
                        jnp.where(state.fully_connected, complete_counts, 0), members)


def chain_path_efficiencies(paths: ChainPaths, reference):
    return jnp.where(jnp.isfinite(paths.lengths) & jnp.isfinite(reference) & (reference < 1e6),
                     jnp.clip(reference / jnp.maximum(paths.lengths, 1e-6), 0., 1.), 0.)


def compute_rewards(
    previous: EnvState,
    current: EnvState,
    cfg: RewardConfig,
    env_cfg: EnvConfig,
    geodesic_distances=None,
    chain_paths=None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute 3D relay rewards with persistent discovery and delivery events.

    The local finder and target-found bonuses are one-shot events. The
    terminal success bonus is emitted when either the held chain succeeds or
    the environment's target-found success mode is enabled and its selected
    target signal occurs.
    """
    n = current.pos.shape[0]
    reward_active = current.active.astype(jnp.float32)
    reward_count = jnp.maximum(jnp.sum(reward_active), 1.0)
    newly_knows = current.target_known & ~previous.target_known
    success_event = current.success & ~previous.success
    if cfg.target_found_requires_delivery:
        target_found_event = (
            current.base_target_known & ~previous.base_target_known
        )
        dynamic_gap_enabled = current.base_target_known
        # A base-adjacent drone may deliver previously known or newly seen information:
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

    if cfg.enable_chain_efficiency_reward and geodesic_distances is None:
        geodesic_distances = chain_distance_matrix(current)
    use_obstacle_reward = cfg.chain_reward_system == "obstacle_geodesic"
    if use_obstacle_reward:
        if geodesic_distances is None:
            geodesic_distances = chain_distance_matrix(current)
        gap_distance, _, _, _ = obstacle_chain_diagnostics(current, geodesic_distances)
        full_distance = geodesic_distances[0, 1]
    else:
        gap_distance, _ = chain_diagnostics(current)
        full_distance = jnp.linalg.norm(current.target_pos - current.base_pos)
    # Dynamic gap shaping starts only after the base has received the
    # target information, not when a remote drone first sees it.
    active_gap = jnp.where(dynamic_gap_enabled, gap_distance, full_distance)
    gap_penalty = -cfg.max_gap_penalty * active_gap / jnp.maximum(full_distance, 1e-6)
    is_contributing = _contributing_chain_agents(current, env_cfg)
    if cfg.allow_redundancy_reward or cfg.enable_chain_efficiency_reward:
        if chain_paths is None:
            chain_paths = simple_chain_paths(current, env_cfg)
        valid_paths = chain_paths.counts > 0
        if cfg.allow_redundancy_reward:
            all_members = jnp.any(chain_paths.members & valid_paths[:, None], axis=0)
            is_contributing = jnp.where(current.fully_connected, all_members, is_contributing)
    terms = {
        # Exploration ends for a drone once it knows the target.
        "coverage": cfg.exploration_bonus * current.coverage_credit
        * (~current.target_known).astype(jnp.float32),
        "collision": -cfg.collision_penalty * current.collided.astype(jnp.float32),
        "finder": cfg.finder_bonus * (
            target_found_event & finder_receivers
        ).astype(jnp.float32),
        "target_found": reward_active * (cfg.target_found_bonus / reward_count) * target_found_event,
        "chain_gap": jnp.where(
            is_contributing,
            gap_penalty / reward_count,
            -cfg.max_gap_penalty / reward_count,
        ),
        "success": reward_active * (cfg.success_bonus / reward_count) * success_event,
        "no_movement_termination": reward_active * (cfg.no_movement_termination_penalty / reward_count)
        * (current.idle_terminated & ~current.success
           & ~(current.step >= jnp.int32(env_cfg.max_steps))).astype(jnp.float32),
    }
    # Agents learning through relayed information still receive the shared event;
    if cfg.enable_chain_efficiency_reward:
        efficiency = chain_path_efficiencies(chain_paths, geodesic_distances[0, 1])
        if cfg.allow_redundancy_reward:
            credit = jnp.max(jnp.where(chain_paths.members, efficiency[:, None], 0.), axis=0)
        else:
            best = jnp.argmin(chain_paths.lengths)
            credit = chain_paths.members[best] * efficiency[best]
        terms["efficiency"] = cfg.chain_efficiency_bonus / reward_count * credit
    # ``newly_knows`` is exposed for diagnostics without double-paying finders.
    terms["newly_knows"] = newly_knows.astype(jnp.float32)
    terms = {name: jnp.where(current.active, value, 0.0) for name, value in terms.items()}
    total = sum(value for name, value in terms.items() if name != "newly_knows")
    return total, terms


def _target_spawn_boxes(
    building: BuildingArrays, cfg: EnvConfig, *, include_excluded: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return valid cell boxes and their volumes for continuous target sampling."""
    dims = building.target_exclusion.shape
    cells = np.stack(np.meshgrid(*[np.arange(size) for size in dims], indexing="ij"), axis=-1)
    cells = cells.reshape(-1, 3)
    excluded = building.target_exclusion[tuple(cells.T)] & (not include_excluded)
    interior = building.interior_cells[tuple(cells.T)]
    clearance = cfg.target_wall_buffer_fraction * building.cell_size_m
    if not 0 <= cfg.target_wall_buffer_fraction < 0.5:
        raise ValueError("target_wall_buffer_fraction must be in [0, 0.5).")
    half_wall = building.wall_thickness_m / 2
    half_tile = building.tile_thickness_m / 2
    lower = cells.astype(np.float32) * building.cell_size_m
    upper = lower + building.cell_size_m
    for index, (x, y, z) in enumerate(cells):
        if building.x_walls[x, y, z]:
            lower[index, 0] += half_wall + clearance
        if building.x_walls[x + 1, y, z]:
            upper[index, 0] -= half_wall + clearance
        if building.y_walls[x, y, z]:
            lower[index, 1] += half_wall + clearance
        if building.y_walls[x, y + 1, z]:
            upper[index, 1] -= half_wall + clearance
        if building.tiles[x, y, z]:
            lower[index, 2] += half_tile + clearance
        if building.tiles[x, y, z + 1]:
            upper[index, 2] -= half_tile + clearance
    volumes = np.prod(np.maximum(upper - lower, 0), axis=-1)
    valid = interior & ~excluded & (volumes > 0)
    if not np.any(valid):
        raise ValueError("Building has no non-excluded target spawn volume.")
    return tuple(
        jnp.asarray(value[valid], dtype=jnp.float32)
        for value in (lower, upper, volumes)
    )


def make_env_fns(building: BuildingArrays, cfg: EnvConfig, *, plan_geodesic: bool = True,
                       randomize_base: bool = False, minimum_geodesic_separation: bool = False,
                       minimum_geodesic_separation_multiplier: float = 2.0,
                       spawn_pair_max_attempts: int = 1024, allow_redundancy_reward: bool = False,
                       target_found_requires_delivery: bool = True):
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
    if cfg.num_obstacles < 0:
        raise ValueError("num_obstacles must be non-negative.")
    if cfg.obstacle_size_min_m <= 0 or cfg.obstacle_size_max_m < cfg.obstacle_size_min_m:
        raise ValueError("Obstacle size bounds must be positive and ordered.")
    coverage_voxel_size, coverage_shape = coverage_grid_geometry(building, cfg)
    directions = jnp.asarray(spherical_directions(cfg.radar_bins))
    spawn_lower, spawn_upper, spawn_volumes = _target_spawn_boxes(building, cfg)
    if not math.isfinite(cfg.roadmap_corner_bonus_m) or cfg.roadmap_corner_bonus_m < 0:
        raise ValueError("roadmap_corner_bonus_m must be finite and nonnegative.")
    corner_merge_distance = building.wall_thickness_m + 2 * (cfg.drone_radius + cfg.obstacle_planning_clearance_m)
    if randomize_base or minimum_geodesic_separation:
        if isinstance(spawn_pair_max_attempts, bool) or not isinstance(spawn_pair_max_attempts, int) or not 1 <= spawn_pair_max_attempts <= np.iinfo(np.int32).max:
            raise ValueError("spawn_pair_max_attempts must be a positive int32 integer.")
    if minimum_geodesic_separation:
        if not plan_geodesic:
            raise ValueError("Minimum geodesic separation requires obstacle_geodesic chain reward.")
        if not math.isfinite(minimum_geodesic_separation_multiplier) or minimum_geodesic_separation_multiplier < 0:
            raise ValueError("Minimum geodesic separation multiplier must be finite and nonnegative.")
    if randomize_base:
        base_lower, base_upper, base_volumes = _target_spawn_boxes(building, cfg, include_excluded=True)
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
    coverage_cell_indices = np.minimum(
        np.floor(np.asarray(coverage_centres) / building.cell_size_m).astype(np.int64),
        np.asarray(building.interior_cells.shape) - 1,
    )
    coverage_interior = jnp.asarray(
        building.interior_cells[tuple(coverage_cell_indices.reshape(-1, 3).T)].reshape(
            coverage_shape
        )
    )
    authored_min = jnp.asarray(building.solid_min_m, dtype=jnp.float32)
    authored_max = jnp.asarray(building.solid_max_m, dtype=jnp.float32)
    has_authored_solids = bool(building.solid_min_m.shape[0])
    geometry = shared_roadmap(
        building.solid_min_m, building.solid_max_m, np.asarray(lower), np.asarray(upper),
        cfg.drone_radius + cfg.obstacle_planning_clearance_m,
        plan=plan_geodesic and cfg.num_obstacles == 0,
        merge_walls=cfg.roadmap_merge_walls, edge_spacing=building.cell_size_m,
        corner_bonus_m=cfg.roadmap_corner_bonus_m if minimum_geodesic_separation else 0.0,
        corner_merge_distance_m=corner_merge_distance,
    )
    authored_vertices = jnp.asarray(geometry.vertices)
    if plan_geodesic and cfg.roadmap_merge_walls and cfg.num_obstacles:
        authored_vertices = jnp.asarray(authored_roadmap_vertices(
            building.solid_min_m, building.solid_max_m, np.asarray(lower), np.asarray(upper),
            cfg.drone_radius + cfg.obstacle_planning_clearance_m, building.cell_size_m))

    def all_solids(obstacle_min, obstacle_max):
        if has_authored_solids:
            return (
                jnp.concatenate((authored_min, obstacle_min), axis=0),
                jnp.concatenate((authored_max, obstacle_max), axis=0),
            )
        return obstacle_min, obstacle_max

    obstacle_z_min = cfg.obstacle_spawn_layer_min * building.cell_size_m
    obstacle_z_max = min(cfg.obstacle_spawn_layer_max * building.cell_size_m, float(world_size[2]))

    def layout_roadmap(obstacle_min, obstacle_max):
        """Build from the full layout, not only its generated cuboids.

        Future generated buildings can use a factory per shared layout, or
        supply their varying solids through these layout-specific arrays.
        """
        if not plan_geodesic or obstacle_min.shape[0] == 0:
            return jnp.zeros((0, 3), dtype=jnp.float32), jnp.zeros((0, 0), dtype=jnp.float32)
        solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
        clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
        if cfg.roadmap_merge_walls:
            vertices = jnp.concatenate((authored_vertices,
                jnp.clip(roadmap_vertices(obstacle_min, obstacle_max, clearance), lower, upper)))
        else:
            vertices = jnp.clip(roadmap_vertices(solid_min, solid_max, clearance), lower, upper)
        distances = roadmap_distances(vertices, solid_min, solid_max, clearance)
        return vertices, distances

    def build_layout(key):
        obstacle_min, obstacle_max = generate_obstacles(
            key, world_size, count=cfg.num_obstacles,
            size_min=cfg.obstacle_size_min_m, size_max=cfg.obstacle_size_max_m,
            z_min=obstacle_z_min, z_max=obstacle_z_max,
            boundary_buffer=cfg.obstacle_boundary_buffer_m,
        )
        if cfg.num_obstacles and plan_geodesic:
            vertices, distances = layout_roadmap(obstacle_min, obstacle_max)
        else:
            vertices = jnp.zeros((0, 3), dtype=jnp.float32)
            distances = jnp.zeros((0, 0), dtype=jnp.float32)
        return obstacle_min, obstacle_max, vertices, distances

    def sample_target(key, obstacle_min, obstacle_max):
        """Sample allowed interior volume, rejecting buffered solid geometry."""
        solid_min, solid_max = all_solids(obstacle_min, obstacle_max)

        def draw(draw_key):
            box_key, point_key = jax.random.split(draw_key)
            box = jax.random.categorical(box_key, jnp.log(spawn_volumes))
            return jax.random.uniform(
                point_key, (3,), minval=spawn_lower[box], maxval=spawn_upper[box]
            )

        key, first_key = jax.random.split(key)
        first = draw(first_key)

        def condition(carry):
            iteration, _, point = carry
            invalid_obstacle = points_inside_aabbs(
                point, solid_min, solid_max, cfg.obstacle_target_buffer_m
            )
            return (iteration < 64) & invalid_obstacle

        def retry(carry):
            iteration, retry_key, _ = carry
            retry_key, draw_key = jax.random.split(retry_key)
            return iteration + 1, retry_key, draw(draw_key)

        point = jax.lax.while_loop(condition, retry, (0, key, first))[2]
        fallback_candidates = (spawn_lower + spawn_upper) / 2
        fallback_valid = ~points_inside_aabbs(
            fallback_candidates, solid_min, solid_max,
            cfg.obstacle_target_buffer_m,
        )
        fallback_index = jnp.argmax(
            jnp.where(fallback_valid, fallback_candidates[:, 2], -jnp.inf)
        )
        point_invalid = points_inside_aabbs(
            point, solid_min, solid_max, cfg.obstacle_target_buffer_m
        )
        return jnp.where(point_invalid, fallback_candidates[fallback_index], point)

    def connectivity(pos, active, base_pos, target_pos, obstacle_min, obstacle_max):
        solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
        delta = pos[:, None, :] - pos[None, :, :]
        distances = jnp.linalg.norm(delta, axis=-1)
        agent_adj = (
            (distances <= cfg.comm_radius)
            & active[:, None]
            & active[None, :]
            & ~jnp.eye(n, dtype=jnp.bool_)
        )
        if cfg.num_obstacles or has_authored_solids:
            agent_adj &= ~segments_blocked(
                pos[:, None, :], pos[None, :, :], solid_min, solid_max
            )
        base_edges = (jnp.linalg.norm(pos - base_pos, axis=-1) <= cfg.comm_radius_base) & active
        sees = (jnp.linalg.norm(pos - target_pos, axis=-1) <= cfg.visual_radius) & active
        if cfg.num_obstacles or has_authored_solids:
            base_edges &= ~segments_blocked(pos, base_pos, solid_min, solid_max)
            sees &= ~segments_blocked(pos, target_pos, solid_min, solid_max)

        reach = agent_adj | jnp.eye(n, dtype=jnp.bool_)
        for _ in range(n):
            reach = (reach.astype(jnp.int32) @ reach.astype(jnp.int32)) > 0
        conn_base = jnp.any(reach & base_edges[None, :], axis=1) & active
        conn_target = jnp.any(reach & sees[None, :], axis=1) & active
        return sees, conn_base, conn_target, agent_adj

    def update_coverage(coverage, pos, active, obstacle_min, obstacle_max):
        solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
        delta = coverage_centres[None, ...] - pos[:, None, None, None, :]
        visible = jnp.linalg.norm(delta, axis=-1) <= cfg.visual_radius
        occupied = jnp.clip(
            jnp.floor(pos / coverage_voxel_size).astype(jnp.int32),
            0,
            jnp.asarray(coverage_shape) - 1,
        )
        visible |= jnp.all(
            coverage_indices[None, ...] == occupied[:, None, None, None, :],
            axis=-1,
        )
        visible &= active[:, None, None, None]
        visible &= coverage_interior[None, ...]
        if cfg.num_obstacles or has_authored_solids:
            visible &= ~segments_blocked(
                pos[:, None, None, None, :], coverage_centres[None, ...],
                solid_min, solid_max,
            )
        newly_covered = ~coverage & jnp.any(visible, axis=0)
        viewers = jnp.sum(visible, axis=0)
        credit_per_voxel = newly_covered / jnp.maximum(viewers, 1)
        credit = jnp.sum(visible * credit_per_voxel[None, ...], axis=(1, 2, 3))
        return coverage | newly_covered, credit

    def reset(
        key,
        target_pos: jax.Array | None = None,
        obstacle_min: jax.Array | None = None,
        obstacle_max: jax.Array | None = None,
        stored_vertices: jax.Array | None = None,
        stored_distances: jax.Array | None = None,
        stored_corner_distances: jax.Array | None = None,
    ):
        """Reset an episode, optionally at one validated external target point.

        The optional target is used by checkpoint inspection: it lets a replay
        reproduce one target sampled by a previous parallel evaluation without
        changing the normal keyed target sampler used for training.
        """
        layout_key, target_key, next_key = jax.random.split(key, 3)
        if obstacle_min is None:
            obstacle_min, obstacle_max, stored_vertices, stored_distances = build_layout(layout_key)
        elif obstacle_min.shape[0] and plan_geodesic:
            expected_vertices = (authored_vertices.shape[0] + 8 * obstacle_min.shape[0]
                                 if cfg.roadmap_merge_walls else
                                 8 * (authored_min.shape[0] + obstacle_min.shape[0]))
            if (stored_vertices is None or stored_distances is None
                    or stored_vertices.shape != (expected_vertices, 3)
                    or stored_distances.shape != (expected_vertices, expected_vertices)):
                stored_vertices, stored_distances = layout_roadmap(obstacle_min, obstacle_max)
                stored_corner_distances = None
        else:
            stored_vertices = jnp.zeros((0, 3), dtype=jnp.float32)
            stored_distances = jnp.zeros((0, 0), dtype=jnp.float32)
        target = sample_target(target_key, obstacle_min, obstacle_max) if target_pos is None else jnp.asarray(target_pos)
        if minimum_geodesic_separation and obstacle_min.shape[0] and cfg.roadmap_corner_bonus_m:
            if stored_corner_distances is None or stored_corner_distances.shape != stored_distances.shape:
                solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
                stored_corner_distances = roadmap_distances(
                    stored_vertices, solid_min, solid_max,
                    cfg.drone_radius + cfg.obstacle_planning_clearance_m,
                    cfg.roadmap_corner_bonus_m, corner_merge_distance)
        else:
            stored_corner_distances = jnp.zeros((0, 0), dtype=jnp.float32)
        base_pos = jnp.asarray(building.base_position_m)
        pair_attempts = jnp.int32(1)
        if randomize_base or minimum_geodesic_separation:
            solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
            vertices = jnp.asarray(geometry.vertices) if cfg.num_obstacles == 0 else stored_vertices
            distances = (jnp.asarray(geometry.corner_distances) if cfg.num_obstacles == 0 else
                         stored_corner_distances if minimum_geodesic_separation and cfg.roadmap_corner_bonus_m else stored_distances)
            fixed_target = target
            if minimum_geodesic_separation:
                margin = geometry.clearance - 1e-4
                planning_min, planning_max = solid_min - margin, solid_max + margin
                # One endpoint is fixed throughout rejection sampling. Attach
                # it and propagate through the graph once, not on every retry.
                fixed_point = fixed_target if randomize_base else base_pos
                fixed_cost = _roadmap_attachment(fixed_point[None], vertices, planning_min, planning_max)
                fixed_routes = _route_costs(
                    fixed_cost + cfg.roadmap_corner_bonus_m,
                    distances.T if randomize_base else distances)[0]

            def draw_pair(draw_key):
                base_key, point_key, target_key = jax.random.split(draw_key, 3)
                base = base_pos
                if randomize_base:
                    box = jax.random.categorical(base_key, jnp.log(base_volumes))
                    base = jax.random.uniform(point_key, (3,), minval=base_lower[box], maxval=base_upper[box])
                # Randomized bases retry against the same episode target.
                # With a fixed base, retry targets instead.
                target = fixed_target if randomize_base else sample_target(target_key, obstacle_min, obstacle_max)
                drone = base + jnp.asarray([0.0, 0.0, cfg.drone_radius])
                valid = (~points_inside_aabbs(base, solid_min, solid_max, cfg.drone_radius)
                         & ~points_inside_aabbs(drone, solid_min, solid_max, cfg.drone_radius)
                         & jnp.all((drone >= lower) & (drone <= upper))) if randomize_base else jnp.bool_(True)
                valid &= ~points_inside_aabbs(target, solid_min, solid_max, cfg.obstacle_target_buffer_m)
                if minimum_geodesic_separation:
                    distance = jnp.linalg.norm(base - target)
                    distance = jnp.where(segments_blocked(base, target, planning_min, planning_max), 1e6, distance)
                    if vertices.shape[0]:
                        moving_point = base if randomize_base else target
                        moving_cost = _roadmap_attachment(moving_point[None], vertices, planning_min, planning_max)[0]
                        distance = jnp.minimum(distance, jnp.min(moving_cost + fixed_routes))
                    valid &= (distance < 1e6) & (distance >= minimum_geodesic_separation_multiplier * cfg.comm_radius)
                return base, target, valid

            pair_key, first_key = jax.random.split(target_key)
            first_base, first_target, valid = draw_pair(first_key)

            def retry_pair(carry):
                count, key, _, _, _ = carry
                key, draw_key = jax.random.split(key)
                base, target, valid = draw_pair(draw_key)
                return count + 1, key, base, target, valid

            pair_attempts, _, base_pos, target, valid = jax.lax.while_loop(
                lambda carry: (~carry[4]) & (carry[0] < spawn_pair_max_attempts), retry_pair,
                (pair_attempts, pair_key, first_base, first_target, valid))

            # Zero marks exhausted sampling. Training checks this on its
            # existing host transfer, avoiding per-environment host callbacks.
            pair_attempts = jnp.where(valid, pair_attempts, jnp.int32(0))
        # The station itself sits on the floor; drone centres start one radius
        # above it so the initial state does not intersect the floor tile.
        drone_spawn = base_pos + jnp.asarray([0.0, 0.0, cfg.drone_radius])
        pos = jnp.broadcast_to(drone_spawn, (n, 3))
        active = jnp.arange(n) * cfg.spawn_delay <= 0
        coverage = jnp.zeros(coverage_shape, dtype=jnp.bool_)
        coverage, _ = update_coverage(coverage, pos, active, obstacle_min, obstacle_max)
        sees, conn_base, conn_target, _ = connectivity(pos, active, base_pos, target, obstacle_min, obstacle_max)
        fully_connected = jnp.any(conn_base & conn_target)
        return EnvState(
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
            obstacle_min=obstacle_min,
            obstacle_max=obstacle_max,
            roadmap_vertices=stored_vertices,
            roadmap_distances=stored_distances,
            shared_geometry=geometry,
            spawn_pair_attempts=pair_attempts,
            roadmap_corner_distances=stored_corner_distances,
        )

    def step(state: EnvState, action: jax.Array):
        next_step = state.step + 1
        active = next_step >= jnp.arange(n) * cfg.spawn_delay
        force = jnp.clip(action, -1.0, 1.0) * cfg.max_force
        # Newly spawned agents first choose an action from their next observation.
        force = jnp.where(state.active[:, None], force, 0.0)
        velocity = state.vel * cfg.drag + force * cfg.dt
        speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= jnp.minimum(1.0, cfg.max_speed / jnp.maximum(speed, 1e-8))
        proposed = state.pos + velocity * cfg.dt
        collided = (proposed < lower) | (proposed > upper)
        if cfg.num_obstacles or has_authored_solids:
            solid_min, solid_max = all_solids(state.obstacle_min, state.obstacle_max)
            motion_min = solid_min - cfg.drone_radius
            motion_max = solid_max + cfg.drone_radius
            obstacle_collision = segments_blocked(state.pos, proposed, motion_min, motion_max)
            collided |= obstacle_collision[:, None]
            proposed = jnp.where(obstacle_collision[:, None], state.pos, proposed)
        pos = jnp.clip(proposed, lower, upper)
        velocity = jnp.where(collided, 0.0, velocity)
        drone_spawn = state.base_pos + jnp.asarray([0.0, 0.0, cfg.drone_radius])
        pos = jnp.where(active[:, None], pos, drone_spawn)
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
        coverage, coverage_credit = update_coverage(
            state.coverage, pos, active, state.obstacle_min, state.obstacle_max
        )
        sees, conn_base, conn_target, _ = connectivity(
            pos, active, state.base_pos, state.target_pos,
            state.obstacle_min, state.obstacle_max,
        )
        known = state.target_known | conn_target
        fully_connected = jnp.any(conn_base & conn_target)
        base_target_known = state.base_target_known | fully_connected
        chain_held_steps = jnp.where(
            fully_connected,
            state.chain_held_steps + jnp.int32(1),
            jnp.int32(0),
        )
        chain_success = chain_held_steps >= jnp.int32(cfg.hold_chain_for)
        target_found_or_delivered = (
            base_target_known if target_found_requires_delivery else jnp.any(known)
        )
        target_success = (
            jnp.bool_(cfg.success_when_target_found_or_delivered)
            & target_found_or_delivered
        )
        success = chain_success | target_success
        done = success | idle_terminated | (next_step >= jnp.int32(cfg.max_steps))
        return EnvState(
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
            obstacle_min=state.obstacle_min,
            obstacle_max=state.obstacle_max,
            roadmap_vertices=state.roadmap_vertices,
            roadmap_distances=state.roadmap_distances,
            shared_geometry=state.shared_geometry,
            spawn_pair_attempts=state.spawn_pair_attempts,
            roadmap_corner_distances=state.roadmap_corner_distances,
        )

    def observations(state: EnvState):
        pos = state.pos
        if cfg.observe_chain_contributor:
            if allow_redundancy_reward:
                paths = simple_chain_paths(state, cfg)
                contributors = jnp.any(paths.members & (paths.counts > 0)[:, None], axis=0)
            else:
                contributors = _contributing_chain_agents(state, cfg)
            chain_contributor = state.target_known & state.fully_connected & contributors
        delta = pos[:, None, :] - pos[None, :, :]
        pair_dist = jnp.linalg.norm(delta, axis=-1)
        _, _, _, agent_adj = connectivity(
            pos, state.active, state.base_pos, state.target_pos,
            state.obstacle_min, state.obstacle_max,
        )

        def one_agent(i):
            origin = pos[i]
            positive = jnp.where(directions > 0, (upper - origin) / directions, jnp.inf)
            negative = jnp.where(directions < 0, (lower - origin) / directions, jnp.inf)
            wall_distance = jnp.min(jnp.minimum(positive, negative), axis=-1)
            if cfg.num_obstacles or has_authored_solids:
                solid_min, solid_max = all_solids(state.obstacle_min, state.obstacle_max)
                ray_end = origin[None, :] + directions * cfg.visual_radius
                lo = jnp.zeros(cfg.radar_bins, dtype=jnp.float32)
                hi = jnp.ones(cfg.radar_bins, dtype=jnp.float32)
                hit = segments_blocked(
                    jnp.broadcast_to(origin, ray_end.shape), ray_end,
                    solid_min, solid_max,
                )
                def refine_radar(_, interval):
                    lo, hi = interval
                    mid = (lo + hi) / 2
                    mid_point = origin[None, :] + directions * (mid * cfg.visual_radius)[:, None]
                    blocked = segments_blocked(
                        jnp.broadcast_to(origin, mid_point.shape), mid_point,
                        solid_min, solid_max,
                    )
                    hi = jnp.where(blocked, mid, hi)
                    lo = jnp.where(blocked, lo, mid)
                    return lo, hi

                # Preserve all eight bisections without tracing eight copies
                # of the wall-intersection calculation into every rollout.
                lo, hi = jax.lax.fori_loop(0, 8, refine_radar, (lo, hi))
                wall_distance = jnp.minimum(
                    wall_distance,
                    jnp.where(hit, hi * cfg.visual_radius, jnp.inf),
                )
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
            if cfg.observe_chain_contributor:
                optional.append(jnp.asarray([chain_contributor[i]], dtype=jnp.float32))
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
            if cfg.observe_current_timestep:
                optional.append(jnp.asarray([state.step / cfg.max_steps], dtype=jnp.float32))
            observation = jnp.concatenate([self_state, *optional, radar])
            return jnp.where(state.active[i], observation, 0.0)

        del agent_adj  # reserved for future LOS-aware radar filtering
        return jax.vmap(one_agent)(jnp.arange(n))

    def metrics(state: EnvState):
        return {
            "coverage_fraction": jnp.sum(state.coverage & coverage_interior)
            / jnp.maximum(jnp.sum(coverage_interior), 1),
            "active_agents": jnp.sum(state.active),
            "target_seen": jnp.any(state.directly_sees_target),
            "success": state.success,
            "fully_connected": state.fully_connected,
            "base_target_known": state.base_target_known,
            "stationary_steps": state.stationary_steps,
            "idle_terminated": state.idle_terminated,
            "done": state.done,
        }

    # Replay/evaluation prepare persisted layouts through the same geometry path.
    reset.build_roadmap = layout_roadmap
    return reset, step, observations, metrics


def make_autoreset_fns(
    building: BuildingArrays,
    cfg: EnvConfig,
    reward_cfg: RewardConfig = RewardConfig(),
    *, randomize_base: bool = False, minimum_geodesic_separation: bool = False,
    minimum_geodesic_separation_multiplier: float = 2.0,
    spawn_pair_max_attempts: int = 1024,
):
    """Return reset and terminal-aware step functions for batched training.

    Terminal rewards and diagnostic state describe the completed transition;
    only the returned carry state is replaced by a fresh episode.
    """
    if not math.isfinite(reward_cfg.chain_efficiency_bonus) or reward_cfg.chain_efficiency_bonus < 0:
        raise ValueError("chain_efficiency_bonus must be finite and nonnegative.")
    if minimum_geodesic_separation and reward_cfg.chain_reward_system != "obstacle_geodesic":
        raise ValueError("Minimum geodesic separation requires obstacle_geodesic chain reward.")
    reset, step, observations, metrics = make_env_fns(
        building, cfg, plan_geodesic=reward_cfg.chain_reward_system == "obstacle_geodesic" or reward_cfg.enable_chain_efficiency_reward,
        randomize_base=randomize_base, minimum_geodesic_separation=minimum_geodesic_separation,
        minimum_geodesic_separation_multiplier=minimum_geodesic_separation_multiplier,
        spawn_pair_max_attempts=spawn_pair_max_attempts,
        allow_redundancy_reward=reward_cfg.allow_redundancy_reward,
        target_found_requires_delivery=reward_cfg.target_found_requires_delivery,
    )

    def transition(state: EnvState, action: jax.Array):
        terminal_state = step(state, action)
        distances = (chain_distance_matrix(terminal_state)
                     if reward_cfg.chain_reward_system == "obstacle_geodesic" or reward_cfg.enable_chain_efficiency_reward else None)
        paths = (simple_chain_paths(terminal_state, cfg)
                 if reward_cfg.allow_redundancy_reward or reward_cfg.enable_chain_efficiency_reward else None)
        reward, reward_terms = compute_rewards(state, terminal_state, reward_cfg, cfg, distances, paths)
        if reward_cfg.chain_reward_system == "obstacle_geodesic":
            chain_gap_dist, chain_progress_pct, route_excess, leaders = obstacle_chain_diagnostics(terminal_state, distances)
        else:
            chain_gap_dist, chain_progress_pct = chain_diagnostics(terminal_state)
            route_excess = jnp.float32(0.0)
            leaders = jnp.asarray([-1, -1], dtype=jnp.int32)
        info = {
            **reward_terms,
            "done": terminal_state.done,
            "idle_terminated": terminal_state.idle_terminated,
            # Keep the reward component distinct from the terminal-success
            # flag so training telemetry can accumulate the actual bonus.
            "reward_success": reward_terms["success"],
            "success": terminal_state.success,
            "fully_connected": terminal_state.fully_connected,
            "global_target_found": (
                terminal_state.base_target_known
                if reward_cfg.target_found_requires_delivery
                else jnp.any(terminal_state.target_known)
            ),
            "chain_gap_dist": chain_gap_dist,
            "chain_progress_pct": chain_progress_pct,
            "chain_route_excess": route_excess,
            "chain_frontier_leaders": leaders,
            "global_coverage": jnp.mean(terminal_state.coverage),
            "terminal_target_pos": terminal_state.target_pos,
            "terminal_coverage_fraction": jnp.mean(terminal_state.coverage),
        }
        if paths is not None:
            info["number_of_valid_paths"] = jnp.sum(paths.counts)
            if reward_cfg.enable_chain_efficiency_reward:
                info["chain_efficiency"] = jnp.max(chain_path_efficiencies(paths, distances[0, 1]))
        return terminal_state, reward, terminal_state.done, info

    def reset_persisted(state):
        return reset(state.key, obstacle_min=state.obstacle_min, obstacle_max=state.obstacle_max,
                     stored_vertices=state.roadmap_vertices, stored_distances=state.roadmap_distances,
                     stored_corner_distances=state.roadmap_corner_distances)

    def autoreset_step(state: EnvState, action: jax.Array):
        """Original scalar API retained as the dense equivalence reference."""
        terminal, reward, done, info = transition(state, action)
        fresh = reset_persisted(terminal)
        result = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), fresh, terminal)
        return result, reward, done, info

    def batched_step(states, actions):
        terminal, rewards, done, info = jax.vmap(transition)(states, actions)
        size = done.shape[0]
        count = jnp.sum(done, dtype=jnp.int32)

        def dense(current):
            fresh = jax.vmap(reset_persisted)(current)
            return jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((size,) + (1,) * (a.ndim - 1)), a, b),
                fresh, current,
            )

        def sparse(current):
            block = min(32, size)
            indices = jnp.nonzero(done, size=size, fill_value=size)[0]
            indices = jnp.pad(indices, (0, block - 1), constant_values=size)

            def reset_chunk(index, result):
                lanes = jax.lax.dynamic_slice_in_dim(indices, index * block, block)
                selected = jax.tree_util.tree_map(lambda a: a[jnp.minimum(lanes, size - 1)], current)
                fresh = jax.vmap(reset_persisted)(selected)
                # Layout and roadmap are invariant across episode reset. Avoid
                # scattering potentially large matrices back into their copies.
                unchanged = {"obstacle_min", "obstacle_max", "roadmap_vertices", "roadmap_corner_distances",
                             "roadmap_distances", "shared_geometry"}
                return result._replace(**{
                    field: getattr(result, field).at[lanes].set(getattr(fresh, field), mode="drop")
                    for field in result._fields if field not in unchanged
                })

            return jax.lax.fori_loop(0, (count + block - 1) // block, reset_chunk, current)

        # This conditional is outside vmap. No reset is executed when count=0;
        # dense reset remains efficient for synchronized episode endings.
        # Rejection sampling in a dense vmap runs until the slowest of all
        # lanes accepts. Keep it in small groups even at synchronized endings.
        result = (sparse(terminal) if randomize_base or minimum_geodesic_separation else
                  jax.lax.cond(count > size // 2, dense, sparse, terminal))
        return result, rewards, done, info

    autoreset_step.batched = batched_step

    return reset, autoreset_step, observations, metrics
