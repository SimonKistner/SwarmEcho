"""Pure-JAX environment, observations, relay rewards, and episode state."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from swarmecho.env import decommissioning as adaptive_population
from swarmecho.env.knowledge import information_counts, new_information_counts
from swarmecho.core.config import CHAIN_SUCCESS_CONDITIONS, DEPLOYED_AGENTS_OBS_REF, SUCCESS_CONDITIONS

from swarmecho.env.buildings import BuildingArrays
from swarmecho.env.obstacles import (
    _roadmap_attachment,
    _route_costs,
    building_candidate_vertices,
    validate_roadmap_settings,
    free_space_distance,
    pairwise_free_space_distance,
    points_inside_aabbs,
    roadmap_distances,
    roadmap_vertices,
    segments_blocked,
    SharedRoadmap,
    shared_roadmap,
    building_roadmap,
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
    base_target_connected: jax.Array
    chain_held_steps: jax.Array
    done: jax.Array
    collided: jax.Array
    coverage_credit: jax.Array
    stationary_steps: jax.Array = jnp.int32(0)
    best_chain_held_steps: jax.Array = jnp.int32(0)
    # Credited gap progress when target delivery first opened the reward gate.
    # Receiver-by-sender attribution for newly shared target knowledge.
    peer_informed_by: jax.Array = jnp.zeros((0, 0), dtype=jnp.bool_)
    # This step's recipient counts [all, peer, base], independent of sender credit.
    new_information_counts: jax.Array = jnp.zeros((3,), dtype=jnp.int32)
    # Persistent record of target delivery to the base by an informed drone
    # on a communication tick. A later chain break cannot revoke delivery.
    base_target_known: jax.Array = jnp.bool_(False)
    obstacle_min: jax.Array = jnp.zeros((0, 3), dtype=jnp.float32)
    obstacle_max: jax.Array = jnp.zeros((0, 3), dtype=jnp.float32)
    roadmap_vertices: jax.Array = jnp.zeros((0, 3), dtype=jnp.float32)
    roadmap_distances: jax.Array = jnp.zeros((0, 0), dtype=jnp.float32)
    shared_geometry: SharedRoadmap | None = None
    spawn_pair_attempts: jax.Array = jnp.int32(1)
    spawn_pair_target_retries: jax.Array = jnp.int32(0)
    map_id: jax.Array = jnp.int32(0)
    building_bank: object = None
    adaptive: adaptive_population.AdaptiveState | None = None
    # Accepted target's estimated relay requirement for evaluation.
    # Zero denotes an episode without dynamic relay-budget filtering.
    target_required_agents: jax.Array = jnp.int32(0)

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
        if self.building_bank is not None:
            return self.building_bank.get("vertices", self.map_id)
        if self.shared_geometry is not None and self.obstacle_min.shape[0] == 0:
            return jnp.asarray(self.shared_geometry.vertices)
        return self.roadmap_vertices

    @property
    def planning_distances(self):
        if self.building_bank is not None:
            return self.building_bank.get("distances", self.map_id)
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


from swarmecho.core.config import (
    EnvConfig, RewardConfig, maximum_chain_distance,
    validate_reward_config,
)


def build_obstacle_roadmap(obstacle_min, obstacle_max, cfg: EnvConfig):
    """Build the compact roadmap arrays for persisted/replayed obstacle bounds."""
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    vertices = roadmap_vertices(obstacle_min, obstacle_max, clearance)
    return vertices, roadmap_distances(vertices, obstacle_min, obstacle_max, clearance)


def spherical_directions(count: int) -> np.ndarray:
    """Return deterministic unit directions, with exact octants for count 8."""
    if count < 4:
        raise ValueError("Radar direction counts must be at least 4.")
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
    """Return the configured actor observation width."""
    # The retained band_sep_counting ablation changes the observation width.
    return (
        6
        + 3 * int(cfg.observe_visually_gated_target_displacement)
        + cfg.wall_sensing_rays
        + (cfg.drone_radar_bins * max(1, cfg.distance_bands) * 3
           if cfg.radar_mode == "band_sep_counting" else cfg.drone_radar_bins * 3)
        + int(cfg.observe_chain_contributor)
        + int(cfg.decommissioning.enabled) * (
            int(cfg.observe_decommission_zone) + int(cfg.observe_base_population)
        )
    )


def maximum_five_drone_chain_distance(cfg: EnvConfig) -> float:
    """Unobstructed base→D1→…→D5→target reach for five mobile drones."""
    if cfg.num_agents != 5:
        raise ValueError("maximum_five_drone_chain_distance requires num_agents=5.")
    return maximum_chain_distance(cfg)


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
    gap = jnp.where(state.base_target_connected, 0.0, gaps[idx_b, idx_t])
    progress = jnp.where(state.base_target_connected, 100.0, jnp.clip(100.0 * (1.0 - gap / jnp.maximum(mission, 1e-6)), 0.0, 100.0))
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
    return jnp.where(state.base_target_connected & (length < 1e6), length, 0.0)


def _contributing_chain_agents(
    state: EnvState,
    cfg: EnvConfig,
    *, return_graph: bool = False, frontier_leaders: jax.Array | None = None,
    progress_distances=None,
) -> jax.Array:
    """Select a shortest chain, or share credit through its equal-hop alternatives."""
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

    if progress_distances is None:
        return trace_path(base_node, target_node)

    # -1 denotes the base or target endpoint rather than a drone.
    has_base_chain = frontier_leaders[0] >= 0
    has_target_chain = frontier_leaders[1] >= 0
    base_tip = jnp.where(has_base_chain, frontier_leaders[0], base_node)
    target_tip = jnp.where(has_target_chain, frontier_leaders[1], target_node)
    # The matrix uses [base, target, agents]; the radio graph uses
    # [agents, base, target]. Reuse distances from frontier selection.
    order = jnp.concatenate((jnp.arange(n) + 2, jnp.asarray([0, 1])))
    progress_distances = progress_distances[order[:, None], order[None, :]]

    def soft_trace_path(start_node, destination_node, opposite_frontier):
        remaining = shortest[:, destination_node]
        length = shortest[start_node, destination_node]
        on_route = ((shortest[start_node, :] + remaining == length)
                    & (remaining < 9999) & (length < 9999))
        successors = ((hops == 1) & on_route[:, None] & on_route[None, :]
                      & (remaining[None, :] == remaining[:, None] - 1))
        # Pretend each successor is this side's frontier leader: smaller
        # gap to the opposite frontier means more closed route metres.
        # A communication radius is the fixed softmax scale: equal progress
        # splits equally, while better progress gets a larger, smooth share.
        gaps = progress_distances[:, opposite_frontier]
        gaps = jnp.where(jnp.isfinite(gaps) & (gaps < 1e6), gaps, 1e6)
        scores = -gaps / max(cfg.comm_radius, 1e-6)
        logits = jnp.where(successors, scores[None, :], -1e30)
        weights = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True)) * successors
        weights /= jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-20)
        initial = jnp.zeros(node_count, dtype=weights.dtype).at[start_node].set(1.0)

        def advance(_, carry):
            flow, visits = carry
            flow = flow @ weights
            return flow, visits + flow

        _, visits = jax.lax.fori_loop(0, node_count - 1, advance, (initial, initial))
        # At a split, credit is shared; at a merge it recombines. Thus each
        # independent patch has its own budget and mandatory links keep 1.
        return jnp.clip(visits[:n], 0.0, 1.0)

    opposite_target = jnp.where(has_target_chain, target_tip, target_node)
    opposite_base = jnp.where(has_base_chain, base_tip, base_node)
    base_path = jnp.where(has_base_chain, soft_trace_path(base_node, base_tip, opposite_target), 0.0)
    target_path = jnp.where(has_target_chain, soft_trace_path(target_node, target_tip, opposite_base), 0.0)
    complete_path = soft_trace_path(base_node, target_node, target_node)
    return jnp.where(state.base_target_connected, complete_path, jnp.maximum(base_path, target_path))

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
    return ChainPaths(jnp.where(state.base_target_connected, complete_lengths, jnp.inf),
                        jnp.where(state.base_target_connected, complete_counts, 0), members)


def chain_path_efficiencies(paths: ChainPaths, reference):
    return jnp.where(jnp.isfinite(paths.lengths) & jnp.isfinite(reference) & (reference < 1e6),
                     jnp.clip(reference / jnp.maximum(paths.lengths, 1e-6), 0., 1.), 0.)


def success_contributors(state, cfg):
    """Return selected relay membership when the chain is complete."""
    return jax.lax.cond(
        state.base_target_connected, lambda _: _contributing_chain_agents(state, cfg),
        lambda _: jnp.zeros_like(state.active), None,
    ) & state.active


def _return_progress_metres(previous, current):
    """Signed geodesic progress to the decommission-zone centre, with no cutoff."""
    distances = free_space_distance(
        current.adaptive.zone_center, jnp.concatenate((previous.pos, current.pos)),
        current.planning_vertices, current.planning_distances,
        current.planning_min, current.planning_max,
    )[0]
    before, after = jnp.split(distances, 2)
    valid = jnp.isfinite(before) & jnp.isfinite(after) & (before < 1e6) & (after < 1e6)
    return jnp.where(valid, before - after, 0.0)


def non_contributor_rewards(previous, current, cfg, env_cfg):
    """Movement guidance and a one-shot exit payment for surplus lifetimes.

    Membership is measured before the action, as in the actor observation.
    A surviving agent must remain surplus, and the chain must survive the
    transition. Removed agents use their prior membership, never the trivially
    false membership of an inactive/replaced slot.
    """
    zero = jnp.zeros_like(current.coverage_credit)
    empty = (zero, zero, zero)
    if not env_cfg.decommissioning.enabled or env_cfg.success_condition not in CHAIN_SUCCESS_CONDITIONS:
        return empty

    def pay(_):
        # On complete chains, frontier leaders cannot change membership.
        # Avoid redundant roadmap queries just to choose partial-chain tips.
        previous_members = success_contributors(previous, env_cfg)
        surplus = previous.active & ~previous_members
        returned = departed = zero
        if cfg.non_contributor_movement_enabled:
            current_members = success_contributors(current, env_cfg)
            continuing = (surplus & current.active & ~current_members
                          & ~current.adaptive.spawned & ~current.adaptive.decommissioned)

            def movement(_):
                progress = _return_progress_metres(previous, current)
                progress = jnp.where(continuing & current.adaptive.zone_valid, progress, 0.0)
                return (cfg.non_contributor_return_reward_per_meter * jnp.maximum(progress, 0.0),
                        cfg.non_contributor_departure_reward_per_meter * jnp.maximum(-progress, 0.0))

            returned, departed = jax.lax.cond(jnp.any(continuing), movement, lambda _: (zero, zero), None)
        removed = (surplus & current.adaptive.decommissioned
                   & (current.adaptive.decommissions > previous.adaptive.decommissions))
        decommissioned = cfg.non_contributor_decommission_reward * removed.astype(jnp.float32)
        return returned, departed, decommissioned

    return jax.lax.cond(
        previous.base_target_connected & current.base_target_connected & ~previous.done,
        pay, lambda _: empty, None,
    )


def compute_rewards(
    previous: EnvState,
    current: EnvState,
    cfg: RewardConfig,
    env_cfg: EnvConfig,
    geodesic_distances=None,
    chain_paths=None,
    chain_diagnostics_result=None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute individual exploration, delivery, relay and cleanup rewards.

    Base delivery and success are one-shot events. Delivery pays the direct
    reporters and active team separately. Relay shaping recurs each step.
    """
    n = current.pos.shape[0]
    reward_active = current.active.astype(jnp.float32)
    time_reward = reward_active * cfg.time_presure_reward
    newly_knows = current.target_known & ~previous.target_known
    success_event = current.success & ~previous.success
    target_found_event = (
        current.base_target_known & ~previous.base_target_known
    )
    dynamic_gap_enabled = current.base_target_known
    # A base-adjacent drone can deliver information it already held before
    # this transition. Reward credit is limited to a direct base neighbor.
    in_base_range = (
        jnp.linalg.norm(previous.pos - previous.base_pos[None, :], axis=-1)
        <= env_cfg.comm_radius_base
    ) & previous.active
    if previous.solid_min.shape[0]:
        in_base_range &= ~segments_blocked(
            previous.pos, previous.base_pos,
            previous.solid_min, previous.solid_max,
        )
    finder_receivers = in_base_range & previous.target_known

    if cfg.chain_efficiency_reward != 0 and geodesic_distances is None:
        geodesic_distances = chain_distance_matrix(current)

    def gap_reward_when_known(_):
        distances = geodesic_distances
        if distances is None:
            distances = chain_distance_matrix(current)
        diagnostics = (chain_diagnostics_result if chain_diagnostics_result is not None
                       else obstacle_chain_diagnostics(current, distances))
        gap_distance, _, _, frontier_leaders = diagnostics
        full_distance = distances[0, 1]
        is_contributing = _contributing_chain_agents(
            current, env_cfg, frontier_leaders=frontier_leaders,
            progress_distances=distances,
        )
        closed_metres = jnp.where(
            jnp.isfinite(full_distance) & (full_distance < 1e6),
            jnp.maximum(full_distance - gap_distance, 0.0), 0.0,
        )
        return is_contributing.astype(jnp.float32) * (cfg.closed_gap_reward_per_meter * closed_metres)

    gap_reward = (jax.lax.cond(dynamic_gap_enabled, gap_reward_when_known,
                              lambda _: jnp.zeros(n, dtype=jnp.float32), None)
                  if cfg.closed_gap_reward_per_meter != 0 else jnp.zeros(n, dtype=jnp.float32))
    success_reward = reward_active * cfg.success_reward * success_event

    if cfg.chain_efficiency_reward != 0 and chain_paths is None:
        chain_paths = simple_chain_paths(current, env_cfg)
    terms = {
        # Exploration ends for a drone once it knows the target.
        "exploration_reward": cfg.exploration_reward * current.coverage_credit
        * (~current.target_known).astype(jnp.float32),
        "collision_reward": cfg.collision_reward * current.collided.astype(jnp.float32),
        "base_reporter_reward": cfg.base_reporter_reward * (
            target_found_event & finder_receivers
        ).astype(jnp.float32),
        "target_delivery_reward": reward_active * cfg.target_delivery_reward * target_found_event,
        "time_presure_reward": time_reward,
        "gap_reward": gap_reward,
        "success_reward": success_reward,
    }
    if cfg.chain_efficiency_reward != 0:
        efficiency = chain_path_efficiencies(chain_paths, geodesic_distances[0, 1])
        best = jnp.argmin(chain_paths.lengths)
        credit = chain_paths.members[best] * efficiency[best]
        terms["chain_efficiency_reward"] = cfg.chain_efficiency_reward * credit
    # ``newly_knows`` is exposed for diagnostics without double-paying finders.
    terms["newly_knows"] = newly_knows.astype(jnp.float32)
    terms = {name: jnp.where(current.active, value, 0.0) for name, value in terms.items()}
    if env_cfg.decommissioning.enabled:
        a = current.adaptive
        acting = previous.active.astype(jnp.float32)
        terms.update({
            # Signed event rewards pay each recipient in full, even after removal.
            "time_presure_reward": cfg.time_presure_reward * acting,
            "all_decommissioned_penalty": cfg.all_decommissioned_penalty
                * a.all_decommissioned * acting,
        })
    if cfg.non_contributor_rewards_enabled:
        returned, departed, decommissioned = non_contributor_rewards(
            previous, current, cfg, env_cfg,
        )
        # Removal payments belong to the old lifetime, including a reused slot.
        terms.update(non_contributor_return_reward=returned, non_contributor_departure_reward=departed,
                     non_contributor_decommission_reward=decommissioned)
    total = sum(value for name, value in terms.items() if name != "newly_knows")
    return total, terms


def _target_spawn_boxes(building, cfg, *, include_excluded=False):
    from swarmecho.env.buildings import target_spawn_boxes
    return tuple(jnp.asarray(value) for value in target_spawn_boxes(building, cfg, include_excluded=include_excluded))

def make_env_fns(building: BuildingArrays, cfg: EnvConfig, *, plan_geodesic: bool = True,
                       randomize_base: bool = False,
                       dynamic_max_geo_distance: bool = False,
                       relay_options=None,
                       spawn_pair_max_attempts: int = 1024, building_bank=None):
    """Create pure reset, step, observation, and metric functions."""
    from swarmecho.core.config import validate_radar_config
    from swarmecho.env.buildings import validate_feature_clearance
    validate_radar_config(cfg)
    validate_feature_clearance(building, cfg.drone_radius + cfg.obstacle_planning_clearance_m)
    if building_bank is not None:
        building = building_bank.envelope
    if cfg.num_agents < 1:
        raise ValueError("num_agents must be positive.")
    from swarmecho.core.config import validate_decommissioning
    validate_decommissioning(cfg)
    if cfg.success_condition not in SUCCESS_CONDITIONS:
        raise ValueError("Unknown env.success_condition.")
    if dynamic_max_geo_distance and min(cfg.comm_radius_base, cfg.comm_radius, cfg.visual_radius) <= 0:
        raise ValueError("Dynamic max distance requires positive communication and visual ranges.")
    plan_geodesic = plan_geodesic or dynamic_max_geo_distance
    if cfg.spawn_delay < 0:
        raise ValueError("spawn_delay must be non-negative.")
    if cfg.hold_chain_for < 1:
        raise ValueError("hold_chain_for must be at least 1.")
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if cfg.movement_epsilon < 0:
        raise ValueError("movement_epsilon must be non-negative.")
    if cfg.drone_radius <= 0:
        raise ValueError("drone_radius must be positive.")
    coverage_voxel_size, coverage_shape = coverage_grid_geometry(building, cfg)
    wall_directions = jnp.asarray(spherical_directions(cfg.wall_sensing_rays))
    drone_directions = jnp.asarray(spherical_directions(cfg.drone_radar_bins))
    distance_bands = max(1, cfg.distance_bands) if cfg.radar_mode == "band_sep_counting" else 1
    band_edges = jnp.arange(1, distance_bands, dtype=jnp.float32) * (cfg.comm_radius / distance_bands)
    spawn_lower, spawn_upper, spawn_volumes = _target_spawn_boxes(building, cfg)
    validate_roadmap_settings(cfg)
    if randomize_base or dynamic_max_geo_distance:
        if isinstance(spawn_pair_max_attempts, bool) or not isinstance(spawn_pair_max_attempts, int) or not 1 <= spawn_pair_max_attempts <= np.iinfo(np.int32).max:
            raise ValueError("spawn_pair_max_attempts must be a positive int32 integer.")
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
    def map_world_size(map_id):
        return world_size if building_bank is None else building_bank.get("world_size", map_id)

    def map_upper(map_id):
        return map_world_size(map_id) - lower
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
    coverage_eligible = building.interior_cells[
        tuple(coverage_cell_indices.reshape(-1, 3).T)
    ].reshape(coverage_shape)
    if len(building.solid_min_m):
        centres_np = np.asarray(coverage_centres)
        inside_solid = np.any(
            np.all(
                (centres_np[..., None, :] >= building.solid_min_m)
                & (centres_np[..., None, :] <= building.solid_max_m),
                axis=-1,
            ),
            axis=-1,
        )
        coverage_eligible &= ~inside_solid
    coverage_interior = jnp.asarray(coverage_eligible)
    def coverage_mask(map_id):
        return (coverage_interior if building_bank is None
                else building_bank.get("coverage_eligible", map_id))

    def coverage_fraction(coverage, map_id):
        eligible = coverage_mask(map_id)
        return jnp.sum(coverage & eligible) / jnp.maximum(jnp.sum(eligible), 1)
    authored_min = jnp.asarray(building.solid_min_m, dtype=jnp.float32)
    authored_max = jnp.asarray(building.solid_max_m, dtype=jnp.float32)
    has_authored_solids = bool(building.solid_min_m.shape[0]) or building_bank is not None
    geometry = building_roadmap(building, cfg, plan=plan_geodesic)
    relay_calculator = None
    if dynamic_max_geo_distance:
        from swarmecho.env.relay import make_calculator
        relay_calculator = make_calculator(cfg, geometry, building, building_bank,
                                           options=relay_options, randomize_base=randomize_base)

    def all_solids(obstacle_min, obstacle_max):
        if has_authored_solids:
            return (
                jnp.concatenate((authored_min, obstacle_min), axis=0),
                jnp.concatenate((authored_max, obstacle_max), axis=0),
            )
        return obstacle_min, obstacle_max

    def decommission_zone(base_pos, obstacle_min, obstacle_max, map_id):
        # Place the zone directly above the base. The station is on the floor
        # and drones spawn one radius above it, so lift the zone by one drone
        # radius beyond the decommission sphere radius to keep the initial
        # drone body outside the zone.
        radius = cfg.decommissioning.decommission_radius_m
        spawn_z = jnp.maximum(base_pos[2] + cfg.drone_radius, lower[2])
        zone_z = spawn_z + radius + cfg.drone_radius
        candidates = jnp.stack([base_pos[0], base_pos[1], zone_z])[None, :]
        valid = jnp.all((candidates - radius >= lower - 1e-5)
                         & (candidates + radius <= map_upper(map_id) + 1e-5), axis=-1)
        solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
        if solid_min.shape[0]:
            closest = jnp.clip(candidates[:, None, :], solid_min, solid_max)
            valid &= ~jnp.any(jnp.linalg.norm(candidates[:, None, :] - closest, axis=-1)
                              < radius + cfg.drone_radius - 1e-5, axis=-1)
            spawn = base_pos + jnp.asarray([0., 0., cfg.drone_radius])
            valid &= ~segments_blocked(spawn, candidates,
                                       solid_min - cfg.drone_radius, solid_max + cfg.drone_radius)
        index = jnp.argmax(valid)
        return candidates[index], jnp.any(valid)

    def layout_roadmap(obstacle_min, obstacle_max):
        """Build from the full layout, not only its generated cuboids.

        Future generated buildings can use a factory per shared layout, or
        supply their varying solids through these layout-specific arrays.
        """
        if not plan_geodesic or obstacle_min.shape[0] == 0:
            return jnp.zeros((0, 3), dtype=jnp.float32), jnp.zeros((0, 0), dtype=jnp.float32)
        solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
        clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
        authored_vertices = jnp.asarray(building_candidate_vertices(building, cfg))
        vertices = jnp.concatenate((authored_vertices,
            jnp.clip(roadmap_vertices(obstacle_min, obstacle_max, clearance), lower, upper)))
        distances = roadmap_distances(vertices, solid_min, solid_max, clearance)
        return vertices, distances

    def target_inside_buffered_solids(point, obstacle_min, obstacle_max):
        """Use wall clearance for building solids and obstacle clearance for cuboids."""
        wall_buffer = cfg.target_wall_buffer_fraction * building.cell_size_m
        if building_bank is not None:
            return points_inside_aabbs(point, obstacle_min, obstacle_max, wall_buffer)
        return (points_inside_aabbs(point, authored_min, authored_max, wall_buffer)
                | points_inside_aabbs(point, obstacle_min, obstacle_max, 0.5))

    def sample_target(key, obstacle_min, obstacle_max, map_id=0, target_spawn_weights=None):
        """Sample allowed interior volume, rejecting buffered solid geometry."""
        target_lower, target_upper, target_volumes = spawn_lower, spawn_upper, spawn_volumes
        if building_bank is not None:
            target_lower, target_upper, target_volumes = (building_bank.get("target_" + suffix, map_id)
                                                         for suffix in ("lower", "upper", "volumes"))
        # None preserves the original volume sampler, including its keyed draws.
        weights = target_volumes if target_spawn_weights is None else target_spawn_weights
        if target_spawn_weights is not None and building_bank is not None:
            # A single dynamic [map, padded box] table is shared by every lane.
            # Map selection happens on device; padding is never spawnable.
            row = target_spawn_weights[map_id] if target_spawn_weights.ndim == 2 else target_spawn_weights
            weights = jnp.where(target_volumes > 0, row, 0.0)

        def draw(draw_key):
            box_key, point_key = jax.random.split(draw_key)
            box = jax.random.categorical(box_key, jnp.log(weights))
            return jax.random.uniform(
                point_key, (3,), minval=target_lower[box], maxval=target_upper[box]
            )

        key, first_key = jax.random.split(key)
        first = draw(first_key)

        def condition(carry):
            iteration, _, point = carry
            invalid_obstacle = target_inside_buffered_solids(point, obstacle_min, obstacle_max)
            return (iteration < 64) & invalid_obstacle

        def retry(carry):
            iteration, retry_key, _ = carry
            retry_key, draw_key = jax.random.split(retry_key)
            return iteration + 1, retry_key, draw(draw_key)

        point = jax.lax.while_loop(condition, retry, (0, key, first))[2]
        fallback_candidates = (target_lower + target_upper) / 2
        fallback_valid = ~target_inside_buffered_solids(fallback_candidates, obstacle_min, obstacle_max)
        fallback_valid &= target_volumes > 0
        fallback_index = jnp.argmax(
            jnp.where(fallback_valid, fallback_candidates[:, 2], -jnp.inf)
        )
        if target_spawn_weights is not None:
            # Keep the rare geometry fallback consistent with adaptive weights.
            fallback_index = jax.random.categorical(
                jax.random.fold_in(key, 1), jnp.where(fallback_valid, jnp.log(weights), -jnp.inf)
            )
        point_invalid = target_inside_buffered_solids(point, obstacle_min, obstacle_max)
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
        if obstacle_min.shape[0] or has_authored_solids:
            agent_adj &= ~segments_blocked(
                pos[:, None, :], pos[None, :, :], solid_min, solid_max
            )
        base_edges = (jnp.linalg.norm(pos - base_pos, axis=-1) <= cfg.comm_radius_base) & active
        sees = (jnp.linalg.norm(pos - target_pos, axis=-1) <= cfg.visual_radius) & active
        if obstacle_min.shape[0] or has_authored_solids:
            base_edges &= ~segments_blocked(pos, base_pos, solid_min, solid_max)
            sees &= ~segments_blocked(pos, target_pos, solid_min, solid_max)

        reach = agent_adj | jnp.eye(n, dtype=jnp.bool_)
        for _ in range(n):
            reach = (reach.astype(jnp.int32) @ reach.astype(jnp.int32)) > 0
        conn_base = jnp.any(reach & base_edges[None, :], axis=1) & active
        conn_target = jnp.any(reach & sees[None, :], axis=1) & active
        return sees, conn_base, conn_target, agent_adj

    def knowledge_links(pos, active, base_pos, obstacle_min, obstacle_max):
        """Direct links for one synchronous store-and-forward communication tick."""
        delta = pos[:, None, :] - pos[None, :, :]
        peer = (
            (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
            & active[:, None] & active[None, :]
            & ~jnp.eye(n, dtype=jnp.bool_)
        )
        base = (jnp.linalg.norm(pos - base_pos, axis=-1) <= cfg.comm_radius_base) & active
        if obstacle_min.shape[0] or has_authored_solids:
            solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
            peer &= ~segments_blocked(pos[:, None, :], pos[None, :, :], solid_min, solid_max)
            base &= ~segments_blocked(pos, base_pos, solid_min, solid_max)
        return peer, base

    def update_coverage(coverage, pos, active, obstacle_min, obstacle_max, map_id):
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
        visible &= coverage_mask(map_id)[None, ...]
        if obstacle_min.shape[0] or has_authored_solids:
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
        map_id=0,
        target_spawn_weights: jax.Array | None = None,
        target_agent_budget: jax.Array | None = None,
        target_min_agents: jax.Array | None = None,
    ):
        """Reset an episode, optionally at one validated external target point.

        The optional target is used by checkpoint inspection: it lets a replay
        reproduce one target sampled by a previous parallel evaluation without
        changing the normal keyed target sampler used for training.
        """
        _, target_key, next_key = jax.random.split(key, 3)
        if building_bank is not None:
            obstacle_min = building_bank.get("solid_min", map_id)
            obstacle_max = building_bank.get("solid_max", map_id)
            stored_vertices = jnp.zeros((0, 3), dtype=jnp.float32)
            stored_distances = jnp.zeros((0, 0), dtype=jnp.float32)
        elif obstacle_min is None:
            obstacle_min = obstacle_max = jnp.zeros((0, 3), dtype=jnp.float32)
            stored_vertices = jnp.zeros((0, 3), dtype=jnp.float32)
            stored_distances = jnp.zeros((0, 0), dtype=jnp.float32)
        elif obstacle_min.shape[0] and plan_geodesic:
            expected_vertices = building_candidate_vertices(building, cfg).shape[0] + 8 * obstacle_min.shape[0]
            if (stored_vertices is None or stored_distances is None
                    or stored_vertices.shape != (expected_vertices, 3)
                    or stored_distances.shape != (expected_vertices, expected_vertices)):
                stored_vertices, stored_distances = layout_roadmap(obstacle_min, obstacle_max)
        else:
            stored_vertices = jnp.zeros((0, 3), dtype=jnp.float32)
            stored_distances = jnp.zeros((0, 0), dtype=jnp.float32)
        target = (sample_target(target_key, obstacle_min, obstacle_max, map_id, target_spawn_weights)
                  if target_pos is None else jnp.asarray(target_pos))
        base_pos = jnp.asarray(building.base_position_m)
        if building_bank is not None:
            base_pos = building_bank.get("base_position", map_id)
        pair_attempts = jnp.int32(1)
        target_retries = jnp.int32(0)
        required_agents = jnp.int32(0)
        if randomize_base or dynamic_max_geo_distance:
            solid_min, solid_max = all_solids(obstacle_min, obstacle_max)
            if randomize_base:
                pair_lower, pair_upper, pair_volumes = base_lower, base_upper, base_volumes
                if building_bank is not None:
                    pair_lower, pair_upper, pair_volumes = (building_bank.get("base_" + suffix, map_id)
                                                           for suffix in ("lower", "upper", "volumes"))
            fixed_target = target
            def draw_pair(draw_key, fixed_target):
                base_key, point_key, target_key = jax.random.split(draw_key, 3)
                base = base_pos
                if randomize_base:
                    box = jax.random.categorical(base_key, jnp.log(pair_volumes))
                    base = jax.random.uniform(point_key, (3,), minval=pair_lower[box], maxval=pair_upper[box])
                # Randomized bases retry against the target for this batch.
                # With a fixed base, retry targets instead.
                target = (fixed_target if (randomize_base and not dynamic_max_geo_distance) or target_pos is not None
                          else sample_target(target_key, obstacle_min, obstacle_max, map_id, target_spawn_weights))
                drone = base + jnp.asarray([0.0, 0.0, cfg.drone_radius])
                valid = (~points_inside_aabbs(base, solid_min, solid_max, cfg.drone_radius)
                         & ~points_inside_aabbs(drone, solid_min, solid_max, cfg.drone_radius)
                         & jnp.all((drone >= lower) & (drone <= map_upper(map_id)))) if randomize_base else jnp.bool_(True)
                valid &= ~target_inside_buffered_solids(target, obstacle_min, obstacle_max)
                required = jnp.int32(0)
                if dynamic_max_geo_distance:
                    # Necessary straight-line bound, followed by a constructive
                    # witness. Curriculum spare slots never enlarge target eligibility.
                    budget = cfg.num_agents if target_agent_budget is None else target_agent_budget
                    bound = cfg.comm_radius_base + (budget - 1) * cfg.comm_radius + cfg.visual_radius
                    valid &= jnp.linalg.norm(base - target) <= bound
                    required, certified = jax.lax.cond(
                        valid, lambda _: relay_calculator(base, target, map_id),
                        lambda _: (jnp.int32(0), jnp.bool_(False)), operand=None)
                    valid &= certified & (required <= budget)
                    if target_min_agents is not None:
                        valid &= required >= target_min_agents
                return base, target, valid, required

            def sample_pair_for_target(fixed_target, key):
                key, draw_key = jax.random.split(key)
                first_base, first_target, first_valid, first_required = draw_pair(draw_key, fixed_target)

                def retry_pair(carry):
                    count, retry_key, _, _, _, _ = carry
                    retry_key, draw_key = jax.random.split(retry_key)
                    base, target, valid, required = draw_pair(draw_key, fixed_target)
                    return count + 1, retry_key, base, target, valid, required

                return jax.lax.while_loop(
                    lambda carry: (~carry[4]) & (carry[0] < spawn_pair_max_attempts), retry_pair,
                    (jnp.int32(1), key, first_base, first_target, first_valid, first_required))

            _, first_key = jax.random.split(target_key)
            pair_attempts, _, base_pos, target, valid, required_agents = sample_pair_for_target(fixed_target, first_key)

            # Zero marks exhausted sampling when skipping is disabled or an
            # externally supplied target cannot be retried. Host callers check it.
            pair_attempts = jnp.where(valid, pair_attempts, jnp.int32(0))
        # The station itself sits on the floor; drone centres start one radius
        # above it so the initial state does not intersect the floor tile.
        drone_spawn = base_pos + jnp.asarray([0.0, 0.0, cfg.drone_radius])
        pos = jnp.broadcast_to(drone_spawn, (n, 3))
        active = jnp.arange(n) * cfg.spawn_delay <= 0
        adaptive = None
        if cfg.decommissioning.enabled:
            zone, valid_zone = decommission_zone(base_pos, obstacle_min, obstacle_max, map_id)
            adaptive = adaptive_population.initial_state(n, zone, valid_zone)
        coverage = jnp.zeros(coverage_shape, dtype=jnp.bool_)
        coverage, _ = update_coverage(coverage, pos, active, obstacle_min, obstacle_max, map_id)
        sees, conn_base, conn_target, _ = connectivity(pos, active, base_pos, target, obstacle_min, obstacle_max)
        base_target_connected = jnp.any(conn_base & conn_target)
        return EnvState(
            adaptive=adaptive,
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
            target_known=sees,
            success=jnp.bool_(False),
            base_target_connected=base_target_connected,
            chain_held_steps=jnp.int32(0),
            best_chain_held_steps=jnp.int32(0),
            peer_informed_by=jnp.zeros((n, n), dtype=jnp.bool_),
            done=jnp.bool_(False),
            collided=jnp.zeros(n, dtype=jnp.bool_),
            coverage_credit=jnp.zeros(n, dtype=jnp.float32),
            stationary_steps=jnp.int32(0),
            base_target_known=jnp.bool_(False),
            obstacle_min=obstacle_min,
            obstacle_max=obstacle_max,
            roadmap_vertices=stored_vertices,
            roadmap_distances=stored_distances,
            shared_geometry=geometry,
            spawn_pair_attempts=pair_attempts,
            spawn_pair_target_retries=target_retries,
            map_id=jnp.asarray(map_id, dtype=jnp.int32),
            building_bank=building_bank,
            target_required_agents=required_agents,
        )

    def step(state: EnvState, action: jax.Array):
        expected_actions = 3
        if action.shape != (n, expected_actions):
            raise ValueError(f"Expected actions shaped {(n, expected_actions)}, got {action.shape}.")
        # The policy communicates from the pre-action state. Knowledge uses
        # those same positions and the same episode clock. New recipients may
        # relay on a later communication tick, never transitively in one tick.
        peer_links, base_links = knowledge_links(
            state.pos, state.active, state.base_pos, state.obstacle_min, state.obstacle_max,
        )
        informed_by = peer_links & state.target_known[None, :]
        base_received = jnp.any(base_links & state.target_known)
        from_base = base_links & state.base_target_known
        peer_received = jnp.any(informed_by, axis=1)
        next_step = state.step + 1
        active = state.active if cfg.decommissioning.enabled else next_step >= jnp.arange(n) * cfg.spawn_delay
        force = jnp.clip(action[..., :3], -1.0, 1.0) * cfg.max_force
        # Newly spawned agents first choose an action from their next observation.
        if cfg.decommissioning.enabled:
            force = jnp.where(state.active[:, None], force, 0.0)
        velocity = state.vel * cfg.drag + force * cfg.dt
        speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= jnp.minimum(1.0, cfg.max_speed / jnp.maximum(speed, 1e-8))
        proposed = state.pos + velocity * cfg.dt
        physical_upper = map_upper(state.map_id)
        collided = (proposed < lower) | (proposed > physical_upper)
        if state.obstacle_min.shape[0] or has_authored_solids:
            solid_min, solid_max = all_solids(state.obstacle_min, state.obstacle_max)
            motion_min = solid_min - cfg.drone_radius
            motion_max = solid_max + cfg.drone_radius
            obstacle_collision = segments_blocked(state.pos, proposed, motion_min, motion_max)
            collided |= obstacle_collision[:, None]
            proposed = jnp.where(obstacle_collision[:, None], state.pos, proposed)
        pos = jnp.clip(proposed, lower, physical_upper)
        velocity = jnp.where(collided, 0.0, velocity)
        drone_spawn = state.base_pos + jnp.asarray([0.0, 0.0, cfg.drone_radius])
        pos = jnp.where(active[:, None], pos, drone_spawn)
        velocity = jnp.where(active[:, None], velocity, 0.0)
        adaptive = state.adaptive
        if cfg.decommissioning.enabled:
            adaptive, active = adaptive_population.transition(
                adaptive, state.active, pos, cfg.decommissioning, spawn_delay=cfg.spawn_delay,
            )
            pos = jnp.where((active & ~adaptive.spawned)[:, None], pos, drone_spawn)
            velocity = jnp.where((active & ~adaptive.spawned)[:, None], velocity, 0.0)
        any_agent_moved = jnp.any(
            (jnp.linalg.norm(pos - state.pos, axis=-1) > cfg.movement_epsilon) & active
            & (state.active if cfg.decommissioning.enabled else True)
        )
        stationary_steps = jnp.where(
            any_agent_moved,
            jnp.int32(0),
            state.stationary_steps + jnp.int32(1),
        )
        coverage, coverage_credit = update_coverage(
            state.coverage, pos, active, state.obstacle_min, state.obstacle_max, state.map_id
        )
        sees, conn_base, conn_target, _ = connectivity(
            pos, active, state.base_pos, state.target_pos,
            state.obstacle_min, state.obstacle_max,
        )
        known = state.target_known | sees | peer_received | from_base
        base_target_connected = jnp.any(conn_base & conn_target)
        base_target_known = state.base_target_known | base_received
        chain_held_steps = jnp.where(
            base_target_connected,
            state.chain_held_steps + jnp.int32(1),
            jnp.int32(0),
        )
        best_chain_held_steps = jnp.minimum(
            jnp.maximum(state.best_chain_held_steps, chain_held_steps),
            jnp.int32(cfg.hold_chain_for),
        )
        # Direct sighting and base replay take precedence over peer credit.
        newly_peer_informed = (~state.target_known) & peer_received & ~sees & ~from_base
        peer_informed_by = informed_by & newly_peer_informed[:, None]
        if cfg.decommissioning.enabled:
            continuing = active & ~adaptive.spawned
            known &= continuing
            peer_informed_by &= continuing[:, None] & continuing[None, :]
            adaptive = adaptive._replace(peer_credit_count=jnp.minimum(
                n - 1, adaptive.peer_credit_count + jnp.sum(jnp.any(peer_informed_by, axis=1))))
        success = chain_held_steps >= jnp.int32(cfg.hold_chain_for)
        done = next_step >= jnp.int32(cfg.max_steps)
        if cfg.decommissioning.enabled:
            done |= adaptive.all_decommissioned
        next_state = EnvState(
            adaptive=adaptive,
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
            base_target_connected=base_target_connected,
            chain_held_steps=chain_held_steps,
            best_chain_held_steps=best_chain_held_steps,
            peer_informed_by=peer_informed_by,
            new_information_counts=new_information_counts(state.target_known, known, active, sees, from_base),
            done=done | success,
            collided=jnp.any(collided, axis=-1) & active,
            coverage_credit=coverage_credit,
            stationary_steps=stationary_steps,
            base_target_known=base_target_known,
            obstacle_min=state.obstacle_min,
            obstacle_max=state.obstacle_max,
            roadmap_vertices=state.roadmap_vertices,
            roadmap_distances=state.roadmap_distances,
            shared_geometry=state.shared_geometry,
            spawn_pair_attempts=state.spawn_pair_attempts,
            spawn_pair_target_retries=state.spawn_pair_target_retries,
            map_id=state.map_id,
            building_bank=state.building_bank,
            target_required_agents=state.target_required_agents,
        )
        if cfg.success_condition == "chain_held_and_cleand_up":
            # Hold timing is independent of cleanup: it keeps advancing while
            # surplus drones return, and any chain break resets it above.
            def cleaned_up(_):
                contributors = success_contributors(next_state, cfg)
                return ~jnp.any(next_state.active & ~contributors)

            success = jax.lax.cond(success, cleaned_up, lambda _: jnp.bool_(False), None)
            next_state = next_state._replace(success=success, done=done | success)
        return next_state

    def observations(state: EnvState):
        pos = state.pos
        if cfg.decommissioning.enabled and cfg.observe_base_population:
            base_population = adaptive_population.base_population_feedback(
                state.active, state.is_conn_base, DEPLOYED_AGENTS_OBS_REF,
            )
        if cfg.observe_chain_contributor:
            chain_contributor = state.target_known & success_contributors(state, cfg)
        delta = pos[:, None, :] - pos[None, :, :]
        pair_dist = jnp.linalg.norm(delta, axis=-1)

        def one_agent(i):
            origin = pos[i]
            positive = jnp.where(wall_directions > 0, (map_upper(state.map_id) - origin) / wall_directions, jnp.inf)
            negative = jnp.where(wall_directions < 0, (lower - origin) / wall_directions, jnp.inf)
            wall_distance = jnp.min(jnp.minimum(positive, negative), axis=-1)
            if state.obstacle_min.shape[0] or has_authored_solids:
                solid_min, solid_max = all_solids(state.obstacle_min, state.obstacle_max)
                ray_end = origin[None, :] + wall_directions * cfg.wall_sensing_ray_max_distance
                lo = jnp.zeros(cfg.wall_sensing_rays, dtype=jnp.float32)
                hi = jnp.ones(cfg.wall_sensing_rays, dtype=jnp.float32)
                hit = segments_blocked(
                    jnp.broadcast_to(origin, ray_end.shape), ray_end,
                    solid_min, solid_max,
                )
                def refine_radar(_, interval):
                    lo, hi = interval
                    mid = (lo + hi) / 2
                    mid_point = origin[None, :] + wall_directions * (mid * cfg.wall_sensing_ray_max_distance)[:, None]
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
                    jnp.where(hit, hi * cfg.wall_sensing_ray_max_distance, jnp.inf),
                )
            wall_signal = jnp.maximum(0.0, 1.0 - wall_distance / cfg.wall_sensing_ray_max_distance)

            rel = pos - origin
            rel_norm = rel / jnp.maximum(pair_dist[i, :, None], 1e-8)
            bin_index = jnp.argmax(rel_norm @ drone_directions.T, axis=-1)
            other = (jnp.arange(n) != i) & state.active
            if state.obstacle_min.shape[0] or has_authored_solids:
                other &= ~segments_blocked(origin, pos, solid_min, solid_max)
            if cfg.radar_mode == "max_proximity":
                signal = jnp.maximum(0.0, 1.0 - pair_dist[i] / cfg.comm_radius) * other

                def scatter(values):
                    return jnp.zeros(cfg.drone_radar_bins, dtype=jnp.float32).at[bin_index].max(values)

                drone_proximity = jnp.stack(
                    [scatter(signal),
                     scatter(signal * state.is_conn_target),
                     scatter(signal * state.is_conn_base)],
                    axis=-1,
                )
                if cfg.wall_sensing_rays == cfg.drone_radar_bins:
                    # Preserve the original checkpoint feature order exactly.
                    radar = jnp.concatenate([wall_signal[:, None], drone_proximity], axis=-1).reshape(-1)
                else:
                    radar = jnp.concatenate([wall_signal, drone_proximity.reshape(-1)])
            else:
                # Retained band_sep_counting ablation.
                visible = other & (pair_dist[i] <= cfg.comm_radius)
                # An internal border belongs to the outer band; the final
                # band includes comm_radius. Each neighbor occupies one cell.
                band_index = jnp.searchsorted(band_edges, pair_dist[i], side="right")
                cell_index = bin_index * distance_bands + band_index

                def count_channel(qualifying):
                    counts = jnp.zeros(cfg.drone_radar_bins * distance_bands, dtype=jnp.float32)
                    counts = counts.at[cell_index].add(qualifying.astype(jnp.float32))
                    return jnp.minimum(counts, cfg.radar_count_cap) / cfg.radar_count_cap

                counts = jnp.stack(
                    [count_channel(visible),
                     count_channel(visible & state.is_conn_target),
                     count_channel(visible & state.is_conn_base)],
                    axis=-1,
                ).reshape(-1)
                radar = jnp.concatenate([wall_signal, counts])
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
            if cfg.decommissioning.enabled:
                if cfg.observe_decommission_zone:
                    in_zone = (jnp.linalg.norm(origin - state.adaptive.zone_center)
                               <= cfg.decommissioning.decommission_radius_m) & state.adaptive.zone_valid
                    optional.append(jnp.asarray([in_zone], dtype=jnp.float32))
                # Instantaneous base-network feedback, never a latched receipt.
                # is_conn_base includes physical relay paths, not only direct LOS.
                if cfg.observe_base_population:
                    optional.append(jnp.asarray([base_population[i]]))
            if cfg.observe_chain_contributor:
                optional.append(jnp.asarray([chain_contributor[i]], dtype=jnp.float32))
            if cfg.observe_visually_gated_target_displacement:
                optional.append(jnp.where(
                    state.directly_sees_target[i],
                    (state.target_pos - origin) / cfg.visual_radius,
                    jnp.zeros(3, dtype=jnp.float32),
                ))
            observation = jnp.concatenate([self_state, *optional, radar])
            return jnp.where(state.active[i], observation, 0.0)

        return jax.vmap(one_agent)(jnp.arange(n))

    def metrics(state: EnvState):
        return {
            "coverage_fraction": coverage_fraction(state.coverage, state.map_id),
            "active_agents": jnp.sum(state.active),
            "target_seen": jnp.any(state.directly_sees_target),
            "success": state.success,
            "base_target_connected": state.base_target_connected,
            "base_target_known": state.base_target_known,
            "best_chain_held_steps": state.best_chain_held_steps,
            "stationary_steps": state.stationary_steps,
            "done": state.done,
        }

    # Replay/evaluation prepare persisted layouts through the same geometry path.
    reset.build_roadmap = layout_roadmap
    return reset, step, observations, metrics


def make_autoreset_fns(
    building: BuildingArrays,
    cfg: EnvConfig,
    reward_cfg: RewardConfig = RewardConfig(),
    *, randomize_base: bool = False,
    dynamic_max_geo_distance: bool = False,
    relay_options=None,
    spawn_pair_max_attempts: int = 1024,
    building_bank=None,
    adaptive_map_sampling: bool = False,
    resample_maps: bool = False,
):
    """Return reset and terminal-aware step functions for batched training.

    Terminal rewards and diagnostic state describe the completed transition;
    only the returned carry state is replaced by a fresh episode.

    Map resampling is independent of adaptation: an optional CDF supplies
    weighted probabilities; without one every prepared map is equally likely.
    adaptive_map_sampling remains an alias for older internal callers.
    """
    if type(adaptive_map_sampling) is not bool:
        raise ValueError("adaptive_map_sampling must be boolean.")
    if type(resample_maps) is not bool:
        raise ValueError("resample_maps must be boolean.")
    resample_maps = resample_maps or adaptive_map_sampling
    if resample_maps and building_bank is None:
        raise ValueError("Episode map resampling requires a prepared building bank.")
    validate_reward_config(reward_cfg)
    reset, step, observations, metrics = make_env_fns(
        building, cfg, plan_geodesic=True,
        randomize_base=randomize_base,
        dynamic_max_geo_distance=dynamic_max_geo_distance,
        relay_options=relay_options,
        spawn_pair_max_attempts=spawn_pair_max_attempts,
        building_bank=building_bank,
    )

    def transition(state: EnvState, action: jax.Array):
        terminal_state = step(state, action)
        distances = chain_distance_matrix(terminal_state)
        if reward_cfg.chain_efficiency_reward != 0:
            def empty_paths(_):
                count = 1 << terminal_state.pos.shape[0]
                return ChainPaths(
                    jnp.full((count,), jnp.inf, dtype=jnp.float32),
                    jnp.zeros((count,), dtype=jnp.int32),
                    jnp.zeros((count, terminal_state.pos.shape[0]), dtype=jnp.bool_),
                )

            paths = jax.lax.cond(
                terminal_state.base_target_connected,
                lambda _: simple_chain_paths(terminal_state, cfg),
                empty_paths, None,
            )
        else:
            paths = None
        diagnostics = obstacle_chain_diagnostics(terminal_state, distances)
        chain_gap_dist, chain_progress_pct, route_excess, leaders = diagnostics
        reward, reward_terms = compute_rewards(
            state, terminal_state, reward_cfg, cfg, distances, paths, diagnostics,
        )
        terminal_metrics = metrics(terminal_state)
        info = {
            **reward_terms,
            "gap_gain": jnp.maximum(reward_terms["gap_reward"], 0.0),
            "gap_loss": jnp.minimum(reward_terms["gap_reward"], 0.0),
            "peer_informed_count": jnp.sum(jnp.any(terminal_state.peer_informed_by, axis=1)),
            "information_counts": information_counts(state, terminal_state),
            "best_chain_held_steps": terminal_state.best_chain_held_steps,
            "done": terminal_state.done,
            "success": terminal_state.success,
            "base_target_connected": terminal_state.base_target_connected,
            "global_target_found": terminal_state.base_target_known,
            "global_visually_found": jnp.any(terminal_state.directly_sees_target),
            "chain_gap_dist": chain_gap_dist,
            "chain_progress_pct": chain_progress_pct,
            "chain_route_excess": route_excess,
            "chain_frontier_leaders": leaders,
            "global_coverage": terminal_metrics["coverage_fraction"],
            "terminal_target_pos": terminal_state.target_pos,
            "terminal_required_agents": terminal_state.target_required_agents,
            "terminal_coverage_fraction": terminal_metrics["coverage_fraction"],
            "agent_terminated": adaptive_population.lifetime_ended(state, terminal_state),
        }
        if cfg.decommissioning.enabled:
            a = terminal_state.adaptive
            info.update({
                "adapt_decommissions": a.decommissions,
                "adapt_end_agents": jnp.sum(terminal_state.active),
                "adapt_all_decommissioned": a.all_decommissioned & ~terminal_state.success,
                "adapt_useless_agents": jnp.sum(terminal_state.active & ~success_contributors(terminal_state, cfg)),
                "adapt_zone_valid": a.zone_valid,
            })
        if paths is not None:
            info["number_of_valid_paths"] = jnp.sum(paths.counts)
            if reward_cfg.chain_efficiency_reward != 0:
                info["chain_efficiency"] = jnp.max(chain_path_efficiencies(paths, distances[0, 1]))
        return terminal_state, reward, terminal_state.done, info

    def reset_persisted(state, target_spawn_weights=None, map_sampling_cdf=None, target_agent_budget=None, target_min_agents=None):
        if resample_maps:
            map_key, reset_key = jax.random.split(state.key)
            num_maps = building_bank.arrays["base_position"].shape[0]
            def choose_map(key):
                if map_sampling_cdf is None:
                    return jax.random.randint(key, (), 0, num_maps)
                if map_sampling_cdf.shape != (num_maps,):
                    raise ValueError("Map-selection CDF must cover the complete prepared training bank.")
                return jnp.minimum(jnp.searchsorted(map_sampling_cdf, jax.random.uniform(key, ()),
                                                    side="right"), num_maps - 1)
            index = choose_map(map_key)
            fresh = reset(reset_key, map_id=index, target_spawn_weights=target_spawn_weights,
                         target_agent_budget=target_agent_budget, target_min_agents=target_min_agents)
            if target_min_agents is not None:
                # Some maps have no targets in a reverse curriculum's allowed
                # group. Retry map selection; never fall back to easier targets.
                def retry(carry):
                    attempt, previous = carry
                    mk, rk = jax.random.split(previous.key)
                    index = choose_map(mk)
                    return attempt + 1, reset(rk, map_id=index, target_spawn_weights=target_spawn_weights,
                        target_agent_budget=target_agent_budget, target_min_agents=target_min_agents)
                _, fresh = jax.lax.while_loop(
                    lambda carry: (carry[0] < 128) & (carry[1].spawn_pair_attempts == 0),
                    retry, (jnp.int32(0), fresh))
            return fresh
        return reset(state.key, obstacle_min=state.obstacle_min, obstacle_max=state.obstacle_max,
                     stored_vertices=state.roadmap_vertices, stored_distances=state.roadmap_distances,
                     map_id=state.map_id,
                     target_spawn_weights=target_spawn_weights, target_agent_budget=target_agent_budget,
                     target_min_agents=target_min_agents)

    def autoreset_step(state: EnvState, action: jax.Array, target_spawn_weights=None, map_sampling_cdf=None, target_agent_budget=None, target_min_agents=None):
        """Original scalar API retained as the dense equivalence reference."""
        terminal, reward, done, info = transition(state, action)
        fresh = reset_persisted(terminal, target_spawn_weights, map_sampling_cdf, target_agent_budget, target_min_agents)
        result = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), fresh, terminal)
        return result, reward, done, info

    def batched_step(states, actions, target_spawn_weights=None, map_sampling_cdf=None, target_agent_budget=None, target_min_agents=None):
        terminal, rewards, done, info = jax.vmap(transition)(states, actions)
        size = done.shape[0]
        count = jnp.sum(done, dtype=jnp.int32)

        def dense(current):
            if resample_maps or target_agent_budget is not None:
                fresh = jax.vmap(lambda state: reset_persisted(state, target_spawn_weights, map_sampling_cdf, target_agent_budget, target_min_agents))(current)
            else:
                fresh = (jax.vmap(reset_persisted)(current) if target_spawn_weights is None else
                         jax.vmap(lambda state: reset_persisted(state, target_spawn_weights))(current))
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
                if resample_maps or target_agent_budget is not None:
                    fresh = jax.vmap(lambda state: reset_persisted(state, target_spawn_weights, map_sampling_cdf, target_agent_budget, target_min_agents))(selected)
                else:
                    fresh = (jax.vmap(reset_persisted)(selected) if target_spawn_weights is None else
                             jax.vmap(lambda state: reset_persisted(state, target_spawn_weights))(selected))
                # Layout and roadmap are invariant across episode reset. Avoid
                # scattering potentially large matrices back into their copies.
                unchanged = {"obstacle_min", "obstacle_max", "roadmap_vertices",
                             "roadmap_distances", "shared_geometry", "building_bank", "map_id"}
                if resample_maps:
                    unchanged = {"shared_geometry", "building_bank"}
                return result._replace(**{
                    field: jax.tree.map(lambda old, new: old.at[lanes].set(new, mode="drop"),
                                        getattr(result, field), getattr(fresh, field))
                    for field in result._fields if field not in unchanged
                })

            return jax.lax.fori_loop(0, (count + block - 1) // block, reset_chunk, current)

        # This conditional is outside vmap. No reset is executed when count=0;
        # dense reset remains efficient for synchronized episode endings.
        # Rejection sampling in a dense vmap runs until the slowest of all
        # lanes accepts. Keep it in small groups even at synchronized endings.
        result = (sparse(terminal) if randomize_base or dynamic_max_geo_distance else
                  jax.lax.cond(count > size // 2, dense, sparse, terminal))
        return result, rewards, done, info

    autoreset_step.batched = batched_step
    reset.resample = reset_persisted

    return reset, autoreset_step, observations, metrics
