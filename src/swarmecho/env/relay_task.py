"""Relay-task runtime and discrete 2D finder-path state transitions."""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from swarmecho.env.maps import MapDefinition
from swarmecho.env.state import EnvState, RelayTaskState


def make_relay_task_fns(
    cfg: DictConfig,
    map_def: MapDefinition,
    num_agents: int,
    world_width: float,
    world_height: float,
    reachability_steps: int,
) -> tuple[Callable[[], RelayTaskState], Callable[..., RelayTaskState]]:
    """Build relay-state reset and transition functions for one environment."""
    use_finders_path = (
        str(cfg.reward.get("chain_reward_system", "euclidean"))
        == "discrete_finders_path"
    )
    maze_cols = int(map_def.maze_cell_cols or 1)
    maze_rows = int(map_def.maze_cell_rows or 1)
    maze_cell_w = float(world_width) / maze_cols
    maze_cell_h = float(world_height) / maze_rows
    max_path_len = maze_cols * maze_rows if use_finders_path else 1
    base_maze_cell = jnp.array(
        [
            int(
                max(
                    0,
                    min(
                        maze_cols - 1,
                        math.floor((float(world_width) / 2.0) / maze_cell_w),
                    ),
                )
            ),
            int(
                max(
                    0,
                    min(
                        maze_rows - 1,
                        math.floor((float(world_height) / 2.0) / maze_cell_h),
                    ),
                )
            ),
        ],
        dtype=jnp.int16,
    )
    maze_offsets = jnp.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=jnp.int16,
    )
    spawn_room_cells = jnp.clip(
        base_maze_cell[None, :] + maze_offsets,
        jnp.array([0, 0], dtype=jnp.int16),
        jnp.array([maze_cols - 1, maze_rows - 1], dtype=jnp.int16),
    )
    delivery_freeze = bool(
        cfg.reward.get("target_found_requires_delivery", True)
    )
    N = int(num_agents)

    def initial_relay_state() -> RelayTaskState:
        return RelayTaskState(
            chain_held_steps=jnp.int32(0),
            finder_path_cells=jnp.zeros(
                (N, max_path_len, 2),
                dtype=jnp.int16,
            ),
            finder_path_lens=jnp.zeros(N, dtype=jnp.int16),
            finder_path_active=jnp.zeros(N, dtype=jnp.bool_),
            target_known_path_cells=jnp.zeros(
                (N, max_path_len, 2),
                dtype=jnp.int16,
            ),
            target_known_path_lens=jnp.zeros(N, dtype=jnp.int16),
            target_known_path_valid=jnp.zeros(N, dtype=jnp.bool_),
            finders_path=jnp.zeros((max_path_len, 2), dtype=jnp.int16),
            finders_path_len=jnp.int16(0),
            finders_path_valid=jnp.bool_(False),
            finders_path_index_grid=jnp.full(
                (maze_cols, maze_rows),
                jnp.int16(-1),
                dtype=jnp.int16,
            ),
        )

    def maze_cell_from_pos(pos):
        cx = jnp.clip(
            jnp.floor(pos[0] / maze_cell_w).astype(jnp.int16),
            0,
            maze_cols - 1,
        )
        cy = jnp.clip(
            jnp.floor(pos[1] / maze_cell_h).astype(jnp.int16),
            0,
            maze_rows - 1,
        )
        return jnp.stack([cx, cy]).astype(jnp.int16)

    def is_spawn_room_cell(cell):
        return jnp.any(
            jnp.all(spawn_room_cells == cell[None, :], axis=-1)
        )

    def append_loop_erased(path, length, cell):
        valid = jnp.arange(max_path_len) < length.astype(jnp.int32)
        matches = valid & jnp.all(path == cell[None, :], axis=-1)
        already = jnp.any(matches)
        match_idx = jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int16)
        last_idx = jnp.maximum(length.astype(jnp.int32) - 1, 0)
        last_cell = path[last_idx]
        same_last = (length > 0) & jnp.all(last_cell == cell)
        can_append = (
            (length.astype(jnp.int32) < max_path_len)
            & ~already
            & ~same_last
        )
        next_len = jnp.where(
            already,
            match_idx + jnp.int16(1),
            jnp.where(can_append, length + jnp.int16(1), length),
        )
        path = jnp.where(
            can_append,
            path.at[length.astype(jnp.int32)].set(cell),
            path,
        )
        return path, next_len

    def update_relay_state(
        state: EnvState,
        mid_state: EnvState,
        directly_sees: jax.Array,
        new_target_known: jax.Array,
        new_base_target_known: jax.Array,
        is_conn_base: jax.Array,
    ) -> RelayTaskState:
        if not use_finders_path:
            return state.relay

        physics = state.physics
        communication = state.communication
        relay = state.relay
        mid_physics = mid_state.physics
        mid_communication = mid_state.communication

        cells = jax.vmap(maze_cell_from_pos)(mid_physics.pos)
        in_spawn = jax.vmap(is_spawn_room_cell)(cells)

        def update_one(i, carry):
            paths, lens, active = carry
            path = paths[i]
            length = lens[i]
            was_active = length > 0
            now_active = was_active | (
                ~in_spawn[i] & mid_physics.active[i]
            )
            reset_path = (
                in_spawn[i]
                & was_active
                & ~communication.target_known[i]
                & ~directly_sees[i]
            )

            prev_cell = maze_cell_from_pos(physics.pos[i])
            seed = (
                path.at[0]
                .set(base_maze_cell)
                .at[1]
                .set(prev_cell)
                .at[2]
                .set(cells[i])
            )
            seed_len = jnp.int16(3)
            path = jnp.where((~was_active) & now_active, seed, path)
            length = jnp.where(
                (~was_active) & now_active,
                seed_len,
                length,
            )

            path, length = jax.lax.cond(
                was_active
                & now_active
                & ~reset_path
                & mid_physics.active[i],
                lambda operands: append_loop_erased(
                    operands[0],
                    operands[1],
                    cells[i],
                ),
                lambda operands: operands,
                (path, length),
            )
            path = jnp.where(reset_path, jnp.zeros_like(path), path)
            length = jnp.where(reset_path, jnp.int16(0), length)
            now_active = length > 0
            return (
                paths.at[i].set(path),
                lens.at[i].set(length),
                active.at[i].set(now_active),
            )

        paths, lens, active = jax.lax.fori_loop(
            0,
            N,
            update_one,
            (
                relay.finder_path_cells,
                relay.finder_path_lens,
                relay.finder_path_active,
            ),
        )

        path_ready = lens > 0
        target_positions = jnp.tile(
            physics.target_pos[None, :],
            (N, 1),
        )

        def append_target(path_len_target):
            path, length, target_pos = path_len_target
            target_cell = maze_cell_from_pos(target_pos)
            last_cell_idx = jnp.maximum(length - 1, 0)
            last_cell = path[last_cell_idx]
            is_same = jnp.all(last_cell == target_cell)
            dx = jnp.abs(last_cell[0] - target_cell[0])
            dy = jnp.abs(last_cell[1] - target_cell[1])
            is_neighbor = (dx <= 1) & (dy <= 1)
            should_append = (
                ~is_same
                & is_neighbor
                & (length < max_path_len)
            )
            path = jnp.where(
                should_append,
                path.at[length.astype(jnp.int32)].set(target_cell),
                path,
            )
            length = jnp.where(
                should_append,
                length + jnp.int16(1),
                length,
            )
            return path, length

        direct_paths, direct_lens = jax.vmap(append_target)(
            (paths, lens, target_positions)
        )
        direct_valid = (
            directly_sees
            & path_ready
            & ~relay.target_known_path_valid
        )

        source_paths = jnp.where(
            direct_valid[:, None, None],
            direct_paths,
            relay.target_known_path_cells,
        )
        source_lens = jnp.where(
            direct_valid,
            direct_lens,
            relay.target_known_path_lens,
        )
        source_valid = relay.target_known_path_valid | direct_valid

        agent_adjacency = (
            mid_communication.adj_matrix[:N, :N]
            | jnp.eye(N, dtype=jnp.bool_)
        )

        def square_agents(reachability, _):
            return (
                reachability.astype(jnp.float32)
                @ reachability.astype(jnp.float32)
                > 0.5
            ), None

        agent_reachability, _ = jax.lax.scan(
            square_agents,
            agent_adjacency,
            None,
            length=reachability_steps,
        )
        source_matrix = agent_reachability & source_valid[None, :]
        source_idx = jnp.argmax(source_matrix.astype(jnp.int32), axis=1)
        has_source = jnp.any(source_matrix, axis=1)
        should_set_known_path = (
            new_target_known
            & ~relay.target_known_path_valid
            & has_source
        )
        known_paths = jnp.where(
            should_set_known_path[:, None, None],
            source_paths[source_idx],
            relay.target_known_path_cells,
        )
        known_lens = jnp.where(
            should_set_known_path,
            source_lens[source_idx],
            relay.target_known_path_lens,
        )
        known_valid = (
            relay.target_known_path_valid | should_set_known_path
        )

        delivered_now = (
            ~communication.base_target_known
            & new_base_target_known
        )
        delivered_known_path = (
            known_valid
            & new_target_known
            & (
                new_base_target_known
                | communication.base_target_known
            )
        )
        base_delivery_candidates = delivered_known_path & is_conn_base
        delivery_candidates = jnp.where(
            delivered_now & jnp.any(base_delivery_candidates),
            base_delivery_candidates,
            delivered_known_path,
        )
        freeze_candidates = jnp.where(
            delivery_freeze,
            delivery_candidates,
            direct_valid,
        )
        first_find_now = (
            ~relay.finders_path_valid
            & jnp.any(freeze_candidates)
        )
        finder_idx = jnp.argmax(freeze_candidates.astype(jnp.int32))
        selected_path = jnp.where(
            delivery_freeze,
            known_paths[finder_idx],
            direct_paths[finder_idx],
        )
        selected_len = jnp.where(
            delivery_freeze,
            known_lens[finder_idx],
            direct_lens[finder_idx],
        )

        def build_index_grid(path_len_path):
            path_len, path = path_len_path
            grid = jnp.full(
                (maze_cols, maze_rows),
                jnp.int16(-1),
                dtype=jnp.int16,
            )

            def set_index(k, current_grid):
                cell = path[k]
                return jax.lax.cond(
                    k < path_len.astype(jnp.int32),
                    lambda value: value.at[
                        cell[0].astype(jnp.int32),
                        cell[1].astype(jnp.int32),
                    ].set(k.astype(jnp.int16)),
                    lambda value: value,
                    current_grid,
                )

            return jax.lax.fori_loop(
                0,
                max_path_len,
                set_index,
                grid,
            )

        new_finders_path = jnp.where(
            first_find_now,
            selected_path,
            relay.finders_path,
        )
        new_finders_len = jnp.where(
            first_find_now,
            selected_len,
            relay.finders_path_len,
        )
        new_index_grid = jnp.where(
            first_find_now,
            build_index_grid((selected_len, selected_path)),
            relay.finders_path_index_grid,
        )
        keep_tracking_global = ~(
            relay.finders_path_valid | first_find_now
        )
        keep_tracking_agents = keep_tracking_global & ~known_valid

        return relay.replace(
            finder_path_cells=jnp.where(
                keep_tracking_agents[:, None, None],
                paths,
                relay.finder_path_cells,
            ),
            finder_path_lens=jnp.where(
                keep_tracking_agents,
                lens,
                relay.finder_path_lens,
            ),
            finder_path_active=jnp.where(
                keep_tracking_agents,
                active,
                relay.finder_path_active,
            ),
            target_known_path_cells=known_paths,
            target_known_path_lens=known_lens,
            target_known_path_valid=known_valid,
            finders_path=new_finders_path,
            finders_path_len=new_finders_len,
            finders_path_valid=(
                relay.finders_path_valid | first_find_now
            ),
            finders_path_index_grid=new_index_grid,
        )

    return initial_relay_state, update_relay_state
