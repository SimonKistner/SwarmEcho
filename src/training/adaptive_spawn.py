from __future__ import annotations

from collections import deque
import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from env.maps import MapDefinition


@dataclass
class AdaptiveSpawnDiagnostics:
    success_rate: dict[int, float]
    configured_rate: dict[int, float]
    actual_count: dict[int, int]
    actual_pct: dict[int, float]


class AdaptiveTargetSpawnController:
    """Training-only adaptive target-cell sampler.

    Categories are Manhattan/4-neighbor BFS path lengths from the base cell.
    The current project does not yet randomize maze geometry per episode; when
    static_maze_optimal_path is false this class exposes the same recalculation
    hook but still uses the currently loaded map until randomized maps exist.
    """

    def __init__(
        self,
        map_def: MapDefinition,
        target_spawn_method: str,
        static_maze_optimal_path: bool = True,
        target_spawn_radius: float = 0.0,
        target_spawn_radius_min: float = 0.0,
        target_invalid_spawn_base_radius: float = 0.0,
    ):
        if not map_def.maze_cell_cols or not map_def.maze_cell_rows:
            raise ValueError("adaptive target spawning requires map.maze_cell_grid with cols/rows")
        if map_def.valid_target_coords is None or len(map_def.valid_target_coords) == 0:
            raise ValueError("adaptive target spawning requires non-empty valid_target_coords")

        self.map_def = map_def
        self.target_spawn_method = str(target_spawn_method)
        self.static_maze_optimal_path = bool(static_maze_optimal_path)
        self.cols = int(map_def.maze_cell_cols)
        self.rows = int(map_def.maze_cell_rows)
        self.cell_w = float(map_def.width) / self.cols
        self.cell_h = float(map_def.height) / self.rows
        self._target_positions = np.asarray(jax.device_get(map_def.valid_target_coords), dtype=np.float32)
        self._target_positions = self._filter_positions_for_spawn_method(
            self._target_positions,
            target_spawn_radius=float(target_spawn_radius),
            target_spawn_radius_min=float(target_spawn_radius_min),
            target_invalid_spawn_base_radius=float(target_invalid_spawn_base_radius),
        )
        self._target_cells = self.positions_to_cells(self._target_positions)

        self._cell_to_category = self._compute_cell_categories()
        self._target_categories = self._categories_for_cells(self._target_cells)
        reachable = self._target_categories >= 0
        self._target_positions = self._target_positions[reachable]
        self._target_cells = self._target_cells[reachable]
        self._target_categories = self._target_categories[reachable]
        self.categories = np.array(sorted(set(int(c) for c in self._target_categories)), dtype=np.int32)
        if len(self.categories) == 0:
            raise ValueError("adaptive target spawning found no reachable target cells")

        self._category_to_index = {int(c): i for i, c in enumerate(self.categories)}
        self.spawn_counts = np.zeros(len(self.categories), dtype=np.int64)
        self.success_counts = np.zeros(len(self.categories), dtype=np.int64)
        self.actual_counts = np.zeros(len(self.categories), dtype=np.int64)
        self._position_probs = self._uniform_position_probs()

    @property
    def target_positions_jax(self) -> jax.Array:
        return jnp.asarray(self._target_positions, dtype=jnp.float32)

    @property
    def position_probs_jax(self) -> jax.Array:
        return jnp.asarray(self._position_probs, dtype=jnp.float32)

    def make_reset(self, base_reset):
        positions = self.target_positions_jax
        probs = self.position_probs_jax

        def reset(key):
            key, sample_key = jax.random.split(key)
            state = base_reset(key)
            idx = jax.random.choice(sample_key, positions.shape[0], p=probs)
            return dataclasses.replace(state, target_pos=positions[idx])

        return reset

    def _filter_positions_for_spawn_method(
        self,
        positions: np.ndarray,
        target_spawn_radius: float,
        target_spawn_radius_min: float,
        target_invalid_spawn_base_radius: float,
    ) -> np.ndarray:
        if self.target_spawn_method not in ("ring", "outside_base"):
            return positions
        base = np.asarray(self.map_def.base_spawn_zone, dtype=np.float32)
        base_pos = np.array([0.5 * (base[0] + base[2]), 0.5 * (base[1] + base[3])], dtype=np.float32)
        dists = np.linalg.norm(positions - base_pos[None, :], axis=-1)
        if self.target_spawn_method == "ring":
            keep = (dists >= target_spawn_radius_min) & (dists <= target_spawn_radius)
        else:
            keep = dists > target_invalid_spawn_base_radius
        filtered = positions[keep]
        if len(filtered) == 0:
            raise ValueError(f"adaptive target spawning found no valid cells for spawn method {self.target_spawn_method!r}")
        return filtered

    def positions_to_cells(self, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float32)
        cx = np.clip(np.floor(pos[:, 0] / self.cell_w).astype(np.int32), 0, self.cols - 1)
        cy = np.clip(np.floor(pos[:, 1] / self.cell_h).astype(np.int32), 0, self.rows - 1)
        return np.stack([cx, cy], axis=-1)

    def record_completed(self, target_positions: np.ndarray, successes: np.ndarray) -> None:
        if len(target_positions) == 0:
            return
        if not self.static_maze_optimal_path:
            self._cell_to_category = self._compute_cell_categories()
        cells = self.positions_to_cells(np.asarray(target_positions, dtype=np.float32))
        cats = self._categories_for_cells(cells)
        succ = np.asarray(successes, dtype=np.float32)
        for cat, ok in zip(cats, succ):
            idx = self._category_to_index.get(int(cat))
            if idx is None:
                continue
            self.spawn_counts[idx] += 1
            self.success_counts[idx] += int(ok > 0.5)
            self.actual_counts[idx] += 1

    def update_probabilities(self) -> AdaptiveSpawnDiagnostics:
        rates = np.divide(
            self.success_counts,
            self.spawn_counts,
            out=np.zeros_like(self.success_counts, dtype=np.float32),
            where=self.spawn_counts > 0,
        )
        k = len(self.categories)
        if k <= 1:
            deltas = np.zeros_like(rates, dtype=np.float32)
        else:
            deltas = (float(k) / float(k - 1)) * (float(np.mean(rates)) - rates)

        cell_weights = np.ones(len(self._target_positions), dtype=np.float32)
        for cat, delta in zip(self.categories, deltas):
            cell_weights[self._target_categories == cat] += float(delta)
        cell_weights = np.maximum(cell_weights, 0.0)
        if float(cell_weights.sum()) <= 0.0:
            self._position_probs = self._uniform_position_probs()
        else:
            self._position_probs = cell_weights / cell_weights.sum()

        configured = self._configured_category_rates()
        total_actual = int(self.actual_counts.sum())
        actual_pct = self.actual_counts / total_actual if total_actual > 0 else np.zeros_like(rates, dtype=np.float32)
        diag = AdaptiveSpawnDiagnostics(
            success_rate={int(c): float(r) for c, r in zip(self.categories, rates)},
            configured_rate={int(c): float(configured[i]) for i, c in enumerate(self.categories)},
            actual_count={int(c): int(v) for c, v in zip(self.categories, self.actual_counts)},
            actual_pct={int(c): float(v) for c, v in zip(self.categories, actual_pct)},
        )
        self.spawn_counts[:] = 0
        self.success_counts[:] = 0
        self.actual_counts[:] = 0
        return diag

    def _uniform_position_probs(self) -> np.ndarray:
        return np.full(len(self._target_positions), 1.0 / len(self._target_positions), dtype=np.float32)

    def _configured_category_rates(self) -> np.ndarray:
        out = np.zeros(len(self.categories), dtype=np.float32)
        for i, cat in enumerate(self.categories):
            out[i] = float(self._position_probs[self._target_categories == cat].sum())
        return out

    def _categories_for_cells(self, cells: np.ndarray) -> np.ndarray:
        cats = []
        for cx, cy in np.asarray(cells, dtype=np.int32):
            cats.append(self._cell_to_category.get((int(cx), int(cy)), -1))
        return np.asarray(cats, dtype=np.int32)

    def _compute_cell_categories(self) -> dict[tuple[int, int], int]:
        base = np.asarray(self.map_def.base_spawn_zone, dtype=np.float32)
        base_pos = np.array([0.5 * (base[0] + base[2]), 0.5 * (base[1] + base[3])], dtype=np.float32)
        base_cell = tuple(self.positions_to_cells(base_pos[None, :])[0])
        passable = np.zeros((self.cols, self.rows), dtype=bool)
        for cx in range(self.cols):
            for cy in range(self.rows):
                center = np.array([(cx + 0.5) * self.cell_w, (cy + 0.5) * self.cell_h], dtype=np.float32)
                passable[cx, cy] = not self._point_is_wall(center)
        if not passable[base_cell]:
            passable[base_cell] = True

        dist: dict[tuple[int, int], int] = {base_cell: 0}
        q: deque[tuple[int, int]] = deque([base_cell])
        while q:
            cx, cy = q.popleft()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if nx < 0 or nx >= self.cols or ny < 0 or ny >= self.rows:
                    continue
                if not passable[nx, ny] or (nx, ny) in dist:
                    continue
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
        return dist

    def _point_is_wall(self, pos: np.ndarray) -> bool:
        occ = self.map_def.occupancy_grid
        if occ is None:
            return False
        ix = int(np.clip(np.floor(pos[0]), 0, occ.shape[0] - 1))
        iy = int(np.clip(np.floor(pos[1]), 0, occ.shape[1] - 1))
        return bool(occ[ix, iy])


def diagnostics_to_wandb(diag: AdaptiveSpawnDiagnostics) -> dict[str, float]:
    logs: dict[str, float] = {}
    for cat, value in diag.success_rate.items():
        label = f"category_{cat:03d}"
        logs[f"diagnostics/spawn_success_rate/{label}"] = value
    for cat, value in diag.configured_rate.items():
        label = f"category_{cat:03d}"
        logs[f"diagnostics/spawn_configured_rate/{label}"] = value
    for cat, value in diag.actual_count.items():
        label = f"category_{cat:03d}"
        logs[f"diagnostics/spawn_actual_count/{label}"] = value
    for cat, value in diag.actual_pct.items():
        label = f"category_{cat:03d}"
        logs[f"diagnostics/spawn_actual_pct/{label}"] = value
    return logs
