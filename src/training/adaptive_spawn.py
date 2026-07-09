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
    active_categories: tuple[int, ...] = ()
    previous_active_categories: tuple[int, ...] = ()


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
        mode: str = "soft_gate",
        hard_gate_success_lower: float = 0.0,
        hard_gate_success_upper: float = 0.8,
        threshold_hold_updates: int = 5,
    ):
        if not map_def.maze_cell_cols or not map_def.maze_cell_rows:
            raise ValueError("adaptive target spawning requires map.maze_cell_grid with cols/rows")
        if map_def.valid_target_coords is None or len(map_def.valid_target_coords) == 0:
            raise ValueError("adaptive target spawning requires non-empty valid_target_coords")

        self.map_def = map_def
        self.target_spawn_method = str(target_spawn_method)
        self.static_maze_optimal_path = bool(static_maze_optimal_path)
        self.mode = str(mode)
        if self.mode not in ("soft_gate", "hard_gate"):
            raise ValueError("adaptive_target_spawn_mode must be 'soft_gate' or 'hard_gate'")
        self.hard_gate_success_lower = float(hard_gate_success_lower)
        self.hard_gate_success_upper = float(hard_gate_success_upper)
        if not (0.0 <= self.hard_gate_success_lower <= self.hard_gate_success_upper <= 1.0):
            raise ValueError("hard-gate adaptive spawn thresholds must satisfy 0 <= lower <= upper <= 1")
        self.threshold_hold_updates = int(threshold_hold_updates)
        if self.threshold_hold_updates < 1:
            raise ValueError("adaptive_spawn_threshold_hold_updates must be >= 1")
        self._upper_hold_count = 0
        self._lower_hold_count = 0
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
        self.active_category_count = 1
        self._position_probs = self._hard_gate_position_probs() if self.mode == "hard_gate" else self._uniform_position_probs()

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

        previous_active_categories = tuple(int(c) for c in self.categories[: self.active_category_count])

        if self.mode == "hard_gate":
            active_counts = self.spawn_counts[: self.active_category_count]
            active_successes = self.success_counts[: self.active_category_count]
            active_total = int(active_counts.sum())
            active_rate = float(active_successes.sum() / active_total) if active_total > 0 else 0.0
            if active_total > 0 and active_rate >= self.hard_gate_success_upper:
                self._upper_hold_count += 1
                self._lower_hold_count = 0
            elif active_total > 0 and active_rate < self.hard_gate_success_lower:
                self._lower_hold_count += 1
                self._upper_hold_count = 0
            else:
                self._upper_hold_count = 0
                self._lower_hold_count = 0

            if self._upper_hold_count >= self.threshold_hold_updates:
                self.active_category_count = min(len(self.categories), self.active_category_count + 1)
                self._upper_hold_count = 0
                self._lower_hold_count = 0
            elif self._lower_hold_count >= self.threshold_hold_updates:
                self.active_category_count = max(1, self.active_category_count - 1)
                self._upper_hold_count = 0
                self._lower_hold_count = 0
            self._position_probs = self._hard_gate_position_probs()
        else:
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
            active_categories=tuple(int(c) for c in self.categories[: self.active_category_count]),
            previous_active_categories=previous_active_categories,
        )
        self.spawn_counts[:] = 0
        self.success_counts[:] = 0
        self.actual_counts[:] = 0
        return diag

    def _uniform_position_probs(self) -> np.ndarray:
        return np.full(len(self._target_positions), 1.0 / len(self._target_positions), dtype=np.float32)

    def _hard_gate_position_probs(self) -> np.ndarray:
        active = set(int(c) for c in self.categories[: self.active_category_count])
        mask = np.asarray([int(c) in active for c in self._target_categories], dtype=bool)
        probs = np.zeros(len(self._target_positions), dtype=np.float32)
        count = int(mask.sum())
        if count <= 0:
            return self._uniform_position_probs()
        probs[mask] = 1.0 / float(count)
        return probs


    def exploration_reward_mask(self) -> np.ndarray:
        """Return a 1 m-grid mask for cells eligible for exploration reward.

        Hard-gate mode disables exploration reward only for currently inactive
        target-spawn-valid cells. Cells excluded from target spawning (for
        example no-spawn/exclude zones or wall-clearance rejects) remain
        exploration-reward eligible regardless of the active path category.
        The map coverage grid still updates normally; this mask only controls
        the reward delta.
        """
        active = set(int(c) for c in self.categories[: self.active_category_count])
        width = int(round(float(self.map_def.width)))
        height = int(round(float(self.map_def.height)))
        enabled = np.ones((width, height), dtype=bool)

        valid_inactive = np.asarray([int(cat) not in active for cat in self._target_categories], dtype=bool)
        inactive_positions = self._target_positions[valid_inactive]
        if len(inactive_positions) > 0:
            ix = np.clip(np.floor(inactive_positions[:, 0]).astype(np.int32), 0, width - 1)
            iy = np.clip(np.floor(inactive_positions[:, 1]).astype(np.int32), 0, height - 1)
            enabled[ix, iy] = False
        return enabled

    def _configured_category_rates(self) -> np.ndarray:
        out = np.zeros(len(self.categories), dtype=np.float32)
        for i, cat in enumerate(self.categories):
            mask = self._target_categories == cat
            count = np.sum(mask)
            if count > 0:
                out[i] = float(self._position_probs[mask].sum()) / float(count)
            else:
                out[i] = 0.0
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
        dist: dict[tuple[int, int], int] = {base_cell: 0}
        q: deque[tuple[int, int]] = deque([base_cell])
        while q:
            cx, cy = q.popleft()
            current_center = self._cell_center(cx, cy)
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if nx < 0 or nx >= self.cols or ny < 0 or ny >= self.rows:
                    continue
                if (nx, ny) in dist:
                    continue
                neighbor_center = self._cell_center(nx, ny)
                if not self._edge_is_clear(current_center, neighbor_center):
                    continue
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
        return dist

    def _cell_center(self, cx: int, cy: int) -> np.ndarray:
        return np.array([(cx + 0.5) * self.cell_w, (cy + 0.5) * self.cell_h], dtype=np.float32)

    def _edge_is_clear(self, start: np.ndarray, end: np.ndarray) -> bool:
        occ = self.map_def.occupancy_grid
        if occ is None:
            return True
        delta = end - start
        # Sample at sub-metre spacing so one-cell-thick rasterized wall edges
        # between maze-cell centers block BFS adjacency.
        n = max(2, int(np.ceil(float(np.max(np.abs(delta))) * 2.0)) + 1)
        pts = np.linspace(start, end, n, dtype=np.float32)
        ix = np.clip(np.floor(pts[:, 0]).astype(np.int32), 0, occ.shape[0] - 1)
        iy = np.clip(np.floor(pts[:, 1]).astype(np.int32), 0, occ.shape[1] - 1)
        return not bool(np.any(occ[ix, iy]))



def diagnostics_to_wandb(diag: AdaptiveSpawnDiagnostics) -> dict[str, float]:
    logs: dict[str, float] = {}
    categories = sorted(
        set(diag.success_rate)
        | set(diag.configured_rate)
        | set(diag.actual_pct)
        | set(diag.actual_count)
    )
    for cat in categories:
        label = f"category_{cat:03d}"
        if cat in diag.success_rate:
            logs[f"adaptive_spawn_control/spawn_success_rate/{label}"] = diag.success_rate[cat]
        if cat in diag.configured_rate:
            logs[f"adaptive_spawn_control/spawn_configured_rate/{label}"] = diag.configured_rate[cat]
        if cat in diag.actual_pct:
            logs[f"adaptive_spawn_control/spawn_actual_pct/{label}"] = diag.actual_pct[cat]
    for cat in categories:
        if cat in diag.actual_count:
            label = f"category_{cat:03d}"
            logs[f"actual count/spawn_actual_count/{label}"] = diag.actual_count[cat]
        logs[f"adaptive_spawn_control/spawn_active/category_{cat:03d}"] = float(cat in diag.active_categories)
    return logs
