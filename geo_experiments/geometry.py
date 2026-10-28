"""Obstacle-aware Manhattan distance on the authored six-neighbour cell graph.

An edge costs one cell transition. Shared-face portals are tested against the
compiled solid AABBs expanded by drone planning clearance, including door
frames and stair treads. This is a cell-topology metric, not a metre geodesic;
it abstracts motion inside a cell and does not reconstruct a flight path.
"""
from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np


def _portal(left, axis, cell_size, solid_min, solid_max):
    """Find a free patch on the face to the positive-axis neighbouring cell.

    Rectangle boundaries partition the face exactly for AABB solids, avoiding
    a raster resolution or a centre-to-centre test that would miss openings.
    """
    plane = (left[axis] + 1) * cell_size
    other = [a for a in range(3) if a != axis]
    low = np.asarray(left, dtype=float)[other] * cell_size
    high = low + cell_size
    crosses = (solid_min[:, axis] <= plane) & (solid_max[:, axis] >= plane)
    lo, hi = solid_min[crosses][:, other], solid_max[crosses][:, other]
    cuts = [np.unique(np.concatenate(([low[a], high[a]],
                                    np.clip(lo[:, a], low[a], high[a]),
                                    np.clip(hi[:, a], low[a], high[a])))) for a in range(2)]
    for i, j in product(range(len(cuts[0]) - 1), range(len(cuts[1]) - 1)):
        lower = np.array([cuts[0][i], cuts[1][j]])
        upper = np.array([cuts[0][i + 1], cuts[1][j + 1]])
        if np.any(upper - lower < 1e-6):
            continue
        point = (lower + upper) / 2
        if np.any(np.all((point >= lo) & (point <= hi), axis=-1)):
            continue
        xyz = np.empty(3)
        xyz[axis], xyz[other] = plane, point
        return xyz
    return None


def build_cell_graph(building, clearance):
    # Fixed order preserves X neighbours, then Y, then storeys in figures.
    cells = np.asarray(sorted(map(tuple, np.argwhere(building.interior_cells)),
                              key=lambda p: (p[2], p[1], p[0])), dtype=np.int32)
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    lo = np.asarray(building.solid_min_m, dtype=float) - clearance
    hi = np.asarray(building.solid_max_m, dtype=float) + clearance
    adjacency = np.zeros((len(cells), len(cells)), dtype=bool)
    portals, edges = [], []
    for i, cell in enumerate(cells):
        for axis in range(3):
            neighbour = cell.copy()
            neighbour[axis] += 1
            j = lookup.get(tuple(neighbour))
            if j is None:
                continue
            point = _portal(cell, axis, building.cell_size_m, lo, hi)
            if point is not None:
                adjacency[i, j] = adjacency[j, i] = True
                edges.append([i, j])
                portals.append(point)
    distances = np.full(adjacency.shape, np.inf, dtype=np.float32)
    for root in range(len(cells)):
        distances[root, root] = 0
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour in np.flatnonzero(adjacency[current]):
                if np.isinf(distances[root, neighbour]):
                    distances[root, neighbour] = distances[root, current] + 1
                    queue.append(int(neighbour))
    return {"cells": cells, "adjacency": adjacency, "distances": distances,
            "edges": np.asarray(edges, dtype=np.int32).reshape(-1, 2),
            "portals_m": np.asarray(portals, dtype=np.float32).reshape(-1, 3),
            "clearance_m": np.asarray(clearance), "cell_size_m": np.asarray(building.cell_size_m)}


def target_labels(dataset, graph, *, position_key="target_position", cell_key="target_cell"):
    size = float(graph["cell_size_m"])
    actual = np.floor(dataset[position_key] / size).astype(np.int32)
    base = np.floor(dataset["base_position"] / size).astype(np.int32)
    if not np.all(base == base[0]):
        raise ValueError("Phase 0 requires a fixed base cell.")
    lookup = {tuple(cell): i for i, cell in enumerate(graph["cells"])}
    try:
        cell_labels = np.asarray([lookup[tuple(cell)] for cell in actual], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"{position_key} outside interior cells: {exc}") from exc
    storey_offsets = np.unique(graph["cells"][:, 2] - base[0, 2])
    storey_labels = np.searchsorted(storey_offsets, actual[:, 2] - base[:, 2]).astype(np.int32)
    return {cell_key: cell_labels, "target_storey": storey_labels,
            "target_cells": actual, "base_cell": base[0], "actual_storey": actual[:, 2] + 1,
            "cell_offsets": graph["cells"] - base[0], "storey_offsets": storey_offsets}


def prediction_errors(task, predicted, indices, labels, graph):
    """Discrete error: valid cell transitions, or absolute storeys off."""
    if task in ("target_cell", "own_cell"):
        true_cells = labels[task][indices]
        return graph["distances"][true_cells, predicted]
    actual = labels["target_storey"][indices]
    return np.abs(labels["storey_offsets"][predicted] - labels["storey_offsets"][actual])
