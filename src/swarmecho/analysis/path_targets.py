"""Reward-aligned reference routes and exact quarter-turn cell-mask coordinates.

The reference uses unpenalized visibility-roadmap lengths. Canonical node ordering
breaks equal-cost ties independently of the original world axes. No policy inputs
are augmented with geometry: geometry is ground truth and display metadata only.
"""
from functools import lru_cache

import numpy as np

from swarmecho.core.config import DEFAULT_PATH_GRID_SHAPE


@lru_cache(maxsize=8)
def path_cell_offsets(grid_shape):
    rx, ry, rz = (size // 2 for size in grid_shape)
    return np.array([(x, y, z) for z in range(-rz, rz + 1)
                     for y in range(-ry, ry + 1) for x in range(-rx, rx + 1)], dtype=np.int32)


OFFSETS = path_cell_offsets(DEFAULT_PATH_GRID_SHAPE)
ROTATIONS = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                      [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]], dtype=np.int32)


class NoReferenceRoute(ValueError):
    """The reward's sparse roadmap cannot label an otherwise valid mission."""
    def __init__(self, base_attachments, target_attachments):
        super().__init__("Sampled base/target have no finite reward-reference roadmap route.")
        self.base_attachments = int(base_attachments)
        self.target_attachments = int(target_attachments)


def transform(points, rotation, anchor, *, inverse=False):
    points = np.asarray(points)
    matrix = ROTATIONS[int(rotation)]
    return points @ matrix + anchor if inverse else (points - anchor) @ matrix.T


def transformed_boxes(lower, upper, rotation, anchor):
    a, b = transform(lower, rotation, anchor), transform(upper, rotation, anchor)
    return np.minimum(a, b), np.maximum(a, b)


def _ordered_rows(array):
    return tuple(sorted(map(tuple, np.round(array, 6))))


def canonical_frame(lower, upper, cells, cell_size, base, target):
    """Choose one identical scene representation for all four yaw orientations."""
    centers = (np.asarray(cells) + .5) * cell_size
    # A base on a grid face belongs to multiple incident cells. Choosing one
    # with a world-axis floor rule would break quarter-turn invariance. Resolve
    # that ambiguity together with orientation using the canonical scene key.
    incident = centers[np.all(np.abs(centers - base) <= cell_size / 2 + 1e-6, axis=1)]
    if not len(incident):
        raise ValueError("The base does not belong to a supported interior cell.")
    choices = []
    for anchor in incident:
        for rotation in range(4):
            lo, hi = transformed_boxes(lower, upper, rotation, anchor)
            key = (_ordered_rows(np.column_stack((lo, hi))),
                   _ordered_rows(transform(centers, rotation, anchor)),
                   tuple(np.round(transform(base, rotation, anchor), 6)),
                   tuple(np.round(transform(target, rotation, anchor), 6)))
            choices.append((key, rotation, anchor))
    _, rotation, anchor = min(choices, key=lambda choice: choice[0])
    return rotation, anchor


def visible(start, ends, lower, upper):
    """Clearance-expanded AABB visibility, matching the environment roadmap."""
    from swarmecho.env.roadmap_cpu import _visible_from
    return _visible_from(np.asarray(start), np.asarray(ends), lower, upper)


class ReferencePlanner:
    def __init__(self, building, cfg, record_path=None, graph=None, *, grid_shape=DEFAULT_PATH_GRID_SHAPE):
        from swarmecho.env.roadmap_cpu import building_roadmap, RoadmapArrays
        if graph is not None:
            self.graph = graph
        elif record_path is not None and record_path.is_file():
            # A verified bank already contains the reward's expensive all-pairs
            # matrix. Reuse it instead of rebuilding every map for the probe.
            with np.load(record_path, allow_pickle=False) as saved:
                if not np.array_equal(saved["solid_min"], building.solid_min_m) or not np.array_equal(saved["solid_max"], building.solid_max_m):
                    raise ValueError("Cached roadmap geometry differs from the verified map.")
                self.graph = RoadmapArrays(saved["solid_min"], saved["solid_max"], saved["vertices"], saved["distances"],
                                           cfg.drone_radius + cfg.obstacle_planning_clearance_m)
        else:
            self.graph = building_roadmap(building, cfg, plan=True, corner_bonus_m=0.)
        self.lower = np.asarray(building.solid_min_m)
        self.upper = np.asarray(building.solid_max_m)
        interior = np.asarray(building.interior_cells)
        self.cells = np.argwhere(interior) if interior.dtype == bool else interior
        self.cell_size = float(building.cell_size_m)
        self.grid_shape = tuple(grid_shape)
        extent = np.ptp(self.cells, axis=0) + 1
        extent[:2] = max(extent[:2])  # A canonical yaw rotation may swap X and Y.
        if any(cells - 1 > size // 2 for cells, size in zip(extent, self.grid_shape)):
            raise ValueError(f"Path grid {self.grid_shape} is too small for map extents {extent}.")
        self.plan_lower = self.lower - self.graph.clearance + 1e-4
        self.plan_upper = self.upper + self.graph.clearance - 1e-4

    def route(self, base, target):
        base, target = np.asarray(base), np.asarray(target)
        rotation, anchor = canonical_frame(self.lower, self.upper, self.cells,
                                            self.cell_size, base, target)
        vertices = np.asarray(self.graph.vertices)
        transformed = np.round(transform(vertices, rotation, anchor), 6)
        base_visible = visible(base, vertices, self.plan_lower, self.plan_upper)
        target_visible = visible(target, vertices, self.plan_lower, self.plan_upper)
        direct = bool(visible(base, target[None], self.plan_lower, self.plan_upper)[0])
        ref = float(np.linalg.norm(base - target)) if direct else np.inf
        via, costs = np.inf, None
        if len(vertices):
            a = np.where(base_visible, np.linalg.norm(vertices - base, axis=1), np.inf)
            b = np.where(target_visible, np.linalg.norm(vertices - target, axis=1), np.inf)
            costs = a[:, None] + self.graph.distances + b[None, :]
            via = float(np.min(costs))
        # Spawn feasibility can use the denser relay-candidate graph, including
        # visual reach to a target with no sparse-roadmap attachment. Such an
        # episode has no label for THIS reference; do not invent a fallback path.
        # Check before successor reconstruction: disconnected matrices can use
        # the environment's finite 1e6 sentinel rather than actual infinity.
        if not np.isfinite(min(ref, via)) or min(ref, via) >= 1e6:
            raise NoReferenceRoute(base_visible.sum(), target_visible.sum())
        route = [base]
        if via < ref - 1e-7:
            ref = via
            choices = np.argwhere(np.isclose(costs, via, rtol=0, atol=1e-5))
            if not len(choices):
                raise ValueError("No finite roadmap reference route exists.")
            start, end = min(map(tuple, choices), key=lambda pair: (tuple(transformed[pair[0]]), tuple(transformed[pair[1]])))
            current = start
            route.append(vertices[current])
            visited = {current}
            while current != end:
                remaining = self.graph.distances[current, end]
                edge_lengths = np.linalg.norm(vertices - vertices[current], axis=1)
                candidates = np.flatnonzero(np.isclose(edge_lengths + self.graph.distances[:, end], remaining, rtol=1e-5, atol=1e-4)
                                            & (self.graph.distances[:, end] < remaining - 1e-7))
                candidates = [i for i in candidates if i not in visited]
                candidates.sort(key=lambda i: tuple(transformed[i]))
                legal = visible(vertices[current], vertices[candidates], self.plan_lower, self.plan_upper)
                if not np.any(legal):
                    raise ValueError("Cannot reconstruct a visible shortest-route successor from the reward matrix.")
                current = candidates[int(np.flatnonzero(legal)[0])]
                visited.add(current)
                route.append(vertices[current])
        route.append(target)
        canonical = np.round(transform(np.asarray(route), rotation, anchor), 6)
        length = float(np.linalg.norm(np.diff(canonical, axis=0), axis=1).sum())
        if not np.isclose(length, ref, rtol=1e-5, atol=1e-4):
            raise ValueError(f"Route length {length} differs from reward reference {ref}.")
        mask = rasterize(canonical, self.cell_size, grid_shape=self.grid_shape)
        return dict(mask=mask, rotation=rotation, anchor=anchor, length=ref,
                    points=transform(canonical, rotation, anchor, inverse=True))


def rasterize(points, cell_size, *, grid_shape=DEFAULT_PATH_GRID_SHAPE):
    """Mark segment interiors and endpoint cells; mere corner/face contact adds no cell."""
    points = np.asarray(points, dtype=float) / cell_size
    cells = set()
    radii = tuple(size // 2 for size in grid_shape)

    def add(point):
        cell = tuple(np.floor(point + .5 + 1e-9).astype(int))
        if any(abs(c) > radius for c, radius in zip(cell, radii)):
            raise ValueError(f"Path leaves the supported base-relative cell range: {cell}.")
        cells.add(cell)

    if np.any(np.abs(points[0]) > .5 + 1e-6):
        raise ValueError("The first route endpoint must be in the canonical base cell.")
    cells.add((0, 0, 0))
    add(points[-1])
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        cuts = [0., 1.]
        for axis, radius in enumerate(radii):
            if abs(delta[axis]) > 1e-12:
                cuts.extend(t for face in np.arange(-radius - .5, radius + 1., 1.)
                            if 0 < (t := (face - start[axis]) / delta[axis]) < 1)
        cuts = np.unique(cuts)
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right - left > 1e-10:
                add(start + (left + right) / 2 * delta)
    return np.asarray([tuple(c) in cells for c in path_cell_offsets(tuple(grid_shape))], dtype=bool)
