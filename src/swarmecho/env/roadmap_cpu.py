"""Host-only visibility roadmap compilation shared by runtime and inspection."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from time import perf_counter
import json
import heapq
import math
import numpy as np
from swarmecho.core.terminal import init_message

@dataclass(frozen=True, eq=False)
class RoadmapArrays:
    solid_min: np.ndarray
    solid_max: np.ndarray
    vertices: np.ndarray
    distances: np.ndarray
    clearance: float
    visible_edges: np.ndarray | None = None
    corner_distances: np.ndarray | None = None


def shortest_path(matrix, directed=False, method="D"):
    """Use SciPy when installed; dependency-light inspection has a Dijkstra fallback."""
    try:
        from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
    except ImportError:
        adjacency = [[(int(j), float(matrix[i, j])) for j in np.flatnonzero(np.isfinite(matrix[i])) if j != i]
                     for i in range(len(matrix))]
        result = np.full_like(matrix, np.inf)
        for root in range(len(matrix)):
            result[root, root] = 0.
            queue = [(0., root)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance > result[root, node] + 1e-4:
                    continue
                for target, weight in adjacency[node]:
                    candidate = distance + weight
                    if candidate < result[root, target]:
                        result[root, target] = candidate
                        heapq.heappush(queue, (float(result[root, target]), target))
        return result
    return scipy_shortest_path(matrix, directed=directed, method=method)

def _merged_boxes(lo, hi):
    """Return the exact rectangular union without filling any openings."""
    boxes = [(a.copy(), b.copy()) for a, b in zip(lo, hi)]
    # Equal cross sections and overlapping/touching intervals guarantee that
    # merging fills no doorway or other empty space. Repeat across all axes.
    changed = True
    while changed:
        before = len(boxes)
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            groups = {}
            for a, b in boxes:
                key = tuple(a[others]) + tuple(b[others])
                groups.setdefault(key, []).append((a, b))
            boxes = []
            for group in groups.values():
                group.sort(key=lambda box: float(box[0][axis]))
                a, b = group[0]
                for next_a, next_b in group[1:]:
                    if next_a[axis] <= b[axis]:
                        b[axis] = max(b[axis], next_b[axis])
                    else:
                        boxes.append((a, b))
                        a, b = next_a, next_b
                boxes.append((a, b))
        changed = len(boxes) < before
    return boxes


def authored_roadmap_vertices(obstacle_min, obstacle_max, lower, upper, clearance, edge_spacing):
    """Merge exact rectangular unions, then sample their clearance-offset edges.

    Only roadmap candidates change; collision geometry remains untouched.
    This host-side operation runs once per authored layout, outside JAX resets.
    """
    lo = np.asarray(obstacle_min, dtype=np.float32).reshape(-1, 3)
    hi = np.asarray(obstacle_max, dtype=np.float32).reshape(-1, 3)
    boxes = _merged_boxes(lo, hi)
    samples = []
    for a, b in boxes:
        a, b = np.clip(a - clearance, lower, upper), np.clip(b + clearance, lower, upper)
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            count = max(1, int(np.ceil((b[axis] - a[axis]) / edge_spacing)))
            for sides in product((0, 1), repeat=2):
                edge = np.broadcast_to(a, (count + 1, 3)).copy()
                edge[:, axis] = np.linspace(a[axis], b[axis], count + 1)
                for other, side in zip(others, sides):
                    edge[:, other] = b[other] if side else a[other]
                samples.append(edge)
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    vertices = np.unique(np.concatenate(samples), axis=0)
    expanded_lo, expanded_hi = lo - clearance + 1e-4, hi + clearance - 1e-4
    exposed = [not np.any(np.all((v >= expanded_lo) & (v <= expanded_hi), axis=-1)) for v in vertices]
    return vertices[np.asarray(exposed, dtype=bool)]


def _visible_from(start, ends, expanded_lo, expanded_hi):
    """Visibility of one point to a small set of candidate points."""
    direction = ends[:, None] - start
    parallel = np.abs(direction) <= 1e-5
    safe = np.where(parallel, 1.0, direction)
    first, last = (expanded_lo - start) / safe, (expanded_hi - start) / safe
    near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
    far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
    outside = np.any(parallel & ((start < expanded_lo) | (start > expanded_hi)), axis=-1)
    blocked = np.any((far >= np.maximum(near, 1e-5)) & (near <= 1 - 1e-5) & ~outside, axis=-1)
    return ~blocked


def _merge_perpendicular_wall_corner_nodes(vertices, building, extra, lower, upper, clearance):
    """Merge only the close pair left outside a solid perpendicular wall join.

    An opening face still has solid jambs at its cell boundaries, so it can
    form an L join. Its opening-rim vertices remain protected separately.
    """
    if not len(vertices):
        return vertices

    x_walls, y_walls = building.x_walls, building.y_walls
    cell = building.cell_size_m
    face_offset = building.wall_thickness_m / 2 + clearance
    opening_spans = []
    for axis_name, axis in (("x", 0), ("y", 1)):
        along = 1 - axis
        for kind in ("doors", "windows"):
            for face in extra.get(f"{axis_name}_{kind}", []):
                opening_spans.append((axis, face[axis] * cell,
                    (face[along] + .3) * cell + clearance,
                    (face[along] + .7) * cell - clearance,
                    face[2] * cell, (face[2] + 1) * cell))

    def on_opening_rim(point):
        for axis, plane, width_lo, width_hi, height_lo, height_hi in opening_spans:
            along = 1 - axis
            if (height_lo - 1e-5 <= point[2] <= height_hi + 1e-5
                    and width_lo - 1e-5 <= point[along] <= width_hi + 1e-5
                    and (np.isclose(point[axis], plane - face_offset, atol=1e-5)
                         or np.isclose(point[axis], plane + face_offset, atol=1e-5)
                         or np.isclose(point[axis], plane, atol=1e-5))):
                return True
        return False

    forbidden_lo = building.solid_min_m - clearance + 1e-4
    forbidden_hi = building.solid_max_m + clearance - 1e-4
    result = vertices.copy()
    for story in range(x_walls.shape[2]):
        for x in range(1, x_walls.shape[0] - 1):
            for y in range(1, y_walls.shape[1] - 1):
                south, north = bool(x_walls[x, y - 1, story]), bool(x_walls[x, y, story])
                west, east = bool(y_walls[x - 1, y, story]), bool(y_walls[x, y, story])
                if south == north or west == east:
                    continue  # Only an L join: one solid face on each axis.
                sx = 1 if east else -1
                sy = 1 if north else -1
                centre_x, centre_y = x * cell, y * cell
                # These are the two exposed-tip candidates that survive the
                # perpendicular wall's clearance filter at the outside corner.
                first_xy = (centre_x - sx * face_offset, centre_y - sy * clearance)
                second_xy = (centre_x - sx * clearance, centre_y - sy * face_offset)
                first_mask = (np.isclose(result[:, 0], first_xy[0], atol=1e-5)
                              & np.isclose(result[:, 1], first_xy[1], atol=1e-5))
                second_mask = (np.isclose(result[:, 0], second_xy[0], atol=1e-5)
                               & np.isclose(result[:, 1], second_xy[1], atol=1e-5))
                for height in np.intersect1d(result[first_mask, 2], result[second_mask, 2]):
                    first = np.asarray((*first_xy, height), dtype=np.float32)
                    second = np.asarray((*second_xy, height), dtype=np.float32)
                    first_indices = np.flatnonzero(np.all(np.isclose(result, first, atol=1e-5), axis=1))
                    second_indices = np.flatnonzero(np.all(np.isclose(result, second, atol=1e-5), axis=1))
                    if not len(first_indices) or not len(second_indices):
                        continue
                    fi, si = int(first_indices[0]), int(second_indices[0])
                    if on_opening_rim(first) or on_opening_rim(second):
                        continue
                    if not _visible_from(first, second[None], forbidden_lo, forbidden_hi)[0]:
                        continue
                    replacement = (first + second) / 2
                    if (np.any(replacement < lower) or np.any(replacement > upper)
                            or np.any(np.all((replacement >= forbidden_lo)
                                              & (replacement <= forbidden_hi), axis=1))):
                        continue
                    visible = (_visible_from(first, result, forbidden_lo, forbidden_hi)
                               | _visible_from(second, result, forbidden_lo, forbidden_hi))
                    visible[[fi, si]] = False
                    if not np.all(_visible_from(replacement, result[visible], forbidden_lo, forbidden_hi)):
                        continue
                    result = np.concatenate((np.delete(result, [fi, si], axis=0), replacement[None]), axis=0)
    return np.unique(result, axis=0)


def validate_roadmap_settings(cfg):
    if cfg.roadmap_approach not in ("full", "minimal"):
        raise ValueError("roadmap_approach must be 'full' or 'minimal'.")
    if (isinstance(cfg.roadmap_node_density, bool)
            or not isinstance(cfg.roadmap_node_density, (int, float, np.number))
            or not math.isfinite(cfg.roadmap_node_density)
            or cfg.roadmap_node_density <= 0):
        raise ValueError("roadmap_node_density must be finite and positive.")
    if type(cfg.roadmap_merge_wall_end_nodes) is not bool:
        raise ValueError("roadmap_merge_wall_end_nodes must be boolean.")


def _stair_feature_vertices(building, stairs, clearance, spacing, lower, upper):
    """Sample stair entrance/exit spans and a route above expanded treads."""
    if not stairs:
        return np.empty((0, 3), dtype=np.float32)
    cell = building.cell_size_m
    tread = cell / 8
    side_offset = building.wall_thickness_m / 2 + clearance
    epsilon = 1e-3
    samples = []
    forbidden_lo = building.solid_min_m - clearance + 1e-4
    forbidden_hi = building.solid_max_m + clearance - 1e-4
    for x, y, z, direction in stairs:
        axis = direction % 2
        upward = 1 if direction < 2 else -1
        first_step = min(8, int(clearance // tread) + 1)
        anchors = [(0., first_step * tread + clearance + epsilon)]
        for step in range(first_step, 8):
            threshold = step * tread - clearance - epsilon
            if threshold > 0:
                anchors.append((threshold, (step + 1) * tread + clearance + epsilon))
        anchors.append((cell, cell + clearance + epsilon))
        flight = []
        for (start_u, start_h), (end_u, end_h) in zip(anchors, anchors[1:]):
            intervals = max(1, int(np.ceil((end_u - start_u) / spacing)))
            for u, height in zip(np.linspace(start_u, end_u, intervals + 1),
                                 np.linspace(start_h, end_h, intervals + 1)):
                point = (np.asarray([x, y, z], dtype=np.float32) + .5) * cell
                point[axis] = ((x if axis == 0 else y) + (u / cell if upward > 0 else 1 - u / cell)) * cell
                point[2] = z * cell + height
                flight.append(point)
        original = []
        for layer in (z, z + 1):
            point = (np.asarray([x, y, layer], dtype=np.float32) + .5) * cell
            point[axis] += (-.375 if layer == z else .375) * upward * cell
            original.append(point)
        # Keep the entrance and exit even if the flight itself fails its
        # visibility check. They are real passage features in either case.
        samples.extend(original)
        if any(np.any(point < lower) or np.any(point > upper) or
               np.any(np.all((point >= forbidden_lo) & (point <= forbidden_hi), axis=1))
               for point in flight):
            continue
        route = [original[0], *flight, original[1]]
        if all(_visible_from(a, b[None], forbidden_lo, forbidden_hi)[0]
               for a, b in zip(route, route[1:])):
            across = 1 - axis
            side = (x, y)[across]
            side_lo = side * cell + side_offset
            side_hi = (side + 1) * cell - side_offset
            intervals = max(1, int(np.ceil((side_hi - side_lo) / spacing)))
            for centre in route:
                rung = np.broadcast_to(centre, (intervals + 1, 3)).copy()
                rung[:, across] = np.linspace(side_lo, side_hi, intervals + 1)
                samples.extend(rung)
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    return np.unique(np.asarray(samples, dtype=np.float32), axis=0)


def _compact_building_vertices(building, extra, clearance):
    """Cell and portal anchors for minimal mode or maps without saved hints."""
    cell = building.cell_size_m
    nodes = (np.argwhere(building.interior_cells).astype(np.float32) + .5) * cell
    points = [nodes]
    for axis_name, axis in (("x", 0), ("y", 1)):
        for kind in ("doors", "windows"):
            for face in extra.get(f"{axis_name}_{kind}", []):
                point = (np.asarray(face, dtype=np.float32) + .5) * cell
                point[axis] -= .5 * cell
                if kind == "doors":
                    point[2] -= .15 * cell
                for sign in (-1, 1):
                    candidate = point.copy()
                    candidate[axis] += sign * (building.wall_thickness_m / 2 + clearance + .01)
                    points.append(candidate[None])
    for x, y, z, direction in extra.get("stairs", []):
        for layer in (z, z + 1):
            point = (np.asarray([x, y, layer], dtype=np.float32) + .5) * cell
            sign = 1 if direction < 2 else -1
            point[direction % 2] += (-.375 if layer == z else .375) * sign * cell
            points.append(point[None])
    return np.unique(np.concatenate(points), axis=0)


def _full_building_feature_vertices(building, extra, clearance, spacing, lower, upper):
    """Sample navigable opening rims and exposed wall tips, not solid-box edges.

    The wall arrays describe the completed building. Looking at their neighbours
    makes a straight extension remove the old tip automatically; a perpendicular
    join leaves only candidates on its open side after collision filtering.
    """
    cell = building.cell_size_m
    half_wall = building.wall_thickness_m / 2
    half_tile = building.tile_thickness_m / 2
    face_offset = half_wall + clearance
    samples = []

    def line(start, end):
        start, end = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
        length = float(np.linalg.norm(end - start))
        if length < 1e-5:
            samples.append(start[None])
            return
        intervals = max(1, int(np.ceil(length / spacing)))
        samples.append(np.linspace(start, end, intervals + 1, dtype=np.float32))

    def story_height(story):
        # Adjacent floor slabs occupy both sides of a story boundary. Keeping
        # the drone centre clear of them also works where a stair omits a tile.
        bottom = max(float(lower[2]), story * cell + half_tile + clearance)
        top = min(float(upper[2]), (story + 1) * cell - half_tile - clearance)
        return bottom, top

    for axis_name, axis in (("x", 0), ("y", 1)):
        along = 1 - axis
        walls = building.x_walls if axis == 0 else building.y_walls
        for kind in ("doors", "windows"):
            for face in extra.get(f"{axis_name}_{kind}", []):
                face = np.asarray(face, dtype=np.int32)
                story = int(face[2])
                floor, ceiling = story_height(story)
                width_lo = (float(face[along]) + .3) * cell + clearance
                width_hi = (float(face[along]) + .7) * cell - clearance
                opening_bottom = 0. if kind == "doors" else .3
                height_lo = max(floor, (story + opening_bottom) * cell + clearance)
                height_hi = min(ceiling, (story + .7) * cell - clearance)
                if width_hi < width_lo or height_hi < height_lo:
                    continue
                plane = float(face[axis]) * cell

                def rim_point(side, width, height):
                    point = np.empty(3, dtype=np.float32)
                    point[axis] = plane + side * face_offset
                    point[along] = width
                    point[2] = height
                    return point

                # Both jambs and the lintel have a rim on each wall face.
                for side in (-1, 1):
                    for width in (width_lo, width_hi):
                        line(rim_point(side, width, height_lo),
                             rim_point(side, width, height_hi))
                    line(rim_point(side, width_lo, height_hi),
                         rim_point(side, width_hi, height_hi))
                    if kind == "windows":
                        line(rim_point(side, width_lo, height_lo),
                             rim_point(side, width_hi, height_lo))

                # A door has no sill: one line across its clear floor opening.
                if kind == "doors":
                    bottom_a = rim_point(0, width_lo, height_lo)
                    bottom_b = rim_point(0, width_hi, height_lo)
                    line(bottom_a, bottom_b)

                # Short front-to-back links need no interior node at density 1.
                # At higher density, sample them at every jamb height too.
                height_intervals = max(1, int(np.ceil((height_hi - height_lo) / spacing)))
                for width in (width_lo, width_hi):
                    for height in np.linspace(height_lo, height_hi, height_intervals + 1):
                        if kind == "windows" or height > height_lo + 1e-5:
                            line(rim_point(-1, width, height),
                                 rim_point(1, width, height))

        # A wall tip exists only where the completed wall grid has no adjacent
        # face continuing the same line. Perpendicular joins are resolved by
        # the shared solid-clearance filter after all candidates are built.
        for face in np.argwhere(walls):
            boundary, segment, story = (int(face[axis]), int(face[along]), int(face[2]))
            if boundary in (0, walls.shape[axis] - 1):
                continue  # Exterior walls are sealed by world bounds.
            floor, ceiling = story_height(story)
            if ceiling < floor:
                continue
            for end, direction in ((segment, -1), (segment + 1, 1)):
                if end in (0, walls.shape[along]):
                    continue  # This end joins the exterior boundary.
                neighbour = face.copy()
                neighbour[along] = segment - 1 if direction < 0 else segment + 1
                if walls[tuple(neighbour)]:
                    continue
                for side in (-1, 1):
                    point = np.empty(3, dtype=np.float32)
                    point[axis] = boundary * cell + side * face_offset
                    point[along] = end * cell + direction * clearance
                    point[2] = floor
                    other = point.copy()
                    other[2] = ceiling
                    line(point, other)
                height_intervals = max(1, int(np.ceil((ceiling - floor) / spacing)))
                for height in np.linspace(floor, ceiling, height_intervals + 1):
                    left = np.empty(3, dtype=np.float32)
                    left[axis] = boundary * cell - face_offset
                    left[along] = end * cell + direction * clearance
                    left[2] = height
                    right = left.copy()
                    right[axis] = boundary * cell + face_offset
                    line(left, right)

    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    return np.unique(np.concatenate(samples), axis=0)


def building_candidate_vertices(building, cfg):
    """Select the same building candidates for host and dynamic-obstacle roadmaps."""
    validate_roadmap_settings(cfg)
    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                       [building.tile_thickness_m / 2 + cfg.drone_radius], dtype=np.float32)
    upper = building.world_size_m - lower
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m
    spacing = building.cell_size_m / cfg.roadmap_node_density
    extra = json.loads(getattr(building, "extra_geometry_json", "{}"))
    compact = extra.get("roadmap_nodes_m")
    if cfg.roadmap_approach == "minimal":
        vertices = (np.asarray(compact, dtype=np.float32).reshape(-1, 3) if compact is not None else
                    _compact_building_vertices(building, extra, clearance))
    else:
        # Full building roadmaps use only genuine wall/opening features. Saved
        # compact hints belong to minimal and must not add cell centres.
        feature_spacing = spacing / 5  # 1 m at density 1 for a 5 m cell.
        wall_nodes = _full_building_feature_vertices(building, extra, clearance,
                                                      feature_spacing, lower, upper)
        stair_nodes = _stair_feature_vertices(building, extra.get("stairs", []), clearance, feature_spacing,
                                              lower, upper)
        vertices = np.unique(np.concatenate((wall_nodes, stair_nodes)), axis=0)
    vertices = np.unique(vertices, axis=0)
    forbidden_lo = building.solid_min_m - clearance + 1e-4
    forbidden_hi = building.solid_max_m + clearance - 1e-4
    valid = [np.all((v >= lower) & (v <= upper)) and
             not np.any(np.all((v >= forbidden_lo) & (v <= forbidden_hi), axis=-1)) for v in vertices]
    vertices = vertices[np.asarray(valid, dtype=bool)]
    if cfg.roadmap_approach == "full" and cfg.roadmap_merge_wall_end_nodes:
        vertices = _merge_perpendicular_wall_corner_nodes(vertices, building, extra, lower, upper, clearance)
    return vertices


def shared_roadmap(obstacle_min, obstacle_max, lower, upper, clearance, *, plan,
                   edge_spacing=5.0, corner_bonus_m=0.0, corner_merge_distance_m=0.0,
                   candidate_vertices=None):
    arrays = [np.asarray(a, dtype=np.float32) for a in (obstacle_min, obstacle_max, lower, upper)]
    return _shared_roadmap(*(a.tobytes() for a in arrays), float(clearance), bool(plan),
                           float(edge_spacing), float(corner_bonus_m), float(corner_merge_distance_m),
                           None if candidate_vertices is None else np.asarray(candidate_vertices, dtype=np.float32).tobytes())


def building_roadmap(building, cfg, *, plan=True, corner_bonus_m=0.0):
    validate_roadmap_settings(cfg)
    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                       [building.tile_thickness_m / 2 + cfg.drone_radius], dtype=np.float32)
    return shared_roadmap(building.solid_min_m, building.solid_max_m, lower, building.world_size_m - lower,
        cfg.drone_radius + cfg.obstacle_planning_clearance_m, plan=plan,
        edge_spacing=building.cell_size_m / cfg.roadmap_node_density,
        corner_bonus_m=corner_bonus_m,
        corner_merge_distance_m=building.wall_thickness_m + 2 * (cfg.drone_radius + cfg.obstacle_planning_clearance_m),
        candidate_vertices=building_candidate_vertices(building, cfg) if plan else None)


@lru_cache(maxsize=8)
def _shared_roadmap(min_bytes, max_bytes, lower_bytes, upper_bytes, clearance, plan, edge_spacing,
                    corner_bonus_m, corner_merge_distance_m, candidate_bytes=None):
    """Build static all-pairs distances on the host, once per geometry/configuration."""
    lo = np.frombuffer(min_bytes, dtype=np.float32).reshape(-1, 3)
    hi = np.frombuffer(max_bytes, dtype=np.float32).reshape(-1, 3)
    lower = np.frombuffer(lower_bytes, dtype=np.float32)
    upper = np.frombuffer(upper_bytes, dtype=np.float32)
    vertices = np.empty((0, 3), dtype=np.float32)
    distances = np.empty((0, 0), dtype=np.float32)
    visible_edges = np.empty((0, 0), dtype=bool)
    corner_distances = distances
    if plan and len(lo):
        started = perf_counter()
        init_message(f"Building shared geodesic roadmap: {len(lo)} authored solids")
        expanded_lo, expanded_hi = lo - clearance + 1e-4, hi + clearance - 1e-4
        if candidate_bytes is not None:
            vertices = np.unique(np.frombuffer(candidate_bytes, dtype=np.float32).reshape(-1, 3), axis=0)
            valid = [np.all((v >= lower) & (v <= upper)) and
                     not np.any(np.all((v >= expanded_lo) & (v <= expanded_hi), axis=-1)) for v in vertices]
            vertices = vertices[np.asarray(valid, dtype=bool)]
        else:
            vertices = authored_roadmap_vertices(lo, hi, lower, upper, clearance, edge_spacing)
        distances = np.full((len(vertices), len(vertices)), np.inf, dtype=np.float32)
        for index, start in enumerate(vertices):
            # Bound temporary segment/solid arrays instead of allocating V*V*S.
            for offset in range(index + 1, len(vertices), 128):
                ends = vertices[offset:offset + 128]
                direction = ends[:, None] - start
                parallel = np.abs(direction) <= 1e-5
                safe = np.where(parallel, 1.0, direction)
                first, last = (expanded_lo - start) / safe, (expanded_hi - start) / safe
                near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
                far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
                outside = np.any(parallel & ((start < expanded_lo) | (start > expanded_hi)), axis=-1)
                blocked = np.any((far >= np.maximum(near, 1e-5)) & (near <= 1 - 1e-5) & ~outside, axis=-1)
                weights = np.where(blocked, np.inf, np.linalg.norm(ends - start, axis=-1))
                distances[index, offset:offset + len(ends)] = weights
                distances[offset:offset + len(ends), index] = weights
        np.fill_diagonal(distances, 0)
        visible_edges = np.isfinite(distances)
        np.fill_diagonal(visible_edges, False)
        # Entering the first waypoint group is charged at query time. Short
        # internal links stay in that group; longer links enter another one.
        corner_distances = distances + np.where(
            visible_edges & (distances > corner_merge_distance_m + 1e-4), corner_bonus_m, 0.)
        if len(vertices):
            distances = shortest_path(distances, directed=False, method="D").astype(np.float32)
            distances = np.minimum(distances, 1e6)
            corner_distances = (np.minimum(shortest_path(corner_distances, directed=False, method="D"), 1e6)
                                .astype(np.float32)) if corner_bonus_m else distances
        init_message(f"Shared geodesic roadmap ready: {len(vertices)} vertices in {perf_counter() - started:.1f}s")
    for array in (vertices, distances, visible_edges, corner_distances):
        array.setflags(write=False)
    return RoadmapArrays(lo, hi, vertices, distances, clearance, visible_edges, corner_distances)


