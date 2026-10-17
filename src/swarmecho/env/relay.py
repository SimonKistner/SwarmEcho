"""Bounded JAX relay witnesses for spawn acceptance; no host callback in reset.

The research record and standalone inspection commands live in max_geo_distance/.
Counts are constructive upper bounds, never claims of global optimality.
"""
import hashlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.env.obstacles import segments_blocked

MARGIN = 1e-3
INF = 1e6


def opening_arrays(building):
    extras = json.loads(building.extra_geometry_json)
    result = []
    cell = building.cell_size_m
    for axis_name, axis in (("x", 0), ("y", 1)):
        along = 1 - axis
        for kind in ("doors", "windows"):
            for face in extras.get(f"{axis_name}_{kind}", []):
                lo = np.asarray(face, float) * cell
                hi = lo + cell
                plane = face[axis] * cell
                lo[axis], hi[axis] = plane - building.wall_thickness_m / 2, plane + building.wall_thickness_m / 2
                lo[along], hi[along] = (face[along] + .3) * cell, (face[along] + .7) * cell
                lo[2] = (face[2] + (0 if kind == "doors" else .3)) * cell
                hi[2] = (face[2] + .7) * cell
                result.append([axis, plane, *lo, *hi])
    return np.asarray(result, np.float32).reshape(-1, 8)


_RECORD_CACHE = {}


def prepare_record(record, cfg, building=None):
    """Reuse map-bank geometry; add predecessor routes and opening metadata once."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from swarmecho.env.roadmap_cpu import _visible_from

    vertices = np.asarray(record["vertices"], np.float32)
    lo, hi = np.asarray(record["solid_min"], np.float32), np.asarray(record["solid_max"], np.float32)
    openings = opening_arrays(building) if building is not None else record.get("relay_openings", np.empty((0, 8), np.float32))
    key = hashlib.sha256()
    for array in (vertices, lo, hi, openings):
        key.update(np.asarray(array).tobytes())
        key.update(str(np.shape(array)).encode())
    key.update(str(cfg.drone_radius + cfg.obstacle_planning_clearance_m).encode())
    identity = key.hexdigest()
    if identity not in _RECORD_CACHE:
        clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4
        expanded_lo, expanded_hi = lo - clearance, hi + clearance
        weights = np.zeros((len(vertices), len(vertices)), np.float64)
        edges = []
        for index, point in enumerate(vertices):
            visible = _visible_from(point, vertices, expanded_lo, expanded_hi)
            visible[index] = False
            weights[index] = np.where(visible, np.linalg.norm(vertices - point, axis=1), 0)
            edges.extend((index, int(other)) for other in np.flatnonzero(visible) if other > index)
        distances, predecessors = dijkstra(csr_matrix(weights), directed=False, return_predecessors=True)
        # Nonempty padded arrays also support empty convex-map roadmaps/opening lists.
        size = max(1, len(vertices))
        data = {
            "relay_distances": np.full((size, size), INF, np.float32),
            "relay_predecessors": np.full((size, size), -1, np.int32),
            "relay_nodes": np.asarray(len(vertices), np.int32),
            "relay_openings": openings if len(openings) else np.zeros((1, 8), np.float32),
            "relay_opening_count": np.asarray(len(openings), np.int32),
            "relay_edges": np.asarray(edges or [(0, 0)], np.int32),
            "relay_edge_count": np.asarray(len(edges), np.int32),
        }
        data["relay_distances"][:len(vertices), :len(vertices)] = np.minimum(distances, INF)
        data["relay_predecessors"][:len(vertices), :len(vertices)] = np.where(predecessors >= 0, predecessors, -1)
        _RECORD_CACHE[identity] = data
    result = {**record, **_RECORD_CACHE[identity]}
    if not len(vertices):
        result["vertices"] = np.full((1, 3), INF, np.float32)
        for name in ("distances", "corner_distances"):
            if name in result:
                result[name] = np.full((1, 1), INF, np.float32)
    return result


def make_calculator(cfg, geometry, building, bank=None, *, return_witness=False):
    """Return (base,target,map_id)->(constructed k, valid), using static map data."""
    if bank is None:
        record = prepare_record(dict(vertices=geometry.vertices, solid_min=geometry.solid_min,
                                     solid_max=geometry.solid_max), cfg, building)
        if not len(record["vertices"]):
            record["vertices"] = np.zeros((1, 3), np.float32)
        arrays = {name: jnp.asarray(value) for name, value in record.items()}
        get = lambda name, map_id: arrays[name]
        max_vertices = arrays["vertices"].shape[0]
    else:
        get = bank.get
        max_vertices = bank.arrays["vertices"].shape[1]
    diagonal = float(np.linalg.norm(np.asarray(building.world_size_m)))
    min_range = min(cfg.comm_radius, cfg.comm_radius_base)
    slots = (max_vertices + 2) * (math.ceil(diagonal / max(min_range * .98, 1e-6)) + 1) + 4
    lower = jnp.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                        [building.tile_thickness_m / 2 + cfg.drone_radius], jnp.float32)
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4

    def radius(value, construction=False):
        return value - (2 if construction else 1) * jnp.minimum(MARGIN, .01 * value)

    def calculate(base, target, map_id=0):
        base, target = jnp.asarray(base, jnp.float32), jnp.asarray(target, jnp.float32)
        vertices = get("vertices", map_id)
        node_count = get("relay_nodes", map_id).astype(jnp.int32)
        distances = get("relay_distances", map_id)
        predecessors = get("relay_predecessors", map_id).astype(jnp.int32)
        lo, hi = get("solid_min", map_id), get("solid_max", map_id)
        plan_lo, plan_hi = lo - clearance, hi + clearance
        upper = (get("world_size", map_id) if bank is not None else jnp.asarray(building.world_size_m)) - lower
        openings = get("relay_openings", map_id)
        opening_count = get("relay_opening_count", map_id).astype(jnp.int32)

        def point_ok(point):
            inside = jnp.any(jnp.all((point >= plan_lo) & (point <= plan_hi), axis=-1))
            body = jnp.any(jnp.all((point >= lo - cfg.drone_radius - MARGIN) &
                                   (point <= hi + cfg.drone_radius + MARGIN), axis=-1))
            return jnp.all(jnp.isfinite(point)) & jnp.all(point >= lower) & jnp.all(point <= upper) & ~inside & ~body

        def link_ok(a, b, reach):
            return ((jnp.linalg.norm(a - b, axis=-1) <= radius(reach)) &
                    ~segments_blocked(a, b, lo - MARGIN, hi + MARGIN))

        start = jnp.clip(base + jnp.asarray([0, 0, cfg.drone_radius]), lower + MARGIN, upper - MARGIN)
        def lift(_, point):
            inside = jnp.all((point >= plan_lo) & (point <= plan_hi), axis=-1)
            top = jnp.maximum(point[2], jnp.max(jnp.where(inside, plan_hi[:, 2] + MARGIN, -jnp.inf), initial=-jnp.inf))
            return point.at[2].set(top)
        def needs_lift(value):
            point, attempts = value
            return (attempts <= lo.shape[0]) & jnp.any(jnp.all((point >= plan_lo) & (point <= plan_hi), axis=-1))
        start, _ = jax.lax.while_loop(needs_lift,
                                     lambda value: (lift(0, value[0]), value[1] + 1),
                                     (start, jnp.int32(0)))
        node_valid = jnp.arange(max_vertices) < node_count
        attach = ~segments_blocked(start, vertices, plan_lo, plan_hi) & node_valid
        entry = jnp.where(attach, jnp.linalg.norm(vertices - start, axis=-1), INF)
        all_costs = entry[:, None] + distances
        roots = jnp.argmin(all_costs, axis=0).astype(jnp.int32)
        costs = jnp.min(all_costs, axis=0)
        endpoint_clear = ~segments_blocked(target, vertices, plan_lo, plan_hi) & node_valid
        endpoint_cost = jnp.where(endpoint_clear, costs + jnp.linalg.norm(vertices - target, axis=-1), INF)
        end = jnp.argmin(endpoint_cost).astype(jnp.int32)
        via = endpoint_cost[end]
        direct = (~segments_blocked(start, target, plan_lo, plan_hi) &
                  (jnp.linalg.norm(target - start) <= via))
        finder = target

        def recover(_):
            edges = get("relay_edges", map_id).astype(jnp.int32)
            edge_count = get("relay_edge_count", map_id).astype(jnp.int32)
            a, b = edges[:, 0], edges[:, 1]
            edge = vertices[b] - vertices[a]
            fraction = jnp.clip(jnp.sum((target - vertices[a]) * edge, axis=-1) /
                                jnp.maximum(jnp.sum(edge * edge, axis=-1), 1e-20), 0, 1)
            projected = vertices[a] + fraction[:, None] * edge
            first = costs[a] + jnp.linalg.norm(projected - vertices[a], axis=-1)
            second = costs[b] + jnp.linalg.norm(projected - vertices[b], axis=-1)
            origins = jnp.concatenate((jnp.asarray([-1], jnp.int32), jnp.arange(max_vertices, dtype=jnp.int32), jnp.where(first <= second, a, b)))
            points = jnp.concatenate((start[None], vertices, projected))
            candidate_cost = jnp.concatenate((jnp.asarray([0.]), costs, jnp.minimum(first, second)))
            real = jnp.concatenate((jnp.asarray([True]), node_valid,
                                    jnp.arange(edges.shape[0]) < edge_count))
            valid = real & (candidate_cost < INF) & jax.vmap(point_ok)(points)
            valid &= link_ok(points, target, cfg.visual_radius)
            choice = jnp.argmin(jnp.where(valid, candidate_cost, INF))
            return origins[choice], points[choice], jnp.any(valid)

        needs_recovery = ~direct & (via >= INF)
        end, finder, recovered = jax.lax.cond(
            needs_recovery, recover, lambda _: (end, finder, jnp.bool_(True)), operand=None)
        root = roots[jnp.maximum(end, 0)]
        path = jnp.zeros((max_vertices + 1, 3), jnp.float32)
        def backward(carry):
            index, count, result = carry
            result = result.at[count].set(vertices[index])
            index = jnp.where(index == root, -1, predecessors[root, index])
            return index, count + 1, result
        _, path_count, path = jax.lax.while_loop(
            lambda carry: (carry[0] >= 0) & (carry[1] < max_vertices), backward,
            (jnp.where(direct | (end < 0), -1, end), jnp.int32(0), path))
        route = jnp.zeros((slots, 3), jnp.float32).at[0].set(base).at[1].set(start)
        count = jnp.int32(1)

        def append_leg(carry, endpoint):
            route, count = carry
            a = route[count]
            delta = endpoint - a
            length = jnp.linalg.norm(delta)
            reach = radius(cfg.comm_radius, True)
            extra = jnp.maximum(0, jnp.ceil(length / reach).astype(jnp.int32) - 1)
            unit = delta / jnp.maximum(length, 1e-20)
            def add(index, value):
                result, n = value
                return result.at[n + 1].set(a + unit * reach * (index + 1)), n + 1
            route, count = jax.lax.fori_loop(0, extra, add, (route, count))
            return route.at[count + 1].set(endpoint), count + 1

        def path_leg(index, carry):
            return append_leg(carry, path[path_count - index - 1])
        route, count = jax.lax.fori_loop(0, path_count, path_leg, (route, count))
        # A recovered edge projection is a protected reachable finder endpoint.
        route, count = jax.lax.cond(needs_recovery & (jnp.linalg.norm(finder - route[count]) > 1e-7),
                                    lambda carry: append_leg(carry, finder), lambda carry: carry, (route, count))
        delta = target - route[count]
        advance = jnp.maximum(0., jnp.linalg.norm(delta) - radius(cfg.visual_radius, True))
        destination = route[count] + delta / jnp.maximum(jnp.linalg.norm(delta), 1e-20) * advance
        route, count = jax.lax.cond(advance > 0, lambda carry: append_leg(carry, destination),
                                  lambda carry: carry, (route, count))
        route = route.at[count + 1].set(target)

        def whole_ok(route, n):
            def inspect(index, valid):
                reach = jnp.where(index == 1, cfg.comm_radius_base, cfg.comm_radius)
                return valid & point_ok(route[index]) & link_ok(route[index - 1], route[index], reach)
            valid = jax.lax.fori_loop(1, n + 1, inspect, jnp.bool_(True))
            return valid & link_ok(route[n], target, cfg.visual_radius)

        baseline_ok = point_ok(start) & recovered & whole_ok(route, count)

        def forbidden(observer, xy, low, high):
            direction = xy - observer[:2]
            parallel = jnp.abs(direction) <= 1e-5
            safe = jnp.where(parallel, 1., direction)
            first, last = (low[:, :2] - observer[:2]) / safe, (high[:, :2] - observer[:2]) / safe
            near = jnp.max(jnp.where(parallel, -jnp.inf, jnp.minimum(first, last)), axis=-1)
            far = jnp.min(jnp.where(parallel, jnp.inf, jnp.maximum(first, last)), axis=-1)
            outside = jnp.any(parallel & ((observer[:2] < low[:, :2]) | (observer[:2] > high[:, :2])), axis=-1)
            enter, leave = jnp.maximum(near, 1e-5), jnp.minimum(far, 1 - 1e-5)
            active = (leave >= enter) & ~outside
            enter, leave = jnp.maximum(enter, 1e-20), jnp.maximum(leave, 1e-20)
            first_z = observer[2] + jnp.minimum((low[:, 2] - observer[2]) / enter,
                                                (low[:, 2] - observer[2]) / leave)
            last_z = observer[2] + jnp.maximum((high[:, 2] - observer[2]) / enter,
                                               (high[:, 2] - observer[2]) / leave)
            return first_z - MARGIN, last_z + MARGIN, active

        def connector(a, b, c, d, left_base, right_target):
            incoming, outgoing = b[:2] - a[:2], c[:2] - d[:2]
            incoming /= jnp.maximum(jnp.linalg.norm(incoming), 1e-20)
            outgoing /= jnp.maximum(jnp.linalg.norm(outgoing), 1e-20)
            determinant = incoming[0] * (-outgoing[1]) - incoming[1] * (-outgoing[0])
            difference = c[:2] - b[:2]
            safe_det = jnp.where(jnp.abs(determinant) > 1e-10, determinant, 1.)
            first = (-difference[0] * outgoing[1] + difference[1] * outgoing[0]) / safe_det
            second = (incoming[0] * difference[1] - incoming[1] * difference[0]) / safe_det
            xy = b[:2] + first * incoming
            valid = ((jnp.linalg.norm(b[:2] - a[:2]) > 1e-8) &
                     (jnp.linalg.norm(c[:2] - d[:2]) > 1e-8) &
                     (jnp.abs(determinant) >= 1e-10) & (first >= 0) & (second >= 0))
            left_range = jnp.where(left_base, radius(cfg.comm_radius_base, True), radius(cfg.comm_radius, True))
            right_range = jnp.where(right_target, radius(cfg.visual_radius, True), radius(cfg.comm_radius, True))
            left_squared, right_squared = jnp.sum((xy - a[:2]) ** 2), jnp.sum((xy - d[:2]) ** 2)
            valid &= (left_squared <= left_range ** 2) & (right_squared <= right_range ** 2)
            extent_a = jnp.sqrt(jnp.maximum(0., left_range ** 2 - left_squared))
            extent_d = jnp.sqrt(jnp.maximum(0., right_range ** 2 - right_squared))
            low = jnp.maximum(lower[2], jnp.maximum(a[2] - extent_a + 1e-8, d[2] - extent_d + 1e-8))
            high = jnp.minimum(upper[2], jnp.minimum(a[2] + extent_a - 1e-8, d[2] + extent_d - 1e-8))
            inside_xy = jnp.all((xy >= plan_lo[:, :2]) & (xy <= plan_hi[:, :2]), axis=-1)
            p = (plan_lo[:, 2] - MARGIN, plan_hi[:, 2] + MARGIN, inside_xy)
            fa = forbidden(a, xy, lo - MARGIN, hi + MARGIN)
            fd = forbidden(d, xy, lo - MARGIN, hi + MARGIN)
            fb = forbidden(b, xy, plan_lo, plan_hi)
            fc = forbidden(c, xy, plan_lo, plan_hi)
            starts = jnp.concatenate([jnp.where(item[2], item[0], jnp.inf) for item in (p, fa, fd, fb, fc)])
            ends = jnp.concatenate([jnp.where(item[2], item[1], jnp.inf) for item in (p, fa, fd, fb, fc)])
            solid_count = lo.shape[0]
            kind = jnp.repeat(jnp.asarray([0, 0, 0, 1, 2]), solid_count)
            # Sweep interval endpoints once. Common blocks must be absent;
            # at least one of the two planning connections must be unblocked.
            z = jnp.concatenate((starts, ends, jnp.asarray([low, high])))
            delta = jnp.concatenate((jax.nn.one_hot(kind, 3, dtype=jnp.int32),
                                      -jax.nn.one_hot(kind, 3, dtype=jnp.int32), jnp.zeros((2, 3), jnp.int32)))
            order = jnp.argsort(z, stable=True)
            z, delta = z[order], delta[order]
            coverage = jnp.cumsum(delta, axis=0)[:-1]
            segment_low = jnp.maximum(z[:-1] + 1e-5, low)
            segment_high = jnp.minimum(z[1:] - 1e-5, high)
            allowed = ((coverage[:, 0] == 0) & ((coverage[:, 1] == 0) | (coverage[:, 2] == 0)) &
                       (segment_low <= segment_high) & jnp.isfinite(segment_low) & jnp.isfinite(segment_high))
            preferred = (b[2] + c[2]) / 2
            height = jnp.clip(preferred, segment_low, segment_high)
            choice = jnp.argmin(jnp.where(allowed, jnp.abs(height - preferred), jnp.inf))
            point = jnp.asarray([xy[0], xy[1], height[choice]])
            return point, valid & jnp.any(allowed)

        def doorway(a, b, c, d, left_base, right_target):
            axes = openings[:, 0].astype(jnp.int32)
            lows, highs = openings[:, 2:5], openings[:, 5:8]
            first, last = b[axes], c[axes]
            rows = jnp.arange(openings.shape[0])
            low_face, high_face = lows[rows, axes], highs[rows, axes]
            matches = (((first <= low_face + 1e-6) & (last >= high_face - 1e-6)) |
                       ((last <= low_face + 1e-6) & (first >= high_face - 1e-6)))
            for faces in (low_face, high_face):
                fraction = (faces - first) / jnp.where(jnp.abs(last - first) > 1e-8, last - first, 1.)
                crossing = b + fraction[:, None] * (c - b)
                matches &= jnp.all((crossing >= lows - 1e-6) & (crossing <= highs + 1e-6), axis=-1)
            matches &= jnp.arange(openings.shape[0]) < opening_count
            index = jnp.argmax(matches)
            opening = openings[index]
            axis = opening[0].astype(jnp.int32)
            tangent = jnp.asarray([1 - axis, 2])
            low_opening, high_opening = opening[2:5], opening[5:8]
            margin = cfg.drone_radius + cfg.obstacle_planning_clearance_m
            low = jnp.maximum(low_opening[tangent] + margin + MARGIN, lower[tangent])
            high = jnp.minimum(high_opening[tangent] - margin - MARGIN, upper[tangent])
            reaches = (jnp.where(left_base, cfg.comm_radius_base, cfg.comm_radius),
                       jnp.where(right_target, cfg.visual_radius, cfg.comm_radius))
            for observer, reach in zip((a, d), reaches):
                denominator = opening[1] - observer[axis]
                for face in (low_opening[axis], high_opening[axis]):
                    f = (face - observer[axis]) / jnp.where(jnp.abs(denominator) > 1e-12, denominator, 1.)
                    apply = (jnp.abs(denominator) > 1e-12) & (f > 0) & (f < 1)
                    safe = jnp.where(apply, f, 1.)
                    low = jnp.maximum(low, jnp.where(apply, (low_opening[tangent] + MARGIN - (1 - f) * observer[tangent]) / safe, -jnp.inf))
                    high = jnp.minimum(high, jnp.where(apply, (high_opening[tangent] - MARGIN - (1 - f) * observer[tangent]) / safe, jnp.inf))
                reach = reach - 2 * jnp.minimum(MARGIN, .01 * reach)
                extent = jnp.sqrt(jnp.maximum(0., reach ** 2 - (observer[axis] - opening[1]) ** 2))
                low, high = jnp.maximum(low, observer[tangent] - extent), jnp.minimum(high, observer[tangent] + extent)
            point = ((b + c) / 2).at[axis].set(opening[1])
            point = point.at[tangent].set(jnp.clip(point[tangent], low, high))
            return point, ((jnp.sum(matches) == 1) & (jnp.linalg.norm(c - b) > 1e-8) &
                           (jnp.linalg.norm(c - b) < 2) & jnp.all(low <= high))

        def scan(carry, mode):
            route, n = carry
            def advance(value):
                route, n, index = value
                a, b, c, d = route[index - 1], route[index], route[index + 1], route[index + 2]
                left_base, right_target = index == 1, index + 2 == n + 1
                if mode == "prune":
                    reach = jnp.where(index + 1 == n + 1, cfg.visual_radius,
                                       jnp.where(left_base, cfg.comm_radius_base, cfg.comm_radius))
                    point, accepted = b, link_ok(a, c, reach)
                    remove_at = index
                else:
                    if mode == "doorway":
                        point, accepted = doorway(a, b, c, d, left_base, right_target)
                    else:
                        point, accepted = connector(a, b, c, d, left_base, right_target)
                        vectors = jnp.stack((b - a, c - b, d - c))
                        lengths = jnp.linalg.norm(vectors, axis=-1)
                        unit = vectors / jnp.maximum(lengths[:, None], 1e-20)
                        horizontal = jnp.all(jnp.abs(unit[:, 2]) < 1e-6)
                        if mode == "horizontal":
                            angles = jnp.degrees(jnp.arccos(jnp.clip(jnp.sum(unit[:-1] * unit[1:], axis=-1), -1, 1)))
                            normals = jnp.cross(unit[:-1], unit[1:])
                            accepted &= horizontal & jnp.all(5 * jnp.floor(angles / 5 + .5) == 45)
                            accepted &= (normals[0, 2] * normals[1, 2] > 0) & (lengths[1] >= 1) & (lengths[1] < 2)
                        elif mode == "lifted":
                            accepted &= ~horizontal & (lengths[1] < 2)
                        accepted &= jnp.all(lengths > 1e-8)
                    accepted &= (point_ok(point) & link_ok(a, point, jnp.where(left_base, cfg.comm_radius_base, cfg.comm_radius)) &
                                 link_ok(point, d, jnp.where(right_target, cfg.visual_radius, cfg.comm_radius)))
                    accepted &= jnp.any(~segments_blocked(point, jnp.stack((b, c)), plan_lo, plan_hi))
                    remove_at = index + 1
                def replace(_):
                    modified = route if mode == "prune" else route.at[index].set(point)
                    source = jnp.minimum(jnp.arange(slots) + (jnp.arange(slots) >= remove_at), slots - 1)
                    return modified[source], n - 1, jnp.maximum(1, index - (1 if mode == "prune" else 2))
                return jax.lax.cond(accepted, replace, lambda _: (route, n, index + 1), operand=None)
            def condition(value):
                _, n, index = value
                return (n > 1) & (index <= n if mode == "prune" else index + 1 <= n)
            route, n, _ = jax.lax.while_loop(condition, advance, (route, n, jnp.int32(1)))
            return route, n

        def old_rules(carry):
            for mode in ("horizontal", "doorway", "lifted", "prune"):
                carry = scan(carry, mode)
            return carry
        def stabilize(carry, general=False):
            def cycle(value):
                route, n, _ = value
                before = n
                if general:
                    route, n = scan((route, n), "general")
                route, n = old_rules((route, n))
                return route, n, n < before
            route, n, _ = jax.lax.while_loop(lambda value: value[2], cycle, (*carry, jnp.bool_(True)))
            return route, n
        baseline, baseline_count = route, count
        route, count = jax.lax.cond(baseline_ok, lambda carry: stabilize(carry), lambda carry: carry, (route, count))
        route, count = jax.lax.cond(baseline_ok, lambda carry: stabilize(carry, general=True), lambda carry: carry, (route, count))
        valid = whole_ok(route, count)
        count = jnp.where(valid, count, baseline_count)
        if return_witness:
            return jnp.where(valid, route, baseline), count, baseline_ok
        return count, baseline_ok

    return calculate
