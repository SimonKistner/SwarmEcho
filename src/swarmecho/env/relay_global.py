"""JAX target attachments to cached global communication hop labels."""
import jax
import jax.numpy as jnp

from swarmecho.env.obstacles import segments_blocked
from swarmecho.env.relay_candidates import INF, MARGIN, prepare_record, range_limit


def make_calculator(cfg, geometry, building, bank, options, *, randomize_base=False,
                    return_witness=False, cache_dir=None):
    if bank is None:
        record = prepare_record(dict(vertices=geometry.vertices, solid_min=geometry.solid_min,
            solid_max=geometry.solid_max, base_position=building.base_position_m),
            cfg, building, options, randomize_base=randomize_base, cache_dir=cache_dir)
        arrays = {name: jnp.asarray(value) for name, value in record.items()}
        get = lambda name, map_id: arrays[name]
        size = len(record["gc_points"])
    else:
        if "gc_points" not in bank.arrays or randomize_base and "gc_radio" not in bank.arrays:
            raise ValueError("Global relay map bank was not prepared for the selected mode/base settings.")
        get = bank.get
        size = bank.arrays["gc_points"].shape[1]
    lower = jnp.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                        [building.tile_thickness_m / 2 + cfg.drone_radius], jnp.float32)
    planning = cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4
    body = max(planning, cfg.drone_radius + MARGIN)
    slots = size + 4

    def calculate(base, target, map_id=0):
        base, target = jnp.asarray(base, jnp.float32), jnp.asarray(target, jnp.float32)
        lo, hi = get("solid_min", map_id), get("solid_max", map_id)
        upper = get("world_size", map_id) - lower
        points = get("gc_points", map_id)
        real = jnp.arange(size) < get("gc_real_size", map_id)

        def visible(a, b, margin=MARGIN):
            return ~segments_blocked(a, b, lo - margin, hi + margin)

        def legal(point):
            inside = jnp.any(jnp.all((point[..., None, :] >= lo - body) &
                                     (point[..., None, :] <= hi + body), axis=-1), axis=-1)
            return jnp.all(jnp.isfinite(point) & (point >= lower) & (point <= upper), axis=-1) & ~inside

        if randomize_base:
            start = jnp.clip(base + jnp.asarray([0, 0, cfg.drone_radius]), lower + MARGIN, upper - MARGIN)
            def lift(value):
                point, attempts = value
                inside = jnp.all((point >= lo - planning) & (point <= hi + planning), axis=-1)
                top = jnp.max(jnp.where(inside, hi[:, 2] + planning + MARGIN, -jnp.inf), initial=-jnp.inf)
                return point.at[2].set(jnp.maximum(point[2], top)), attempts + 1
            def blocked(value):
                point, attempts = value
                return (attempts <= lo.shape[0]) & jnp.any(jnp.all(
                    (point >= lo - planning) & (point <= hi + planning), axis=-1))
            start, _ = jax.lax.while_loop(blocked, lift, (start, jnp.int32(0)))
            start_ok = legal(start)
            labels = get("gc_components", map_id).astype(jnp.int32)
            seed = visible(start, points, planning) & real & start_ok
            touched = jnp.zeros((size,), jnp.int32).at[labels].max(seed.astype(jnp.int32)) > 0
            reachable = touched[labels] & real
            start_links = visible(start, points) & (jnp.linalg.norm(points - start, axis=-1) <= range_limit(cfg.comm_radius))
            graph = jnp.pad(get("gc_radio", map_id), ((0, 1), (0, 1)))
            graph = graph.at[-1, :-1].set(start_links).at[:-1, -1].set(start_links)
            points = jnp.concatenate((points, start[None]))
            reachable = jnp.concatenate((reachable, jnp.asarray([start_ok])))
            graph &= reachable[:, None] & reachable[None, :]
            first = reachable & visible(base, points) & (
                jnp.linalg.norm(points - base, axis=-1) <= range_limit(cfg.comm_radius_base))
            counts = jnp.where(first, 1, INF).astype(jnp.int32)
            parents = jnp.where(first, -1, -2).astype(jnp.int32)
            def condition(value):
                frontier, _, _, depth = value
                return jnp.any(frontier) & (depth < size + 1)
            def step(value):
                frontier, counts, parents, depth = value
                edges = graph & frontier[:, None]
                fresh = jnp.any(edges, axis=0) & (counts == INF)
                origins = jnp.argmax(edges, axis=0).astype(jnp.int32)
                return fresh, jnp.where(fresh, depth + 1, counts), jnp.where(fresh, origins, parents), depth + 1
            _, counts, parents, _ = jax.lax.while_loop(condition, step, (first, counts, parents, jnp.int32(1)))
            base_ok = start_ok
        else:
            start = get("gc_start", map_id)
            reachable = get("gc_flight_reachable", map_id) & real
            counts, parents = get("gc_hops", map_id).astype(jnp.int32), get("gc_parents", map_id).astype(jnp.int32)
            base_ok = legal(start) & jnp.all(jnp.abs(base - get("gc_base", map_id)) <= 1e-5)
        distances = jnp.linalg.norm(points - target, axis=-1)
        clear = visible(target, points)
        optical = jnp.where(clear & (distances <= range_limit(cfg.visual_radius)), counts, INF)
        optical_index = jnp.argmin(optical).astype(jnp.int32)
        # A finder at the target must be reachable by a drone, independently
        # of whether radio can cross a body-inaccessible opening.
        target_reachable = visible(start, target, planning) | jnp.any(
            visible(target, points, planning) & reachable)
        target_legal = legal(target) & target_reachable
        radio = jnp.where(target_legal & clear & (distances <= range_limit(cfg.comm_radius)) &
                          (counts < INF), counts + 1, INF)
        radio_index = jnp.argmin(radio).astype(jnp.int32)
        add_target = radio[radio_index] < optical[optical_index]
        chosen = jnp.where(add_target, radio_index, optical_index)
        count = jnp.minimum(optical[optical_index], radio[radio_index])
        direct = target_legal & visible(base, target) & (
            jnp.linalg.norm(base - target) <= range_limit(cfg.comm_radius_base))
        count = jnp.where(direct, 1, count).astype(jnp.int32)
        valid = base_ok & (count < INF) & (count >= 1)
        if not return_witness:
            return jnp.where(valid, count, 0), valid
        # Legacy diagnostic layout: base, count drones, target, unused padding.
        route = jnp.zeros((slots, 3), jnp.float32).at[0].set(base)
        nodes = jnp.where(direct, 0, count - add_target.astype(jnp.int32))
        def condition(value):
            index, written, _ = value
            return valid & ~direct & (index >= 0) & (written < nodes) & (written < size + 1)
        def step(value):
            index, written, route = value
            route = route.at[jnp.clip(nodes - written, 1, slots - 2)].set(points[index])
            return parents[index], written + 1, route
        _, written, route = jax.lax.while_loop(condition, step, (chosen, jnp.int32(0), route))
        route = jax.lax.cond(direct, lambda x: x.at[1].set(target), lambda x: x, route)
        route = jax.lax.cond(valid & add_target & ~direct,
            lambda x: x.at[jnp.clip(count, 1, slots - 2)].set(target), lambda x: x, route)
        valid &= direct | (written == nodes)
        route = route.at[jnp.clip(count + 1, 1, slots - 1)].set(target)
        return route, jnp.where(valid, count, 0), valid

    return calculate
