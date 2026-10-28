"""Host preparation and persistent derived caches for global relay candidates.

No JAX import: geometry preparation is shared by pool workers and diagnostics.
Flight connectivity uses body clearance; radio connectivity uses physical LOS.
"""
from collections import deque
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from zipfile import BadZipFile

import numpy as np

VERSION = "global-candidates-v2-random"
SAMPLING_SEED = 456789
INF = 1000000
MARGIN = .001
_CACHE = {}


@lru_cache(maxsize=1)
def source_identity():
    digest = hashlib.sha256()
    for name in ("relay_candidates.py", "relay_global.py", "obstacles.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def settings(options):
    return {"version": VERSION, "source": source_identity(),
            "sampling": "seeded_uniform_random", "sampling_seed": SAMPLING_SEED,
            "candidate_spacing_fraction": options.candidate_spacing_fraction,
            "max_candidates_per_map": options.max_candidates_per_map}


def range_limit(radius):
    return radius - min(MARGIN, .01 * radius)


def visible(start, ends, lo, hi, margin):
    """Slab LOS in bounded host chunks; same endpoint epsilon as simulation."""
    ends = np.asarray(ends, np.float32).reshape(-1, 3)
    result = np.ones(len(ends), bool)
    for offset in range(0, len(ends), 128):
        endpoint = ends[offset:offset + 128]
        delta = endpoint[:, None] - start
        parallel = np.abs(delta) <= 1e-5
        safe = np.where(parallel, 1., delta)
        first, last = (lo - margin - start) / safe, (hi + margin - start) / safe
        near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
        far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
        outside = np.any(parallel & ((start < lo - margin) | (start > hi + margin)), axis=-1)
        result[offset:offset + len(endpoint)] = ~np.any(
            (far >= np.maximum(near, 1e-5)) & (near <= 1 - 1e-5) & ~outside, axis=-1)
    return result


def legal(points, lo, hi, lower, upper, clearance):
    points = np.asarray(points, np.float32).reshape(-1, 3)
    result = np.all(np.isfinite(points) & (points >= lower) & (points <= upper), axis=1)
    for offset in range(0, len(points), 128):
        chunk = points[offset:offset + 128]
        result[offset:offset + len(chunk)] &= ~np.any(np.all(
            (chunk[:, None] >= lo - clearance) & (chunk[:, None] <= hi + clearance), axis=-1), axis=-1)
    return result


def start_position(base, lower, upper, lo, hi, cfg):
    clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4
    start = np.clip(base + np.asarray([0, 0, cfg.drone_radius], np.float32),
                    lower + MARGIN, upper - MARGIN).astype(np.float32)
    for _ in range(len(lo) + 1):
        inside = np.all((start >= lo - clearance) & (start <= hi + clearance), axis=-1)
        if not np.any(inside):
            break
        start[2] = max(start[2], np.max(hi[inside, 2]) + clearance + MARGIN)
    return start


def hops(graph, first, allowed=None):
    allowed = np.ones(len(graph), bool) if allowed is None else allowed
    counts = np.full(len(graph), INF, np.int32)
    parents = np.full(len(graph), -2, np.int32)
    frontier = np.flatnonzero(first & allowed)
    counts[frontier], parents[frontier] = 1, -1
    depth = 1
    while len(frontier):
        edges = graph[frontier]
        fresh = np.flatnonzero(np.any(edges, axis=0) & (counts == INF) & allowed)
        if len(fresh):
            parents[fresh] = frontier[np.argmax(edges[:, fresh], axis=0)]
            counts[fresh] = depth + 1
        frontier, depth = fresh, depth + 1
    return counts, parents


def components(graph):
    labels = np.full(len(graph), -1, np.int32)
    label = 0
    for root in range(len(graph)):
        if labels[root] >= 0:
            continue
        labels[root] = label
        queue = deque([root])
        while queue:
            current = queue.popleft()
            fresh = np.flatnonzero(graph[current] & (labels < 0))
            labels[fresh] = label
            queue.extend(fresh.tolist())
        label += 1
    return labels


def sample_pools(record, cfg, building, fractions, *, seed=SAMPLING_SEED,
                 max_candidates_per_map=None):
    """Nested continuous-uniform legal pools, plus fixed roadmap/launch anchors.

    A fraction requests ceil(world volume / (cell size * fraction)^3) accepted
    random points. Radio range constrains edges, never the sampling budget.
    The fixed draw chunks preserve sample prefixes across density sweeps.
    """
    lo, hi = (np.asarray(record[key], np.float32) for key in ("solid_min", "solid_max"))
    world = np.asarray(record.get("world_size", building.world_size_m), np.float32)
    base = np.asarray(record.get("base_position", building.base_position_m), np.float32)
    vertices = np.asarray(record["vertices"], np.float32).reshape(-1, 3)
    vertices = vertices[np.all(np.isfinite(vertices) & (np.abs(vertices) < INF), axis=1)]
    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                       [building.tile_thickness_m / 2 + cfg.drone_radius], np.float32)
    upper = world - lower
    if np.any(lower >= upper):
        raise ValueError("Building is too small for legal random relay candidates.")
    clearance = max(cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4,
                    cfg.drone_radius + MARGIN)
    start = start_position(base, lower, upper, lo, hi, cfg)
    anchors = np.unique(np.concatenate((vertices, start[None])), axis=0)
    anchors = anchors[legal(anchors, lo, hi, lower, upper, clearance)]
    volume = float(np.prod(world.astype(np.float64)))
    budgets = {}
    for fraction in fractions:
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("Candidate fractions must be finite and positive.")
        try:
            nominal_volume = (building.cell_size_m * fraction) ** 3
        except OverflowError:
            nominal_volume = math.inf
        requested = volume / nominal_volume if nominal_volume > 0 else math.inf
        if not math.isfinite(requested) or requested > np.iinfo(np.int32).max:
            raise ValueError("Requested random candidate budget is too large; increase candidate_spacing_fraction.")
        budgets[fraction] = max(1, math.ceil(requested))
    if not budgets:
        raise ValueError("At least one candidate fraction is required.")
    needed = max(budgets.values())
    total = len(anchors) + needed
    if max_candidates_per_map is not None and total > max_candidates_per_map:
        raise ValueError(f"Global random relay pool requires {total} legal candidates "
                         f"({needed} random + {len(anchors)} anchors), exceeding "
                         f"max_candidates_per_map={max_candidates_per_map}; increase "
                         "candidate_spacing_fraction or the explicit limit. No candidates were truncated.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 104729]))
    accepted = []
    seen = {point.tobytes() for point in anchors}
    drawn = 0
    while len(accepted) < needed:
        proposals = rng.uniform(lower, upper, (2048, 3)).astype(np.float32)
        proposals = proposals[legal(proposals, lo, hi, lower, upper, clearance)]
        for point in proposals:
            key = point.tobytes()
            if key not in seen:
                seen.add(key)
                accepted.append(point)
                if len(accepted) == needed:
                    break
        drawn += 2048
        if drawn >= max(1000000, needed * 1000) and len(accepted) < needed:
            raise ValueError("Cannot fill the requested legal random candidate pool.")
    sequence = np.asarray(accepted, np.float32)
    pools = {fraction: np.concatenate((anchors, sequence[:count])) for fraction, count in budgets.items()}
    metadata = {"sampling": "seeded_uniform_random", "seed_sequence": [seed, 104729],
                "fixed_anchors": len(anchors), "world_volume_m3": volume,
                "random_point_budgets": {str(f): n for f, n in budgets.items()},
                "budget_rule": "ceil(world_volume / (cell_size * fraction)^3)"}
    return pools, start, metadata


def flight_reachable(points, start, lo, hi, clearance):
    """Exact flight visibility BFS without allocating a quadratic graph."""
    remaining = np.ones(len(points), bool)
    frontier = deque([start])
    while frontier and np.any(remaining):
        point = frontier.popleft()
        indices = np.flatnonzero(remaining)
        fresh = indices[visible(point, points[indices], lo, hi, clearance)]
        remaining[fresh] = False
        frontier.extend(points[fresh])
    return ~remaining


def radio_hops(points, base, lo, hi, cfg, reachable):
    """Exact fixed-base unit-hop BFS, generating radio edges only as needed."""
    counts = np.full(len(points), INF, np.int32)
    parents = np.full(len(points), -2, np.int32)
    indices = np.flatnonzero(reachable & (
        np.linalg.norm(points - base, axis=1) <= range_limit(cfg.comm_radius_base)))
    first = indices[visible(base, points[indices], lo, hi, MARGIN)]
    counts[first], parents[first] = 1, -1
    frontier = deque(first.tolist())
    remaining = reachable & (counts == INF)
    while frontier and np.any(remaining):
        parent = frontier.popleft()
        indices = np.flatnonzero(remaining)
        nearby = indices[np.linalg.norm(points[indices] - points[parent], axis=1) <= range_limit(cfg.comm_radius)]
        fresh = nearby[visible(points[parent], points[nearby], lo, hi, MARGIN)]
        counts[fresh], parents[fresh] = counts[parent] + 1, parent
        remaining[fresh] = False
        frontier.extend(fresh.tolist())
    return counts, parents


def prepare_record(record, cfg, building, options, *, randomize_base=False, cache_dir=None):
    """Return unpadded map data plus cached global relay records.

    Stored separately from pool geometry, whose identity and float32 record
    format remain unchanged. No agent-budget dependency or silent truncation.
    """
    lo, hi = (np.asarray(record[key], np.float32) for key in ("solid_min", "solid_max"))
    world = np.asarray(record.get("world_size", building.world_size_m), np.float32)
    base = np.asarray(record.get("base_position", building.base_position_m), np.float32)
    vertices = np.asarray(record["vertices"], np.float32).reshape(-1, 3)
    vertices = vertices[np.all(np.isfinite(vertices) & (np.abs(vertices) < INF), axis=1)]
    lower = np.asarray([building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                       [building.tile_thickness_m / 2 + cfg.drone_radius], np.float32)
    upper = world - lower
    identity = hashlib.sha256()
    parameters = {**settings(options), "randomize_base": randomize_base,
                  "cell_size": building.cell_size_m, "drone_radius": cfg.drone_radius,
                  "planning_clearance": cfg.obstacle_planning_clearance_m,
                  "comm_radius": cfg.comm_radius, "base_radius": cfg.comm_radius_base,
                  "visual_radius": cfg.visual_radius, "margin": MARGIN}
    identity.update(json.dumps(parameters, sort_keys=True).encode())
    for value in (lo, hi, world, base, vertices, lower):
        identity.update(str(value.shape).encode())
        identity.update(value.tobytes())
    key = identity.hexdigest()
    folder = (Path(cache_dir) if cache_dir is not None else
              Path(__file__).resolve().parents[3] / "outputs" / "relay_cache") / VERSION
    path = folder / f"{key}.npz"
    memory_key = (folder.resolve(), key)
    data = _CACHE.get(memory_key)
    if data is None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as saved:
                data = {name: saved[name] for name in saved.files}
            expected = np.frombuffer(bytes.fromhex(key), np.uint8)
            size = int(data["gc_size"])
            if (not np.array_equal(data["gc_identity"], expected)
                    or data["gc_points"].shape != (size, 3)
                    or data["gc_hops"].shape != (size,)
                    or data["gc_parents"].shape != (size,)
                    or data["gc_hops"].dtype != np.int32
                    or data["gc_parents"].dtype != np.int32
                    or data["gc_flight_reachable"].shape != (size,)
                    or int(data["gc_random_size"]) + int(data["gc_anchor_size"]) != size
                    or randomize_base and (data["gc_radio"].shape != (size, size)
                                           or data["gc_components"].shape != (size,))):
                data = None
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            data = None
    if data is None:
        if min(cfg.comm_radius, cfg.comm_radius_base, cfg.visual_radius) <= 0:
            raise ValueError("Global candidates require positive communication and visual ranges.")
        clearance = max(cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4,
                        cfg.drone_radius + MARGIN)
        plan_clearance = cfg.drone_radius + cfg.obstacle_planning_clearance_m - 1e-4
        pools, start, sampling = sample_pools(record, cfg, building,
            (options.candidate_spacing_fraction,), max_candidates_per_map=options.max_candidates_per_map)
        points = pools[options.candidate_spacing_fraction]
        real_size = len(points)
        start_ok = bool(legal(start[None], lo, hi, lower, upper, clearance)[0])
        if randomize_base:
            # Randomized-base JAX queries need the stored radio graph and flight
            # components. Fixed-base maps retain only reachability/hop labels.
            flight = np.zeros((real_size, real_size), bool)
            radio = np.zeros_like(flight)
            for index, point in enumerate(points):
                others = points[index + 1:]
                clear = visible(point, others, lo, hi, plan_clearance)
                flight[index, index + 1:] = clear
                flight[index + 1:, index] = clear
                nearby = np.flatnonzero(np.linalg.norm(others - point, axis=1) <= range_limit(cfg.comm_radius))
                connected = nearby[visible(point, others[nearby], lo, hi, MARGIN)] + index + 1
                radio[index, connected] = radio[connected, index] = True
            labels = components(flight)
            starts = visible(start, points, lo, hi, plan_clearance) & start_ok
            reached_labels = np.zeros(real_size, bool)
            reached_labels[labels[starts]] = True
            reachable = reached_labels[labels]
            first = visible(base, points, lo, hi, MARGIN) & (
                np.linalg.norm(points - base, axis=1) <= range_limit(cfg.comm_radius_base))
            counts, parents = hops(radio, first, reachable)
        else:
            reachable = (flight_reachable(points, start, lo, hi, plan_clearance) if start_ok
                         else np.zeros(real_size, bool))
            counts, parents = radio_hops(points, base, lo, hi, cfg, reachable)
        data = {"gc_points": points.astype(np.float32),
                "gc_hops": counts, "gc_parents": parents,
                "gc_flight_reachable": reachable, "gc_size": np.asarray(len(points), np.int32),
                "gc_real_size": np.asarray(real_size, np.int32),
                "gc_base": base, "gc_start": start,
                "gc_random_size": np.asarray(sampling["random_point_budgets"][str(options.candidate_spacing_fraction)], np.int32),
                "gc_anchor_size": np.asarray(sampling["fixed_anchors"], np.int32),
                "gc_identity": np.frombuffer(bytes.fromhex(key), np.uint8).copy()}
        if randomize_base:
            data["gc_radio"] = radio
            data["gc_components"] = labels
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".{key}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    _CACHE[memory_key] = data
    result = {**record, "world_size": world, **data}
    if not len(vertices):
        result["vertices"] = np.full((1, 3), INF, np.float32)
        for name in ("distances", "corner_distances"):
            if name in result:
                result[name] = np.full((1, 1), INF, np.float32)
    return result
