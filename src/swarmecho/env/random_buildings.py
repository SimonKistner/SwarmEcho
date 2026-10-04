"""Seeded partition-and-connect buildings shared by training and inspection.

Generation is host-only. No filesystem access or RNG occurs in a rollout.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from swarmecho.core.config import RandomBuildingConfig
from swarmecho.env.buildings import BUILDING_FORMAT, compile_building

GENERATOR_VERSION = "partition_connect_v5"

_STAIR_STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _stair_face(x: int, y: int, z: int, direction: int):
    """Return the shared face on one side of a cell."""
    if direction == 0:
        return "x", (x + 1, y, z)
    if direction == 1:
        return "y", (x, y + 1, z)
    if direction == 2:
        return "x", (x, y, z)
    return "y", (x, y, z)


def _stair_footprints(stairs, nx: int, ny: int, nz: int):
    """Check that each stair serves connected, ordinary cells on both levels."""
    occupied = [set() for _ in range(nz)]
    for x, y, z, _ in stairs:
        occupied[z].add((x, y))
        occupied[z + 1].add((x, y))
    cells = {(x, y) for x in range(nx) for y in range(ny)}
    free = [cells - floor for floor in occupied]
    for x, y, z, direction in stairs:
        dx, dy = _STAIR_STEPS[direction]
        if (x - dx, y - dy) not in free[z] or (x + dx, y + dy) not in free[z + 1]:
            return None
    for floor in free:
        if not floor:
            return None
        reached = {min(floor)}
        pending = list(reached)
        while pending:
            x, y = pending.pop()
            for dx, dy in _STAIR_STEPS:
                neighbor = (x + dx, y + dy)
                if neighbor in floor and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        if reached != floor:
            return None
    return free


def _connect_floor(free, z: int, walls, doors, rng):
    """Make every ordinary cell reachable without passing through a window."""
    while True:
        reached = {min(free)}
        pending = list(reached)
        while pending:
            x, y = pending.pop()
            for direction, (dx, dy) in enumerate(_STAIR_STEPS):
                neighbor = (x + dx, y + dy)
                if neighbor not in free or neighbor in reached:
                    continue
                axis, face = _stair_face(x, y, z, direction)
                if face not in walls[axis] or face in doors[axis]:
                    reached.add(neighbor)
                    pending.append(neighbor)
        if reached == free:
            return
        candidates = []
        for x, y in sorted(reached):
            for direction, (dx, dy) in enumerate(_STAIR_STEPS):
                if (x + dx, y + dy) in free - reached:
                    candidates.append(_stair_face(x, y, z, direction))
        axis, face = candidates[int(rng.integers(len(candidates)))]
        doors[axis].add(face)


def random_building_grid(settings: RandomBuildingConfig) -> tuple[int, int, int]:
    """Validate generation settings and return the building grid dimensions."""
    for key in ("length_m", "width_m", "cell_size_m", "wall_thickness_m", "tile_thickness_m"):
        value = getattr(settings, key)
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"random_buildings.{key} must be finite and positive.")
    counts = np.asarray([settings.length_m, settings.width_m]) / settings.cell_size_m
    if not np.allclose(counts, np.round(counts)) or np.any(counts < 2):
        raise ValueError("Random building length/width must be cell-size multiples of at least two cells.")
    for key in ("stories", "rooms_per_story_min", "rooms_per_story_max", "staircase_max"):
        value = getattr(settings, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid random_buildings.{key}.")
    if not 2 <= settings.rooms_per_story_min <= settings.rooms_per_story_max <= int(np.prod(counts)):
        raise ValueError("Random buildings require 2 <= room minimum <= room maximum <= floor cell count.")
    for key in ("extra_door_probability", "window_probability"):
        if not 0 <= getattr(settings, key) <= 1:
            raise ValueError(f"random_buildings.{key} must be in [0,1].")
    if max(settings.wall_thickness_m, settings.tile_thickness_m) >= settings.cell_size_m:
        raise ValueError("Building thickness must be smaller than a cell.")
    if settings.staircase_max >= int(np.prod(counts)):
        raise ValueError("staircase_max must leave at least one cell for the next storey's stairs and the base.")
    for key in ("enabled", "save_training_maps", "load_maps_from_bank"):
        if type(getattr(settings, key)) is not bool:
            raise ValueError(f"random_buildings.{key} must be boolean.")
    return int(round(counts[0])), int(round(counts[1])), settings.stories


def map_seed(seed: int, index: int, *, evaluation: bool = False) -> int:
    return int(np.random.SeedSequence([int(seed), int(index), 1 if evaluation else 0]).generate_state(1)[0])


def generate_building(settings: RandomBuildingConfig, seed: int, *, name: str = "random_building", drone_clearance: float = .35):
    """Return an editable map document and its compiled building.

    Doors form a spanning tree of adjacent rectangular rooms. Windows and
    extra doors add loops. Stair cells have a full-width flight path above the treads.
    """
    nx, ny, nz = random_building_grid(settings)
    c = settings.cell_size_m
    if .4 * c <= 2 * drone_clearance + .01:
        raise ValueError("Centred door/window apertures are too small for drone planning clearance.")
    rng = np.random.default_rng(seed)
    tiles = {(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz + 1)}
    walls = {"x": {(x, y, z) for x in (0, nx) for y in range(ny) for z in range(nz)},
             "y": {(x, y, z) for x in range(nx) for y in (0, ny) for z in range(nz)}}
    doors, windows = {"x": set(), "y": set()}, {"x": set(), "y": set()}
    window_candidates = {"x": set(), "y": set()}
    stair_directions = {
        (x, y): [direction for direction, (dx, dy) in enumerate(_STAIR_STEPS)
                 if 0 <= x - dx < nx and 0 <= y - dy < ny
                 and 0 <= x + dx < nx and 0 <= y + dy < ny]
        for x in range(nx) for y in range(ny)
    }
    if nz > 1 and not any(stair_directions.values()):
        raise ValueError("Through stairs need at least three cells along one horizontal axis.")
    stairs = []
    free_floors = None
    for _ in range(128):
        proposal = []
        previous_stair_cells = set()
        for z in range(nz - 1):
            available = [(int(i) // ny, int(i) % ny) for i in rng.permutation(nx * ny)
                         if stair_directions[(int(i) // ny, int(i) % ny)]
                         and (int(i) // ny, int(i) % ny) not in previous_stair_cells]
            if not available:
                break
            count = int(rng.integers(1, min(settings.staircase_max, len(available)) + 1))
            stair_cells = available[:count]
            for x, y in stair_cells:
                direction = int(rng.choice(stair_directions[(x, y)]))
                proposal.append([x, y, z, direction])
            previous_stair_cells = set(stair_cells)
        if len({stair[2] for stair in proposal}) == nz - 1:
            free_floors = _stair_footprints(proposal, nx, ny, nz)
            if free_floors is not None:
                stairs = proposal
                break
    if free_floors is None:
        raise ValueError("Could not place stairs with clear entrances, exits, and connected rooms; enlarge the grid or lower staircase_max.")
    connectors = set()
    for x, y, z, _ in stairs:
        tiles.remove((x, y, z + 1))
        connectors.update(((x, y, z), (x, y, z + 1)))
    for z in range(nz):
        rooms = [(0, 0, nx, ny)]
        wanted = int(rng.integers(settings.rooms_per_story_min, settings.rooms_per_story_max + 1))
        while len(rooms) < wanted:
            candidates = [i for i, (x, y, w, h) in enumerate(rooms) if w > 1 or h > 1]
            idx = int(rng.choice(candidates))
            x, y, w, h = rooms.pop(idx)
            axis = int(rng.integers(2)) if w > 1 and h > 1 else (0 if w > 1 else 1)
            k = int(rng.integers(1, w if axis == 0 else h))
            rooms.extend(((x, y, k, h), (x + k, y, w - k, h)) if axis == 0 else
                         ((x, y, w, k), (x, y + k, w, h - k)))
        labels = np.zeros((nx, ny), dtype=int)
        for i, (x, y, w, h) in enumerate(rooms):
            labels[x:x + w, y:y + h] = i
        boundaries = {}
        for x in range(nx):
            for y in range(ny):
                for axis, dx, dy in (("x", 1, 0), ("y", 0, 1)):
                    if x + dx >= nx or y + dy >= ny or labels[x, y] == labels[x + dx, y + dy]:
                        continue
                    coord = (x + dx, y + dy, z)
                    walls[axis].add(coord)
                    pair = tuple(sorted((int(labels[x, y]), int(labels[x + dx, y + dy]))))
                    boundaries.setdefault(pair, []).append((axis, coord))
        parent = list(range(len(rooms)))
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        pairs = list(boundaries)
        for idx in rng.permutation(len(pairs)):
            a, b = pairs[int(idx)]
            faces = boundaries[(a, b)]
            if root(a) != root(b) or rng.random() < settings.extra_door_probability:
                parent[root(a)] = root(b)
                axis, face = faces[int(rng.integers(len(faces)))]
                doors[axis].add(face)
        for faces in boundaries.values():
            for axis, face in faces:
                window_candidates[axis].add(face)
    # Each stair has one lower entrance and one upper exit. Both cells have
    # solid side walls; preserve an existing entrance/exit wall as a door.
    stair_wall_faces = {"x": set(), "y": set()}
    for x, y, z, direction in stairs:
        for layer, open_side in ((z, (direction + 2) % 4), (z + 1, direction)):
            for side in range(4):
                axis, face = _stair_face(x, y, layer, side)
                if side == open_side:
                    if face in walls[axis]:
                        doors[axis].add(face)
                    else:
                        doors[axis].discard(face)
                else:
                    walls[axis].add(face)
                    doors[axis].discard(face)
                    stair_wall_faces[axis].add(face)
    for z, free in enumerate(free_floors):
        _connect_floor(free, z, walls, doors, rng)
    # Windows are optional after every room has a door or an open-wall route.
    for axis in ("x", "y"):
        for face in sorted(window_candidates[axis]):
            if (face in walls[axis] and face not in doors[axis]
                    and face not in stair_wall_faces[axis]
                    and rng.random() < settings.window_probability):
                windows[axis].add(face)
    # Keep the saved base on the ground floor, centred in a free corner cell.
    base_choices = [(x, y, 0) for x in (0, nx - 1) for y in (0, ny - 1)
                    if (x, y) in free_floors[0]]
    if not base_choices:
        raise ValueError("Could not place a ground-floor base in a free corner cell.")
    base = base_choices[int(rng.integers(len(base_choices)))]
    geometry = {"tiles": [list(v) for v in sorted(tiles)], "stairs": stairs,
                "stair_full_width": True}
    for axis in ("x", "y"):
        for suffix, values in (("walls", walls), ("doors", doors), ("windows", windows)):
            geometry[f"{axis}_{suffix}"] = [list(v) for v in sorted(values[axis])]
    data = {"format": BUILDING_FORMAT, "studio_coordinate_convention": "east_x_north_y_v1",
            "name": name, "width": nx * c, "height": ny * c,
            "depth": nz * c, "building_cell_grid": dict(cols=nx, rows=ny, layers=nz),
            "cell_size_m": c, "tile_thickness_m": settings.tile_thickness_m,
            "wall_thickness_m": settings.wall_thickness_m, "geometry": geometry,
            "base_position_m": [(base[0] + .5) * c, (base[1] + .5) * c, settings.tile_thickness_m / 2],
            "target_exclusion_cells": [list(v) for v in sorted(connectors | {base})],
            "generation": {"version": GENERATOR_VERSION, "seed": int(seed), "settings": asdict(settings),
                           "planning_clearance_m": drone_clearance}}
    # A compact, architecture-aware visibility graph. Cell centres and portal
    # centres avoid thousands of redundant wall-edge candidates per building.
    nodes = [(np.array([x, y, z]) + .5) * c for x in range(nx) for y in range(ny) for z in range(nz)]
    for axis in ("x", "y"):
        a = 0 if axis == "x" else 1
        for kind, faces in (("doors", doors[axis]), ("windows", windows[axis])):
            for face in faces:
                point = (np.array(face, dtype=float) + .5) * c
                point[a] -= .5 * c
                if kind == "doors":
                    point[2] -= .15 * c
                for sign in (-1, 1):
                    p = point.copy(); p[a] += sign * (settings.wall_thickness_m / 2 + drone_clearance + .01)
                    nodes.append(p)
    for x, y, z, direction in stairs:
        for layer in (z, z + 1):
            p = (np.array([x, y, layer], dtype=float) + .5) * c
            sign = 1 if direction < 2 else -1
            p[direction % 2] += (-.375 if layer == z else .375) * sign * c
            nodes.append(p)
    data["roadmap_nodes_m"] = np.asarray(nodes).tolist()
    return data, compile_building(data)


def save_generated_map(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        text = yaml.safe_dump(data, sort_keys=False)
    except ImportError:
        text = json.dumps(data, indent=2)  # JSON is also a valid YAML document.
    path.write_text(text, encoding="utf-8")
    return path


def generate_maps(settings, seed, count, *, evaluation=False, directory=None, log=lambda message: None, drone_clearance=.35):
    buildings, records = [], []
    for index in range(count):
        name = f"{'eval' if evaluation else 'map'}_{index:04d}"
        child_seed = map_seed(seed, index, evaluation=evaluation)
        data, building = generate_building(settings, child_seed, name=name, drone_clearance=drone_clearance)
        if directory is not None:
            save_generated_map(data, Path(directory) / f"{name}.yaml")
        buildings.append(building)
        records.append({"id": name, "seed": child_seed})
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == count:
            log(f"Generated {index + 1}/{count} buildings; latest {len(building.solid_min_m)} solids")
    return buildings, {"version": GENERATOR_VERSION, "settings": asdict(settings), "maps": records}
