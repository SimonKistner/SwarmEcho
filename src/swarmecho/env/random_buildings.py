"""Seeded partition-and-connect buildings shared by training and inspection.

Generation is host-only. No filesystem access or RNG occurs in a rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np

from swarmecho.env.buildings import BUILDING_FORMAT, compile_building

GENERATOR_VERSION = "partition_connect_v1"


@dataclass(frozen=True)
class RandomBuildingConfig:
    enabled: bool = False
    length_m: float = 30.0
    width_m: float = 30.0
    stories: int = 3
    cell_size_m: float = 5.0
    wall_thickness_m: float = 0.25
    tile_thickness_m: float = 0.25
    rooms_per_story_min: int = 4
    rooms_per_story_max: int = 8
    extra_door_probability: float = 0.15
    window_probability: float = 0.2
    staircase_max: int = 1
    shaft_max: int = 1
    save_training_maps: bool = False

    def grid(self):
        for key in ("length_m", "width_m", "cell_size_m", "wall_thickness_m", "tile_thickness_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"random_buildings.{key} must be finite and positive.")
        counts = np.asarray([self.length_m, self.width_m]) / self.cell_size_m
        if not np.allclose(counts, np.round(counts)) or np.any(counts < 2):
            raise ValueError("Random building length/width must be cell-size multiples of at least two cells.")
        for key in ("stories", "rooms_per_story_min", "rooms_per_story_max", "staircase_max", "shaft_max"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key.endswith("max") and key != "rooms_per_story_max" else 1):
                raise ValueError(f"Invalid random_buildings.{key}.")
        if not 2 <= self.rooms_per_story_min <= self.rooms_per_story_max <= int(np.prod(counts)):
            raise ValueError("Random buildings require 2 <= room minimum <= room maximum <= floor cell count.")
        for key in ("extra_door_probability", "window_probability"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"random_buildings.{key} must be in [0,1].")
        if max(self.wall_thickness_m, self.tile_thickness_m) >= self.cell_size_m:
            raise ValueError("Building thickness must be smaller than a cell.")
        if max(1, self.staircase_max) + self.shaft_max >= int(np.prod(counts)):
            raise ValueError("Vertical connection maxima must leave at least one non-connector cell.")
        for key in ("enabled", "save_training_maps"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"random_buildings.{key} must be boolean.")
        return int(round(counts[0])), int(round(counts[1])), self.stories


def map_seed(seed: int, index: int, *, evaluation: bool = False) -> int:
    return int(np.random.SeedSequence([int(seed), int(index), 1 if evaluation else 0]).generate_state(1)[0])


def generate_building(settings: RandomBuildingConfig, seed: int, *, name: str = "random_building", drone_clearance: float = .35):
    """Return an editable map document and its compiled building.

    Doors form a spanning tree of adjacent rectangular rooms. Windows and
    extra doors add loops. Stair cells have an open side lane for flying agents.
    """
    nx, ny, nz = settings.grid()
    c = settings.cell_size_m
    if .4 * c <= 2 * drone_clearance + .01:
        raise ValueError("Centred door/window apertures are too small for drone planning clearance.")
    if .275 * c - settings.wall_thickness_m / 2 <= 2 * drone_clearance + .01:
        raise ValueError("Stair side passages are too narrow for drone planning clearance.")
    rng = np.random.default_rng(seed)
    tiles = {(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz + 1)}
    walls = {"x": {(x, y, z) for x in (0, nx) for y in range(ny) for z in range(nz)},
             "y": {(x, y, z) for x in range(nx) for y in (0, ny) for z in range(nz)}}
    doors, windows = {"x": set(), "y": set()}, {"x": set(), "y": set()}
    stairs, connectors = [], set()
    # Use disjoint columns for stairs and shafts, also across story boundaries.
    # This avoids stacked stairs obstructing each other's landing space.
    order = rng.permutation(nx * ny)
    stair_cap = settings.staircase_max or (1 if not settings.shaft_max else 0)
    stair_cells = [(int(i) // ny, int(i) % ny) for i in order[:stair_cap]]
    shaft_cells = [(int(i) // ny, int(i) % ny) for i in order[stair_cap:stair_cap + settings.shaft_max]]
    for z in range(nz - 1):
        ns = int(rng.integers(1, stair_cap + 1)) if stair_cap else 0
        nh = int(rng.integers(0 if ns else 1, settings.shaft_max + 1)) if settings.shaft_max else 0
        for x, y in stair_cells[:ns] + shaft_cells[:nh]:
            tiles.remove((x, y, z + 1))
            connectors.update(((x, y, z), (x, y, z + 1)))
        stairs.extend([[x, y, z, int(rng.integers(4))] for x, y in stair_cells[:ns]])
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
                if face not in doors[axis] and rng.random() < settings.window_probability:
                    windows[axis].add(face)
    # Reserve clear access around vertical modules on both levels.
    for x, y, z in connectors:
        for axis, face in (("x", (x, y, z)), ("x", (x + 1, y, z)),
                           ("y", (x, y, z)), ("y", (x, y + 1, z))):
            boundary = face[0 if axis == "x" else 1]
            if boundary not in (0, nx if axis == "x" else ny):
                walls[axis].discard(face); doors[axis].discard(face); windows[axis].discard(face)
    base_choices = [(x, y, 0) for x in range(nx) for y in range(ny) if (x, y, 0) not in connectors]
    base = base_choices[int(rng.integers(len(base_choices)))]
    geometry = {"tiles": [list(v) for v in sorted(tiles)], "stairs": stairs}
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
            p[0 if direction % 2 else 1] += .375 * c
            nodes.append(p)
    # Opposite stair orientations on successive levels must share a clear
    # vertical route. Four corner lanes stay outside every tread orientation.
    for x, y, z in connectors:
        for dx in (.125, .875):
            for dy in (.125, .875):
                nodes.append((np.array([x, y, z]) + [dx, dy, .5]) * c)
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
