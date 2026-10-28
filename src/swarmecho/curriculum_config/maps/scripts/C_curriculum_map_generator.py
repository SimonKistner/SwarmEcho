"""Generate the complete C curriculum with only the Python standard library.

Run directly (no JAX, GPU, environment, or map compiler is imported):
    python src/swarmecho/curriculum_config/maps/scripts/C_curriculum_map_generator.py

Target-band certificates use the ENTIRE cell, including its corners. The
runtime's wall buffers only shrink that volume, so they cannot invalidate
the certificates. Radii must match the level YAMLs (B02cs1 defaults).
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import random


MAP_ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = [(axis, sign) for axis in range(3) for sign in (1, -1)]


def direction_name(direction):
    axis, sign = direction
    return "xyz"[axis] + ("_positive" if sign > 0 else "_negative")


def advance(cell, direction, distance=1):
    result = list(cell)
    axis, sign = direction
    result[axis] += sign * distance
    return tuple(result)


def distance_bounds(cell, origin, size):
    """Exact min/max Euclidean distance from a cell-centre point to a full cell."""
    nearest, farthest = [], []
    for value, point in zip(cell, origin):
        low, high, centre = value * size, (value + 1) * size, (point + .5) * size
        nearest.append(max(low - centre, centre - high, 0.))
        farthest.append(max(abs(low - centre), abs(high - centre)))
    return math.sqrt(sum(x * x for x in nearest)), math.sqrt(sum(x * x for x in farthest))


def capacity(agents, base_radius, comm_radius, visual_radius):
    return base_radius + (agents - 1) * comm_radius + visual_radius


def target_band(first, second, args):
    """Prove necessity by triangle inequality, sufficiency by a clear LOS chain.

    Straight: all links lie on the base-to-target segment in a convex shaft.
    Corner: four cells per leg share the bend (seven cells total). Drone 1
    sits halfway from base to bend, drone 2 at the bend, and drone 3 plus
    the visual link cover the outgoing leg. This
    proves a three-drone construction even when diagonal corner cuts are
    possible. A one-drone solution is excluded for the entire target volume;
    the minimum is therefore either two or three, on the SAME map.
    """
    base = (0, 0, 0)
    bend = advance(base, first, 3) if second is not None else base
    if second is not None:
        half_leg = 1.5 * args.cell_size
        if half_leg > min(args.base_radius, args.comm_radius) - args.buffer:
            raise ValueError("The first-leg midpoint must link safely to both base and bend.")
    previous_capacity = capacity(1, args.base_radius, args.comm_radius, args.visual_radius)
    available = (args.comm_radius + args.visual_radius if second is not None
                 else capacity(3, args.base_radius, args.comm_radius, args.visual_radius))
    candidates = []
    indices = range(1, 4) if second is not None else range(1, math.ceil(available / args.cell_size) + 2)
    for index in indices:
        cell = advance(bend, second if second is not None else first, index)
        minimum, _ = distance_bounds(cell, base, args.cell_size)
        _, maximum = distance_bounds(cell, bend, args.cell_size)
        if minimum > previous_capacity + args.buffer and maximum <= available - args.buffer:
            candidates.append((index, cell, minimum, maximum))
    if not candidates:
        raise ValueError("No whole-cell target band satisfies the two-or-three-drone certificate.")
    return candidates, {
        "minimum_agents": 2,
        "maximum_required_agents": 3,
        "one_agent_reach_m": previous_capacity,
        "minimum_base_target_distance_m": min(row[2] for row in candidates),
        "maximum_bend_target_distance_m": max(row[3] for row in candidates),
        "available_bend_target_reach_m": available,
        "certificate_buffer_m": args.buffer,
        "cells_per_leg_including_corner": 4 if second else None,
        "base_to_bend_distance_m": 3 * args.cell_size if second else 0.,
        "proof": "full-cell Euclidean lower bound; relays at first-leg midpoint and bend centre" if second else
                 "full-cell Euclidean bounds; clear straight chain",
    }


def sealed_map(name, cells, base, targets, size):
    """Seal every exposed face; preserve every face shared by hallway cells."""
    cells = set(cells)
    offset = tuple(-min(cell[axis] for cell in cells) for axis in range(3))
    shift = lambda cell: tuple(value + delta for value, delta in zip(cell, offset))
    cells = {shift(cell) for cell in cells}
    targets = {shift(cell) for cell in targets}
    base = shift(base)
    dimensions = [max(cell[axis] for cell in cells) + 1 for axis in range(3)]
    faces = {"tiles": set(), "x_walls": set(), "y_walls": set()}
    for cell in cells:
        for axis, sign in DIRECTIONS:
            if advance(cell, (axis, sign)) in cells:
                continue
            face = advance(cell, (axis, 1)) if sign > 0 else cell
            faces[("x_walls", "y_walls", "tiles")[axis]].add(face)
    return {
        "format": "swarmecho-map/v1", "name": name,
        "width": dimensions[0] * size, "height": dimensions[1] * size, "depth": dimensions[2] * size,
        "building_cell_grid": dict(zip(("cols", "rows", "layers"), dimensions)),
        "cell_size_m": size, "tile_thickness_m": .25, "wall_thickness_m": .25,
        "interior_cells": [list(cell) for cell in sorted(cells)],
        "target_exclusion_cells": [list(cell) for cell in sorted(cells - targets)],
        "geometry": {key: [list(face) for face in sorted(value)] for key, value in faces.items()},
        "base_position_m": [(value + .5) * size for value in base],
        "studio_coordinate_convention": "east_x_north_y_v1",
    }


def yaml_text(data, indent=0):
    """Small readable YAML emitter for this generator's JSON-compatible maps."""
    lines = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(pad + key + ":\n" + yaml_text(value, indent + 2).rstrip())
        elif isinstance(value, list) and value:
            lines.append(pad + key + ":")
            lines.extend(pad + "- " + json.dumps(item) for item in value)
        else:
            lines.append(pad + key + ": " + json.dumps(value))
    return "\n".join(lines) + "\n"


def generate_hallways(stage, args, output):
    records = []
    for first in DIRECTIONS:
        seconds = [None] if stage == "C01" else [d for d in DIRECTIONS if d[0] != first[0]]
        for second in seconds:
            band, certificate = target_band(first, second, args)
            base = (0, 0, 0)
            bend = advance(base, first, 3) if second is not None else base
            cells = {base, bend}
            if second is not None:
                cells.update(advance(base, first, index) for index in range(1, 3))
            travel = second if second is not None else first
            last_index = 3 if second is not None else max(row[0] for row in band)
            cells.update(advance(bend, travel, index) for index in range(1, last_index + 1))
            name = stage + "_" + direction_name(first) + ("_then_" + direction_name(second) if second else "")
            data = sealed_map(name, cells, base, [row[1] for row in band], args.cell_size)
            path = output / stage / (name + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_text(data), encoding="utf-8")
            records.append({"path": str(path.relative_to(output)).replace("\\", "/"),
                            "first_direction": direction_name(first),
                            "second_direction": direction_name(second) if second else None,
                            "target_cells": [cell for cell in data["interior_cells"]
                                             if cell not in data["target_exclusion_cells"]],
                            **certificate})
    return records


def generate_station(args, output):
    # Centre plus 3 hallway cells, then the 4th cell at a face centre.
    # Face centres branch in all 4 perpendicular directions to edge midpoints;
    # those branch in both remaining directions to corners. Cube edges close.
    centre = (4, 4, 4)
    cells = set()
    for cell in itertools.product(range(9), repeat=3):
        boundary = sum(value in (0, 8) for value in cell)
        mid = sum(value == 4 for value in cell)
        if mid >= 2 or boundary >= 2 or (boundary == 1 and mid >= 1):
            cells.add(cell)
    data = sealed_map("C03_station", cells, centre, cells, args.cell_size)
    rng = random.Random(args.seed)
    added = {"doors": 0, "windows": 0}
    for axis in (0, 1):  # Doors/windows exist on vertical walls; z faces have no aperture format.
        prefix = "xy"[axis]
        data["geometry"][prefix + "_doors"] = []
        data["geometry"][prefix + "_windows"] = []
        for cell in sorted(cells):
            neighbour = advance(cell, (axis, 1))
            if neighbour not in cells or rng.random() >= .5:
                continue
            kind = "doors" if rng.random() < .5 else "windows"
            data["geometry"][prefix + "_walls"].append(list(neighbour))
            data["geometry"][prefix + "_" + kind].append(list(neighbour))
            added[kind] += 1
    path = output / "C03" / "C03_station.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text(data), encoding="utf-8")
    return {"path": str(path.relative_to(output)).replace("\\", "/"), "seed": args.seed,
            "interior_cells": len(cells), "passable_walls": added,
            "target_minimum_geodesic_distance_m": args.comm_radius + args.visual_radius + 1.,
            "target_filter": "runtime geodesic rejection; every interior cell is eligible",
            "vertical_connections": "open shafts; the map format supports doors/windows on x/y walls only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MAP_ROOT / "C_curriculum")
    parser.add_argument("--seed", type=int, default=35461354)
    parser.add_argument("--cell-size", type=float, default=5.)
    parser.add_argument("--base-radius", type=float, default=8.)
    parser.add_argument("--comm-radius", type=float, default=15.)
    parser.add_argument("--visual-radius", type=float, default=4.)
    parser.add_argument("--buffer", type=float, default=.25, help="Safety margin for exact-agent certificates (m).")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.cell_size, args.base_radius, args.comm_radius, args.visual_radius, args.buffer)):
        parser.error("Cell size, radii, and certificate buffer must be positive and finite.")
    manifest = {"format": "swarmecho-c-curriculum/v1", "seed": args.seed,
                "radii_m": {"base": args.base_radius, "drone": args.comm_radius, "visual": args.visual_radius},
                "C01": generate_hallways("C01", args, args.output),
                "C02": generate_hallways("C02", args, args.output),
                "C03": generate_station(args, args.output)}
    cellar_path = args.output / "C05" / "C05_cellar.yaml"
    cellar_path.parent.mkdir(parents=True, exist_ok=True)
    cellar = (MAP_ROOT / "B02c_static_map_cellar.yaml").read_text(encoding="utf-8")
    cellar_path.write_text(cellar.replace("name: B02c_static_map_cellar", "name: C05_cellar"), encoding="utf-8")
    manifest["C05"] = {"path": "C05/C05_cellar.yaml", "source": "B02c_static_map_cellar.yaml"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Generated C01: %s maps; C02: %s maps; C03: 1 station; C05: 1 cellar." %
          (len(manifest["C01"]), len(manifest["C02"])))
    print("Maps and full-cell target certificates: %s" % args.output)


if __name__ == "__main__":
    main()
