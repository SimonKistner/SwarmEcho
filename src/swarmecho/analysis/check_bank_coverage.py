"""Certify that every coverage voxel in every selected bank map is observable.

Run with ``uv run python -m swarmecho.analysis.check_bank_coverage``. A pass is
a sufficient geometric certificate: each interior voxel centre outside solids
is visible from a
base-reachable roadmap vertex, or its own centre is directly reachable from
one. An uncertified voxel may still be reachable by a curved route; its map
and coordinate are reported for inspection rather than called impossible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from swarmecho.core.config import load_level
from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
    find_compatible_pool,
)
from swarmecho.env.buildings import compile_building
from swarmecho.env.environment import coverage_grid_geometry


def _blocked(start, end, lower, upper, epsilon=1e-5):
    """NumPy equivalent of the environment's segment/AABB slab test."""
    direction = end - start
    parallel = np.abs(direction) <= epsilon
    safe = np.where(parallel, 1.0, direction)
    first = (lower - start) / safe
    last = (upper - start) / safe
    near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
    far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
    outside = np.any(parallel & ((start < lower) | (start > upper)), axis=-1)
    return bool(np.any((far >= np.maximum(near, epsilon)) & (near <= 1.0 - epsilon) & ~outside))


def _visible_from_vertex(point, vertices, lower, upper, radius):
    distances = np.linalg.norm(vertices - point, axis=-1)
    for index in np.argsort(distances):
        if distances[index] > radius:
            break
        if not _blocked(vertices[index], point, lower, upper):
            return True
    return False


def _centre_reachable(point, vertices, lower, upper, drone_radius, world_lower, world_upper):
    if np.any(point < world_lower) or np.any(point > world_upper):
        return False
    if np.any(np.all((point >= lower - drone_radius) & (point <= upper + drone_radius), axis=-1)):
        return False
    expanded_lower, expanded_upper = lower - drone_radius, upper + drone_radius
    for index in np.argsort(np.linalg.norm(vertices - point, axis=-1)):
        if not _blocked(vertices[index], point, expanded_lower, expanded_upper):
            return True
    return False


def check_map(map_file: Path, record_file: Path, level) -> tuple[int, list[list[float]]]:
    building = compile_building(yaml.safe_load(map_file.read_text(encoding="utf-8")))
    voxel_size, grid_shape = coverage_grid_geometry(building, level.env)
    indices = np.stack(np.meshgrid(*[np.arange(size) for size in grid_shape], indexing="ij"), axis=-1)
    centres = (indices.astype(np.float32) + 0.5) * voxel_size
    coarse = np.minimum(
        np.floor(centres / building.cell_size_m).astype(np.int64),
        np.asarray(building.interior_cells.shape) - 1,
    )
    eligible = building.interior_cells[coarse[..., 0], coarse[..., 1], coarse[..., 2]]
    if len(building.solid_min_m):
        inside_solid = np.any(
            np.all(
                (centres[..., None, :] >= building.solid_min_m)
                & (centres[..., None, :] <= building.solid_max_m),
                axis=-1,
            ),
            axis=-1,
        )
        eligible &= ~inside_solid
    points = centres[eligible]
    point_cells = coarse[eligible]
    with np.load(record_file, allow_pickle=False) as record:
        vertices = np.asarray(record["vertices"], dtype=np.float32)
        reachable = np.asarray(record["distances"][0] < 1e6)
        vertices = vertices[reachable]
        lower = np.asarray(record["solid_min"], dtype=np.float32)
        upper = np.asarray(record["solid_max"], dtype=np.float32)
    if not len(vertices):
        raise ValueError(f"No base-reachable roadmap vertices in {record_file}")
    world_lower = np.asarray([
        building.wall_thickness_m / 2 + level.env.drone_radius,
        building.wall_thickness_m / 2 + level.env.drone_radius,
        building.tile_thickness_m / 2 + level.env.drone_radius,
    ])
    world_upper = building.world_size_m - world_lower
    # A cell centre that attaches to the base-reachable roadmap certifies many
    # coverage voxels in that cell with one visibility segment each.
    cell_viewpoints = {}
    for cell in np.unique(point_cells, axis=0):
        viewpoint = (cell.astype(np.float32) + 0.5) * building.cell_size_m
        if _centre_reachable(viewpoint, vertices, lower, upper, level.env.drone_radius,
                             world_lower, world_upper):
            cell_viewpoints[tuple(cell)] = viewpoint
    uncertified = []
    for point, cell in zip(points, point_cells, strict=True):
        viewpoint = cell_viewpoints.get(tuple(cell))
        if viewpoint is not None and np.linalg.norm(viewpoint - point) <= level.env.visual_radius \
                and not _blocked(viewpoint, point, lower, upper):
            continue
        if _visible_from_vertex(point, vertices, lower, upper, level.env.visual_radius):
            continue
        if _centre_reachable(point, vertices, lower, upper, level.env.drone_radius,
                             world_lower, world_upper):
            continue
        uncertified.append(point.tolist())
    return len(points), uncertified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="B02a_random_buildings_find_only")
    parser.add_argument("--pool", type=Path, help="Bank pool directory; default: largest compatible pool")
    parser.add_argument("--limit", type=int, help="Check only the first N maps for a quick trial")
    args = parser.parse_args()
    level = load_level(args.level)
    if args.pool is None:
        found = find_compatible_pool(level)
        if found is None:
            parser.error("No compatible bank pool exists; pass --pool or build the bank first.")
        pool, metadata = found
    else:
        pool = args.pool
        metadata = yaml.safe_load((pool / "metadata.json").read_text(encoding="utf-8"))
    maps = metadata["maps"][:args.limit]
    uncertified_maps = []
    eligible_total = 0
    for index, item in enumerate(maps, 1):
        name = item["id"]
        eligible, uncertified = check_map(
            pool / "maps" / f"{name}.yaml", pool / "records" / f"{name}.npz", level,
        )
        eligible_total += eligible
        if uncertified:
            uncertified_maps.append((name, eligible, uncertified))
            if len(uncertified_maps) <= 5:
                print(f"UNCERTIFIED {name}: {len(uncertified)}/{eligible} voxels; "
                      f"first centres={uncertified[:5]}", flush=True)
        if index % 10 == 0 or index == len(maps):
            print(f"checked {index}/{len(maps)} maps; uncertified={len(uncertified_maps)}", flush=True)
    print(f"Eligible voxels: {eligible_total}; certified maps: {len(maps) - len(uncertified_maps)}/{len(maps)}")
    for name, eligible, points in uncertified_maps[5:20]:
        print(f"UNCERTIFIED {name}: {len(points)}/{eligible} voxels; first centres={points[:5]}")
    if uncertified_maps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
