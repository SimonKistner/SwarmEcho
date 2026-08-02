"""3D building authoring contract and validation.

This module deliberately does not alter the active 2D runtime.  It establishes
the Sprint 0 file and array contracts that the 3D runtime will consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


BUILDING_FORMAT = "swarmecho-map/v1"
DISTANCE_COMMENT_PREFIX = "# Maximum base-to-top-corner distance:"


class BuildingValidationError(ValueError):
    """Raised when a building cannot be compiled safely."""


@dataclass(frozen=True)
class BuildingArrays:
    """Fixed-shape geometry and mission arrays produced by a building file.

    Tiles and walls describe *solid rectangular prisms*, not zero-width planes.
    Boolean array indices identify prism centres on lattice boundaries; their
    physical thicknesses are stored separately.
    """

    tiles: np.ndarray
    x_walls: np.ndarray
    y_walls: np.ndarray
    target_exclusion: np.ndarray
    base_position_m: np.ndarray
    world_size_m: np.ndarray
    cell_size_m: float
    tile_thickness_m: float
    wall_thickness_m: float
    max_base_to_top_corner_m: float


def _triples(values: Any, label: str) -> list[tuple[int, int, int]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise BuildingValidationError(f"{label} must be a list of [x, y, z] coordinates.")
    result: list[tuple[int, int, int]] = []
    for index, value in enumerate(values):
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise BuildingValidationError(f"{label}[{index}] must contain exactly three integers.")
        result.append(tuple(value))
    if len(set(result)) != len(result):
        raise BuildingValidationError(f"{label} contains duplicate coordinates.")
    return result


def _positive_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise BuildingValidationError(f"{key} must be a positive number.")
    return float(value)


def _set_coords(array: np.ndarray, coords: Iterable[tuple[int, int, int]], label: str) -> None:
    for coord in coords:
        if any(value < 0 or value >= limit for value, limit in zip(coord, array.shape)):
            raise BuildingValidationError(
                f"{label} coordinate {list(coord)} is outside valid bounds {array.shape}."
            )
        array[coord] = True


def compile_building(data: dict[str, Any]) -> BuildingArrays:
    """Validate and compile one ``swarmecho-map/v1`` mapping.

    Coordinate conventions:

    - cells are ``[x, y, z]`` in ``(X, Y, Z)``;
    - tiles are ``[x_cell, y_cell, z_boundary]`` in ``(X, Y, Z+1)``;
    - X walls are ``[x_boundary, y_cell, z_cell]`` in ``(X+1, Y, Z)``;
    - Y walls are ``[x_cell, y_boundary, z_cell]`` in ``(X, Y+1, Z)``.

    A tile is a square prism of ``cell_size_m × cell_size_m ×
    tile_thickness_m``. A wall is a rectangular prism of ``cell_size_m ×
    cell_size_m × wall_thickness_m`` whose centre lies on the boundary between
    two cells, extending half its thickness into each neighbouring cell.
    """
    if not isinstance(data, dict):
        raise BuildingValidationError("Building root must be a mapping.")
    if data.get("format") != BUILDING_FORMAT:
        raise BuildingValidationError(f"format must be {BUILDING_FORMAT!r}.")

    cell_size = _positive_number(data, "cell_size_m")
    tile_thickness = _positive_number(data, "tile_thickness_m")
    wall_thickness = _positive_number(data, "wall_thickness_m")
    if tile_thickness >= cell_size or wall_thickness >= cell_size:
        raise BuildingValidationError("Tile and wall thickness must be smaller than cell_size_m.")

    grid = data.get("building_cell_grid")
    if not isinstance(grid, dict):
        raise BuildingValidationError(
            "building_cell_grid must contain positive integer cols, rows, and layers."
        )
    dimensions = tuple(grid.get(axis) for axis in ("cols", "rows", "layers"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise BuildingValidationError(
            "building_cell_grid must contain positive integer cols, rows, and layers."
        )
    size_x, size_y, size_z = dimensions
    declared_world = tuple(data.get(axis) for axis in ("width", "height", "depth"))
    expected_world = tuple(size * cell_size for size in dimensions)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isclose(float(value), expected)
        for value, expected in zip(declared_world, expected_world)
    ):
        raise BuildingValidationError(
            "width, height, and depth must equal their cell counts times cell_size_m."
        )

    geometry = data.get("geometry")
    if not isinstance(geometry, dict):
        raise BuildingValidationError("geometry must define tiles, x_walls, and y_walls.")
    tiles = np.zeros((size_x, size_y, size_z + 1), dtype=np.bool_)
    x_walls = np.zeros((size_x + 1, size_y, size_z), dtype=np.bool_)
    y_walls = np.zeros((size_x, size_y + 1, size_z), dtype=np.bool_)
    _set_coords(tiles, _triples(geometry.get("tiles"), "geometry.tiles"), "geometry.tiles")
    _set_coords(x_walls, _triples(geometry.get("x_walls"), "geometry.x_walls"), "geometry.x_walls")
    _set_coords(y_walls, _triples(geometry.get("y_walls"), "geometry.y_walls"), "geometry.y_walls")

    if not (tiles[:, :, 0].all() and tiles[:, :, size_z].all()):
        raise BuildingValidationError("The bottom floor and top roof must contain complete tile layers.")
    if not (x_walls[0, :, :].all() and x_walls[size_x, :, :].all()):
        raise BuildingValidationError("Both outer X walls must be complete.")
    if not (y_walls[:, 0, :].all() and y_walls[:, size_y, :].all()):
        raise BuildingValidationError("Both outer Y walls must be complete.")

    base_cell = data.get("base_cell")
    base_coords = _triples([base_cell] if base_cell is not None else None, "base_cell")
    if len(base_coords) != 1:
        raise BuildingValidationError("base_cell must be one [x, y, z] cell coordinate.")
    base_cell_coord = base_coords[0]
    for value, limit in zip(base_cell_coord, dimensions):
        if value < 0 or value >= limit:
            raise BuildingValidationError(f"base_cell {list(base_cell_coord)} is outside the building grid.")

    target_exclusion = np.zeros(dimensions, dtype=np.bool_)
    _set_coords(
        target_exclusion,
        _triples(data.get("target_exclusion_cells", []), "target_exclusion_cells"),
        "target_exclusion_cells",
    )

    base_position = (np.asarray(base_cell_coord, dtype=np.float32) + 0.5) * cell_size
    world_size = np.asarray(dimensions, dtype=np.float32) * cell_size
    top_corners = [
        np.asarray([x, y, world_size[2]], dtype=np.float32)
        for x in (0.0, world_size[0])
        for y in (0.0, world_size[1])
    ]
    max_distance = max(dist(base_position, corner) for corner in top_corners)

    return BuildingArrays(
        tiles=tiles,
        x_walls=x_walls,
        y_walls=y_walls,
        target_exclusion=target_exclusion,
        base_position_m=base_position,
        world_size_m=world_size,
        cell_size_m=cell_size,
        tile_thickness_m=tile_thickness,
        wall_thickness_m=wall_thickness,
        max_base_to_top_corner_m=max_distance,
    )


def load_building(path: str | Path) -> BuildingArrays:
    """Load and compile a building YAML file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    try:
        return compile_building(data)
    except BuildingValidationError as exc:
        raise BuildingValidationError(f"{source}: {exc}") from exc


def distance_comment(building: BuildingArrays) -> str:
    """Return the diagnostic comment the map builder writes above a YAML map."""
    return f"{DISTANCE_COMMENT_PREFIX} {building.max_base_to_top_corner_m:.3f} m"


def make_cuboid_building(
    grid: tuple[int, int, int],
    *,
    cell_size_m: float = 5.0,
    tile_thickness_m: float = 0.25,
    wall_thickness_m: float = 0.25,
) -> BuildingArrays:
    """Construct a sealed empty cuboid without materializing verbose YAML.

    This is primarily useful for benchmarks and generated templates. The base
    is centred horizontally in a bottom-layer cell and that cell is the only
    explicit target exclusion.
    """
    size_x, size_y, size_z = grid
    if min(grid) < 1:
        raise BuildingValidationError("Cuboid grid dimensions must be positive.")
    tiles = np.zeros((size_x, size_y, size_z + 1), dtype=np.bool_)
    tiles[:, :, (0, size_z)] = True
    x_walls = np.zeros((size_x + 1, size_y, size_z), dtype=np.bool_)
    x_walls[(0, size_x), :, :] = True
    y_walls = np.zeros((size_x, size_y + 1, size_z), dtype=np.bool_)
    y_walls[:, (0, size_y), :] = True
    base_cell = np.asarray([size_x // 2, size_y // 2, 0])
    target_exclusion = np.zeros(grid, dtype=np.bool_)
    target_exclusion[tuple(base_cell)] = True
    base_position = (base_cell.astype(np.float32) + 0.5) * cell_size_m
    world_size = np.asarray(grid, dtype=np.float32) * cell_size_m
    top_corners = [
        np.asarray([x, y, world_size[2]], dtype=np.float32)
        for x in (0.0, world_size[0])
        for y in (0.0, world_size[1])
    ]
    return BuildingArrays(
        tiles=tiles,
        x_walls=x_walls,
        y_walls=y_walls,
        target_exclusion=target_exclusion,
        base_position_m=base_position,
        world_size_m=world_size,
        cell_size_m=float(cell_size_m),
        tile_thickness_m=float(tile_thickness_m),
        wall_thickness_m=float(wall_thickness_m),
        max_base_to_top_corner_m=max(dist(base_position, corner) for corner in top_corners),
    )
