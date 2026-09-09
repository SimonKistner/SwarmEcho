"""Document helpers for the layered 3D building editor."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from swarmecho.env.buildings import BUILDING_FORMAT, compile_building, distance_comment

DEFAULT_CELL_SIZE_M = 5.0
DEFAULT_TILE_THICKNESS_M = 0.25
DEFAULT_WALL_THICKNESS_M = 0.25
MAX_GRID_AXIS = 120
MAP_DIR = Path(__file__).resolve().parents[2]


def _coordinates(
    values: Any, label: str, shape: tuple[int, int, int]
) -> list[list[int]]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list.")
    result: set[tuple[int, int, int]] = set()
    for index, value in enumerate(values):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f"{label}[{index}] must contain three integers.")
        coordinate = tuple(value)
        if any(item < 0 or item >= limit for item, limit in zip(coordinate, shape)):
            raise ValueError(f"{label} coordinate {list(coordinate)} is outside {shape}.")
        result.add(coordinate)
    return [list(value) for value in sorted(result)]


def new_document(cols: int = 6, rows: int = 4, layers: int = 1) -> dict[str, Any]:
    """Create a sealed rectangular starter building."""
    cols, rows, layers = int(cols), int(rows), int(layers)
    if min(cols, rows, layers) < 1 or max(cols, rows, layers) > MAX_GRID_AXIS:
        raise ValueError(f"Grid axes must be between 1 and {MAX_GRID_AXIS} cells.")
    interior = [[x, y, z] for x in range(cols) for y in range(rows) for z in range(layers)]
    tiles = [
        [x, y, boundary]
        for x in range(cols)
        for y in range(rows)
        for boundary in (0, layers)
    ]
    x_walls = [
        [boundary, y, z]
        for boundary in (0, cols)
        for y in range(rows)
        for z in range(layers)
    ]
    y_walls = [
        [x, boundary, z]
        for x in range(cols)
        for boundary in (0, rows)
        for z in range(layers)
    ]
    return {
        "format": BUILDING_FORMAT,
        "name": "custom_3d_building",
        "cols": cols,
        "rows": rows,
        "layers": layers,
        "cell_size_m": DEFAULT_CELL_SIZE_M,
        "tile_thickness_m": DEFAULT_TILE_THICKNESS_M,
        "wall_thickness_m": DEFAULT_WALL_THICKNESS_M,
        "interior_cells": interior,
        "tiles": tiles,
        "x_walls": x_walls,
        "y_walls": y_walls,
        "base_cell": [cols // 2, rows // 2, 0],
        "target_exclusion_cells": [[cols // 2, rows // 2, 0]],
    }


def normalize_document(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize an editor document."""
    if not isinstance(data, dict):
        raise ValueError("Building document must be an object.")
    cols, rows, layers = (int(data.get(key, 0)) for key in ("cols", "rows", "layers"))
    if min(cols, rows, layers) < 1 or max(cols, rows, layers) > MAX_GRID_AXIS:
        raise ValueError(f"Grid axes must be between 1 and {MAX_GRID_AXIS} cells.")
    cell_shape = (cols, rows, layers)
    tile_shape = (cols, rows, layers + 1)
    x_wall_shape = (cols + 1, rows, layers)
    y_wall_shape = (cols, rows + 1, layers)
    base = data.get("base_cell")
    if not isinstance(base, (list, tuple)) or len(base) != 3:
        raise ValueError("base_cell must contain [x, y, z].")
    base = [int(value) for value in base]
    if any(value < 0 or value >= limit for value, limit in zip(base, cell_shape)):
        raise ValueError(f"base_cell {base} is outside {cell_shape}.")
    result = {
        "format": BUILDING_FORMAT,
        "name": str(data.get("name") or "custom_3d_building").strip(),
        "cols": cols,
        "rows": rows,
        "layers": layers,
        "cell_size_m": float(data.get("cell_size_m", DEFAULT_CELL_SIZE_M)),
        "tile_thickness_m": float(
            data.get("tile_thickness_m", DEFAULT_TILE_THICKNESS_M)
        ),
        "wall_thickness_m": float(
            data.get("wall_thickness_m", DEFAULT_WALL_THICKNESS_M)
        ),
        "interior_cells": _coordinates(
            data.get("interior_cells", []), "interior_cells", cell_shape
        ),
        "tiles": _coordinates(data.get("tiles", []), "tiles", tile_shape),
        "x_walls": _coordinates(data.get("x_walls", []), "x_walls", x_wall_shape),
        "y_walls": _coordinates(data.get("y_walls", []), "y_walls", y_wall_shape),
        "base_cell": base,
        "target_exclusion_cells": _coordinates(
            data.get("target_exclusion_cells", []),
            "target_exclusion_cells",
            cell_shape,
        ),
    }
    if data.get("base_position_m") is not None:
        position = data["base_position_m"]
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in position)
        ):
            raise ValueError("base_position_m must contain three numbers.")
        result["base_position_m"] = [float(value) for value in position]
    if result["cell_size_m"] <= 0:
        raise ValueError("cell_size_m must be positive.")
    return result


def document_from_map_data(data: dict[str, Any]) -> dict[str, Any]:
    """Open an existing v1 map without requiring a format migration."""
    grid = data.get("building_cell_grid") or {}
    cols, rows, layers = (int(grid.get(key, 0)) for key in ("cols", "rows", "layers"))
    cell = float(data.get("cell_size_m", DEFAULT_CELL_SIZE_M))
    base_position = data.get("base_position_m", [cell / 2, cell / 2, 0])
    base_cell = [
        min(limit - 1, max(0, int(float(value) // cell)))
        for value, limit in zip(base_position, (cols, rows, layers))
    ]
    geometry = data.get("geometry") or {}
    interior = data.get("interior_cells")
    if interior is None:
        interior = [
            [x, y, z]
            for x in range(cols)
            for y in range(rows)
            for z in range(layers)
        ]
    return normalize_document(
        {
            "name": data.get("name"),
            "cols": cols,
            "rows": rows,
            "layers": layers,
            "cell_size_m": cell,
            "tile_thickness_m": data.get("tile_thickness_m"),
            "wall_thickness_m": data.get("wall_thickness_m"),
            "interior_cells": interior,
            "tiles": geometry.get("tiles", []),
            "x_walls": geometry.get("x_walls", []),
            "y_walls": geometry.get("y_walls", []),
            "base_cell": base_cell,
            "base_position_m": base_position,
            "target_exclusion_cells": data.get("target_exclusion_cells", []),
        }
    )


def map_data_from_document(document: dict[str, Any]) -> dict[str, Any]:
    """Compile the editor representation into the existing v1 YAML contract."""
    doc = normalize_document(document)
    cell = doc["cell_size_m"]
    x, y, z = doc["base_cell"]
    tile_thickness = doc["tile_thickness_m"]
    return {
        "format": BUILDING_FORMAT,
        "name": doc["name"],
        "width": doc["cols"] * cell,
        "height": doc["rows"] * cell,
        "depth": doc["layers"] * cell,
        "building_cell_grid": {
            "cols": doc["cols"],
            "rows": doc["rows"],
            "layers": doc["layers"],
        },
        "cell_size_m": cell,
        "tile_thickness_m": tile_thickness,
        "wall_thickness_m": doc["wall_thickness_m"],
        "interior_cells": doc["interior_cells"],
        "target_exclusion_cells": doc["target_exclusion_cells"],
        "geometry": {
            "tiles": doc["tiles"],
            "x_walls": doc["x_walls"],
            "y_walls": doc["y_walls"],
        },
        "base_position_m": doc.get(
            "base_position_m",
            [
                (x + 0.5) * cell,
                (y + 0.5) * cell,
                z * cell + tile_thickness / 2,
            ],
        ),
    }


def add_outer_walls(document: dict[str, Any], layer: int) -> dict[str, Any]:
    """Add the horizontal perimeter of the selected interior footprint."""
    doc = normalize_document(deepcopy(document))
    layer = int(layer)
    if not 0 <= layer < doc["layers"]:
        raise ValueError("Selected layer is outside the building.")
    interior = {tuple(value) for value in doc["interior_cells"]}
    x_walls = {tuple(value) for value in doc["x_walls"]}
    y_walls = {tuple(value) for value in doc["y_walls"]}
    for x, y, z in interior:
        if z != layer:
            continue
        if (x - 1, y, z) not in interior:
            x_walls.add((x, y, z))
        if (x + 1, y, z) not in interior:
            x_walls.add((x + 1, y, z))
        if (x, y - 1, z) not in interior:
            y_walls.add((x, y, z))
        if (x, y + 1, z) not in interior:
            y_walls.add((x, y + 1, z))
    doc["x_walls"] = [list(value) for value in sorted(x_walls)]
    doc["y_walls"] = [list(value) for value in sorted(y_walls)]
    return doc


def add_roof(document: dict[str, Any]) -> dict[str, Any]:
    """Add tiles over every exposed top face in the interior volume."""
    doc = normalize_document(deepcopy(document))
    interior = {tuple(value) for value in doc["interior_cells"]}
    tiles = {tuple(value) for value in doc["tiles"]}
    for x, y, z in interior:
        if (x, y, z + 1) not in interior:
            tiles.add((x, y, z + 1))
    doc["tiles"] = [list(value) for value in sorted(tiles)]
    return doc


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Run the same map compiler used by 3D level loading."""
    data = map_data_from_document(document)
    building = compile_building(data)
    return {
        "valid": True,
        "world_size_m": building.world_size_m.tolist(),
        "interior_cells": int(building.interior_cells.sum()),
        "authored_solids": int(building.solid_min_m.shape[0]),
        "target_candidate_cells": int(
            (building.interior_cells & ~building.target_exclusion).sum()
        ),
        "max_base_to_top_corner_m": building.max_base_to_top_corner_m,
    }


def load_document(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return document_from_map_data(yaml.safe_load(source.read_text(encoding="utf-8")))


def save_document(document: dict[str, Any], path: str | Path) -> Path:
    """Validate and atomically save one v1 map."""
    target = Path(path)
    data = map_data_from_document(document)
    building = compile_building(data)
    text = distance_comment(building) + "\n" + yaml.safe_dump(data, sort_keys=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def available_maps(directory: Path = MAP_DIR) -> Iterable[str]:
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data.get("format") == BUILDING_FORMAT:
                yield path.stem
        except (OSError, AttributeError, yaml.YAMLError):
            continue
