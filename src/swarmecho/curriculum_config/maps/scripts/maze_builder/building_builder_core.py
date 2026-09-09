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
        "base_cell": None,
        "target_exclusion_cells": [],
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
    if base is not None:
        if not isinstance(base, (list, tuple)) or len(base) != 3:
            raise ValueError("base_cell must contain [x, y, z] or be unset.")
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
    base_position = data.get("base_position_m")
    base_cell = None
    if base_position is not None:
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
            **({"base_position_m": base_position} if base_position is not None else {}),
            "target_exclusion_cells": data.get("target_exclusion_cells", []),
        }
    )


def map_data_from_document(document: dict[str, Any]) -> dict[str, Any]:
    """Compile the editor representation into the existing v1 YAML contract."""
    doc = normalize_document(document)
    cell = doc["cell_size_m"]
    if doc["base_cell"] is None:
        raise ValueError("Place the base before validating or saving the map.")
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


def add_layer(document: dict[str, Any]) -> dict[str, Any]:
    """Append a storey that inherits the editable volume, but not its tile plane."""
    doc = normalize_document(deepcopy(document))
    source_layer = doc["layers"] - 1
    new_layer = doc["layers"]
    doc["layers"] += 1
    # The old roof occupies the new storey's editable boundary.  Remove it;
    # explicitly authored tiles must never become an implicit floor.
    doc["tiles"] = [tile for tile in doc["tiles"] if tile[2] != new_layer]
    doc["interior_cells"].extend(
        [[x, y, new_layer] for x, y, z in doc["interior_cells"] if z == source_layer]
    )
    doc["x_walls"].extend(
        [[x, y, new_layer] for x, y, z in doc["x_walls"] if z == source_layer]
    )
    doc["y_walls"].extend(
        [[x, y, new_layer] for x, y, z in doc["y_walls"] if z == source_layer]
    )
    doc["target_exclusion_cells"].extend(
        [
            [x, y, new_layer]
            for x, y, z in doc["target_exclusion_cells"]
            if z == source_layer
        ]
    )
    return normalize_document(doc)


def delete_layer(document: dict[str, Any], layer: int) -> dict[str, Any]:
    """Delete a storey, or delete the roof when its inspection level is selected."""
    doc = normalize_document(deepcopy(document))
    layer = int(layer)
    if layer == doc["layers"]:
        doc["tiles"] = [tile for tile in doc["tiles"] if tile[2] != layer]
        return normalize_document(doc)
    if not 0 <= layer < doc["layers"]:
        raise ValueError("Selected layer is outside the building.")
    if doc["layers"] == 1:
        raise ValueError("A building must retain at least one storey.")

    for key in ("interior_cells", "target_exclusion_cells", "x_walls", "y_walls"):
        doc[key] = [
            [x, y, z - 1 if z > layer else z]
            for x, y, z in doc[key]
            if z != layer
        ]
    doc["tiles"] = [
        [x, y, z - 1 if z > layer + 1 else z]
        for x, y, z in doc["tiles"]
        if z != layer + 1
    ]
    base = doc["base_cell"]
    if base is not None:
        if base[2] == layer:
            doc["base_cell"] = None
        elif base[2] > layer:
            base[2] -= 1
        doc.pop("base_position_m", None)
    doc["layers"] -= 1
    return normalize_document(doc)


def expand_document(
    document: dict[str, Any], direction: str, delta: int = 1
) -> dict[str, Any]:
    """Resize one horizontal edge, copying its neighbor when growing."""
    doc = normalize_document(deepcopy(document))
    direction = str(direction).lower()
    if direction not in {"west", "east", "north", "south"}:
        raise ValueError("direction must be west, east, north, or south.")
    if isinstance(delta, bool) or int(delta) != delta or int(delta) not in {-1, 1}:
        raise ValueError("delta must be -1 or 1.")
    delta = int(delta)
    axis = 0 if direction in {"west", "east"} else 1
    prepend = direction in {"west", "north"}
    dimension_key = "cols" if axis == 0 else "rows"
    old_size = doc[dimension_key]
    if delta == -1:
        if old_size == 1:
            raise ValueError(f"{dimension_key} cannot shrink below one cell.")
        removed = 0 if prepend else old_size - 1

        def crop_cells(values: list[list[int]], value_axis: int = axis) -> list[list[int]]:
            return [
                [coordinate - 1 if prepend and i == value_axis else coordinate
                 for i, coordinate in enumerate(value)]
                for value in values if value[value_axis] != removed
            ]

        for key in ("interior_cells", "target_exclusion_cells", "tiles"):
            doc[key] = crop_cells(doc[key])
        perpendicular = "y_walls" if axis == 0 else "x_walls"
        parallel = "x_walls" if axis == 0 else "y_walls"
        doc[perpendicular] = crop_cells(doc[perpendicular], axis)

        # Preserve the removed exterior wall state on the new outside edge,
        # while dropping the old shared boundary.
        exterior = 0 if prepend else old_size
        shared = 1 if prepend else old_size - 1
        rebuilt = []
        for wall in doc[parallel]:
            boundary = wall[axis]
            if boundary == shared:
                continue
            value = wall.copy()
            if boundary == exterior:
                value[axis] = 0 if prepend else old_size - 1
            elif prepend:
                value[axis] -= 1
            rebuilt.append(value)
        doc[parallel] = rebuilt
        if doc["base_cell"] is not None:
            if doc["base_cell"][axis] == removed:
                doc["base_cell"] = None
            elif prepend:
                doc["base_cell"][axis] -= 1
            doc.pop("base_position_m", None)
        doc[dimension_key] -= 1
        return normalize_document(doc)

    if old_size >= MAX_GRID_AXIS:
        raise ValueError(f"{dimension_key} cannot exceed {MAX_GRID_AXIS}.")

    def shifted(values: list[list[int]], coordinate_axis: int = axis) -> list[list[int]]:
        result = deepcopy(values)
        if prepend:
            for value in result:
                value[coordinate_axis] += 1
        return result

    doc["interior_cells"] = shifted(doc["interior_cells"])
    doc["target_exclusion_cells"] = shifted(doc["target_exclusion_cells"])
    doc["tiles"] = shifted(doc["tiles"])
    doc["x_walls"] = shifted(doc["x_walls"], axis)
    doc["y_walls"] = shifted(doc["y_walls"], axis)
    if doc["base_cell"] is not None and prepend:
        doc["base_cell"][axis] += 1
    if doc.get("base_position_m") is not None and prepend:
        doc["base_position_m"][axis] += doc["cell_size_m"]

    # Move the old exterior wall outward. Parallel segments are duplicated
    # below, while the shared boundary remains open like an extended room.
    walls = doc["x_walls"] if axis == 0 else doc["y_walls"]
    if prepend:
        for wall in walls:
            if wall[axis] == 1:
                wall[axis] = 0
    else:
        for wall in walls:
            if wall[axis] == old_size:
                wall[axis] += 1

    source = 1 if prepend else old_size - 1
    destination = 0 if prepend else old_size

    def duplicate(values: list[list[int]], value_axis: int = axis) -> None:
        values.extend(
            [
                [destination if i == value_axis else coordinate for i, coordinate in enumerate(value)]
                for value in list(values)
                if value[value_axis] == source
            ]
        )

    duplicate(doc["interior_cells"])
    duplicate(doc["target_exclusion_cells"])
    duplicate(doc["tiles"])
    if axis == 0:
        duplicate(doc["y_walls"], 0)
    else:
        duplicate(doc["x_walls"], 1)

    doc[dimension_key] += 1
    return normalize_document(doc)


def document_yaml(document: dict[str, Any]) -> str:
    """Return exactly the normalized map payload consumed by validation/save."""
    return yaml.safe_dump(map_data_from_document(document), sort_keys=False)


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
