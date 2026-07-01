"""Core conversion helpers for the SwarmEcho browser maze builder.

The public import/export format intentionally mirrors the machine-readable JSON
format used in the manual screenshot workflow:

- width/height are maze dimensions in cells.
- h_walls is a (height + 1) x width matrix of 0/1 horizontal wall edges.
- v_walls is a height x (width + 1) matrix of 0/1 vertical wall edges.
- target_exclude_cells is a list of [row, column] cells painted red in the UI.

Generated map YAML uses the existing SwarmEcho map convention: world units are
metres, each maze cell is 10 m by default, and base/drone spawn at map centre.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

CELL_SIZE_M = 10.0
TARGET_EDGE_MARGIN_M = 0.833
TARGET_WALL_CLEARANCE_M = 1.25
MAP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_SCRIPT_DIR = Path(__file__).resolve().parent
_MAPS_DIR = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _SCRIPT_DIR.parents[3]
_LEVELS_DIR = _SRC_ROOT / "curriculum_config" / "levels"


def _f(value: float, digits: int = 3) -> float:
    rounded = round(float(value), digits)
    if rounded == -0.0:
        return 0.0
    return rounded


def validate_map_name(name: str) -> str:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Map name is required.")
    if not MAP_NAME_RE.fullmatch(name):
        raise ValueError("Map name may only contain letters, numbers, underscores, and hyphens.")
    return name


def normalize_machine_maze(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a machine-readable maze JSON payload."""
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive cell counts.")

    h_walls = data.get("h_walls")
    v_walls = data.get("v_walls")
    if not isinstance(h_walls, list) or len(h_walls) != height + 1:
        raise ValueError(f"h_walls must have {height + 1} rows for height={height}.")
    if not isinstance(v_walls, list) or len(v_walls) != height:
        raise ValueError(f"v_walls must have {height} rows for height={height}.")

    norm_h: list[list[int]] = []
    for row_i, row in enumerate(h_walls):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"h_walls row {row_i} must have {width} entries.")
        norm_h.append([1 if int(v) else 0 for v in row])

    norm_v: list[list[int]] = []
    for row_i, row in enumerate(v_walls):
        if not isinstance(row, list) or len(row) != width + 1:
            raise ValueError(f"v_walls row {row_i} must have {width + 1} entries.")
        norm_v.append([1 if int(v) else 0 for v in row])

    excluded: set[tuple[int, int]] = set()
    for item in data.get("target_exclude_cells", []) or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("target_exclude_cells entries must be [row, column].")
        row = int(item[0])
        col = int(item[1])
        if 0 <= row < height and 0 <= col < width:
            excluded.add((row, col))

    return {
        "width": width,
        "height": height,
        "coordinate_system": "row,column with 0,0 at top-left",
        "encoding": {"1": "wall", "0": "open"},
        "h_walls": norm_h,
        "v_walls": norm_v,
        "target_exclude_cells": [[row, col] for row, col in sorted(excluded)],
        "cell_size_m": float(data.get("cell_size_m", CELL_SIZE_M)),
        "map_name": str(data.get("map_name", "")).strip(),
    }


def crop_canvas_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Crop free-canvas wall/cell data to the outermost active walls.

    Expected payload fields:
    - edges: [{type: "h"|"v", x: int, y: int}, ...]
    - target_exclude_cells: [[row, col], ...]
    """
    edges = payload.get("edges") or []
    if not edges:
        raise ValueError("Draw at least one wall before creating a map.")

    min_x: int | None = None
    max_x: int | None = None
    min_y: int | None = None
    max_y: int | None = None
    clean_edges: list[tuple[str, int, int]] = []

    for edge in edges:
        edge_type = str(edge.get("type", ""))
        x = int(edge.get("x", 0))
        y = int(edge.get("y", 0))
        if edge_type not in {"h", "v"}:
            continue
        clean_edges.append((edge_type, x, y))
        if edge_type == "h":
            ex0, ex1, ey0, ey1 = x, x + 1, y, y
        else:
            ex0, ex1, ey0, ey1 = x, x, y, y + 1
        min_x = ex0 if min_x is None else min(min_x, ex0)
        max_x = ex1 if max_x is None else max(max_x, ex1)
        min_y = ey0 if min_y is None else min(min_y, ey0)
        max_y = ey1 if max_y is None else max(max_y, ey1)

    if min_x is None or max_x is None or min_y is None or max_y is None:
        raise ValueError("No valid walls found.")

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError("Detected maze bounds are empty.")

    h_walls = [[0 for _ in range(width)] for _ in range(height + 1)]
    v_walls = [[0 for _ in range(width + 1)] for _ in range(height)]

    for edge_type, x, y in clean_edges:
        nx = x - min_x
        ny = y - min_y
        if edge_type == "h" and 0 <= ny <= height and 0 <= nx < width:
            h_walls[ny][nx] = 1
        elif edge_type == "v" and 0 <= ny < height and 0 <= nx <= width:
            v_walls[ny][nx] = 1

    excluded: list[list[int]] = []
    ignored = 0
    for item in payload.get("target_exclude_cells", []) or []:
        row = int(item[0])
        col = int(item[1])
        nrow = row - min_y
        ncol = col - min_x
        if 0 <= nrow < height and 0 <= ncol < width:
            excluded.append([nrow, ncol])
        else:
            ignored += 1

    machine = normalize_machine_maze({
        "width": width,
        "height": height,
        "h_walls": h_walls,
        "v_walls": v_walls,
        "target_exclude_cells": excluded,
        "cell_size_m": payload.get("cell_size_m", CELL_SIZE_M),
        "map_name": payload.get("map_name", ""),
    })
    machine["source_canvas_bounds"] = {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}
    machine["ignored_target_exclude_cells"] = ignored
    return machine


def walls_from_machine_maze(machine: dict[str, Any]) -> list[list[float]]:
    data = normalize_machine_maze(machine)
    width = data["width"]
    height = data["height"]
    cell = float(data.get("cell_size_m", CELL_SIZE_M))
    walls: list[list[float]] = []

    for y in range(height + 1):
        x = 0
        while x < width:
            if not data["h_walls"][y][x]:
                x += 1
                continue
            start = x
            while x < width and data["h_walls"][y][x]:
                x += 1
            walls.append([_f(start * cell), _f(y * cell), _f(x * cell), _f(y * cell)])

    for x in range(width + 1):
        y = 0
        while y < height:
            if not data["v_walls"][y][x]:
                y += 1
                continue
            start = y
            while y < height and data["v_walls"][y][x]:
                y += 1
            walls.append([_f(x * cell), _f(start * cell), _f(x * cell), _f(y * cell)])

    return walls


def target_exclude_zones_from_cells(machine: dict[str, Any]) -> list[list[float]]:
    data = normalize_machine_maze(machine)
    width = data["width"]
    height = data["height"]
    cell = float(data.get("cell_size_m", CELL_SIZE_M))
    remaining = {(int(row), int(col)) for row, col in data.get("target_exclude_cells", [])}
    zones: list[list[float]] = []

    while remaining:
        top, left = min(remaining)
        right = left
        while (top, right + 1) in remaining:
            right += 1

        bottom = top
        while True:
            candidate = bottom + 1
            if candidate >= height:
                break
            if all((candidate, col) in remaining for col in range(left, right + 1)):
                bottom = candidate
            else:
                break

        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                remaining.discard((row, col))

        zones.append([
            _f(left * cell),
            _f(top * cell),
            _f((right + 1) * cell),
            _f((bottom + 1) * cell),
        ])

    return zones


def build_map_data(machine: dict[str, Any], map_name: str | None = None) -> dict[str, Any]:
    data = normalize_machine_maze(machine)
    name = validate_map_name(map_name or data.get("map_name"))
    cols = data["width"]
    rows = data["height"]
    cell = float(data.get("cell_size_m", CELL_SIZE_M))
    width_m = _f(cols * cell)
    height_m = _f(rows * cell)
    cx = _f(width_m / 2.0)
    cy = _f(height_m / 2.0)
    margin = min(TARGET_EDGE_MARGIN_M, max(0.0, min(width_m, height_m) / 2.0))

    return {
        "name": name,
        "width": width_m,
        "height": height_m,
        "maze_cell_grid": {"cols": cols, "rows": rows},
        "spawn_zones": {
            "base": [cx, cy, cx, cy],
            "drone": [cx, cy, cx, cy],
            "target": [_f(margin), _f(margin), _f(width_m - margin), _f(height_m - margin)],
        },
        "target_wall_clearance": TARGET_WALL_CLEARANCE_M,
        "target_exclude_zones": target_exclude_zones_from_cells(data),
        "rooms": [],
        "hallways": [],
        "walls": walls_from_machine_maze(data),
    }


def build_level_data(map_name: str) -> dict[str, Any]:
    name = validate_map_name(map_name)
    return {
        "env": {
            "map_names": [name],
            "num_agents": 7,
            "num_targets": 1,
            "num_bases": 1,
            "max_steps": 1000,
            "visual_radius": 5.0,
            "max_speed": 6.0,
            "max_force": 20.83,
            "comm_radius": 20.83,
            "comm_radius_base": 16.67,
            "spawn_delay": 5,
            "use_random_base_spawn": False,
            "use_random_drone_spawn": False,
            "target_spawn_method": "map_defined",
            "adaptive_target_spawn": False,
            "static_maze_optimal_path": True,
            "precover_base_comm": True,
            "hold_chain_for": 50,
            "observe_target_vector": False,
            "observe_base_vector": False,
        },
        "reward": {
            "every_reward_global": False,
            "only_explor_individual": False,
            "only_shortest_path_chain_reward": True,
            "target_found_requires_delivery": True,
            "back_to_target_after_delivery": False,
            "chain_reward_system": "discrete_finders_path",
            "only_reward_chain_from_target": False,
            "exploration_bonus": 0.25,
            "finder_bonus": 50.0,
            "max_gap_penalty": 5.0,
            "target_found_bonus": 100.0,
            "success_bonus": 500.0,
        },
        "training": {
            "total_timesteps": 500_000_000,
            "num_envs": 4000,
            "num_steps": 100,
            "num_epochs": 4,
            "num_minibatches": 20,
            "seed": 6302026950,
            "eval_parallel": True,
            "eval_parallel_envs": 4000,
            "eval_parallel_early_exit_threshold": 1.0,
        },
        "network": {
            "critic_type": "agent_centric",
            "num_layers": 3,
            "actor_num_layers": 3,
            "actor_memory": True,
            "critic_memory": True,
            "memory_comm_enabled": True,
            "memory_comm_every_k_steps": 5,
            "tarmac_sig_dim": 64,
            "tarmac_val_dim": 128,
            "tarmac_include_self": False,
        },
        "curriculum": {"success_threshold": 0.95, "metric": "success", "mode": "eval"},
        "logging": {
            "adaptive_spawn_diagnostics": True,
            "checkpoint_freq": 50,
            "checkpoint_offset": 0,
            "eval_freq": 50,
            "eval_offset": 1,
            "eval_not_delivered_or_visually_found_heatmap": True,
            "eval_video": False,
            "eval_video_freq": 100,
            "eval_video_offset": 1,
        },
    }


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)


def create_map_and_level(payload: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    machine = crop_canvas_payload(payload) if payload.get("edges") is not None else normalize_machine_maze(payload)
    map_name = validate_map_name(payload.get("map_name") or machine.get("map_name"))
    machine["map_name"] = map_name
    map_data = build_map_data(machine, map_name)
    level_data = build_level_data(map_name)

    map_path = _MAPS_DIR / f"{map_name}.yaml"
    level_path = _LEVELS_DIR / f"{map_name}.yaml"
    if not overwrite:
        existing = [str(p) for p in (map_path, level_path) if p.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing file(s): " + ", ".join(existing))

    _MAPS_DIR.mkdir(parents=True, exist_ok=True)
    _LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    map_path.write_text(dump_yaml(map_data), encoding="utf-8")
    level_path.write_text(dump_yaml(level_data), encoding="utf-8")

    preview_command = f"uv run python src/visualize/render_preview.py {map_name} --mode image --format png --no-spawns"
    return {
        "map_path": str(map_path),
        "level_path": str(level_path),
        "preview_command": preview_command,
        "machine_maze": machine,
        "map_data": map_data,
        "level_data": level_data,
    }
