"""
Verify a SwarmEcho single-target chain-maze map without requiring JAX/NumPy.

Checks:
- base center is in free space;
- target spawn-zone free cells are reachable from the base;
- deterministic samples from the level's outside_base target sampler are free;
- approximate turn count from base to target-zone/sample cells is <= max turns.

The turn count uses orientation-aware BFS on a 1m grid: straight moves cost 0,
direction changes cost 1. This is a conservative proxy for the "no more than
7 corners" design constraint for an 8-agent chain.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random

import yaml


ROOT = Path(__file__).resolve().parents[4]
MAP_DIR = ROOT / "src" / "curriculum_config" / "maps"
LEVEL_DIR = ROOT / "src" / "curriculum_config" / "levels"
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
INF = 1_000_000


def _resolve_map_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"}:
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return MAP_DIR / f"{name_or_path}.yaml"


def _resolve_level_path(name_or_path: str | None) -> Path | None:
    if name_or_path is None:
        return None
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"}:
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return LEVEL_DIR / f"{name_or_path}.yaml"


def _compile_segments(data: dict) -> list[list[float]]:
    segments = list(data.get("walls", []))
    door_width = 10.0

    for room in data.get("rooms", []):
        x, y, w, h = float(room["x"]), float(room["y"]), float(room["w"]), float(room["h"])
        door_side = room.get("door_side", "none")

        def add_wall(p1, p2, side):
            if door_side == side:
                if side in {"N", "S"}:
                    segments.append([p1[0], p1[1], x + w / 2 - door_width / 2, p1[1]])
                    segments.append([x + w / 2 + door_width / 2, p1[1], p2[0], p2[1]])
                else:
                    segments.append([p1[0], p1[1], p1[0], y + h / 2 - door_width / 2])
                    segments.append([p1[0], y + h / 2 + door_width / 2, p2[0], p2[1]])
            else:
                segments.append([p1[0], p1[1], p2[0], p2[1]])

        add_wall([x, y + h], [x + w, y + h], "N")
        add_wall([x, y], [x + w, y], "S")
        add_wall([x, y], [x, y + h], "W")
        add_wall([x + w, y], [x + w, y + h], "E")

    for hall in data.get("hallways", []):
        x1, y1, x2, y2 = map(float, [hall["x1"], hall["y1"], hall["x2"], hall["y2"]])
        width = float(hall["width"])
        dx, dy = x2 - x1, y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 0:
            ux, uy = -dy / dist * (width / 2), dx / dist * (width / 2)
            segments.append([x1 + ux, y1 + uy, x2 + ux, y2 + uy])
            segments.append([x1 - ux, y1 - uy, x2 - ux, y2 - uy])

    return segments


def _rasterize(data: dict) -> list[list[bool]]:
    width = int(float(data["width"]))
    height = int(float(data["height"]))
    wall = [[False for _ in range(height)] for _ in range(width)]

    for x1, y1, x2, y2 in _compile_segments(data):
        dx, dy = x2 - x1, y2 - y1
        steps = max(1, int(max(abs(dx), abs(dy)) * 3))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(x1 + dx * t))
            y = int(round(y1 + dy * t))
            if 0 <= x < width and 0 <= y < height:
                wall[x][y] = True
    return wall


def _cell(point: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int]:
    return (
        min(max(int(point[0]), 0), shape[0] - 1),
        min(max(int(point[1]), 0), shape[1] - 1),
    )


def _is_free(wall: list[list[bool]], cell: tuple[int, int]) -> bool:
    return not wall[cell[0]][cell[1]]


def _reachable(wall: list[list[bool]], start: tuple[int, int]) -> list[list[bool]]:
    width, height = len(wall), len(wall[0])
    seen = [[False for _ in range(height)] for _ in range(width)]
    q: deque[tuple[int, int]] = deque()
    if _is_free(wall, start):
        seen[start[0]][start[1]] = True
        q.append(start)
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not wall[nx][ny] and not seen[nx][ny]:
                seen[nx][ny] = True
                q.append((nx, ny))
    return seen


def _turns(wall: list[list[bool]], start: tuple[int, int]) -> list[list[int]]:
    width, height = len(wall), len(wall[0])
    dist = [[[INF for _ in range(4)] for _ in range(height)] for _ in range(width)]
    dq: deque[tuple[int, int, int]] = deque()

    if not _is_free(wall, start):
        return [[INF for _ in range(height)] for _ in range(width)]

    for direction in range(4):
        dist[start[0]][start[1]][direction] = 0
        dq.append((start[0], start[1], direction))

    while dq:
        x, y, direction = dq.popleft()
        cur = dist[x][y][direction]
        for nd, (dx, dy) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or wall[nx][ny]:
                continue
            cost = 0 if nd == direction else 1
            nxt = cur + cost
            if nxt < dist[nx][ny][nd]:
                dist[nx][ny][nd] = nxt
                if cost == 0:
                    dq.appendleft((nx, ny, nd))
                else:
                    dq.append((nx, ny, nd))

    return [[min(dist[x][y]) for y in range(height)] for x in range(width)]


def _zone_cells(zone: list[float], shape: tuple[int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = zone
    cells = []
    for x in range(max(0, int(x1)), min(shape[0], int(x2) + 1)):
        for y in range(max(0, int(y1)), min(shape[1], int(y2) + 1)):
            cells.append((x, y))
    return cells


def _sample_outside_base(data: dict, level: dict, wall: list[list[bool]], samples: int) -> list[tuple[float, float]]:
    env = level.get("env", {})
    if env.get("target_spawn_method") != "outside_base":
        return []

    base_zone = data["spawn_zones"]["base"]
    base = ((base_zone[0] + base_zone[2]) / 2.0, (base_zone[1] + base_zone[3]) / 2.0)
    invalid_radius = float(env.get("target_invalid_spawn_base_radius", 0.0))
    width, height = len(wall), len(wall[0])
    rng = random.Random(123)

    sampled = []
    for _ in range(samples):
        candidates = [(rng.uniform(2.0, width - 2.0), rng.uniform(2.0, height - 2.0)) for _ in range(64)]
        valid = []
        free = []
        for p in candidates:
            cell = _cell(p, (width, height))
            if not wall[cell[0]][cell[1]]:
                free.append((p, ((p[0] - base[0]) ** 2 + (p[1] - base[1]) ** 2) ** 0.5))
            dist = ((p[0] - base[0]) ** 2 + (p[1] - base[1]) ** 2) ** 0.5
            if dist > invalid_radius and not wall[cell[0]][cell[1]]:
                valid.append(p)
        if valid:
            sampled.append(valid[0])
        elif free:
            sampled.append(max(free, key=lambda item: item[1])[0])
        else:
            sampled.append(candidates[0])
    return sampled


def verify(map_path: Path, level_path: Path | None, max_turns: int, samples: int) -> int:
    data = yaml.safe_load(map_path.read_text())
    wall = _rasterize(data)
    shape = (len(wall), len(wall[0]))
    base_zone = data["spawn_zones"]["base"]
    base = ((base_zone[0] + base_zone[2]) / 2.0, (base_zone[1] + base_zone[3]) / 2.0)
    base_cell = _cell(base, shape)
    seen = _reachable(wall, base_cell)
    turns = _turns(wall, base_cell)

    target_cells = [cell for cell in _zone_cells(data["spawn_zones"]["target"], shape) if _is_free(wall, cell)]
    target_reachable = [cell for cell in target_cells if seen[cell[0]][cell[1]]]
    target_turn_values = [turns[x][y] for x, y in target_cells if turns[x][y] < INF]

    level = yaml.safe_load(level_path.read_text()) if level_path else {}
    sampled = _sample_outside_base(data, level, wall, samples) if level else []
    sampled_cells = [_cell(p, shape) for p in sampled]
    sampled_free = [cell for cell in sampled_cells if _is_free(wall, cell)]
    sampled_reachable = [cell for cell in sampled_cells if seen[cell[0]][cell[1]]]
    sampled_turn_values = [turns[x][y] for x, y in sampled_cells]

    total_free = sum(1 for x in range(shape[0]) for y in range(shape[1]) if not wall[x][y])
    total_seen = sum(1 for x in range(shape[0]) for y in range(shape[1]) if seen[x][y])

    errors = []
    if not _is_free(wall, base_cell):
        errors.append(f"base center cell {base_cell} is in a wall")
    if not target_cells:
        errors.append("target spawn zone has no free cells")
    if len(target_reachable) != len(target_cells):
        errors.append(f"target zone reachable free cells {len(target_reachable)}/{len(target_cells)}")
    if target_turn_values and max(target_turn_values) > max_turns:
        errors.append(f"target zone max turns {max(target_turn_values)} > {max_turns}")
    if sampled and len(sampled_free) != len(sampled):
        errors.append(f"sampled targets free {len(sampled_free)}/{len(sampled)}")
    if sampled and len(sampled_reachable) != len(sampled):
        errors.append(f"sampled targets reachable {len(sampled_reachable)}/{len(sampled)}")
    if sampled_turn_values and max(sampled_turn_values) > max_turns:
        errors.append(f"sampled target max turns {max(sampled_turn_values)} > {max_turns}")

    print(f"map: {data['name']}")
    print(f"grid: {shape[0]} x {shape[1]}")
    print(f"base center: {base} cell={base_cell} free={_is_free(wall, base_cell)}")
    print(f"free cells reachable from base: {total_seen}/{total_free}")
    print(f"target zone free cells reachable: {len(target_reachable)}/{len(target_cells)}")
    print(f"target zone max turns: {max(target_turn_values) if target_turn_values else 'n/a'}")
    if sampled:
        print(f"sampled target cells free: {len(sampled_free)}/{len(sampled)}")
        print(f"sampled target cells reachable: {len(sampled_reachable)}/{len(sampled)}")
        print(f"sampled target max turns: {max(sampled_turn_values)}")

    if errors:
        print("FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", help="Map name or YAML path")
    parser.add_argument("--level", default=None, help="Optional level name/YAML path for target sampling checks")
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()

    map_path = _resolve_map_path(args.map)
    level_path = _resolve_level_path(args.level)
    raise SystemExit(verify(map_path, level_path, args.max_turns, args.samples))


if __name__ == "__main__":
    main()
