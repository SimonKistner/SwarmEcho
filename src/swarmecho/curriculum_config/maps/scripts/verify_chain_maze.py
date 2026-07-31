"""
Verify a SwarmEcho single-target chain-maze map without requiring JAX/NumPy.

Checks:
- base center is in free space;
- map-defined valid target cells are reachable from the base;
- approximate turn count from base to target cells is <= max turns.

The turn count uses orientation-aware BFS on a 1m grid: straight moves cost 0,
direction changes cost 1. This is a conservative proxy for the "no more than
7 corners" design constraint for an 8-agent chain.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import yaml

from swarmecho.core.config import MAP_DIR

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
INF = 1_000_000


def _resolve_map_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"}:
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return MAP_DIR / f"{name_or_path}.yaml"


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


def _target_cell_allowed(data: dict, cell: tuple[int, int]) -> bool:
    x = float(cell[0]) + 0.5
    y = float(cell[1]) + 0.5
    for x1, y1, x2, y2 in data.get("target_exclude_zones", []):
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return False
    for centre_x, centre_y, radius in data.get("target_exclude_circles", []):
        if (x - float(centre_x)) ** 2 + (y - float(centre_y)) ** 2 <= float(radius) ** 2:
            return False
    return True


def verify(map_path: Path, max_turns: int) -> int:
    data = yaml.safe_load(map_path.read_text())
    wall = _rasterize(data)
    shape = (len(wall), len(wall[0]))
    base_zone = data["spawn_zones"]["base"]
    base = ((base_zone[0] + base_zone[2]) / 2.0, (base_zone[1] + base_zone[3]) / 2.0)
    base_cell = _cell(base, shape)
    seen = _reachable(wall, base_cell)
    turns = _turns(wall, base_cell)

    target_cells = [
        cell
        for cell in _zone_cells(data["spawn_zones"]["target"], shape)
        if _is_free(wall, cell) and _target_cell_allowed(data, cell)
    ]
    target_reachable = [cell for cell in target_cells if seen[cell[0]][cell[1]]]
    target_turn_values = [turns[x][y] for x, y in target_cells if turns[x][y] < INF]

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

    print(f"map: {data['name']}")
    print(f"grid: {shape[0]} x {shape[1]}")
    print(f"base center: {base} cell={base_cell} free={_is_free(wall, base_cell)}")
    print(f"free cells reachable from base: {total_seen}/{total_free}")
    print(f"target zone free cells reachable: {len(target_reachable)}/{len(target_cells)}")
    print(f"target zone max turns: {max(target_turn_values) if target_turn_values else 'n/a'}")
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
    parser.add_argument("--max-turns", type=int, default=7)
    args = parser.parse_args()

    map_path = _resolve_map_path(args.map)
    raise SystemExit(verify(map_path, args.max_turns))


if __name__ == "__main__":
    main()
