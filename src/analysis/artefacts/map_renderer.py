"""
map_renderer.py — Lightweight map preview renderer for SwarmEcho Artefacts.

IMPORTANT: This module intentionally has NO JAX imports.
It uses only yaml + cv2 + numpy, which are fast and lightweight.
This is the alternative to spawning render_preview.py as a subprocess,
which loads the full JAX runtime (~60 GB RAM issue).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Wall segment compilation (copied from render_preview.py, no JAX deps)
# ---------------------------------------------------------------------------

def _compile_segments(data: dict) -> list[list[float]]:
    segments = list(data.get("walls", []))
    door_width = 10.0
    for room in data.get("rooms", []):
        x, y, w, h = float(room["x"]), float(room["y"]), float(room["w"]), float(room["h"])
        door_side = room.get("door_side", "none")

        def add_wall(p1, p2, side, x=x, y=y, w=w, h=h, ds=door_side):
            if ds == side:
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
        x1, y1, x2, y2 = (
            float(hall["x1"]), float(hall["y1"]),
            float(hall["x2"]), float(hall["y2"]),
        )
        width = float(hall["width"])
        dx, dy = x2 - x1, y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 0:
            ux, uy = -dy / dist * (width / 2), dx / dist * (width / 2)
            segments.append([x1 + ux, y1 + uy, x2 + ux, y2 + uy])
            segments.append([x1 - ux, y1 - uy, x2 - ux, y2 - uy])

    return segments


# ---------------------------------------------------------------------------
# Main renderer — clean map: walls + grid + base station only
# ---------------------------------------------------------------------------

def render_and_save(map_yaml_path: Path, out_path: Path, scale: float = 4.0) -> bool:
    """
    Render a clean map preview PNG (no spawn zones, no target markers).
    Returns True on success, False on error.

    Fast: <1 second, ~30 MB RAM. No JAX.
    """
    try:
        with open(map_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        width = float(data["width"])
        height = float(data["height"])
        img_w = int(width * scale)
        img_h = int(height * scale)

        # Background
        img = np.full((img_h, img_w, 3), (248, 250, 252), dtype=np.uint8)

        # Grid lines
        maze_grid = data.get("maze_cell_grid") or {}
        cols = int(maze_grid.get("cols", max(1, int(width // 10))))
        rows_grid = int(maze_grid.get("rows", max(1, int(height // 10))))

        for i in range(cols + 1):
            px = int(i * width / cols * scale)
            cv2.line(img, (px, 0), (px, img_h), (225, 213, 203), 1, cv2.LINE_AA)
        for i in range(rows_grid + 1):
            py = int((1.0 - i / rows_grid) * height * scale)
            cv2.line(img, (0, py), (img_w, py), (225, 213, 203), 1, cv2.LINE_AA)

        # Solid walls
        wall_w = max(2, int(scale))
        for seg in _compile_segments(data):
            x1, y1, x2, y2 = seg
            p1 = (int(x1 * scale), int((height - y1) * scale))
            p2 = (int(x2 * scale), int((height - y2) * scale))
            cv2.line(img, p1, p2, (31, 41, 55), wall_w, cv2.LINE_AA)

        # Base station marker (from spawn zone centre — no random spawn needed)
        base_pos = None
        if "spawn_zones" in data and "base" in data["spawn_zones"]:
            bx1, by1, bx2, by2 = data["spawn_zones"]["base"]
            base_pos = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

        if base_pos is not None:
            px = int(base_pos[0] * scale)
            py = int((height - base_pos[1]) * scale)
            bs = max(5, int(6 * scale / 4))
            # Filled square
            cv2.rectangle(img, (px - bs, py - bs), (px + bs, py + bs), (216, 78, 29), -1)
            # Border
            cv2.rectangle(img, (px - bs, py - bs), (px + bs, py + bs), (100, 24, 17), 1, cv2.LINE_AA)
            # "B" label
            font_scale = max(0.3, 0.38 * scale / 4)
            cv2.putText(
                img, "B",
                (px - int(3 * scale / 4), py + int(4 * scale / 4)),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), img)
        return bool(ok)

    except Exception as exc:
        print(f"[map_renderer] ERROR: {exc}")
        return False


def find_map_yaml(map_name: str, repo_root: Path) -> Path | None:
    """Resolve map_name to its YAML path."""
    candidate = repo_root / "src" / "curriculum_config" / "maps" / f"{map_name}.yaml"
    return candidate if candidate.exists() else None


def load_map_data(map_name: str, repo_root: Path) -> dict | None:
    """Load and return the raw map YAML as a dict, or None if not found."""
    yaml_path = find_map_yaml(map_name, repo_root)
    if yaml_path is None:
        return None
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        print(f"[map_renderer] Failed to load {yaml_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

def _dist_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Minimum distance from point (px, py) to line segment (x1,y1)→(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    rx = x1 + t * dx - px
    ry = y1 + t * dy - py
    return (rx * rx + ry * ry) ** 0.5


def validate_target(data: dict, x: float, y: float) -> tuple[bool, str]:
    """
    Full target placement validation against map constraints.

    Checks (in order):
      1. Within map bounds
      2. target_wall_clearance from all four world borders
      3. Not inside any rectangular or circular target exclusion
      4. target_wall_clearance from every wall segment

    Returns (is_valid, human_readable_reason).
    """
    width = float(data.get("width", 0.0))
    height = float(data.get("height", 0.0))

    if width <= 0 or height <= 0:
        return False, "map dimensions unknown"

    # 1. Basic bounds
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        return False, "outside map bounds"

    clearance = float(data.get("target_wall_clearance", 0.0))

    # 2. World border clearance
    if clearance > 0:
        if x < clearance:
            return False, f"too close to west border"
        if x > width - clearance:
            return False, f"too close to east border"
        if y < clearance:
            return False, f"too close to south border"
        if y > height - clearance:
            return False, f"too close to north border"

    # 3. Target exclude zones  (rectangular no-spawn areas)
    for zone in data.get("target_exclude_zones", []):
        zx1, zy1, zx2, zy2 = zone
        if (min(zx1, zx2) <= x <= max(zx1, zx2)) and (min(zy1, zy2) <= y <= max(zy1, zy2)):
            return False, "inside excluded zone (e.g. spawn room)"
    for centre_x, centre_y, radius in data.get("target_exclude_circles", []):
        if (x - float(centre_x)) ** 2 + (y - float(centre_y)) ** 2 <= float(radius) ** 2:
            return False, "inside circular excluded zone"

    # 4. Clearance from every interior wall segment
    if clearance > 0:
        for seg in _compile_segments(data):
            d = _dist_to_segment(x, y, *seg)
            if d < clearance:
                return False, f"too close to wall ({d:.1f} m < {clearance:.1f} m required)"

    return True, "valid"
