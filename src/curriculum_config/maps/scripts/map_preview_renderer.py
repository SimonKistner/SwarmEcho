"""
Render a static preview for any SwarmEcho map YAML.

This is a lightweight CLI companion to the Streamlit map builder. It draws the
same room/hallway/wall primitives plus spawn zones, without constructing an
environment rollout.

Usage
-----
python src/curriculum_config/maps/scripts/map_preview_renderer.py M01_grid_maze
python src/curriculum_config/maps/scripts/map_preview_renderer.py src/curriculum_config/maps/open_field.yaml --scale 3

Valid target spawn cells are always overlaid as a red pixel cloud so you can
verify wall clearance and exclude zones at a glance.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
MAP_DIR = ROOT / "src" / "curriculum_config" / "maps"
PREVIEW_DIR = ROOT / "outputs" / "map_previews"

# Make sure env/ is on the path so MapDefinition can be imported
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _resolve_map_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"}:
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return MAP_DIR / f"{name_or_path}.yaml"


def _to_px(x: float, y: float, height: float, scale: float) -> tuple[float, float]:
    return x * scale, (height - y) * scale


def _zone_to_px(zone: list[float], height: float, scale: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = zone
    px1, py1 = _to_px(x1, y2, height, scale)
    px2, py2 = _to_px(x2, y1, height, scale)
    return px1, py1, px2 - px1, py2 - py1


def _compile_segments(data: dict) -> list[list[float]]:
    segments = list(data.get("walls", []))

    # Match MapDefinition.rasterize room semantics.
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
        x1, y1, x2, y2 = (
            float(hall["x1"]),
            float(hall["y1"]),
            float(hall["x2"]),
            float(hall["y2"]),
        )
        width = float(hall["width"])
        dx, dy = x2 - x1, y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 0:
            ux, uy = -dy / dist * (width / 2), dx / dist * (width / 2)
            segments.append([x1 + ux, y1 + uy, x2 + ux, y2 + uy])
            segments.append([x1 - ux, y1 - uy, x2 - ux, y2 - uy])

    return segments


def render_map(map_path: Path, out_path: Path | None, scale: float) -> Path:
    with open(map_path, "r") as f:
        data = yaml.safe_load(f)

    width = float(data["width"])
    height = float(data["height"])
    img_w = width * scale
    img_h = height * scale

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{img_w:.0f}" height="{img_h:.0f}" '
        f'viewBox="0 0 {img_w:.3f} {img_h:.3f}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]

    # ── Valid target cell overlay (computed by MapDefinition.load) ──────────
    # Drawn first so walls/zones render on top.
    try:
        from env.maps import MapDefinition
        map_def = MapDefinition.load(map_path, cell_size=1.0)
        if map_def.valid_target_coords is not None:
            import numpy as np
            coords = np.array(map_def.valid_target_coords)
            # Each cell is 1 m wide; draw as a small square centred on the coord
            cell_px = max(1.0, scale)  # pixel size = 1 world-metre * scale
            for cx, cy in coords:
                px, py = _to_px(cx, cy, height, scale)
                # shift by half a cell so the rect is centred
                rx = px - cell_px / 2
                ry = py - cell_px / 2
                svg.append(
                    f'<rect x="{rx:.2f}" y="{ry:.2f}" '
                    f'width="{cell_px:.2f}" height="{cell_px:.2f}" '
                    f'fill="#ef4444" fill-opacity="0.55" stroke="none"/>'
                )
        # Draw target_exclude_zones as cross-hatched areas
        for ez in data.get("target_exclude_zones", []):
            ex, ey, ew, eh = _zone_to_px(ez, height, scale)
            svg.append(
                f'<rect x="{ex:.3f}" y="{ey:.3f}" width="{ew:.3f}" height="{eh:.3f}" '
                f'fill="url(#excl_hatch)" fill-opacity="0.6" '
                f'stroke="#7f1d1d" stroke-width="2" stroke-dasharray="6,3"/>'
            )
    except Exception as e:
        print(f"[warn] Could not overlay valid target coords: {e}")


    # ── Hatching pattern definition ────────────────────────────────────────
    svg.insert(1,
        '<defs>'
        '<pattern id="excl_hatch" patternUnits="userSpaceOnUse" width="8" height="8">'
        '<line x1="0" y1="8" x2="8" y2="0" stroke="#b91c1c" stroke-width="1.5" stroke-opacity="0.7"/>'
        '</pattern>'
        '</defs>'
    )

    # Light grid every 10m.
    for x in range(0, int(width) + 1, 10):
        px, _ = _to_px(float(x), 0.0, height, scale)
        svg.append(f'<line x1="{px:.3f}" y1="0" x2="{px:.3f}" y2="{img_h:.3f}" stroke="#cbd5e1" stroke-opacity="0.45" stroke-width="1"/>')
    for y in range(0, int(height) + 1, 10):
        _, py = _to_px(0.0, float(y), height, scale)
        svg.append(f'<line x1="0" y1="{py:.3f}" x2="{img_w:.3f}" y2="{py:.3f}" stroke="#cbd5e1" stroke-opacity="0.45" stroke-width="1"/>')

    wall_width = max(2.0, scale)
    for seg in _compile_segments(data):
        if len(seg) != 4:
            continue
        x1, y1, x2, y2 = map(float, seg)
        px1, py1 = _to_px(x1, y1, height, scale)
        px2, py2 = _to_px(x2, y2, height, scale)
        svg.append(
            f'<line x1="{px1:.3f}" y1="{py1:.3f}" x2="{px2:.3f}" y2="{py2:.3f}" '
            f'stroke="#1f2937" stroke-width="{wall_width:.3f}" stroke-linecap="square"/>'
        )

    zones = data.get("spawn_zones", {})
    zone_specs = [
        ("drone", "#6b7280", "#374151", "D"),
        ("base", "#2563eb", "#1d4ed8", "B"),
        ("target", "#ef4444", "#b91c1c", "T zone"),
    ]
    for key, fill, outline, label in zone_specs:
        if key not in zones:
            continue
        x, y, w, h = _zone_to_px(zones[key], height, scale)
        svg.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}" fill-opacity="0.25" stroke="{outline}" stroke-width="2"/>')
        svg.append(f'<text x="{x + 5:.3f}" y="{max(14, y - 5):.3f}" fill="{outline}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" font-weight="700">{html.escape(label)}</text>')

    points = data.get("spawn_points", {})
    point_specs = [
        ("drone", "#4b5563", "D"),
        ("target", "#dc2626", "T"),
        ("anti_target", "#581c87", "A"),
    ]
    for key, color, prefix in point_specs:
        for idx, point in enumerate(points.get(key, []) or []):
            px, py = _to_px(float(point[0]), float(point[1]), height, scale)
            r = max(4, int(round(2.5 * scale)))
            svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="{r}" fill="{color}" fill-opacity="0.9" stroke="#ffffff" stroke-width="2"/>')
            svg.append(f'<text x="{px + 5:.3f}" y="{py - 8:.3f}" fill="{color}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" font-weight="700">{prefix}{idx}</text>')

    if out_path is None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PREVIEW_DIR / f"map_preview_{data['name']}.svg"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", help="Map name or path to YAML")
    parser.add_argument("--out", type=Path, default=None, help="Output SVG path")
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Preview resolution in pixels per metre; does not affect map geometry.",
    )
    args = parser.parse_args()

    map_path = _resolve_map_path(args.map)
    if not map_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {map_path}")

    out = render_map(map_path, args.out, args.scale)
    print(out.resolve())


if __name__ == "__main__":
    main()
