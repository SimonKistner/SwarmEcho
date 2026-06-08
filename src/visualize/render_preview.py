"""
SwarmEcho Map Preview and Simulation Renderer
=============================================
Consolidated utility to render static image previews (PNG, SVG) or short rollout
videos/GIFs of any map.

Usage:
------
# 1. Render default simulated video rollout (MP4, 100 frames, 3 drones, 1 target, 1 base):
uv run python src/visualize/render_preview.py M01_grid_maze

# 2. Render a static PNG blueprint image (showing spawn zones and real spawn positions):
uv run python src/visualize/render_preview.py M01_grid_maze --mode image

# 3. Render a clean architectural SVG blueprint without spawns/zones:
uv run python src/visualize/render_preview.py M01_grid_maze --mode image --format svg --no-spawns --no-zones

# 4. Render a snappy simulated GIF preview using a specific level configuration:
uv run python src/visualize/render_preview.py memory_t_maze_8 --mode gif --level MEM_T8_memory
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from PIL import Image
from omegaconf import OmegaConf

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import MAP_DIR, load_config, validate_config
from env.maps import MapDefinition
from env.physics import make_env_fns
from visualize.renderer import render_video


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


def render_svg(
    data: dict,
    map_def: Optional[MapDefinition],
    state: Optional[jax.Array],
    show_zones: bool,
    show_spawns: bool,
    scale: float,
) -> str:
    width = float(data["width"])
    height = float(data["height"])
    img_w = width * scale
    img_h = height * scale

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{img_w:.0f}" height="{img_h:.0f}" '
        f'viewBox="0 0 {img_w:.3f} {img_h:.3f}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<defs>',
        '<pattern id="excl_hatch" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<line x1="0" y1="8" x2="8" y2="0" stroke="#b91c1c" stroke-width="1.5" stroke-opacity="0.7"/>',
        '</pattern>',
        '</defs>'
    ]

    # Grid lines every 10m
    for x in range(0, int(width) + 1, 10):
        px = x * scale
        svg.append(f'<line x1="{px:.3f}" y1="0" x2="{px:.3f}" y2="{img_h:.3f}" stroke="#cbd5e1" stroke-opacity="0.45" stroke-width="1"/>')
    for y in range(0, int(height) + 1, 10):
        py = (height - y) * scale
        svg.append(f'<line x1="0" y1="{py:.3f}" x2="{img_w:.3f}" y2="{py:.3f}" stroke="#cbd5e1" stroke-opacity="0.45" stroke-width="1"/>')

    # Valid target coords (Red cloud overlay)
    if show_zones and map_def is not None and map_def.valid_target_coords is not None:
        cell_px = max(1.0, scale)
        for cx, cy in map_def.valid_target_coords:
            px = cx * scale
            py = (height - cy) * scale
            rx = px - cell_px / 2
            ry = py - cell_px / 2
            svg.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{cell_px:.2f}" height="{cell_px:.2f}" fill="#ef4444" fill-opacity="0.1375" stroke="none"/>')

    # Exclude zones (cross-hatched)
    if show_zones:
        for ez in data.get("target_exclude_zones", []):
            x1, y1, x2, y2 = ez
            ex = x1 * scale
            ey = (height - y2) * scale
            ew = (x2 - x1) * scale
            eh = (y2 - y1) * scale
            svg.append(f'<rect x="{ex:.3f}" y="{ey:.3f}" width="{ew:.3f}" height="{eh:.3f}" fill="url(#excl_hatch)" fill-opacity="0.6" stroke="#7f1d1d" stroke-width="2" stroke-dasharray="6,3"/>')

    # Walls
    wall_width = max(2.0, scale)
    segments = _compile_segments(data)
    for seg in segments:
        x1, y1, x2, y2 = seg
        px1, py1 = x1 * scale, (height - y1) * scale
        px2, py2 = x2 * scale, (height - y2) * scale
        svg.append(f'<line x1="{px1:.3f}" y1="{py1:.3f}" x2="{px2:.3f}" y2="{py2:.3f}" stroke="#1f2937" stroke-width="{wall_width:.3f}" stroke-linecap="square"/>')

    # Mesh walls: movement/visual blockers, but communication-transparent.
    for seg in data.get("mesh_walls", data.get("mesh", [])):
        x1, y1, x2, y2 = seg
        px1, py1 = x1 * scale, (height - y1) * scale
        px2, py2 = x2 * scale, (height - y2) * scale
        svg.append(f'<line x1="{px1:.3f}" y1="{py1:.3f}" x2="{px2:.3f}" y2="{py2:.3f}" stroke="#0ea5e9" stroke-width="{wall_width:.3f}" stroke-linecap="square"/>')

    # Zones
    if show_zones:
        zones = data.get("spawn_zones", {})
        zone_specs = [
            ("drone", "#6b7280", "#374151", "D"),
            ("base", "#2563eb", "#1d4ed8", "B"),
            ("target", "#ef4444", "#b91c1c", "T zone"),
        ]
        for key, fill, outline, label in zone_specs:
            if key not in zones:
                continue
            x1, y1, x2, y2 = zones[key]
            zx = x1 * scale
            zy = (height - y2) * scale
            zw = (x2 - x1) * scale
            zh = (y2 - y1) * scale
            alpha = 0.0625 if key == "target" else 0.25
            svg.append(f'<rect x="{zx:.3f}" y="{zy:.3f}" width="{zw:.3f}" height="{zh:.3f}" fill="{fill}" fill-opacity="{alpha:.4f}" stroke="{outline}" stroke-width="2"/>')
            svg.append(f'<text x="{zx + 5:.3f}" y="{max(14, zy - 5):.3f}" fill="{outline}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" font-weight="700">{label}</text>')

    # Base Station (Always drawn as a reference landmark if resolvable)
    # TODO: Support random base spawns when show_spawns is False (do not rely on zone average)
    base_pos = None
    if state is not None and getattr(state, "base_pos", None) is not None:
        base_pos = (float(state.base_pos[0]), float(state.base_pos[1]))
    elif "spawn_zones" in data and "base" in data["spawn_zones"]:
        x1, y1, x2, y2 = data["spawn_zones"]["base"]
        base_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    if base_pos is not None:
        px = base_pos[0] * scale
        py = (height - base_pos[1]) * scale
        bs = max(4.0, 5.0 * scale / 4.0)
        svg.append(f'<rect x="{px - bs:.3f}" y="{py - bs:.3f}" width="{2*bs:.3f}" height="{2*bs:.3f}" fill="#2563eb" stroke="#1d4ed8" stroke-width="1.5"/>')
        svg.append(f'<text x="{px:.3f}" y="{py + bs/2:.3f}" fill="#ffffff" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" font-weight="700" text-anchor="middle">B</text>')

    # Real Spawns
    if show_spawns and state is not None:

        # Target
        if getattr(state, "target_pos", None) is not None:
            target_points = np.asarray(state.target_pos)
            if target_points.ndim == 1:
                target_points = target_points[None, :]
            for k, tp in enumerate(target_points):
                tx, ty = float(tp[0]), float(tp[1])
                px = tx * scale
                py = (height - ty) * scale
                tm = max(6.0, 6.0 * scale / 4.0)
                svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="{tm:.3f}" fill="#ef4444" stroke="#b91c1c" stroke-width="1.5"/>')
                label = "T" if len(target_points) == 1 else f"T{k}"
                svg.append(f'<text x="{px:.3f}" y="{py + tm/2 - 1:.3f}" fill="#ffffff" font-family="DejaVu Sans, Arial, sans-serif" font-size="9" font-weight="700" text-anchor="middle">{label}</text>')

        # Drones
        dr = max(4.0, 4.0 * scale / 4.0)
        for i in range(len(state.pos)):
            dx, dy = float(state.pos[i, 0]), float(state.pos[i, 1])
            px = dx * scale
            py = (height - dy) * scale
            svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="{dr:.3f}" fill="#6b7280" stroke="#ffffff" stroke-width="1.5"/>')
            svg.append(f'<text x="{px + dr + 4:.3f}" y="{py + dr/2:.3f}" fill="#374151" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" font-weight="700">{i}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def render_png(
    data: dict,
    map_def: Optional[MapDefinition],
    state: Optional[jax.Array],
    show_zones: bool,
    show_spawns: bool,
    scale: float,
) -> np.ndarray:
    width = float(data["width"])
    height = float(data["height"])
    img_w = int(width * scale)
    img_h = int(height * scale)

    # 1. Blank background (light slate background)
    img = np.full((img_h, img_w, 3), (252, 250, 248), dtype=np.uint8)

    # 2. Grid lines
    for x in range(0, int(width) + 1, 10):
        px = int(x * scale)
        cv2.line(img, (px, 0), (px, img_h), (225, 213, 203), 1, cv2.LINE_AA)
    for y in range(0, int(height) + 1, 10):
        py = int((height - y) * scale)
        cv2.line(img, (0, py), (img_w, py), (225, 213, 203), 1, cv2.LINE_AA)

    # 3. Valid target coords (Red cloud overlay)
    if show_zones and map_def is not None and map_def.valid_target_coords is not None:
        cell_px = max(1, int(scale))
        overlay = img.copy()
        for cx, cy in map_def.valid_target_coords:
            px = int(cx * scale)
            py = int((height - cy) * scale)
            rx1 = int(px - cell_px / 2)
            ry1 = int(py - cell_px / 2)
            rx2 = rx1 + cell_px
            ry2 = ry1 + cell_px
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (68, 68, 239), -1)
        cv2.addWeighted(overlay, 0.1375, img, 0.8625, 0, img)

    # 4. Target exclude zones (cross-hatched equivalent)
    if show_zones:
        for ez in data.get("target_exclude_zones", []):
            x1, y1, x2, y2 = ez
            px1 = int(x1 * scale)
            py1 = int((height - y2) * scale)
            px2 = int(x2 * scale)
            py2 = int((height - y1) * scale)
            overlay = img.copy()
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (28, 28, 185), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.rectangle(img, (px1, py1), (px2, py2), (28, 28, 185), 2, cv2.LINE_AA)

    # 5. Walls
    wall_width = max(2, int(scale))
    segments = _compile_segments(data)
    for seg in segments:
        x1, y1, x2, y2 = seg
        p1 = (int(x1 * scale), int((height - y1) * scale))
        p2 = (int(x2 * scale), int((height - y2) * scale))
        cv2.line(img, p1, p2, (55, 41, 31), wall_width, cv2.LINE_AA)

    # Mesh walls: movement/visual blockers, but communication-transparent.
    for seg in data.get("mesh_walls", data.get("mesh", [])):
        x1, y1, x2, y2 = seg
        p1 = (int(x1 * scale), int((height - y1) * scale))
        p2 = (int(x2 * scale), int((height - y2) * scale))
        cv2.line(img, p1, p2, (235, 165, 14), wall_width, cv2.LINE_AA)

    # 6. Spawn Zones
    if show_zones:
        zones = data.get("spawn_zones", {})
        zone_specs = [
            ("drone", (128, 114, 107), (81, 65, 55), "D"),
            ("base", (235, 99, 37), (216, 78, 29), "B"),
            ("target", (68, 68, 239), (28, 28, 185), "T zone"),
        ]
        for key, fill, outline, label in zone_specs:
            if key not in zones:
                continue
            x1, y1, x2, y2 = zones[key]
            px1 = int(x1 * scale)
            py1 = int((height - y2) * scale)
            px2 = int(x2 * scale)
            py2 = int((height - y1) * scale)
            overlay = img.copy()
            cv2.rectangle(overlay, (px1, py1), (px2, py2), fill, -1)
            alpha = 0.0625 if key == "target" else 0.25
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
            cv2.rectangle(img, (px1, py1), (px2, py2), outline, 2, cv2.LINE_AA)
            cv2.putText(img, label, (px1 + 5, max(15, py1 - 5)), cv2.FONT_HERSHEY_DUPLEX, 0.4, outline, 1, cv2.LINE_AA)

    # Base Station (Always drawn as a reference landmark if resolvable)
    # TODO: Support random base spawns when show_spawns is False (do not rely on zone average)
    base_pos = None
    if state is not None and getattr(state, "base_pos", None) is not None:
        base_pos = (float(state.base_pos[0]), float(state.base_pos[1]))
    elif "spawn_zones" in data and "base" in data["spawn_zones"]:
        x1, y1, x2, y2 = data["spawn_zones"]["base"]
        base_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    if base_pos is not None:
        px = int(base_pos[0] * scale)
        py = int((height - base_pos[1]) * scale)
        bs = max(4, int(5 * scale / 4))
        cv2.rectangle(img, (px - bs, py - bs), (px + bs, py + bs), (216, 78, 29), -1)
        cv2.rectangle(img, (px - bs, py - bs), (px + bs, py + bs), (100, 24, 17), 1, cv2.LINE_AA)
        cv2.putText(img, "B", (px - 4, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Real Spawns (drones, target)
    if show_spawns and state is not None:

        # Target
        if getattr(state, "target_pos", None) is not None:
            target_points = np.asarray(state.target_pos)
            if target_points.ndim == 1:
                target_points = target_points[None, :]
            for k, tp in enumerate(target_points):
                tx, ty = float(tp[0]), float(tp[1])
                px = int(tx * scale)
                py = int((height - ty) * scale)
                tm = max(6, int(6 * scale / 4))
                cv2.circle(img, (px, py), tm, (68, 68, 239), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), tm, (28, 28, 185), 1, cv2.LINE_AA)
                label = "T" if len(target_points) == 1 else f"T{k}"
                cv2.putText(img, label, (px - 4, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Drones
        dr = max(4, int(4 * scale / 4))
        for i in range(len(state.pos)):
            dx, dy = float(state.pos[i, 0]), float(state.pos[i, 1])
            px = int(dx * scale)
            py = int((height - dy) * scale)
            cv2.circle(img, (px, py), dr, (128, 114, 107), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), dr, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(i), (px + dr + 4, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.35, (81, 65, 55), 1, cv2.LINE_AA)

    return img


def convert_mp4_to_gif(input_vid: Path, output_gif: Path, frame_stride: int = 2) -> None:
    cap = cv2.VideoCapture(str(input_vid))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            target_w = 600
            target_h = int(h * (target_w / w))
            frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(frame_resized))

        count += 1

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from the generated video for GIF conversion.")

    # Keep 2x subjective speed
    duration_ms = int(1000 / fps)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidated Map Preview and Simulation Renderer")
    parser.add_argument("map", help="Map name or path to YAML blueprint")
    parser.add_argument(
        "--mode",
        choices=["video", "image", "gif"],
        default="image",
        help="Preview mode: 'image' (default, blueprint), 'video' (simulated MP4 rollout), or 'gif' (simulated GIF rollout)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output image format for 'image' mode (default 'png')",
    )
    parser.add_argument(
        "--renderer",
        choices=["fast", "slow"],
        default="fast",
        help="Renderer to use for video/gif: 'fast' (OpenCV, default) or 'slow' (Matplotlib)",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Optional level name or YAML path to load environment variables from",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames for video/gif output (default 10)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Preview resolution in pixels per world-meter (only affects image mode)",
    )
    parser.add_argument(
        "--show-zones",
        action="store_true",
        default=True,
        help="Render spawn zones on the blueprint (default True)",
    )
    parser.add_argument(
        "--no-zones",
        action="store_false",
        dest="show_zones",
        help="Do not render spawn zones on the blueprint",
    )
    parser.add_argument(
        "--show-spawns",
        action="store_true",
        default=True,
        help="Render spawned positions of drones/base/targets (default True)",
    )
    parser.add_argument(
        "--no-spawns",
        action="store_false",
        dest="show_spawns",
        help="Do not render spawned entities or spawn zones (clean blueprint mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for spawning drones and targets in the environment (default 42)",
    )

    args = parser.parse_args()

    # 1. Resolve map path and load basic YAML
    map_path = _resolve_map_path(args.map)
    if not map_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {map_path}")

    with open(map_path, "r") as f:
        map_data = yaml.safe_load(f)
    map_name = map_data["name"]

    # If --no-spawns was passed, we also imply no spawn zones to give a clean blueprint
    if not args.show_spawns:
        args.show_zones = False

    # 2. Setup config
    if args.level:
        cfg = load_config(overrides=[f"level={args.level}"])
    else:
        cfg = load_config(cli_overrides=False)
        # Override to 3 drones, 1 target, 1 base when no level config is provided
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["env"]["num_agents"] = 3
        cfg["env"]["num_targets"] = 1
        cfg["env"]["num_bases"] = 1
        cfg = OmegaConf.create(cfg)

    # Force map override in curriculum config
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["env"]["map_names"] = [map_name]
    cfg = OmegaConf.create(cfg)
    validate_config(cfg)

    # 3. Setup environment and reset to spawn objects
    state = None
    env_step = None
    map_def = None
    try:
        env_step, env_reset, _, (W, H, occ_grid, comm_occ_grid) = make_env_fns(cfg)
        state = env_reset(jax.random.PRNGKey(args.seed))
        map_def = MapDefinition.load(map_path, cell_size=1.0)
    except Exception as e:
        print(f"[warn] Could not initialize environment or sample spawns: {e}")

    # 4. Resolve Automatic Output Path
    out_dir = ROOT / "outputs" / "map_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "image":
        out_path = out_dir / f"map_preview_{map_name}.{args.format}"
    elif args.mode == "gif":
        out_path = out_dir / f"map_preview_{map_name}.gif"
    else:
        out_path = out_dir / f"map_preview_{map_name}.mp4"

    # 5. Execution: Image Mode
    if args.mode == "image":
        if args.format == "svg":
            svg_content = render_svg(
                data=map_data,
                map_def=map_def,
                state=state,
                show_zones=args.show_zones,
                show_spawns=args.show_spawns,
                scale=args.scale,
            )
            out_path.write_text(svg_content, encoding="utf-8")
        else:
            img_bgr = render_png(
                data=map_data,
                map_def=map_def,
                state=state,
                show_zones=args.show_zones,
                show_spawns=args.show_spawns,
                scale=args.scale,
            )
            cv2.imwrite(str(out_path), img_bgr)

        print(f"✓ Static map preview image saved to: {out_path.resolve()}")

    # 6. Execution: Video or GIF Mode
    else:
        if state is None or env_step is None:
            raise RuntimeError("Environment must be fully initialized to render video/GIF rollouts.")

        # GIF needs a temporary MP4 rendered first
        render_path = out_path if args.mode == "video" else out_path.with_suffix(".mp4")

        # Rollout generation (Always simulated using random actions)
        states = [state]
        for step_idx in range(args.num_frames - 1):
            key, subkey = jax.random.split(state.key)
            actions = jax.random.uniform(subkey, (cfg.env.num_agents, 2), minval=-1.0, maxval=1.0)
            state = env_step(state, actions)
            states.append(state)

        # Stack states to form trajectory
        trajectory = jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        # Call the standard project visualizer
        print(f"Rendering {args.num_frames} frames simulation rollout using '{args.renderer}' renderer...")
        actual_mp4 = render_video(
            trajectory=trajectory,
            cfg=cfg,
            filename=render_path,
            fps=20,
            renderer=args.renderer,
        )

        # Post-process to GIF if requested
        if args.mode == "gif":
            print(f"Converting temporary video to snappy GIF...")
            convert_mp4_to_gif(Path(actual_mp4), out_path, frame_stride=2)
            if os.path.exists(actual_mp4):
                os.remove(actual_mp4)
            print(f"✓ Snappy GIF preview saved to: {out_path.resolve()}")
        else:
            print(f"✓ Video preview saved to: {Path(actual_mp4).resolve()}")


if __name__ == "__main__":
    main()
