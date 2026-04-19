"""
swarmecho/visualize/renderer_cv2.py
===================================
Fast OpenCV-based video renderer for SwarmEcho environment states.

This backend draws directly to NumPy arrays via C++ routines in OpenCV.
It is highly optimized for creating intermediate training videos that
still look professional but render in ~15-20 seconds instead of minutes.
"""

from __future__ import annotations

import datetime
import warnings
import multiprocessing
from functools import partial
from collections import deque
from pathlib import Path
from typing import NamedTuple

import imageio
import numpy as np
from omegaconf import DictConfig

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "opencv-python-headless is required for the fast CV2 renderer.\n"
        "Install it with:  uv add opencv-python-headless"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_DPI = 150
FRAME_STRIDE = 1
# ---------------------------------------------------------------------------
# BGR colour palette  (OpenCV uses BGR, not RGB)
# ---------------------------------------------------------------------------

def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

_C = {
    "bg_outer":     _hex_to_bgr("#f0f4f8"),
    "bg_inner":     _hex_to_bgr("#f8fafc"),
    "border":       _hex_to_bgr("#cbd5e1"),
    "grid":         _hex_to_bgr("#e2e8f0"),
    "text":         _hex_to_bgr("#0f172a"),
    "text_grey":    _hex_to_bgr("#64748b"),
    "base_chain":   _hex_to_bgr("#3b82f6"),   # blue
    "tgt_chain":    _hex_to_bgr("#ef4444"),   # red
    "both_chain":   _hex_to_bgr("#a855f7"),   # purple
    "iso":          _hex_to_bgr("#6b7280"),   # grey
    "base_mkr":     _hex_to_bgr("#1d4ed8"),   # dark blue
    "tgt_mkr":      _hex_to_bgr("#dc2626"),   # dark red
    "coverage":     _hex_to_bgr("#0d9488"),   # teal
    "link_grey":    _hex_to_bgr("#94a3b8"),
    "reward":       _hex_to_bgr("#10b981"),   # green
    "reward_bg":    _hex_to_bgr("#ffffff"),
    "white":        (255, 255, 255),
    "black":        (0, 0, 0),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FrameData(NamedTuple):
    pos:           np.ndarray   # (N, 2)
    vel:           np.ndarray   # (N, 2)
    base_pos:      np.ndarray   # (2,)
    target_pos:    np.ndarray   # (2,)
    coverage_grid: np.ndarray   # (G, G) bool
    step:          int
    active:        np.ndarray | None  # (N,) bool
    occ_grid:      np.ndarray | None  # (W_px, H_px) bool


def _bfs(adj: np.ndarray, source: int) -> set[int]:
    visited: set[int] = {source}
    queue   = deque([source])
    while queue:
        node = queue.popleft()
        for nb in np.where(adj[node])[0]:
            nb = int(nb)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def _dda_raycast_np(p1, p2, occ_grid):
    """NumPy version of DDA raycast for the renderer."""
    gx0, gy0 = p1; gx1, gy1 = p2
    dx = gx1 - gx0; dy = gy1 - gy0
    steps = int(max(abs(dx), abs(dy), 1) * 2) # Over-sample for safety in drawing
    if steps > 1000: steps = 1000
    
    xs = np.linspace(gx0, gx1, steps)
    ys = np.linspace(gy0, gy1, steps)
    
    ixs = np.floor(xs).astype(int)
    iys = np.floor(ys).astype(int)
    
    # Clip to bounds
    W, H = occ_grid.shape
    mask = (ixs >= 0) & (ixs < W) & (iys >= 0) & (iys < H)
    if not np.all(mask):
        # We allow out of bounds rays as long as they don't hit a wall
        # but realistically they shouldn't happen much.
        ixs = np.clip(ixs, 0, W-1)
        iys = np.clip(iys, 0, H-1)

    hit = occ_grid[ixs, iys]
    return not np.any(hit)

def _build_adjacency(pos, base_pos, target_pos, comm_r, vis_r, occ_grid, world_size):
    N = pos.shape[0];  M = N + 2
    ents  = np.concatenate([base_pos[None], target_pos[None], pos], axis=0)
    dists = np.linalg.norm(ents[:, None] - ents[None, :], axis=-1)
    
    # Distance based adjacency
    adj = (dists <= comm_r) & ~np.eye(M, dtype=bool)
    
    # Target visibility
    tmask = dists[1] <= vis_r
    adj[1, :] &= tmask;  adj[:, 1] &= tmask
    adj[1, 1] = adj[0, 1] = adj[1, 0] = False
    
    # Raycast check
    if occ_grid is not None:
        W_m, H_m = world_size
        sx, sy = occ_grid.shape[0] / W_m, occ_grid.shape[1] / H_m
        for i in range(M):
            for j in range(i + 1, M):
                if adj[i, j]:
                    p1 = ents[i] * np.array([sx, sy])
                    p2 = ents[j] * np.array([sx, sy])
                    if not _dda_raycast_np(p1, p2, occ_grid):
                        adj[i, j] = adj[j, i] = False
                        
    return adj


def _drone_col(idx_in_adj: int, base_comp: set, target_comp: set) -> tuple:
    ib = idx_in_adj in base_comp
    it = idx_in_adj in target_comp
    if ib and it: return _C["both_chain"]
    if ib:        return _C["base_chain"]
    if it:        return _C["tgt_chain"]
    return _C["iso"]


def _cfg_bgr(color_str: str, fallback: tuple) -> tuple:
    s = str(color_str)
    if s.startswith("#") and len(s) == 7: return _hex_to_bgr(s)
    return fallback


# ---------------------------------------------------------------------------
# Drawing Utilities
# ---------------------------------------------------------------------------

def _draw_text(img, text, pos, scale, color, thickness=1, center=False):
    font = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font than SIMPLEX
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if center:
        x -= size[0] // 2
        y += size[1] // 2
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thickness, cv2.LINE_AA)

def _draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=10):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    points = int(dist // gap)
    if points > 1:
        xs = np.linspace(pt1[0], pt2[0], points)
        ys = np.linspace(pt1[1], pt2[1], points)
        for i in range(0, points - 1, 2):
            cv2.line(img, (int(xs[i]), int(ys[i])), (int(xs[i+1]), int(ys[i+1])), color, thickness, cv2.LINE_AA)
    else:
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class _Layout:
    def __init__(self, W_world: float, H_world: float, has_reward: bool):
        scale = RENDER_DPI / 100.0
        # Standard margins for a symmetric, premium feel
        ml = int(80 * scale);  mr = int(80 * scale)
        mt = int(60 * scale);  mb = int(60 * scale)
        if has_reward:
            # Increase bottom margin significantly to fit reward plot + axis labels
            mb = int(220 * scale)

        MAX_LONG = int(1100 * scale)
        MIN_SHORT = int(200 * scale)
        aspect = W_world / H_world

        if aspect >= 1.0:
            pw = MAX_LONG; ph = max(int(MAX_LONG / aspect), MIN_SHORT)
        else:
            ph = MAX_LONG; pw = max(int(MAX_LONG * aspect), MIN_SHORT)

        # H.264 macroblock encoding requires strictly even dimensions. 
        # We absorb any odd-pixel remainder into the right/bottom margins.
        if (pw + ml + mr) % 2 != 0: mr += 1
        if (ph + mt + mb) % 2 != 0: mb += 1

        self.ml = ml;  self.mr = mr;  self.mt = mt;  self.mb = mb
        self.pw = pw;  self.ph = ph
        self.scale = pw / float(W_world)
        self.total_w = pw + ml + mr
        self.total_h = ph + mt + mb
        self.has_reward = has_reward
        self.rew_h = int(100 * scale) if has_reward else 0
        # Increased gap between map bottom and reward plot top
        self.rew_top = self.total_h - mb + int(60 * scale) if has_reward else 0
        
        # Base scale factor for fonts
        self.fs = scale * 0.45

    def w2p(self, x: float, y: float) -> tuple[int, int]:
        return (int(self.ml + x * self.scale), int(self.mt + self.ph - y * self.scale))

    def w2r(self, r: float) -> int:
        return max(1, int(r * self.scale))


# ---------------------------------------------------------------------------
# Per-frame draw
# ---------------------------------------------------------------------------

def _draw_frame_cv2(
    frame:          _FrameData,
    cfg:            DictConfig,
    lay:            _Layout,
    rewards_so_far: np.ndarray | None,
    total_steps:    int,
) -> np.ndarray:
    W = float(cfg.env.box_width); H = float(cfg.env.box_height)
    N = int(cfg.env.num_agents)
    vis_r = float(cfg.env.visual_radius); comm_r = float(cfg.env.comm_radius)
    B = int(cfg.env.radar_bins); v_cfg = cfg.visualize

    # Backgrounds
    img = np.full((lay.total_h, lay.total_w, 3), _C["bg_outer"], dtype=np.uint8)
    # Status markers
    cv2.rectangle(img, (lay.ml, lay.mt), (lay.ml + lay.pw, lay.mt + lay.ph), _C["bg_inner"], -1)
    # DRAW FRAME BORDER (Solid 2px border around the world)
    cv2.rectangle(img, (lay.ml, lay.mt), (lay.ml + lay.pw, lay.mt + lay.ph), _C["border"], 2)

    # ── Walls (Occupancy Grid) ───────────────────────────────────────────
    # We load the grid from the state (if available) or assume a static one
    # For now, we try to get it from the state if we were to add it, or 
    # we'll have to load it from the map.
    occ_grid = getattr(frame, "occ_grid", None)
    if occ_grid is not None:
        wall_img = np.zeros((occ_grid.shape[1], occ_grid.shape[0]), dtype=np.uint8)
        # grid is (W, H), opencv wants (H, W)
        wall_img[occ_grid.T[::-1]] = 50 # Dark grey walls
        wall_full = cv2.resize(wall_img, (lay.pw, lay.ph), interpolation=cv2.INTER_NEAREST)
        mask = wall_full > 0
        roi = img[lay.mt:lay.mt+lay.ph, lay.ml:lay.ml+lay.pw]
        roi[mask] = 50 # Solid walls

    # Axis labels removed per request


    # ── Coverage grid ─────────────────────────────────────────────────────
    if frame.coverage_grid.any():
        cell_size = float(cfg.env.grid_cell_size)
        gh_active = int(H / cell_size)
        gw_active = int(W / cell_size)
        # Use local world dimensions to mask the active region
        GW_active = int(W / cell_size)
        GH_active = int(H / cell_size)
        
        active_cov = frame.coverage_grid[:GW_active, :GH_active]
        
        # grid is (GW, GH), opencv wants (GH, GW)
        cov_img_bool = active_cov.T[::-1]
        gh_px, gw_px = cov_img_bool.shape
        cov_img = np.zeros((gh_px, gw_px, 3), dtype=np.uint8)
        cov_img[cov_img_bool] = _C["coverage"]
        cov_full = cv2.resize(cov_img, (lay.pw, lay.ph), interpolation=cv2.INTER_NEAREST)
        mask = cov_full.sum(axis=2) > 0
        roi = img[lay.mt:lay.mt+lay.ph, lay.ml:lay.ml+lay.pw]
        blend = (roi.astype(np.float32) * 0.75 + cov_full.astype(np.float32) * 0.25).astype(np.uint8)
        roi[mask] = blend[mask]

    # ── Connectivity ──────────────────────────────────────────────────────
    num_bases = int(cfg.env.num_bases or 0)
    num_targets = int(cfg.env.num_targets or 0)
    
    ents_comp = []
    base_idx = -1; target_idx = -1
    if num_bases > 0: 
        base_idx = len(ents_comp)
        ents_comp.append(frame.base_pos)
    if num_targets > 0:
        target_idx = len(ents_comp)
        ents_comp.append(frame.target_pos)
    
    drone_start = len(ents_comp)
    for i in range(N):
        ents_comp.append(frame.pos[i])
    ents = np.array(ents_comp)
    M = len(ents)

    # Build adjacency only for present entities
    adj = np.zeros((M, M), dtype=bool)
    dists = np.linalg.norm(ents[:, None] - ents[None, :], axis=-1)
    
    for i in range(M):
        for j in range(i + 1, M):
            if dists[i, j] <= comm_r:
                # Wall check
                p1_grid = ents[i] / cell_size
                p2_grid = ents[j] / cell_size
                if _dda_raycast_np(p1_grid, p2_grid, occ_grid):
                    adj[i, j] = adj[j, i] = True
    
    base_comp = _bfs(adj, base_idx) if base_idx >= 0 else set()
    target_comp = _bfs(adj, target_idx) if target_idx >= 0 else set()
    
    drone_cols = [_drone_col(i + drone_start, base_comp, target_comp) for i in range(N)]

    comm_col_cfg = _cfg_bgr(str(v_cfg.comm_color), None); vis_col_cfg = _cfg_bgr(str(v_cfg.vis_color), None)
    comm_fa = float(v_cfg.comm_fill_alpha); vis_fa = float(v_cfg.vis_fill_alpha)
    comm_r_px = lay.w2r(comm_r); vis_r_px = lay.w2r(vis_r)

    # ── Filled circles (batch alpha blend) ────────────────────────────────
    for fill_alpha, r_px, col_cfg in [(comm_fa, comm_r_px, comm_col_cfg), (vis_fa, vis_r_px, vis_col_cfg)]:
        if fill_alpha < 0.01: continue
        overlay = img.copy()
        for i in range(N):
            if frame.active is not None and not frame.active[i]: continue
            col = col_cfg if col_cfg else drone_cols[i]
            cv2.circle(overlay, lay.w2p(frame.pos[i, 0], frame.pos[i, 1]), r_px, col, -1)
        cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0, img)

    # ── Connectivity links ────────────────────────────────────────────────
    for i in range(M):
        for j in range(i + 1, M):
            if not adj[i, j]: continue
            ib = i in base_comp; jb = j in base_comp; it = i in target_comp; jt = j in target_comp
            if ib and jb and it and jt: ec, lw = _C["both_chain"], 2
            elif ib and jb: ec, lw = _C["base_chain"], 1
            elif it and jt: ec, lw = _C["tgt_chain"], 1
            else: ec, lw = _C["link_grey"], 1
            _draw_dashed_line(img, lay.w2p(ents[i, 0], ents[i, 1]), lay.w2p(ents[j, 0], ents[j, 1]), ec, lw, gap=6)

    # ── Circle outlines + radar lines ─────────────────────────────────────
    radar_angles = [b * 2.0 * np.pi / B - np.pi for b in range(B)]
    for i in range(N):
        cx, cy = lay.w2p(frame.pos[i, 0], frame.pos[i, 1]); col = drone_cols[i]
        cr_col = comm_col_cfg if comm_col_cfg else col; vr_col = vis_col_cfg if vis_col_cfg else col
        is_act = bool(frame.active[i]) if frame.active is not None else True
        if is_act:
            # Comm/Vis circles
            cv2.circle(img, (cx, cy), comm_r_px, cr_col, 1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), vis_r_px, vr_col, 1, cv2.LINE_AA)
            
            # 8 Radar bins: Draw as distinct radial pings
            for ang in radar_angles:
                ex = frame.pos[i, 0] + vis_r * np.cos(ang)
                ey = frame.pos[i, 1] + vis_r * np.sin(ang)
                # Use a thin line (1px) for the "beams" per request
                cv2.line(img, (cx, cy), lay.w2p(ex, ey), vr_col, 1, cv2.LINE_AA)

    # ── Base station ──────────────────────────────────────────────────────
    if int(cfg.env.num_bases) > 0:
        bpx, bpy = lay.w2p(frame.base_pos[0], frame.base_pos[1]); bs = lay.w2r(2.5)
        cv2.rectangle(img, (bpx - bs, bpy - bs), (bpx + bs, bpy + bs), _C["base_mkr"], -1)
        cv2.rectangle(img, (bpx - bs, bpy - bs), (bpx + bs, bpy + bs), (17, 24, 100), 1)
        _draw_text(img, "B", (bpx, bpy), lay.fs * 0.8, _C["white"], center=True)

    # ── Target ────────────────────────────────────────────────────────────
    if int(cfg.env.num_targets) > 0:
        tpx, tpy = lay.w2p(frame.target_pos[0], frame.target_pos[1]); tm = lay.w2r(3.5)
        cv2.drawMarker(img, (tpx, tpy), _C["tgt_mkr"], cv2.MARKER_STAR, tm*2, 2, cv2.LINE_AA)
        _draw_text(img, "T", (tpx + tm + 2, tpy + tm + 2), lay.fs * 0.8, _C["tgt_mkr"], center=True)

    # ── Drones ────────────────────────────────────────────────────────────
    dr = lay.w2r(1.5)
    for i in range(N):
        is_act = bool(frame.active[i]) if frame.active is not None else True
        alpha = 1.0 if is_act else 0.3
        cx, cy = lay.w2p(frame.pos[i, 0], frame.pos[i, 1]); col = drone_cols[i]
        if alpha < 1.0:
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), dr, col, -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), dr, _C["white"], 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        else:
            cv2.circle(img, (cx, cy), dr, col, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), dr, _C["white"], 1, cv2.LINE_AA)
        _draw_text(img, str(i), (cx + dr + 2, cy - dr - 2), lay.fs * 0.7, col)

    # ── Title ─────────────────────────────────────────────────────────────
    cell_size = float(cfg.env.grid_cell_size)
    GW_active = int(W / cell_size)
    GH_active = int(H / cell_size)
    
    # Coverage calculation on the active region
    cov_cnt = frame.coverage_grid[:GW_active, :GH_active].sum()
    cov_pct = 100.0 * cov_cnt / (GW_active * GH_active)
    
    # Logic for successful chain: intersection of base and target sets (excluding themselves)
    full_chain = bool(base_idx >= 0 and target_idx >= 0 and (base_comp & target_comp - {base_idx, target_idx}))
    status = " | CHAIN FORMED!" if full_chain else ""
    
    title = f"Step {frame.step:04d}   |   Coverage {cov_pct:.1f}%{status}"
    _draw_text(img, title, (lay.ml, lay.mt - 25), lay.fs * 1.0, _C["text"])

    # ── Legend ────────────────────────────────────────────────────────────
    lx = lay.ml + lay.pw + 20; ly = lay.mt + 20
    _draw_text(img, "Legend", (lx, ly), lay.fs * 0.9, _C["text"])
    legend_items = []
    if num_bases > 0:   legend_items.append((_C["base_mkr"],   "Base station"))
    if num_targets > 0: legend_items.append((_C["tgt_mkr"],    "Target"))
    
    if num_bases > 0:   legend_items.append((_C["base_chain"], "Base-connected"))
    if num_targets > 0: legend_items.append((_C["tgt_chain"],  "Target-connected"))
    if num_bases > 0 and num_targets > 0:
        legend_items.append((_C["both_chain"], "Both (bridge)"))
    
    legend_items.append((_C["iso"],        "Isolated"))
    legend_items.append((_C["coverage"],   "Coverage"))
    
    ly += 25
    bs = int(12 * lay.fs * 2)
    for col, label in legend_items:
        cv2.rectangle(img, (lx, ly - bs//2), (lx + bs, ly + bs//2), col, -1)
        _draw_text(img, label, (lx + bs + 8, ly + bs//4), lay.fs * 0.7, _C["text"])
        ly += int(25 * lay.fs * 2)

    # ── Reward plot ───────────────────────────────────────────────────────
    if lay.has_reward and rewards_so_far is not None:
        rt = lay.rew_top; rh = lay.rew_h
        rl = lay.ml; rr = lay.ml + lay.pw; rb = rt + rh

        cv2.rectangle(img, (rl, rt), (rr, rb), _C["reward_bg"], -1)
        cv2.rectangle(img, (rl, rt), (rr, rb), _C["border"], 1)
        
        # Title and Axis Labelling
        _draw_text(img, "Cumulative Team Reward", (rl, rt - 12), lay.fs * 0.8, _C["text"])
        _draw_text(img, "Return", (rl - 50, rt + rh//2), lay.fs * 0.7, _C["text_grey"])

        # Gridlines (Both H and V now)
        for i in range(1, 4):
            yy = rt + i * rh // 4
            cv2.line(img, (rl, yy), (rr, yy), _C["grid"], 1)
        for i in range(1, 5):
            xx = rl + i * (rr - rl) // 5
            cv2.line(img, (xx, rt), (xx, rb), _C["grid"], 1)

        if len(rewards_so_far) > 0:
            cum = np.cumsum(rewards_so_far)
            cmin = float(cum.min()); cmax = float(cum.max())
            if abs(cmax - cmin) < 1e-4: cmax += 1.0
            
            # Y-Axis Ticks (Min/Max)
            _draw_text(img, f"{cmax:.0f}", (rl - 8, rt + 5), lay.fs * 0.6, _C["text_grey"], center=False)
            _draw_text(img, f"{cmin:.0f}", (rl - 8, rb - 5), lay.fs * 0.6, _C["text_grey"], center=False)

            # X-Axis Ticks (Steps)
            for i in range(6):
                tx = rl + i * (rr - rl) // 5
                step_val = i * total_steps // 5
                cv2.line(img, (tx, rb), (tx, rb + 5), _C["border"], 1)
                _draw_text(img, str(step_val), (tx, rb + 18), lay.fs * 0.6, _C["text_grey"], center=True)

            if len(rewards_so_far) > 1:
                xs = np.linspace(rl, rr, len(cum)).astype(np.int32)
                ys = (rb - (cum - cmin) / (cmax - cmin) * rh).astype(np.int32)
                pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, _C["reward"], 2, cv2.LINE_AA)

    # ── Final Border ──────────────────────────────────────────────────────
    cv2.rectangle(img, (lay.ml, lay.mt), (lay.ml + lay.pw - 1, lay.mt + lay.ph - 1), _C["border"], 2)

    return img


# ---------------------------------------------------------------------------
# Render Implementation
# ---------------------------------------------------------------------------

def render_video_cv2(
    trajectory,
    cfg:          DictConfig,
    filename:     str | Path,
    fps:          int  = 20,
    frame_stride: int | None = None,
    rewards:      np.ndarray | None = None,
) -> str:
    filename = Path(filename).resolve()
    
    sample = getattr(trajectory, "pos", None)
    if sample is not None and isinstance(sample, np.ndarray):
        traj_cpu = trajectory; rewards_cpu = rewards
    else:
        import jax as _jax
        traj_cpu = _jax.device_get(trajectory)
        rewards_cpu = _jax.device_get(rewards) if rewards is not None else None

    if frame_stride is None:
        frame_stride = FRAME_STRIDE

    T = traj_cpu.pos.shape[0]
    idxs = list(range(0, T, frame_stride))
    
    # Extract dynamic world dimensions from the first frame of the trajectory
    W = float(traj_cpu.box_width[0])
    H = float(traj_cpu.box_height[0])
    
    lay = _Layout(W, H, rewards_cpu is not None)
    
    # Load map once for static rendering data (walls)
    from env.maps import MapDefinition
    occ_grid_static = None
    if cfg.env.map_names and len(cfg.env.map_names) > 0:
        active_map_name = cfg.env.map_names[0]
        map_path = Path("maps") / f"{active_map_name}.yaml"
        if not map_path.exists():
            map_path = Path(__file__).resolve().parents[2] / "maps" / f"{active_map_name}.yaml"
        if map_path.exists():
            map_def = MapDefinition.load(map_path, cell_size=float(cfg.env.grid_cell_size))
            occ_grid_static = map_def.occupancy_grid

    # 1. Prepare frame data for parallel processing
    frame_args = []
    for t in idxs:
        frame_args.append(_FrameData(
            pos           = np.array(traj_cpu.pos[t]),
            vel           = np.array(traj_cpu.vel[t]),
            base_pos      = np.array(traj_cpu.base_pos[t]),
            target_pos    = np.array(traj_cpu.target_pos[t]),
            coverage_grid = np.array(traj_cpu.coverage_grid[t]),
            step          = int(traj_cpu.step[t]),
            active        = (np.array(traj_cpu.active[t]) if hasattr(traj_cpu, "active") else None),
            occ_grid      = occ_grid_static,
            box_width     = float(traj_cpu.box_width[t]),
            box_height    = float(traj_cpu.box_height[t]),
        ))

    # 2. Determine parallelism (daemonic processes cannot spawn children)
    is_daemon = multiprocessing.current_process().daemon
    num_vcpus = multiprocessing.cpu_count()
    pool_size = max(1, num_vcpus - 1) if not is_daemon else 0
    
    # Partial function for the worker
    worker = partial(_draw_frame_cv2, cfg=cfg, lay=lay, total_steps=T)
    worker_full = partial(worker, rewards_so_far=rewards_cpu)

    print(f"Frame size: {lay.total_w}×{lay.total_h} px  (CV2 renderer, pool={pool_size}, world {W:.0f}×{H:.0f} m)")
    print(f"Rendering {len(idxs)} frames ({T} steps) → {filename}")

    last_img = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="os.fork.*is incompatible with multithreaded code")
        
        writer = imageio.get_writer(
            str(filename), fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None, quality=7,
        )
        
        try:
            if pool_size > 1:
                # Parallel path
                with multiprocessing.Pool(processes=pool_size) as pool:
                    for k, bgr in enumerate(pool.imap(worker_full, frame_args)):
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        writer.append_data(rgb)
                        last_img = rgb
                        if (k + 1) % 50 == 0 or k == len(idxs) - 1:
                            print(f"  {k+1}/{len(idxs)} frames rendered", flush=True)
            else:
                # Sequential fallback (for daemon workers or single-core)
                for k, arg in enumerate(frame_args):
                    bgr = worker_full(arg)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb)
                    last_img = rgb
                    if (k + 1) % 50 == 0 or k == len(idxs) - 1:
                        print(f"  {k+1}/{len(idxs)} frames rendered (sequential)", flush=True)

            # 5-second freeze hold
            if last_img is not None:
                for _ in range(fps * 5):
                    writer.append_data(last_img)
        finally:
            writer.close()

    return str(filename)


