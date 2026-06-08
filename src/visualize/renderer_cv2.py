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

from visualize.renderer_config import RendererConfig

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
    "anti_mkr":     _hex_to_bgr("#581c87"),   # dark purple
    "coverage":     _hex_to_bgr("#0d9488"),   # teal
    "link_grey":    _hex_to_bgr("#94a3b8"),
    "reward":       _hex_to_bgr("#10b981"),   # green
    "reward_bg":    _hex_to_bgr("#ffffff"),
    "white":        (255, 255, 255),
    "black":        (0, 0, 0),
}

_TAB10 = [
    _hex_to_bgr("#1f77b4"), _hex_to_bgr("#ff7f0e"), _hex_to_bgr("#2ca02c"),
    _hex_to_bgr("#d62728"), _hex_to_bgr("#9467bd"), _hex_to_bgr("#8c564b"),
    _hex_to_bgr("#e377c2"), _hex_to_bgr("#7f7f7f"), _hex_to_bgr("#bcbd22"),
    _hex_to_bgr("#17becf")
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FrameData(NamedTuple):
    pos:           np.ndarray   # (N, 2)
    vel:           np.ndarray   # (N, 2)
    base_pos:      np.ndarray   # (2,)
    target_pos:    np.ndarray   # (2,) normally; (N, 2) only for MEM_T8 diagnostics
    anti_target_pos: np.ndarray | None  # MEM_T8-only diagnostic markers
    coverage_grid: np.ndarray   # (G, G) bool
    step:          int
    active:        np.ndarray | None  # (N,) bool
    collides:      np.ndarray | None  # (N,) bool
    occ_grid:      np.ndarray | None  # (W_px, H_px) bool
    comm_occ_grid: np.ndarray | None  # communication blockers; mesh walls are transparent
    mesh_walls:    np.ndarray | None  # (M, 4), drawn blue for communication-transparent blockers
    box_width:     float
    box_height:    float
    extra_metrics: dict[str, any]
    target_known:  np.ndarray | None
    base_target_known: bool | None
    adj_matrix:    np.ndarray | None


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


from env.raycast import dda_raycast_np

def _dda_raycast_np(p1, p2, occ_grid):
    """NumPy version of DDA raycast for the renderer."""
    return dda_raycast_np(p1, p2, occ_grid)


def _build_adjacency(pos, base_pos, target_pos, comm_r, vis_r, occ_grid, world_size):
    N = pos.shape[0];  M = N + 2
    ents  = np.concatenate([base_pos[None], target_pos[None], pos], axis=0)
    dists = np.linalg.norm(ents[:, None] - ents[None, :], axis=-1)

    # Distance based adjacency
    adj = (dists <= comm_r) & ~np.eye(M, dtype=bool)

    # Base visibility
    bmask = dists[0] <= vis_r
    adj[0, :] &= bmask;  adj[:, 0] &= bmask

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

    # Identify indices
    base_idx = 0; target_idx = 1; drone_start = 2

    return adj, base_idx, target_idx, drone_start, ents, dists



def _get_shortest_path_distances(adj: np.ndarray, source: int) -> np.ndarray:
    """Computes shortest path distances (hops) from a single source using a queue-based BFS."""
    V = adj.shape[0]
    dists = np.full(V, 999, dtype=np.int32)
    if source < 0 or source >= V:
        return dists
    dists[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        d_u = dists[u]
        for v in np.where(adj[u])[0]:
            v = int(v)
            if dists[v] == 999:
                dists[v] = d_u + 1
                queue.append(v)
    return dists

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

def _draw_text(img, text, pos, scale, color, thickness=1, center=False, font=cv2.FONT_HERSHEY_DUPLEX):
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if center:
        x -= size[0] // 2
        y += size[1] // 2
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thickness, cv2.LINE_AA)

def _draw_monospace_text(img, text, pos, scale, color, thickness=1):
    x, y = pos
    # Measure a reference character to determine spacing
    (w, h), _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    char_step = w + int(4 * scale)
    for i, char in enumerate(text):
        if char != " ":
            cv2.putText(img, char, (int(x + i * char_step), int(y)), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

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
    def __init__(self, W_world: float, H_world: float, has_reward: bool, has_indiv_reward: bool = False):
        scale = RendererConfig.RENDER_DPI / 100.0

        ml = int(RendererConfig.MARGIN_LEFT * scale)
        mr = int(RendererConfig.MARGIN_RIGHT * scale)
        mt = int(RendererConfig.MARGIN_TOP * scale)
        mb = int(RendererConfig.MARGIN_BOTTOM * scale)

        # Fit map within designated display width and height
        max_w = int(RendererConfig.MAP_DISPLAY_WIDTH * scale)
        max_h = int(RendererConfig.MAP_DISPLAY_HEIGHT * scale)

        aspect = W_world / H_world
        if aspect >= 1.0:
            pw = max_w
            ph = int(max_w / aspect)
        else:
            ph = max_h
            pw = int(max_h * aspect)

        self.ml = ml;  self.mr = mr;  self.mt = mt;  self.mb = mb
        self.pw = pw;  self.ph = ph
        self.scale = pw / float(W_world)
        self.has_reward = has_reward
        self.has_indiv_reward = has_indiv_reward

        # Compute reward plots layout using static values from configuration
        self.rew_h = int(RendererConfig.TEAM_REWARD_HEIGHT * scale) if has_reward else 0
        self.indiv_rew_h = int(RendererConfig.INDIV_REWARD_HEIGHT * scale) if has_indiv_reward else 0

        gap_map_rew = int(RendererConfig.GAP_MAP_REWARDS * scale)
        gap_between = int(RendererConfig.GAP_BETWEEN_REWARDS * scale)

        self.rew_top = mt + ph + gap_map_rew if has_reward else 0
        self.indiv_rew_top = self.rew_top + self.rew_h + gap_between if has_indiv_reward else 0

        # Calculate total total_w and total_h
        self.total_w = pw + ml + mr

        if has_indiv_reward:
            self.total_h = self.indiv_rew_top + self.indiv_rew_h + mb
        elif has_reward:
            self.total_h = self.rew_top + self.rew_h + mb
        else:
            self.total_h = mt + ph + mb

        # H.264 macroblock encoding requires strictly even dimensions.
        # We absorb any odd-pixel remainder into the margins.
        if self.total_w % 2 != 0:
            self.total_w += 1
            self.mr += 1
        if self.total_h % 2 != 0:
            self.total_h += 1
            self.mb += 1

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
    W = float(np.asarray(getattr(frame, "box_width", cfg.env.box_width)).flatten()[0])
    H = float(np.asarray(getattr(frame, "box_height", cfg.env.box_height)).flatten()[0])
    N = int(cfg.env.num_agents)
    vis_r = float(cfg.env.visual_radius); comm_r = float(cfg.env.comm_radius)
    # Base station uses its own comm radius for first-hop connectivity
    comm_r_base = float(cfg.env.get("comm_radius_base", cfg.env.comm_radius))
    cell_size = 1.0
    B = int(cfg.env.radar_bins); v_cfg = cfg.visualize
    rew_cfg = cfg.reward
    use_shortest_path_visuals = (
        bool(rew_cfg.get("only_shortest_path_chain_reward", False))
        and not bool(rew_cfg.get("only_explor_individual", False))
        and not bool(rew_cfg.get("every_reward_global", False))
    )

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
    comm_occ_grid = getattr(frame, "comm_occ_grid", occ_grid)
    if occ_grid is not None:
        wall_img = np.zeros((occ_grid.shape[1], occ_grid.shape[0]), dtype=np.uint8)
        # grid is (W, H), opencv wants (H, W)
        wall_img[occ_grid.T[::-1]] = 50 # Dark grey walls
        wall_full = cv2.resize(wall_img, (lay.pw, lay.ph), interpolation=cv2.INTER_NEAREST)
        mask = wall_full > 0
        roi = img[lay.mt:lay.mt+lay.ph, lay.ml:lay.ml+lay.pw]
        roi[mask] = 50 # Solid walls

    mesh_walls = getattr(frame, "mesh_walls", None)
    if mesh_walls is not None:
        mesh_width = max(2, int(round(lay.scale)))
        for x1, y1, x2, y2 in np.asarray(mesh_walls):
            cv2.line(img, lay.w2p(float(x1), float(y1)), lay.w2p(float(x2), float(y2)), (235, 165, 14), mesh_width, cv2.LINE_AA)

    # Axis labels removed per request


    # ── Coverage grid ─────────────────────────────────────────────────────
    if frame.coverage_grid.any():
        gh_active = int(H / cell_size)
        gw_active = int(W / cell_size)
        # Use local world dimensions to mask the active region
        GW_active = int(W // cell_size)
        GH_active = int(H // cell_size)

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

    target_points = np.asarray(frame.target_pos)
    if target_points.ndim == 1:
        target_points = target_points[None, :]

    ents_comp = []
    base_idx = -1; target_idx = -1; target_indices = []
    if num_bases > 0:
        base_idx = len(ents_comp)
        ents_comp.append(frame.base_pos)
    if num_targets > 0:
        for tp in target_points:
            if target_idx < 0:
                target_idx = len(ents_comp)
            target_indices.append(len(ents_comp))
            ents_comp.append(tp)

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
            # First hop logic:
            #   - Drone-to-base first hop uses comm_r_base
            #   - Drone-to-target first hop uses vis_r
            #   - Drone-to-drone edges use comm_r
            if i == base_idx or j == base_idx:
                threshold = comm_r_base
            elif i in target_indices or j in target_indices:
                threshold = vis_r
            else:
                threshold = comm_r

            if dists[i, j] <= threshold:
                if comm_occ_grid is not None:
                    # Communication raycast: mesh walls are transparent only for comm.
                    p1_grid = ents[i] / cell_size
                    p2_grid = ents[j] / cell_size
                    if _dda_raycast_np(p1_grid, p2_grid, comm_occ_grid):
                        adj[i, j] = adj[j, i] = True
                else:
                    adj[i, j] = adj[j, i] = True

    base_comp = _bfs(adj, base_idx) if base_idx >= 0 else set()
    target_comp = set()
    for ti in target_indices:
        target_comp |= _bfs(adj, ti)

    # Calculate shortest path distances only when the reward mode actually uses
    # shortest-path chain gating; otherwise render component links uniformly.
    if use_shortest_path_visuals:
        dist_from_base = _get_shortest_path_distances(adj, base_idx) if base_idx >= 0 else np.full(M, 999, dtype=np.int32)
        dist_from_target = _get_shortest_path_distances(adj, target_idx) if target_idx >= 0 else np.full(M, 999, dtype=np.int32)
        full_chain = bool(base_idx >= 0 and target_idx >= 0 and dist_from_base[target_idx] < 999)
    else:
        dist_from_base = np.full(M, 999, dtype=np.int32)
        dist_from_target = np.full(M, 999, dtype=np.int32)
        full_chain = bool(base_idx >= 0 and target_idx >= 0 and target_idx in base_comp)
    sp_nodes = set()
    if use_shortest_path_visuals and full_chain:
        for i in range(M):
            if dist_from_base[i] + dist_from_target[i] == dist_from_base[target_idx]:
                sp_nodes.add(i)

    drone_cols = []
    for i in range(N):
        idx = i + drone_start
        ib = idx in base_comp
        it = idx in target_comp
        if use_shortest_path_visuals and full_chain:
            if idx in sp_nodes: col = _C["both_chain"]
            elif dist_from_base[idx] <= dist_from_target[idx]: col = _C["base_chain"]
            else: col = _C["tgt_chain"]
        else:
            if ib and it: col = _C["both_chain"]
            elif ib:      col = _C["base_chain"]
            elif it:      col = _C["tgt_chain"]
            else:         col = _C["iso"]
        drone_cols.append(col)

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
    # Identify tips (drones closest to hubs) for shortest path calc
    idx_base_tip = -1
    idx_target_tip = -1
    if use_shortest_path_visuals and base_idx >= 0 and target_idx >= 0 and not full_chain:
        d_to_t = dists[drone_start:, target_idx]
        valid_b = [i for i in range(N) if (i + drone_start) in base_comp]
        if valid_b:
            idx_base_tip = valid_b[np.argmin(d_to_t[valid_b])] + drone_start

        d_to_b = dists[drone_start:, base_idx]
        valid_t = [i for i in range(N) if (i + drone_start) in target_comp]
        if valid_t:
            idx_target_tip = valid_t[np.argmin(d_to_b[valid_t])] + drone_start

    dist_from_base_tip = _get_shortest_path_distances(adj, idx_base_tip) if use_shortest_path_visuals and idx_base_tip >= 0 else np.full(M, 999, dtype=np.int32)
    dist_from_target_tip = _get_shortest_path_distances(adj, idx_target_tip) if use_shortest_path_visuals and idx_target_tip >= 0 else np.full(M, 999, dtype=np.int32)

    for i in range(M):
        for j in range(i + 1, M):
            if not adj[i, j]: continue
            ib = i in base_comp; jb = j in base_comp; it = i in target_comp; jt = j in target_comp

            is_both = False
            on_base_sp = False
            on_tgt_sp = False

            if use_shortest_path_visuals and full_chain:
                # On full shortest path
                if dist_from_base[i] + 1 + dist_from_target[j] == dist_from_base[target_idx] or dist_from_base[j] + 1 + dist_from_target[i] == dist_from_base[target_idx]:
                    is_both = True

                if is_both: ec, lw = _C["both_chain"], 2
                else:
                    if min(dist_from_base[i], dist_from_base[j]) <= min(dist_from_target[i], dist_from_target[j]):
                        ec, lw = _C["base_chain"], 1
                    else:
                        ec, lw = _C["tgt_chain"], 1
            else:
                is_both = ib and jb and it and jt
                if is_both: ec, lw = _C["both_chain"], 2
                elif ib and jb: ec, lw = _C["base_chain"], 1
                elif it and jt: ec, lw = _C["tgt_chain"], 1
                else: ec, lw = _C["link_grey"], 1

                if ib and jb and idx_base_tip >= 0:
                    if dist_from_base[i] + 1 + dist_from_base_tip[j] == dist_from_base[idx_base_tip] or dist_from_base[j] + 1 + dist_from_base_tip[i] == dist_from_base[idx_base_tip]:
                        on_base_sp = True

                if it and jt and idx_target_tip >= 0:
                    if dist_from_target[i] + 1 + dist_from_target_tip[j] == dist_from_target[idx_target_tip] or dist_from_target[j] + 1 + dist_from_target_tip[i] == dist_from_target[idx_target_tip]:
                        on_tgt_sp = True

            # Draw the subtle "glow" for shortest paths
            if use_shortest_path_visuals and (on_base_sp or on_tgt_sp or is_both):
                glow_col = _C["both_chain"] if is_both else (_C["base_chain"] if on_base_sp else _C["tgt_chain"])
                # Draw a thicker, semi-transparent line behind
                overlay = img.copy()
                cv2.line(overlay, lay.w2p(ents[i, 0], ents[i, 1]), lay.w2p(ents[j, 0], ents[j, 1]), glow_col, 5, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

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

    scale = RendererConfig.RENDER_DPI / 100.0

    # ── Base station ──────────────────────────────────────────────────────
    if int(cfg.env.num_bases) > 0:
        bpx, bpy = lay.w2p(frame.base_pos[0], frame.base_pos[1])
        bs = int(RendererConfig.BASE_MARKER_SIZE * scale)
        # Draw base comm radius circle (same style as drone comm circles)
        comm_r_base_px = lay.w2r(comm_r_base)
        comm_col_cfg = _cfg_bgr(str(v_cfg.comm_color), None)
        overlay = img.copy()
        base_col = comm_col_cfg if comm_col_cfg else _C["base_mkr"]
        cv2.circle(overlay, (bpx, bpy), comm_r_base_px, base_col, -1)
        cv2.addWeighted(overlay, float(v_cfg.comm_fill_alpha), img, 1.0 - float(v_cfg.comm_fill_alpha), 0, img)
        cv2.circle(img, (bpx, bpy), comm_r_base_px, base_col, 1, cv2.LINE_AA)
        # Draw base station marker on top
        cv2.rectangle(img, (bpx - bs, bpy - bs), (bpx + bs, bpy + bs), _C["base_mkr"], -1)
        cv2.rectangle(img, (bpx - bs, bpy - bs), (bpx + bs, bpy + bs), (17, 24, 100), 1)
        _draw_text(img, "B", (bpx, bpy + int(2 * scale)), scale * RendererConfig.CV2_FONT_SCALE_LABELS, _C["white"], center=True)

    # ── Target ────────────────────────────────────────────────────────────
    if int(cfg.env.num_targets) > 0:
        tm = int(RendererConfig.TARGET_MARKER_SIZE * scale)
        for k, tp in enumerate(target_points):
            tpx, tpy = lay.w2p(tp[0], tp[1])
            cv2.drawMarker(img, (tpx, tpy), _C["tgt_mkr"], cv2.MARKER_STAR, tm * 2, 2, cv2.LINE_AA)
            label = "T" if len(target_points) == 1 else f"T{k}"
            _draw_text(img, label, (tpx + tm + int(4 * scale), tpy - int(2 * scale)), scale * RendererConfig.CV2_FONT_SCALE_LABELS, _C["tgt_mkr"])

    # MEM_T8-only diagnostic markers for wrong-branch decoys.
    if frame.anti_target_pos is not None:
        am = int(RendererConfig.TARGET_MARKER_SIZE * scale)
        anti_points = np.asarray(frame.anti_target_pos)
        if anti_points.ndim == 1:
            anti_points = anti_points[None, :]
        for k, ap in enumerate(anti_points):
            apx, apy = lay.w2p(ap[0], ap[1])
            cv2.drawMarker(img, (apx, apy), _C["anti_mkr"], cv2.MARKER_TILTED_CROSS, am * 2, 2, cv2.LINE_AA)
            label = "A" if len(anti_points) == 1 else f"A{k}"
            _draw_text(img, label, (apx + am + int(4 * scale), apy - int(2 * scale)), scale * RendererConfig.CV2_FONT_SCALE_LABELS, _C["anti_mkr"])

    # ── Drones ────────────────────────────────────────────────────────────
    dr = int(RendererConfig.DRONE_MARKER_SIZE * scale)
    for i in range(N):
        is_act = bool(frame.active[i]) if frame.active is not None else True
        is_coll = bool(frame.collides[i]) if frame.collides is not None else False
        alpha = 1.0 if is_act else 0.3
        cx, cy = lay.w2p(frame.pos[i, 0], frame.pos[i, 1])
        col = _C["tgt_mkr"] if is_coll else drone_cols[i]  # Red if colliding

        # Collision glow
        if is_coll:
            glow_r = int(RendererConfig.COLLISION_GLOW_RADIUS * scale)
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), glow_r, _C["tgt_mkr"], -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        if alpha < 1.0:
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), dr, col, -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), dr, _C["white"], 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        else:
            cv2.circle(img, (cx, cy), dr, col, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), dr, _C["white"], 1, cv2.LINE_AA)
        _draw_text(img, str(i), (cx + dr + int(4 * scale), cy - dr - int(2 * scale)), scale * RendererConfig.CV2_FONT_SCALE_LABELS, col)

    # ── Title ─────────────────────────────────────────────────────────────
    GW_active = int(W // cell_size)
    GH_active = int(H // cell_size)

    # Coverage calculation on the active region
    cov_cnt = frame.coverage_grid[:GW_active, :GH_active].sum()
    cov_pct = 100.0 * cov_cnt / (GW_active * GH_active)

    # Logic for successful chain: intersection of base and target sets (excluding themselves)
    full_chain = bool(base_idx >= 0 and target_idx >= 0 and (base_comp & target_comp - {base_idx, target_idx}))
    status = " | CHAIN FORMED!" if full_chain else ""

    metrics_str = ""
    if frame.extra_metrics:
      em = frame.extra_metrics
      r_explor = em.get("r_explor", 0.0)
      r_gap    = em.get("r_gap", 0.0)
      r_total  = em.get("r_total", 0.0)
      chain_pct = em.get("chain_pct", 0.0)

      stats = [
          f"Step {frame.step:04d}",
          f"Coverage {cov_pct:4.1f}%",
          f"r_explor {r_explor:4.1f}",
          f"r_gap {r_gap:4.1f}",
          f"r_total {r_total:4.1f}",
          f"Chain {chain_pct:4.1f}%",
      ]
      title = " | ".join(stats) + "   " + status
    else:
      title = f"Step {frame.step:04d} | Coverage {cov_pct:.1f}% | {status}"

    _draw_text(img, title, (lay.ml, lay.mt - int(25 * scale)), scale * RendererConfig.CV2_FONT_SCALE_TITLE, _C["text"])

    # ── Legend ────────────────────────────────────────────────────────────
    lx = lay.ml + lay.pw + int(20 * scale)
    ly = lay.mt + int(20 * scale)
    _draw_text(img, "Legend", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND * 1.2, _C["text"])
    legend_items = []
    if num_bases > 0:   legend_items.append((_C["base_mkr"],   "Base station"))
    if num_targets > 0: legend_items.append((_C["tgt_mkr"],    "Target"))
    # MEM_T8-only diagnostic legend entry.
    if frame.anti_target_pos is not None:
        legend_items.append((_C["anti_mkr"], "Anti-target"))

    if num_bases > 0:   legend_items.append((_C["base_chain"], "Base-connected"))
    if num_targets > 0: legend_items.append((_C["tgt_chain"],  "Target-connected"))
    if num_bases > 0 and num_targets > 0:
        legend_items.append((_C["both_chain"], "Both (bridge)"))

    legend_items.append((_C["iso"],        "Isolated"))
    legend_items.append((_C["coverage"],   "Coverage"))

    ly += int(25 * scale)
    bs = int(10 * scale)
    for col, label in legend_items:
        cv2.rectangle(img, (lx, ly - bs//2), (lx + bs, ly + bs//2), col, -1)
        _draw_text(img, label, (lx + bs + int(10 * scale), ly + bs//4), scale * RendererConfig.CV2_FONT_SCALE_LEGEND, _C["text"])
        ly += int(25 * scale)

    # ── Target Known Info Text ────────────────────────────────────────────
    ly += int(30 * scale)
    _draw_text(img, "Target known to:", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND * 1.1, _C["text"])

    known_indices = []
    if frame.target_known is not None:
        known_indices = [i for i in range(N) if bool(frame.target_known[i])]
        known_indices.sort()

    ly += int(20 * scale)
    has_any = False
    if getattr(frame, "base_target_known", False):
        _draw_text(img, "- Base station", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND, _C["text"])
        ly += int(20 * scale)
        has_any = True

    if known_indices:
        for idx in known_indices:
            _draw_text(img, f"- Drone {idx}", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND, _C["text"])
            ly += int(20 * scale)
        has_any = True

    if not has_any:
        _draw_text(img, "  (none)", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND, _C["text_grey"])
        ly += int(20 * scale)

    # ── Connections matrix ───────────────────────────────────────────────
    render_conn = bool(cfg.visualize.get("render_conn_matrix", True))
    if render_conn:
        ly += int(15 * scale)
        _draw_text(img, "Connections (0..N-1, B):", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND * 1.05, _C["text"])
        ly += int(20 * scale)

        if frame.adj_matrix is not None and frame.adj_matrix.size > 0:
            adj = frame.adj_matrix
            # Header
            header = "   " + " ".join(str(i) for i in range(N)) + " B"
            _draw_monospace_text(img, header, (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND * 0.8, _C["text"])
            ly += int(18 * scale)

            for i in range(N + 1):
                row_label = f"{i} " if i < N else "B "
                row_vals = []
                for j in range(N + 1):
                    if i == j:
                        row_vals.append(".")
                    else:
                        row_vals.append("1" if bool(adj[i, j]) else "0")
                row_str = f"{row_label} " + " ".join(row_vals)
                _draw_monospace_text(img, row_str, (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND * 0.8, _C["text"])
                ly += int(16 * scale)
        else:
            _draw_text(img, "  (not available)", (lx, ly), scale * RendererConfig.CV2_FONT_SCALE_LEGEND, _C["text_grey"])
            ly += int(20 * scale)

    # ── Reward plot ───────────────────────────────────────────────────────
    if lay.has_reward and rewards_so_far is not None:
        # Slice rewards to current step to show progression
        curr_rewards = rewards_so_far[:frame.step + 1]

        if curr_rewards.ndim == 2:
            team_rewards = curr_rewards.sum(axis=1)
            indiv_rewards = curr_rewards
        else:
            team_rewards = curr_rewards
            indiv_rewards = None

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

        if len(team_rewards) > 0:
            cum = np.cumsum(team_rewards)
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

            if len(team_rewards) > 1:
                xs = np.linspace(rl, rr, len(cum)).astype(np.int32)
                ys = (rb - (cum - cmin) / (cmax - cmin) * rh).astype(np.int32)
                pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, _C["reward"], 2, cv2.LINE_AA)

        # ── Individual Reward Plot ────────────────────────────────────────────
        if lay.has_indiv_reward and indiv_rewards is not None:
            irt = lay.indiv_rew_top; irh = lay.indiv_rew_h
            irl = lay.ml; irr = lay.ml + lay.pw; irb = irt + irh

            cv2.rectangle(img, (irl, irt), (irr, irb), _C["reward_bg"], -1)
            cv2.rectangle(img, (irl, irt), (irr, irb), _C["border"], 1)

            for i in range(1, 4):
                yy = irt + i * irh // 4
                cv2.line(img, (irl, yy), (irr, yy), _C["grid"], 1)
            for i in range(1, 5):
                xx = irl + i * (irr - irl) // 5
                cv2.line(img, (xx, irt), (xx, irb), _C["grid"], 1)

            if len(indiv_rewards) > 0:
                # To highlight differences, we subtract the step-wise minimum reward
                # This removes the large shared penalty and shows who is "pulling ahead"
                rel_indiv = indiv_rewards - indiv_rewards.min(axis=1, keepdims=True)
                cum_indiv = np.cumsum(rel_indiv, axis=0) # (t, N)

                cmin = float(cum_indiv.min()); cmax = float(cum_indiv.max())
                if abs(cmax - cmin) < 1e-4: cmax += 1.0

                _draw_text(img, f"Relative Return", (irl - 50, irt + irh//2), lay.fs * 0.7, _C["text_grey"])
                _draw_text(img, f"Individual Advantage", (irl, irt - 12), lay.fs * 0.8, _C["text"])
                if abs(cmax - cmin) < 1e-4: cmax += 1.0

                _draw_text(img, f"{cmax:.0f}", (irl - 8, irt + 5), lay.fs * 0.6, _C["text_grey"], center=False)
                _draw_text(img, f"{cmin:.0f}", (irl - 8, irb - 5), lay.fs * 0.6, _C["text_grey"], center=False)

                for i in range(6):
                    tx = irl + i * (irr - irl) // 5
                    step_val = i * total_steps // 5
                    cv2.line(img, (tx, irb), (tx, irb + 5), _C["border"], 1)
                    _draw_text(img, str(step_val), (tx, irb + 18), lay.fs * 0.6, _C["text_grey"], center=True)

                if len(indiv_rewards) > 1:
                    xs = np.linspace(irl, irr, len(cum_indiv)).astype(np.int32)
                    for n in range(indiv_rewards.shape[1]):
                        ys = (irb - (cum_indiv[:, n] - cmin) / (cmax - cmin) * irh).astype(np.int32)
                        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                        col = _TAB10[n % len(_TAB10)]
                        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)

            # Draw tiny legend for individual drones to the right of the individual plot
            lg_x = irr + 20
            lg_y = irt
            _draw_text(img, "Drones", (lg_x, lg_y - 12), lay.fs * 0.8, _C["text"])
            for n in range(N):
                col = _TAB10[n % len(_TAB10)]
                cv2.rectangle(img, (lg_x, lg_y), (lg_x + 12, lg_y + 12), col, -1)
                _draw_text(img, f"D{n}", (lg_x + 20, lg_y + 10), lay.fs * 0.6, _C["text"])
                lg_y += 18

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
    extra_metrics: dict[str, np.ndarray] | None = None,
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
        frame_stride = RendererConfig.FRAME_STRIDE

    T = traj_cpu.pos.shape[0]
    idxs = list(range(0, T, frame_stride))

    # Extract dynamic world dimensions from the first frame
    W = float(traj_cpu.box_width[0]) if hasattr(traj_cpu, "box_width") else float(cfg.env.box_width)
    H = float(traj_cpu.box_height[0]) if hasattr(traj_cpu, "box_height") else float(cfg.env.box_height)

    has_indiv_reward = (rewards_cpu is not None and getattr(rewards_cpu, "ndim", 0) == 2)
    lay = _Layout(W, H, rewards_cpu is not None, has_indiv_reward=has_indiv_reward)

    # Load map once for static rendering data (walls)
    from env.maps import MapDefinition
    from core.config import MAP_DIR
    occ_grid_static = None
    comm_occ_grid_static = None
    mesh_walls_static = None
    anti_target_static = None
    if cfg.env.map_names and len(cfg.env.map_names) > 0:
        active_map_name = cfg.env.map_names[0]
        map_path = MAP_DIR / f"{active_map_name}.yaml"
        if map_path.exists():
            map_def = MapDefinition.load(map_path, cell_size=1.0)
            occ_grid_static = map_def.occupancy_grid
            comm_occ_grid_static = map_def.communication_occupancy_grid
            mesh_walls_static = np.array(map_def.mesh_walls, dtype=np.float32) if map_def.mesh_walls else None
            # MEM_T8-only diagnostic marker overlay.
            if map_def.anti_target_spawn_points is not None:
                anti_target_static = np.array(map_def.anti_target_spawn_points)

    # 1. Prepare frame data for parallel processing
    frame_args = []
    for t in idxs:
        frame_args.append(_FrameData(
            pos           = np.array(traj_cpu.pos[t]),
            vel           = np.array(traj_cpu.vel[t]),
            base_pos      = np.array(traj_cpu.base_pos[t]),
            target_pos    = np.array(traj_cpu.target_pos[t]),
            anti_target_pos = anti_target_static,
            coverage_grid = np.array(traj_cpu.coverage_grid[t]),
            step          = int(traj_cpu.step[t]),
            active        = (np.array(traj_cpu.active[t]) if hasattr(traj_cpu, "active") else None),
            collides      = (np.array(traj_cpu.collides[t]) if hasattr(traj_cpu, "collides") else None),
            occ_grid      = occ_grid_static,
            comm_occ_grid = comm_occ_grid_static,
            mesh_walls    = mesh_walls_static,
            box_width     = float(traj_cpu.box_width[t]),
            box_height    = float(traj_cpu.box_height[t]),
            extra_metrics = {k: float(v[t]) for k, v in extra_metrics.items()} if extra_metrics else {},
            target_known  = (np.array(traj_cpu.target_known[t]) if hasattr(traj_cpu, "target_known") else None),
            base_target_known = (bool(traj_cpu.base_target_known[t]) if hasattr(traj_cpu, "base_target_known") else None),
            adj_matrix    = (np.array(traj_cpu.adj_matrix[t]) if hasattr(traj_cpu, "adj_matrix") else None),
        ))

    # 2. Determine parallelism (daemonic processes cannot spawn children)
    is_daemon = multiprocessing.current_process().daemon
    num_vcpus = multiprocessing.cpu_count()
    pool_size = max(1, num_vcpus - 1) if not is_daemon else 0

    # Partial function for the worker
    worker = partial(_draw_frame_cv2, cfg=cfg, lay=lay, total_steps=T, rewards_so_far=rewards_cpu)

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
                    for k, bgr in enumerate(pool.imap(worker, frame_args)):
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        writer.append_data(rgb)
                        last_img = rgb
                        if (k + 1) % 50 == 0 or k == len(idxs) - 1:
                            print(f"  {k+1}/{len(idxs)} frames rendered", flush=True)
            else:
                # Sequential fallback (for daemon workers or single-core)
                for k, arg in enumerate(frame_args):
                    bgr = worker(arg)
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
