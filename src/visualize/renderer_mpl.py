"""
swarmecho/visualize/renderer.py
================================
CPU-side video renderer for SwarmEcho environment states.

Performance design
------------------
* Frame streaming: imageio writer is kept open during the loop — frames are
  written immediately and never accumulated in a Python list. Peak RAM is
  O(1 frame) instead of O(T frames).

* Figure reuse: a single Matplotlib figure + axes pair is created once and
  cleared between frames with ax.clear(). This avoids the expensive Figure
  constructor and Agg canvas allocation on every frame.

* Frame stride: only every `frame_stride`-th step is rendered, reducing
  the frame count without changing the FPS of the output video.

Two quality presets:
    These are now controlled by RENDER_DPI and FRAME_STRIDE constants
    at the top of the file for global easy tweaking.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple

import imageio
import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Force headless backend — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf import DictConfig

from visualize.renderer_config import RendererConfig

# Use a clean, modern font stack
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["text.color"] = "#0f172a"
plt.rcParams["axes.labelcolor"] = "#64748b"

# To avoid 50GB GPU memory spikes in worker processes, we import jax locally
import warnings
import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_DPI = 150
FRAME_STRIDE = 1


# ---------------------------------------------------------------------------
# RGB colour palette
# ---------------------------------------------------------------------------

_C = {
    "white":        "#ffffff",
    "bg_outer":     "#f0f4f8",
    "bg_inner":     "#f8fafc",
    "border":       "#cbd5e1",
    "grid":         "#e2e8f0",
    "text":         "#0f172a",
    "text_grey":    "#64748b",
    "base_chain":   "#3b82f6",   # blue
    "tgt_chain":    "#ef4444",   # red
    "both_chain":   "#a855f7",   # purple
    "iso":          "#6b7280",   # grey
    "base_mkr":     "#1d4ed8",   # dark blue
    "tgt_mkr":      "#dc2626",   # dark red
    "comm_fill":    "#22d3ee",   # cyan
    "vis_fill":     "#fbbf24",   # amber
    "coverage":     "#0ea5e9",   # light blue
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FrameData(NamedTuple):
    pos:           np.ndarray
    vel:           np.ndarray
    base_pos:      np.ndarray
    target_pos:    np.ndarray  # (2,) normally; (N, 2) only for MEM_T8 diagnostics
    anti_target_pos: np.ndarray | None  # MEM_T8-only diagnostic markers
    coverage_grid: np.ndarray
    step:          int
    active:        np.ndarray | None
    collides:      np.ndarray | None
    occ_grid:      np.ndarray | None
    comm_occ_grid: np.ndarray | None
    mesh_walls:    np.ndarray | None
    box_width:     float
    box_height:    float
    extra_metrics: dict[str, any]
    target_known:  np.ndarray | None
    base_target_known: bool | None
    adj_matrix:    np.ndarray | None
    finders_path:  np.ndarray | None
    finders_path_len: int
    maze_cell_grid: tuple[int, int] | None


def _bfs(adj: np.ndarray, source: int) -> set[int]:
    visited: set[int] = {source}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for nb in np.where(adj[node])[0]:
            nb = int(nb)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


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


from env.raycast import dda_raycast_np

def _dda_raycast_np(p1, p2, occ_grid):
    """NumPy version of DDA raycast for the renderer (shared with CV2)."""
    return dda_raycast_np(p1, p2, occ_grid)


def _build_adjacency(pos, base_pos, target_pos, comm_radius, visual_radius, comm_radius_base, occ_grid, world_size, cfg):
    N = int(pos.shape[0])
    num_bases = int(cfg.env.num_bases)
    num_targets = int(cfg.env.num_targets)
    cell_size = 1.0

    # Build filtered entity list
    ents_comp = []
    target_points = np.asarray(target_pos)
    if target_points.ndim == 1:
        target_points = target_points[None, :]
    elif target_points.ndim == 2:
        mask = ~np.all(target_points == 0.0, axis=1)
        target_points = target_points[mask]
        _, indices = np.unique(target_points, axis=0, return_index=True)
        target_points = target_points[np.sort(indices)]

    base_idx = -1; target_idx = -1; target_indices = []
    if num_bases > 0: 
        base_idx = len(ents_comp)
        ents_comp.append(base_pos)
    if num_targets > 0:
        for tp in target_points:
            if target_idx < 0:
                target_idx = len(ents_comp)
            target_indices.append(len(ents_comp))
            ents_comp.append(tp)
    
    drone_start = len(ents_comp)
    for i in range(N):
        ents_comp.append(pos[i])
        
    ents = np.array(ents_comp)
    M = len(ents)
    
    diff  = ents[:, None, :] - ents[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Start with full comm_radius adjacency
    adj = (dists <= comm_radius) & ~np.eye(M, dtype=bool)
    
    # Override base first-hop to use comm_radius_base
    if base_idx >= 0:
        base_mask = dists[base_idx] <= comm_radius_base
        adj[base_idx, :] = base_mask & ~np.eye(M, dtype=bool)[base_idx]
        adj[:, base_idx] = base_mask & ~np.eye(M, dtype=bool)[:, base_idx]
        
    # Target first-hop uses visual_radius
    for ti in target_indices:
        tmask = dists[ti] <= visual_radius
        adj[ti, :] &= tmask
        adj[:, ti] &= tmask

    # Base and Target cannot connect directly
    if base_idx >= 0:
        for ti in target_indices:
            adj[base_idx, ti] = adj[ti, base_idx] = False
    
    # Raycast check for walls
    if occ_grid is not None:
        for i in range(M):
            for j in range(i + 1, M):
                if adj[i, j]:
                    p1_grid = ents[i] / cell_size
                    p2_grid = ents[j] / cell_size
                    if not _dda_raycast_np(p1_grid, p2_grid, occ_grid):
                        adj[i, j] = adj[j, i] = False
    
    return adj, base_idx, target_idx, target_indices, drone_start, ents, dists





# ---------------------------------------------------------------------------
# Figure setup — called ONCE per render_video invocation
# ---------------------------------------------------------------------------

def _make_figure(width_px, height_px, pw, ph, dpi, has_reward, has_indiv_reward=False):
    scale = dpi / 100.0
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#f0f4f8")
    canvas = FigureCanvasAgg(fig)

    ml = int(RendererConfig.MARGIN_LEFT * scale)
    mr = int(RendererConfig.MARGIN_RIGHT * scale)
    mt = int(RendererConfig.MARGIN_TOP * scale)
    mb = int(RendererConfig.MARGIN_BOTTOM * scale)

    # Pre-compute exact fractional axes
    map_bot = (height_px - mt - ph) / height_px
    ax = fig.add_axes([ml / width_px, map_bot, pw / width_px, ph / height_px])

    # Legend Axes (to the right of the map)
    leg_left = (ml + pw + int(20 * scale)) / width_px
    leg_w = (mr - int(40 * scale)) / width_px
    leg_ax = fig.add_axes([leg_left, map_bot, leg_w, ph / height_px])

    rew_ax = None
    indiv_rew_ax = None
    
    gap_map_rew = int(RendererConfig.GAP_MAP_REWARDS * scale)
    gap_between = int(RendererConfig.GAP_BETWEEN_REWARDS * scale)
    
    rew_h = int(RendererConfig.TEAM_REWARD_HEIGHT * scale)
    indiv_rew_h = int(RendererConfig.INDIV_REWARD_HEIGHT * scale)

    rew_top = mt + ph + gap_map_rew
    indiv_rew_top = rew_top + rew_h + gap_between

    if has_reward:
        rew_bot_pos = (height_px - rew_top - rew_h) / height_px
        rew_ax = fig.add_axes([ml / width_px, rew_bot_pos, pw / width_px, rew_h / height_px])
        
        if has_indiv_reward:
            indiv_bot_pos = (height_px - indiv_rew_top - indiv_rew_h) / height_px
            indiv_rew_ax = fig.add_axes([ml / width_px, indiv_bot_pos, pw / width_px, indiv_rew_h / height_px])

    return fig, canvas, ax, leg_ax, rew_ax, indiv_rew_ax


# ---------------------------------------------------------------------------
# Per-frame draw (reuses existing figure — axes are cleared, not recreated)
# ---------------------------------------------------------------------------
_THETA = np.linspace(0, 2 * np.pi, 96)


def _draw_frame(
    fig, ax, leg_ax, rew_ax, indiv_rew_ax,
    frame:          _FrameData,
    cfg:            DictConfig,
    rewards_so_far: np.ndarray | None,
    total_steps:    int,
) -> None:
    """Clear axes and redraw — no new Figure objects allocated."""

    # Explicitly pull dimensions or fallback to config to prevent AttributeErrors
    W = float(np.asarray(getattr(frame, "box_width", cfg.env.box_width)).flatten()[0])
    H = float(np.asarray(getattr(frame, "box_height", cfg.env.box_height)).flatten()[0])
    N = int(cfg.env.num_agents)
    vis_r  = float(cfg.env.visual_radius)
    comm_r = float(cfg.env.comm_radius)
    comm_r_base = float(cfg.env.get("comm_radius_base", cfg.env.comm_radius))
    B      = int(cfg.env.radar_bins)
    v_cfg  = cfg.visualize
    rew_cfg = cfg.reward
    use_shortest_path_visuals = (
        bool(rew_cfg.get("only_shortest_path_chain_reward", False))
        and str(rew_cfg.get("chain_reward_system", "euclidean")) == "euclidean"
        and not bool(rew_cfg.get("only_explor_individual", False))
        and not bool(rew_cfg.get("every_reward_global", False))
    )

    # Clear only the dynamic axes
    ax.clear()
    leg_ax.clear()
    if rew_ax is not None:
        rew_ax.clear()
    if indiv_rew_ax is not None:
        indiv_rew_ax.clear()

    # --- Axes base style ---
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(-2, H + 2)
    # Force 1:1 coordinate scaling and a box shape that matches the world aspect
    ax.set_aspect("equal", adjustable="datalim")
    try:
        ax.set_box_aspect(H / W)
    except AttributeError:
        pass # Older matplotlib
    ax.tick_params(labelsize=7, colors="#64748b")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cbd5e1")
    ax.set_xlabel("x [m]", fontsize=7, color="#64748b")
    ax.set_ylabel("y [m]", fontsize=7, color="#64748b")

    # --- World box ---
    ax.add_patch(mpatches.Rectangle(
        (0, 0), W, H, linewidth=2,
        edgecolor="#334155", facecolor="none", zorder=10,
    ))

    # ── Walls ─────────────────────────────────────────────────────────────
    if frame.occ_grid is not None:
        occ = frame.occ_grid
        # Transpose/flip correctly for matplotlib's origin bottom
        ax.imshow(occ.T, extent=(0, W, 0, H), origin="lower", 
                  cmap="Greys", alpha=0.5, interpolation="nearest", zorder=1)

    if frame.mesh_walls is not None:
        for x1, y1, x2, y2 in np.asarray(frame.mesh_walls):
            ax.plot([x1, x2], [y1, y2], color="#0ea5e9", lw=2.0, solid_capstyle="butt", zorder=2)

    # ── Coverage ──────────────────────────────────────────────────────────
    if frame.coverage_grid.any():
        cell_size = 1.0
        gh_active = int(H / cell_size)
        gw_active = int(W / cell_size)

        # Slice the grid to only the active part
        active_cov = frame.coverage_grid[:gw_active, :gh_active]

        # grid is (GW, GH), matplotlib expects (GH, GW) for imshow with origin='lower'
        # We need to map World X to Matrix Column, World Y to Matrix Row.
        # But imshow(A) maps A[row, col] -> y, x.
        # So A[: , :] = coverage_grid.T  -> A[y, x] = coverage_grid[x, y]
        cov_img_bool = active_cov.T
        gh_px, gw_px = cov_img_bool.shape
        cov = np.zeros((gh_px, gw_px, 4), dtype=np.float32)
        # Using [R, G, B, A] for Teal-ish coverage marker
        cov[cov_img_bool] = [0.04, 0.78, 0.73, 0.25]
        ax.imshow(cov, extent=[0, W, 0, H],
                  origin="lower", aspect="auto", zorder=1, interpolation="nearest")

    if (
        str(cfg.reward.get("chain_reward_system", "euclidean")) == "discrete_finders_path"
        and frame.finders_path is not None
        and frame.finders_path_len > 1
        and frame.maze_cell_grid is not None
    ):
        cols, rows = frame.maze_cell_grid
        cell_w = W / cols
        cell_h = H / rows
        pts = np.array([
            [(c[0] + 0.5) * cell_w, (c[1] + 0.5) * cell_h]
            for c in frame.finders_path[:frame.finders_path_len]
        ], dtype=np.float32)
        ax.plot(pts[:, 0], pts[:, 1], color="#22c55e", alpha=0.5, lw=3.0, zorder=2)

    # ── Adjacency ─────────────────────────────────────────────────────────
    adj, base_idx, target_idx, target_indices, drone_start, ents, dists = _build_adjacency(
        frame.pos, frame.base_pos, frame.target_pos, 
        comm_r, vis_r, comm_r_base, frame.comm_occ_grid if frame.comm_occ_grid is not None else frame.occ_grid, (W, H), cfg
    )
    M = len(ents)
    
    base_comp   = _bfs(adj, source=base_idx) if base_idx >= 0 else set()
    target_comp = set()
    for ti in target_indices:
        target_comp |= _bfs(adj, source=ti)

    # --- Shortest Path Highlighting ---
    # Only use shortest-path highlighting when the active reward mode uses the
    # shortest-path chain gate; otherwise draw component links uniformly.
    if use_shortest_path_visuals:
        dist_from_base = _get_shortest_path_distances(adj, base_idx) if base_idx >= 0 else np.full(M, 999, dtype=np.int32)
        dist_from_target = _get_shortest_path_distances(adj, target_idx) if target_idx >= 0 else np.full(M, 999, dtype=np.int32)
        full_chain = bool(base_idx >= 0 and target_idx >= 0 and dist_from_base[target_idx] < 999)
    else:
        dist_from_base = np.full(M, 999, dtype=np.int32)
        dist_from_target = np.full(M, 999, dtype=np.int32)
        full_chain = bool(base_idx >= 0 and any(ti in base_comp for ti in target_indices))
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
            if idx in sp_nodes: col = "#a855f7"
            elif dist_from_base[idx] <= dist_from_target[idx]: col = "#3b82f6"
            else: col = "#ef4444"
        else:
            if ib and it: col = "#a855f7"
            elif ib:      col = "#3b82f6"
            elif it:      col = "#ef4444"
            else:         col = "#6b7280"
        drone_cols.append(col)

    # Identify tips
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
            if adj[i, j]:
                xi, yi = ents[i]; xj, yj = ents[j]
                ib = i in base_comp;  jb = j in base_comp
                it = i in target_comp; jt = j in target_comp
                
                is_both = False
                on_base_sp = False
                on_tgt_sp = False
                
                if use_shortest_path_visuals and full_chain:
                    if dist_from_base[i] + 1 + dist_from_target[j] == dist_from_base[target_idx] or dist_from_base[j] + 1 + dist_from_target[i] == dist_from_base[target_idx]:
                        is_both = True
                        
                    if is_both: ec, lw = "#a855f7", 1.5
                    else:
                        if min(dist_from_base[i], dist_from_base[j]) <= min(dist_from_target[i], dist_from_target[j]):
                            ec, lw = "#3b82f6", 1.2
                        else:
                            ec, lw = "#ef4444", 1.2
                else:
                    is_both = ib and jb and it and jt
                    if is_both: ec, lw = "#a855f7", 1.5
                    elif ib and jb:              ec, lw = "#3b82f6", 1.2
                    elif it and jt:              ec, lw = "#ef4444", 1.2
                    else:                        ec, lw = "#94a3b8", 0.8
                    
                    if ib and jb and idx_base_tip >= 0:
                        if dist_from_base[i] + 1 + dist_from_base_tip[j] == dist_from_base[idx_base_tip] or dist_from_base[j] + 1 + dist_from_base_tip[i] == dist_from_base[idx_base_tip]:
                            on_base_sp = True
                    
                    if it and jt and idx_target_tip >= 0:
                        if dist_from_target[i] + 1 + dist_from_target_tip[j] == dist_from_target[idx_target_tip] or dist_from_target[j] + 1 + dist_from_target_tip[i] == dist_from_target[idx_target_tip]:
                            on_tgt_sp = True

                if use_shortest_path_visuals and (on_base_sp or on_tgt_sp or is_both):
                    glow_col = "#a855f7" if is_both else ("#3b82f6" if on_base_sp else "#ef4444")
                    # Draw a slightly thicker, semi-transparent line behind for the shine
                    ax.plot([xi, xj], [yi, yj], "-", color=glow_col, lw=lw*3, alpha=0.15, zorder=2)

                ax.plot([xi, xj], [yi, yj], "--", color=ec, lw=lw,
                        alpha=0.7, zorder=3)


    # --- Drone radii + radar lines ---
    radar_angles = [b * 2.0 * np.pi / B - np.pi for b in range(B)]
    for i in range(N):
        x, y     = frame.pos[i]
        col      = drone_cols[i]
        is_act   = frame.active[i] if frame.active is not None else True
        am       = 1.0 if is_act else 0.3
        cr_col   = col if v_cfg.comm_color == "match_drone" else v_cfg.comm_color
        vr_col   = col if v_cfg.vis_color  == "match_drone" else v_cfg.vis_color

        ax.fill(x + comm_r*np.cos(_THETA),
                y + comm_r*np.sin(_THETA),
                color=cr_col, alpha=float(v_cfg.comm_fill_alpha)*am, zorder=3)
        ax.plot(x + comm_r*np.cos(_THETA), y + comm_r*np.sin(_THETA),
                color=cr_col, alpha=float(v_cfg.comm_edge_alpha)*am, lw=1.0,
                linestyle="--", zorder=3)

        ax.fill(x + vis_r*np.cos(_THETA),
                y + vis_r*np.sin(_THETA),
                color=vr_col, alpha=float(v_cfg.vis_fill_alpha)*am, zorder=4)
        ax.plot(x + vis_r*np.cos(_THETA), y + vis_r*np.sin(_THETA),
                color=vr_col, alpha=float(v_cfg.vis_edge_alpha)*am, lw=1.0,
                linestyle=":", zorder=4)

        if is_act:
            for ang in radar_angles:
                ax.plot([x, x + vis_r*np.cos(ang)], [y, y + vis_r*np.sin(ang)],
                        color=vr_col, alpha=float(v_cfg.vis_edge_alpha)*0.5,
                        lw=0.6, linestyle=":", zorder=4)

    # --- Base station ---
    if int(cfg.env.num_bases) > 0:
        bx, by = frame.base_pos
        # Draw base comm radius circle (same visual style as drone circles)
        cr_col = v_cfg.comm_color if v_cfg.comm_color != "match_drone" else "#1d4ed8"
        ax.fill(bx + comm_r_base*np.cos(_THETA),
                by + comm_r_base*np.sin(_THETA),
                color=cr_col, alpha=float(v_cfg.comm_fill_alpha), zorder=3)
        ax.plot(bx + comm_r_base*np.cos(_THETA), by + comm_r_base*np.sin(_THETA),
                color=cr_col, alpha=float(v_cfg.comm_edge_alpha), lw=1.0,
                linestyle="--", zorder=3)
        ax.scatter(bx, by, s=RendererConfig.MPL_BASE_S, marker="s",
                   facecolor="#1d4ed8", edgecolor="#1e3a8a", lw=1.5, zorder=5)
        ax.text(bx, by, "B", color="white", fontsize=RendererConfig.MPL_FONT_SIZE_LABELS, fontweight="bold",
                ha="center", va="center", zorder=6)

    # --- Target ---
    if int(cfg.env.num_targets) > 0:
        target_points = np.asarray(frame.target_pos)
        if target_points.ndim == 1:
            target_points = target_points[None, :]
        elif target_points.ndim == 2:
            mask = ~np.all(target_points == 0.0, axis=1)
            target_points = target_points[mask]
            _, indices = np.unique(target_points, axis=0, return_index=True)
            target_points = target_points[np.sort(indices)]
        for k, tp in enumerate(target_points):
            tx, ty = tp
            ax.scatter(tx, ty, s=RendererConfig.MPL_TARGET_S, marker="*", color="#dc2626",
                       edgecolors="#7f1d1d", linewidths=0.8, zorder=5)
            label = "T" if len(target_points) == 1 else f"T{k}"
            ax.annotate(label, xy=(tx, ty), xytext=(6, 6), textcoords="offset points",
                        color="#dc2626", fontsize=RendererConfig.MPL_FONT_SIZE_LABELS, fontweight="bold",
                        ha="left", va="bottom", zorder=6)

    # MEM_T8-only diagnostic markers for wrong-branch decoys.
    if frame.anti_target_pos is not None:
        anti_points = np.asarray(frame.anti_target_pos)
        if anti_points.ndim == 1:
            anti_points = anti_points[None, :]
        elif anti_points.ndim == 2:
            mask = ~np.all(anti_points == 0.0, axis=1)
            anti_points = anti_points[mask]
            _, indices = np.unique(anti_points, axis=0, return_index=True)
            anti_points = anti_points[np.sort(indices)]
        for k, ap in enumerate(anti_points):
            anti_x, anti_y = ap
            ax.scatter(anti_x, anti_y, s=RendererConfig.MPL_TARGET_S * 0.75, marker="X",
                       color="#581c87", edgecolors="#2e1065", linewidths=0.8, zorder=5)
            label = "A" if len(anti_points) == 1 else f"A{k}"
            ax.annotate(label, xy=(anti_x, anti_y), xytext=(6, -8), textcoords="offset points",
                        color="#581c87", fontsize=RendererConfig.MPL_FONT_SIZE_LABELS, fontweight="bold",
                        ha="left", va="top", zorder=6)

    # --- Drones ---
    for i in range(N):
        x, y   = frame.pos[i]
        is_act = frame.active[i] if frame.active is not None else True
        is_coll = frame.collides[i] if frame.collides is not None else False
        alpha  = 1.0 if is_act else 0.3
        col = _C["tgt_mkr"] if is_coll else drone_cols[i]
        
        if is_coll:
            ax.scatter(x, y, s=RendererConfig.MPL_GLOW_S, color=_C["tgt_mkr"], alpha=0.4, zorder=6, lw=0)
            
        ax.scatter(x, y, s=RendererConfig.MPL_DRONE_S, color=col,
                   edgecolors="white", linewidths=0.8, zorder=7, alpha=alpha)
        ax.annotate(str(i), xy=(x, y), xytext=(5, 5), textcoords="offset points",
                    color=col, fontsize=RendererConfig.MPL_FONT_SIZE_LABELS, zorder=8, alpha=alpha)

    # --- Title ---
    cell_size = 1.0
    GW_active = int(W // cell_size)
    GH_active = int(H // cell_size)
    cov_cnt = frame.coverage_grid[:GW_active, :GH_active].sum()
    cov_pct = 100.0 * cov_cnt / (GW_active * GH_active)

    status  = "✓ CHAIN FORMED" if full_chain else ""
    
    em = frame.extra_metrics
    r_explor = em.get("r_explor", 0.0)
    r_gap    = em.get("r_gap", 0.0)
    chain_pct = em.get("chain_pct", 0.0)
    r_total  = em.get("r_total", 0.0)

    title_str = (
        f"Step {frame.step:04d}  |  "
        f"Coverage {cov_pct:4.1f}%  |  "
        f"r_explor {r_explor:4.1f}  |  "
        f"r_gap {r_gap:4.1f}  |  "
        f"r_total {r_total:4.1f}  |  "
        f"Chain {chain_pct:4.1f}%   {status}"
    )

    ax.set_title(title_str, fontsize=RendererConfig.MPL_FONT_SIZE_TITLE, pad=12, color="#0f172a", loc='left')

    # --- Legend ---
    leg_ax.axis("off")
    leg_ax.set_title("Legend", fontsize=RendererConfig.MPL_FONT_SIZE_LEGEND_TITLE, color="#0f172a", loc="left", pad=10)
    legend_items = []
    num_bases = int(cfg.env.num_bases)
    num_targets = int(cfg.env.num_targets)
    
    if num_bases > 0:   legend_items.append((mpatches.Patch(color="#1d4ed8"), "Base station"))
    if num_targets > 0: legend_items.append((mpatches.Patch(color="#dc2626"), "Target"))
    # MEM_T8-only diagnostic legend entry.
    if frame.anti_target_pos is not None:
        legend_items.append((mpatches.Patch(color="#581c87"), "Anti-target"))
    
    if num_bases > 0:   legend_items.append((mpatches.Patch(color="#3b82f6"), "Base-connected"))
    if num_targets > 0: legend_items.append((mpatches.Patch(color="#ef4444"), "Target-connected"))
    if num_bases > 0 and num_targets > 0:
        legend_items.append((mpatches.Patch(color="#a855f7"), "Both (bridge)"))
        
    legend_items.append((mpatches.Patch(color="#6b7280"),         "Isolated"))
    legend_items.append((mpatches.Patch(color="#0d9488", alpha=.3), "Coverage"))

    leg_ax.legend(
        handles=[p for p, _ in legend_items], labels=[l for _, l in legend_items],
        loc="upper left", fontsize=RendererConfig.MPL_FONT_SIZE_LEGEND, frameon=True,
        framealpha=0.8, edgecolor="#e2e8f0",
    )

    # --- Target Known Info Text ---
    known_str = "Target known to:"
    has_any = False
    if getattr(frame, "base_target_known", False):
        known_str += "\n- Base station"
        has_any = True
    if frame.target_known is not None:
        known_indices = [i for i in range(N) if bool(frame.target_known[i])]
        known_indices.sort()
        if known_indices:
            for idx in known_indices:
                known_str += f"\n- Drone {idx}"
            has_any = True
    if not has_any:
        known_str += "\n  (none)"

    # --- Finder Path Debug Print ---
    render_fp_debug = bool(cfg.visualize.get("render_finders_path_debug", False))
    if render_fp_debug:
        known_str += "\n\nFinder Path:"
        if frame.finders_path is not None and frame.finders_path_len > 0:
            path_cells = [tuple(c) for c in frame.finders_path[:frame.finders_path_len]]
            path_str = ", ".join(f"({c[0]},{c[1]})" for c in path_cells)
            
            # Wrap lines for text box
            max_chars = 22
            wrapped_lines = [path_str[i:i+max_chars] for i in range(0, len(path_str), max_chars)]
            for line in wrapped_lines:
                known_str += f"\n  {line}"
        else:
            known_str += "\n  (not valid)"

    leg_ax.text(
        0.0, 0.40, known_str,
        fontsize=RendererConfig.MPL_FONT_SIZE_LEGEND,
        color="#0f172a",
        ha="left", va="top",
        transform=leg_ax.transAxes,
        linespacing=1.4
    )

    # --- Connection Matrix Text ---
    render_conn = bool(cfg.visualize.get("render_conn_matrix", True))
    if render_conn:
        lines_count = known_str.count("\n") + 1
        conn_y = 0.40 - lines_count * 0.035 - 0.03
        conn_str = "Connections (0..N-1, B):"
        if getattr(frame, "adj_matrix", None) is not None and frame.adj_matrix.size > 0:
            adj = frame.adj_matrix
            # Header
            header = "   " + " ".join(str(i) for i in range(N)) + " B"
            conn_str += f"\n{header}"
            for i in range(N + 1):
                row_label = f"{i} " if i < N else "B "
                row_vals = []
                for j in range(N + 1):
                    if i == j:
                        row_vals.append(".")
                    else:
                        row_vals.append("1" if bool(adj[i, j]) else "0")
                conn_str += f"\n{row_label} " + " ".join(row_vals)
        else:
            conn_str += "\n  (not available)"

        leg_ax.text(
            0.0, conn_y, conn_str,
            fontfamily="monospace",
            fontsize=RendererConfig.MPL_FONT_SIZE_LEGEND * 0.8,
            color="#0f172a",
            ha="left", va="top",
            transform=leg_ax.transAxes,
            linespacing=1.2
        )

    # --- Reward plot ---
    if rew_ax is not None and rewards_so_far is not None:
        if rewards_so_far.ndim == 2:
            team_rewards = rewards_so_far.sum(axis=1)
            indiv_rewards = rewards_so_far
        else:
            team_rewards = rewards_so_far
            indiv_rewards = None

        rew_ax.plot(np.arange(len(team_rewards)), np.cumsum(team_rewards),
                    color="#10b981", lw=1.8)
        rew_ax.set_xlim(0, total_steps)
        rew_ax.set_title("Cumulative Team Reward", fontsize=16, color="#0f172a",
                          loc="left", pad=10)
        rew_ax.set_ylabel("Return", fontsize=7, color="#64748b")
        rew_ax.tick_params(labelsize=6, colors="#64748b")
        for sp in rew_ax.spines.values():
            sp.set_edgecolor("#cbd5e1")
        rew_ax.grid(True, linestyle=":", alpha=0.5, color="#94a3b8")

        if indiv_rew_ax is not None and indiv_rewards is not None:
            # Show individual advantage relative to the worst-performing agent at each step
            rel_indiv = indiv_rewards - indiv_rewards.min(axis=1, keepdims=True)
            cum_indiv = np.cumsum(rel_indiv, axis=0)
            
            for n in range(N):
                indiv_rew_ax.plot(np.arange(len(cum_indiv)), cum_indiv[:, n],
                                  lw=1.2, alpha=0.8, label=f"D{n}")
            
            indiv_rew_ax.set_xlim(0, total_steps)
            indiv_rew_ax.set_title("Individual Advantage (Relative Return)", fontsize=16, color="#0f172a",
                                   loc="left", pad=10)
            indiv_rew_ax.set_ylabel("Return", fontsize=7, color="#64748b")
            indiv_rew_ax.tick_params(labelsize=6, colors="#64748b")
            for sp in indiv_rew_ax.spines.values():
                sp.set_edgecolor("#cbd5e1")
            indiv_rew_ax.grid(True, linestyle=":", alpha=0.5, color="#94a3b8")
            indiv_rew_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_video(
    trajectory,
    cfg:          DictConfig,
    filename:     str | Path = "outputs/videos/rollout.mp4",
    fps:          int  = 20,
    frame_stride: int | None  = None,
    rewards:      np.ndarray | None = None,
    extra_metrics: dict[str, np.ndarray] | None = None,
) -> str:
    """
    Render a trajectory to an MP4 file.

    Parameters
    ----------
    trajectory   : stacked EnvState PyTree — each field shape (T, ...)
    cfg          : OmegaConf config
    filename     : output path
    fps          : frames per second in the output video
    dpi          : override DPI (default: 80 for training, 150 for hires)
    frame_stride : render every N-th step (default 2 → half temporal resolution)
    hires        : if True, use HIRES_PRESET (dpi=150, frame_stride=1)
    rewards      : optional (T,) reward array — adds a cumulative reward plot

    Memory usage
    ------------
    Frames are streamed directly to the ffmpeg writer; no frame list is
    stored in memory. Peak additional RAM per call is ~1 frame.
    """
    if frame_stride is None: frame_stride = RendererConfig.FRAME_STRIDE

    filename = Path(filename).resolve()
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename.with_name(f"{filename.stem}_{ts}{filename.suffix}")
    filename.parent.mkdir(parents=True, exist_ok=True)

    # --- Pull trajectory to CPU (once) ---
    sample = getattr(trajectory, 'pos', None)
    if sample is not None and isinstance(sample, np.ndarray):
        traj_cpu    = trajectory          # already numpy — no jax needed
        rewards_cpu = rewards if rewards is not None else None
    else:
        import jax as _jax
        traj_cpu    = _jax.device_get(trajectory)
        rewards_cpu = _jax.device_get(rewards) if rewards is not None else None

    # Use actual world dimensions from the first frame of the trajectory
    W = float(traj_cpu.box_width[0])
    H = float(traj_cpu.box_height[0])

    # --- Compute frame dimensions ---
    scale = RendererConfig.RENDER_DPI / 100.0
    
    ml = int(RendererConfig.MARGIN_LEFT * scale)
    mr = int(RendererConfig.MARGIN_RIGHT * scale)
    mt = int(RendererConfig.MARGIN_TOP * scale)
    mb = int(RendererConfig.MARGIN_BOTTOM * scale)

    # Fit map within designated display width and height
    max_w = int(RendererConfig.MAP_DISPLAY_WIDTH * scale)
    max_h = int(RendererConfig.MAP_DISPLAY_HEIGHT * scale)
    
    aspect = W / H
    if aspect >= 1.0:
        pw = max_w
        ph = int(max_w / aspect)
    else:
        ph = max_h
        pw = int(max_h * aspect)

    has_reward = (rewards is not None)
    has_indiv_reward = (rewards is not None and rewards.ndim == 2)

    # Compute reward plots layout using static values from configuration
    rew_h = int(RendererConfig.TEAM_REWARD_HEIGHT * scale) if has_reward else 0
    indiv_rew_h = int(RendererConfig.INDIV_REWARD_HEIGHT * scale) if has_indiv_reward else 0
    
    gap_map_rew = int(RendererConfig.GAP_MAP_REWARDS * scale)
    gap_between = int(RendererConfig.GAP_BETWEEN_REWARDS * scale)

    rew_top = mt + ph + gap_map_rew if has_reward else 0
    indiv_rew_top = rew_top + rew_h + gap_between if has_indiv_reward else 0

    # Calculate total width_px and height_px
    width_px = pw + ml + mr
    
    if has_indiv_reward:
        height_px = indiv_rew_top + indiv_rew_h + mb
    elif has_reward:
        height_px = rew_top + rew_h + mb
    else:
        height_px = mt + ph + mb

    # Macroblock correction
    if width_px % 2 != 0:
        width_px += 1
        mr += 1
    if height_px % 2 != 0:
        height_px += 1
        mb += 1

    # (JAX/CPU extraction)
    traj_cpu = jax.device_get(trajectory)
    rewards_cpu = jax.device_get(rewards) if rewards is not None else None
    
    T = traj_cpu.pos.shape[0]
    frames = range(0, T, frame_stride)
    n_out = len(frames)
    
    print(f"Frame size: {width_px}×{height_px} px  "
          f"(Matplotlib renderer, dpi={RendererConfig.RENDER_DPI}, stride={frame_stride}, world {W:.0f}×{H:.0f} m)")
    print(f"Rendering {n_out} frames ({T} steps) → {filename}")
    
    # --- Create figure ONCE ---
    fig, canvas, ax, leg_ax, rew_ax, indiv_rew_ax = _make_figure(
        width_px, height_px, pw, ph, RendererConfig.RENDER_DPI,
        has_reward=(rewards_cpu is not None),
        has_indiv_reward=has_indiv_reward
    )

    crop_w = crop_h = None
    last_img = None

    # Load map once for static rendering data (walls)
    from env.maps import MapDefinition
    from core.config import MAP_DIR
    occ_grid_static = None
    comm_occ_grid_static = None
    mesh_walls_static = None
    anti_target_static = None
    maze_cell_grid_static = None
    if cfg.env.map_names and len(cfg.env.map_names) > 0:
        active_map_name = cfg.env.map_names[0]
        map_path = MAP_DIR / f"{active_map_name}.yaml"
        if map_path.exists():
            map_def = MapDefinition.load(map_path, cell_size=1.0)
            occ_grid_static = np.array(map_def.occupancy_grid)
            comm_occ_grid_static = np.array(map_def.communication_occupancy_grid)
            mesh_walls_static = np.array(map_def.mesh_walls, dtype=np.float32) if map_def.mesh_walls else None
            if map_def.maze_cell_cols and map_def.maze_cell_rows:
                maze_cell_grid_static = (int(map_def.maze_cell_cols), int(map_def.maze_cell_rows))
            # MEM_T8-only diagnostic marker overlay.
            if map_def.anti_target_spawn_points is not None:
                anti_target_static = np.array(map_def.anti_target_spawn_points)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message="os.fork.*is incompatible with multithreaded code")
        writer = imageio.get_writer(
            str(filename), fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None, quality=7,
        )
        try:
            for idx, t in enumerate(frames):
                frame = _FrameData(
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
                    finders_path  = (np.array(traj_cpu.finders_path[t]) if hasattr(traj_cpu, "finders_path") else None),
                    finders_path_len = (int(traj_cpu.finders_path_len[t]) if hasattr(traj_cpu, "finders_path_len") else 0),
                    maze_cell_grid = maze_cell_grid_static,
                )
                rew_hist = rewards_cpu[:t+1] if rewards_cpu is not None else None

                _draw_frame(fig, ax, leg_ax, rew_ax, indiv_rew_ax, frame, cfg, rew_hist, T)

                canvas.draw()
                buf = canvas.buffer_rgba()
                img = np.asarray(buf)[:, :, :3].copy()

                # H.264 needs even dimensions — measure on first frame only
                if crop_w is None:
                    actual_h, actual_w = img.shape[:2]
                    crop_w = actual_w if actual_w % 2 == 0 else actual_w - 1
                    crop_h = actual_h if actual_h % 2 == 0 else actual_h - 1

                last_img = img[:crop_h, :crop_w]
                writer.append_data(last_img)

                if (idx + 1) % 50 == 0 or idx == n_out - 1:
                    print(f"  {idx + 1}/{n_out} frames", flush=True)

            # 5-second freeze hold — write last frame repeatedly without copy
            if last_img is not None:
                hold = fps * 5
                print(f"  Hold: {hold} freeze frames (5 s)")
                for _ in range(hold):
                    writer.append_data(last_img)

        finally:
            writer.close()

    plt.close(fig)

    size_mb  = filename.stat().st_size / 1e6
    print(f"Saved {filename}  ({size_mb:.1f} MB, {n_out} frames @ {fps} fps)")
    return str(filename)
