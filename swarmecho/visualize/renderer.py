"""
swarmecho/visualize/renderer.py
================================
CPU-side video renderer for SwarmEcho environment states.

Converts a JAX `EnvState` trajectory (stacked PyTree from lax.scan) into
an MP4 file using Matplotlib for frame generation and imageio for encoding.

Design
------
* All computation here is NumPy / pure Python — runs entirely on CPU after
  `jax.device_get()` pulls arrays off the GPU.
* Connectivity is computed with a fast NumPy BFS to colour-code drones:
    - Blue  : reachable from base station via comm links
    - Red   : reachable from target via comm links
    - Purple: reachable from BOTH (chain is nearly closed)
    - Grey  : isolated (not part of any connected component)
* Coverage grid is shown as a semi-transparent teal overlay.
* Comm links are drawn between any pair of entities within comm_radius.

Public API
----------
    render_video(trajectory, cfg, filename, fps=20, dpi=100) -> str
        `trajectory` is the stacked EnvState returned directly from
        jax.lax.scan (each field has shape (T, ...)). Calls jax.device_get
        internally so it's safe to call before or after.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple

import imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf import DictConfig

import jax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FrameData(NamedTuple):
    """Flat NumPy arrays for a single timestep — avoids array indexing in loop."""
    pos:           np.ndarray   # (N, 2)
    vel:           np.ndarray   # (N, 2)
    base_pos:      np.ndarray   # (2,)
    target_pos:    np.ndarray   # (2,)
    coverage_grid: np.ndarray   # (G, G) bool
    step:          int


def _bfs(adj: np.ndarray, source: int) -> set[int]:
    """NumPy adjacency matrix BFS — returns set of reachable node indices."""
    visited: set[int] = {source}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        neighbours = np.where(adj[node])[0]
        for nb in neighbours:
            nb = int(nb)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def _build_adjacency(
    pos: np.ndarray,        # (N, 2)
    base_pos: np.ndarray,   # (2,)
    target_pos: np.ndarray, # (2,)
    comm_radius: float,
) -> np.ndarray:            # (N+2, N+2) bool
    """
    Build symmetric adjacency matrix for [base, target, drones].
    Entity index layout: 0=base, 1=target, 2..N+1=drones.
    """
    N = pos.shape[0]
    M = N + 2
    entities = np.concatenate([
        base_pos[None, :],    # (1, 2)
        target_pos[None, :],  # (1, 2)
        pos,                  # (N, 2)
    ], axis=0)                # (M, 2)

    # Pairwise L2 distances
    diff = entities[:, None, :] - entities[None, :, :]   # (M, M, 2)
    dists = np.linalg.norm(diff, axis=-1)                # (M, M)
    adj = (dists <= comm_radius) & ~np.eye(M, dtype=bool)
    return adj


def _drone_colours(
    adj: np.ndarray,
    N: int,
    base_component: set[int],
    target_component: set[int],
) -> list[str]:
    """Return a colour string per drone based on component membership."""
    colours = []
    for i in range(2, N + 2):   # drone indices start at 2
        in_base   = i in base_component
        in_target = i in target_component
        if in_base and in_target:
            colours.append("#a855f7")   # purple — bridging both
        elif in_base:
            colours.append("#3b82f6")   # blue
        elif in_target:
            colours.append("#ef4444")   # red
        else:
            colours.append("#6b7280")   # grey — isolated
    return colours


def _render_frame(
    frame: _FrameData,
    cfg:   DictConfig,
    fig_size_px: tuple[int, int],
    dpi: int,
) -> np.ndarray:     # (H, W, 3) uint8
    """Render one frame to a NumPy RGB image."""

    W      = float(cfg.env.box_width)
    H      = float(cfg.env.box_height)
    N      = int(cfg.env.num_agents)
    G      = int(cfg.env.grid_resolution)
    vis_r  = float(cfg.env.visual_radius)
    comm_r = float(cfg.env.comm_radius)

    width_px, height_px = fig_size_px
    fig_w  = width_px / dpi
    fig_h  = height_px / dpi
    fig    = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    range_x = W + 4.0
    range_y = H + 4.0

    scale = dpi / 100.0

    # Compute exact layout to preserve 1:1 aspect ratio:
    MARGIN_LEFT   = int(60 * scale)
    MARGIN_RIGHT  = int(180 * scale)
    MARGIN_BOTTOM = int(50 * scale)
    MARGIN_TOP    = int(50 * scale)

    plot_w_px = width_px - MARGIN_LEFT - MARGIN_RIGHT
    plot_h_px = height_px - MARGIN_BOTTOM - MARGIN_TOP

    target_aspect = range_x / range_y
    current_aspect = plot_w_px / plot_h_px

    if current_aspect > target_aspect:
        # Too wide, shrink width
        new_w = plot_h_px * target_aspect
        plot_w_px = new_w
    else:
        # Too tall, shrink height
        new_h = plot_w_px / target_aspect
        MARGIN_BOTTOM += (plot_h_px - new_h) / 2
        plot_h_px = new_h

    ax_left   = MARGIN_LEFT / width_px
    ax_bottom = MARGIN_BOTTOM / height_px
    ax_width  = plot_w_px / width_px
    ax_height = plot_h_px / height_px

    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    # --- Background ---------------------------------------------------------
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(-2, H + 2)
    ax.tick_params(labelsize=7)

    # --- Bounding box -------------------------------------------------------
    box = mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=2, edgecolor="#334155", facecolor="none", zorder=2,
    )
    ax.add_patch(box)

    # --- Coverage grid (semi-transparent teal overlay) ----------------------
    if frame.coverage_grid.any():
        cov_rgba = np.zeros((G, G, 4), dtype=np.float32)
        cov_rgba[frame.coverage_grid] = [0.04, 0.78, 0.73, 0.25]  # teal, 25% alpha
        # coverage_grid[i,j] → x=i, y=j; imshow wants [row=y, col=x]
        ax.imshow(
            cov_rgba.transpose(1, 0, 2),
            extent=[0, W, 0, H],
            origin="lower",
            aspect="auto",
            zorder=1,
            interpolation="nearest",
        )

    # --- Connectivity -------------------------------------------------------
    adj          = _build_adjacency(frame.pos, frame.base_pos, frame.target_pos, comm_r)
    base_comp    = _bfs(adj, source=0)
    target_comp  = _bfs(adj, source=1)
    drone_cols   = _drone_colours(adj, N, base_comp, target_comp)
    full_chain   = bool(base_comp & target_comp - {0, 1})   # drones in both

    # Draw comm links between all connected pairs
    M        = N + 2
    entities_pos = np.concatenate([
        frame.base_pos[None, :],
        frame.target_pos[None, :],
        frame.pos,
    ], axis=0)   # (M, 2)

    for i in range(M):
        for j in range(i + 1, M):
            if adj[i, j]:
                xi, yi = entities_pos[i]
                xj, yj = entities_pos[j]
                # Edge colour based on component membership
                i_base = i in base_comp
                j_base = j in base_comp
                i_tgt  = i in target_comp
                j_tgt  = j in target_comp
                if (i_base and j_base) and (i_tgt and j_tgt):
                    edge_col, lw = "#a855f7", 1.5   # purple
                elif i_base and j_base:
                    edge_col, lw  = "#3b82f6", 1.2   # blue
                elif i_tgt and j_tgt:
                    edge_col, lw  = "#ef4444", 1.2   # red
                else:
                    edge_col, lw  = "#94a3b8", 0.8   # grey
                ax.plot(
                    [xi, xj], [yi, yj],
                    linestyle="--", color=edge_col, linewidth=lw,
                    alpha=0.7, zorder=3,
                )

    # --- Radii circles per drone --------------------------------------------
    _theta = np.linspace(0, 2 * np.pi, 128)
    
    v_cfg = cfg.visualize

    for i in range(N):
        x, y   = frame.pos[i]
        col    = drone_cols[i]
        
        cr_col = col if v_cfg.comm_color == "match_drone" else v_cfg.comm_color
        vr_col = col if v_cfg.vis_color == "match_drone" else v_cfg.vis_color

        # Outer ring — comm_radius
        cr_x = x + comm_r * np.cos(_theta)
        cr_y = y + comm_r * np.sin(_theta)
        ax.fill(cr_x, cr_y, color=cr_col, alpha=float(v_cfg.comm_fill_alpha), zorder=3)
        ax.plot(cr_x, cr_y, color=cr_col, alpha=float(v_cfg.comm_edge_alpha), linewidth=1.2,
                linestyle="--", zorder=3)

        # Inner ring — visual_radius
        vr_x = x + vis_r * np.cos(_theta)
        vr_y = y + vis_r * np.sin(_theta)
        ax.fill(vr_x, vr_y, color=vr_col, alpha=float(v_cfg.vis_fill_alpha), zorder=4)
        ax.plot(vr_x, vr_y, color=vr_col, alpha=float(v_cfg.vis_edge_alpha), linewidth=1.2,
                linestyle=":", zorder=4)

    # --- Base station (blue square) -----------------------------------------
    bx, by = frame.base_pos
    base_patch = mpatches.FancyBboxPatch(
        (bx - 2.5, by - 2.5), 5, 5,
        boxstyle="square,pad=0.3",
        facecolor="#1d4ed8", edgecolor="#1e3a8a", linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(base_patch)
    ax.text(bx, by, "B", color="white", fontsize=7, fontweight="bold",
            ha="center", va="center", zorder=6)

    # --- Target (red star) --------------------------------------------------
    tx, ty = frame.target_pos
    ax.plot(
        tx, ty, marker="*", color="#dc2626",
        markersize=14, markeredgecolor="#7f1d1d", markeredgewidth=0.8,
        zorder=5,
    )
    ax.text(tx + 3, ty + 3, "T", color="#dc2626", fontsize=7, fontweight="bold",
            ha="left", va="bottom", zorder=6)

    # --- Drones -------------------------------------------------------------
    for i in range(N):
        x, y = frame.pos[i]
        ax.scatter(x, y, s=60, color=drone_cols[i],
                   edgecolors="white", linewidths=0.8, zorder=7)
        ax.text(x + 1.5, y + 1.5, str(i), color=drone_cols[i],
                fontsize=6, ha="left", va="bottom", zorder=8)

    # --- Title / info -------------------------------------------------------
    coverage_pct = 100.0 * frame.coverage_grid.sum() / (G * G)
    chain_status = "✓ CHAIN FORMED" if full_chain else ""
    ax.set_title(
        f"Step {frame.step:04d}   |   Coverage {coverage_pct:.1f}%   {chain_status}",
        fontsize=8, pad=4, color="#0f172a",
    )
    ax.set_xlabel("x [m]", fontsize=7, color="#64748b")
    ax.set_ylabel("y [m]", fontsize=7, color="#64748b")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")

    # --- Legend (right panel) -----------------------------------------------
    leg_left = ax_left + ax_width + (20 / width_px)
    legend_ax = fig.add_axes([leg_left, 0.20, min(160/width_px, 0.98-leg_left), 0.60])
    legend_ax.axis("off")
    legend_ax.set_title("Legend", fontsize=7, color="#0f172a", loc="left", pad=4)
    legend_items = [
        (mpatches.Patch(color="#1d4ed8"),        "Base station"),
        (mpatches.Patch(color="#dc2626"),        "Target"),
        (mpatches.Patch(color="#3b82f6"),        "Base-connected"),
        (mpatches.Patch(color="#ef4444"),        "Target-connected"),
        (mpatches.Patch(color="#a855f7"),        "Both (bridge)"),
        (mpatches.Patch(color="#6b7280"),        "Isolated"),
        (mpatches.Patch(color="#0d9488", alpha=0.3), "Coverage"),
    ]
    legend_ax.legend(
        handles=[p for p, _ in legend_items],
        labels=[l for _, l in legend_items],
        loc="upper left", fontsize=6.5,
        frameon=True, framealpha=0.8,
        edgecolor="#e2e8f0",
    )

    # --- Extract to numpy ---------------------------------------------------
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()   # RGBA → RGB, contiguous
    plt.close(fig)
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_video(
    trajectory,           # stacked EnvState PyTree (T, ...) from lax.scan
    cfg: DictConfig,
    filename: str | Path = "outputs/videos/rollout.mp4",
    fps: int  = 20,
    dpi: int | None = None,
    width_px: int | None  = None,   # None → auto-computed from box aspect ratio
    height_px: int | None = None,   # None → auto-computed from box aspect ratio
) -> str:
    """
    Render a trajectory to an MP4 file.

    Parameters
    ----------
    trajectory : stacked EnvState (each field has a leading time dimension T)
                 Can still be on GPU — this function calls jax.device_get().
    cfg        : OmegaConf config
    filename   : output path (created if it doesn't exist)
    fps        : frames per second
    dpi        : matplotlib DPI (affects render quality)
    width_px   : total frame width in pixels. Default: auto from box aspect ratio.
    height_px  : total frame height in pixels. Default: auto from box aspect ratio.

    Returns
    -------
    str : absolute path to the saved video
    """
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = int(cfg.visualize.dpi)

    W = float(cfg.env.box_width)
    H = float(cfg.env.box_height)
    
    scale = dpi / 100.0

    # -----------------------------------------------------------------------
    # Auto-compute frame dimensions from world aspect ratio
    # Strategy: fix the longest world dimension to MAX_LONG_SIDE_PX pixels,
    # scale the other proportionally, enforce a minimum, then add fixed
    # margins for the axes decorations and legend panel.
    # -----------------------------------------------------------------------
    if width_px is None or height_px is None:
        MAX_LONG_SIDE_PX = int(1200 * scale)
        MIN_SHORT_SIDE_PX = int(220 * scale)

        aspect = W / H
        if aspect >= 1.0:         # landscape or square
            plot_w = MAX_LONG_SIDE_PX
            plot_h = max(int(MAX_LONG_SIDE_PX / aspect), MIN_SHORT_SIDE_PX)
        else:                     # portrait
            plot_h = MAX_LONG_SIDE_PX
            plot_w = max(int(MAX_LONG_SIDE_PX * aspect), MIN_SHORT_SIDE_PX)

        # Add fixed pixel budget for: axes labels, title, legend panel
        MARGIN_W = int(260 * scale)   # legend panel + left/right axis padding
        MARGIN_H = int(120 * scale)   # title + top/bottom axis padding
        width_px  = plot_w + MARGIN_W
        height_px = plot_h + MARGIN_H

    print(f"Frame size: {width_px}×{height_px} px  "
          f"(world {W:.0f}×{H:.0f} m, aspect {W/H:.2f})")

    # Pull everything to CPU once
    traj_cpu = jax.device_get(trajectory)

    T = traj_cpu.pos.shape[0]
    print(f"Rendering {T} frames → {filename} ...")

    frames: list[np.ndarray] = []
    # H.264 (libx264) requires width AND height to be divisible by 2.
    # Matplotlib does not guarantee exact pixel counts, so we measure the
    # first rendered frame and crop to the nearest even dimensions.
    crop_w: int | None = None
    crop_h: int | None = None

    for t in range(T):
        frame = _FrameData(
            pos           = np.array(traj_cpu.pos[t]),
            vel           = np.array(traj_cpu.vel[t]),
            base_pos      = np.array(traj_cpu.base_pos[t]),
            target_pos    = np.array(traj_cpu.target_pos[t]),
            coverage_grid = np.array(traj_cpu.coverage_grid[t]),
            step          = int(traj_cpu.step[t]),
        )
        img = _render_frame(frame, cfg, (width_px, height_px), dpi)

        # Determine even crop bounds from the first frame
        if crop_w is None:
            actual_h, actual_w = img.shape[:2]
            crop_w = actual_w if actual_w % 2 == 0 else actual_w - 1
            crop_h = actual_h if actual_h % 2 == 0 else actual_h - 1
            if crop_w != actual_w or crop_h != actual_h:
                print(f"  Note: frame cropped {actual_w}×{actual_h} → {crop_w}×{crop_h} "
                      f"(H.264 even-dimension requirement)")

        frames.append(img[:crop_h, :crop_w])

        if (t + 1) % 50 == 0 or t == T - 1:
            print(f"  {t + 1}/{T} frames rendered")

    # Write MP4 via imageio + ffmpeg
    with imageio.get_writer(
        str(filename),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",   # maximum player compatibility
        macro_block_size=None,   # allow arbitrary frame dimensions
        quality=8,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    size_mb = filename.stat().st_size / 1e6
    print(f"Saved {filename}  ({size_mb:.1f} MB, {T} frames @ {fps} fps)")
    return str(filename)
