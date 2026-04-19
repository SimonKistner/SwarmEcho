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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf import DictConfig

# To avoid 50GB GPU memory spikes in worker processes, we import jax locally
import warnings
import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_DPI = 150
FRAME_STRIDE = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FrameData(NamedTuple):
    pos:           np.ndarray
    vel:           np.ndarray
    base_pos:      np.ndarray
    target_pos:    np.ndarray
    coverage_grid: np.ndarray
    step:          int
    active:        np.ndarray | None
    occ_grid:      np.ndarray | None
    box_width:     float
    box_height:    float


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


def _dda_raycast_np(p1, p2, occ_grid):
    """NumPy version of DDA raycast for the renderer (shared with CV2)."""
    gx0, gy0 = p1; gx1, gy1 = p2
    dx = gx1 - gx0; dy = gy1 - gy0
    steps = int(max(abs(dx), abs(dy), 1) * 2)
    if steps > 1000: steps = 1000
    xs = np.linspace(gx0, gx1, steps)
    ys = np.linspace(gy0, gy1, steps)
    ixs = np.floor(xs).astype(int)
    iys = np.floor(ys).astype(int)
    W, H = occ_grid.shape
    mask = (ixs >= 0) & (ixs < W) & (iys >= 0) & (iys < H)
    if not np.all(mask):
        ixs = np.clip(ixs, 0, W-1); iys = np.clip(iys, 0, H-1)
    hit = occ_grid[ixs, iys]
    return not np.any(hit)

def _build_adjacency(pos, base_pos, target_pos, comm_radius, visual_radius, occ_grid, world_size, cfg):
    N = int(pos.shape[0])
    num_bases = int(cfg.env.num_bases)
    num_targets = int(cfg.env.num_targets)
    cell_size = float(cfg.env.grid_cell_size)

    # Build filtered entity list
    ents_comp = []
    base_idx = -1; target_idx = -1
    if num_bases > 0: 
        base_idx = len(ents_comp)
        ents_comp.append(base_pos)
    if num_targets > 0:
        target_idx = len(ents_comp)
        ents_comp.append(target_pos)
    
    drone_start = len(ents_comp)
    for i in range(N):
        ents_comp.append(pos[i])
        
    ents = np.array(ents_comp)
    M = len(ents)
    
    diff  = ents[:, None, :] - ents[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    adj = (dists <= comm_radius) & ~np.eye(M, dtype=bool)
    
    # Raycast check for walls
    if occ_grid is not None:
        for i in range(M):
            for j in range(i + 1, M):
                if adj[i, j]:
                    p1_grid = ents[i] / cell_size
                    p2_grid = ents[j] / cell_size
                    if not _dda_raycast_np(p1_grid, p2_grid, occ_grid):
                        adj[i, j] = adj[j, i] = False
    
    return adj, base_idx, target_idx, drone_start, ents


def _drone_colours(adj, N, base_comp, target_comp, drone_start):
    colours = []
    for i in range(drone_start, drone_start + N):
        ib = i in base_comp
        it = i in target_comp
        if ib and it:   colours.append("#a855f7")
        elif ib:        colours.append("#3b82f6")
        elif it:        colours.append("#ef4444")
        else:           colours.append("#6b7280")
    return colours


# ---------------------------------------------------------------------------
# Figure setup — called ONCE per render_video invocation
# ---------------------------------------------------------------------------

def _make_figure(width_px, height_px, dpi, has_reward):
    scale       = dpi / 100.0
    fig_w       = width_px / dpi
    fig_h       = height_px / dpi

    fig    = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#f0f4f8")
    canvas = FigureCanvasAgg(fig)

    # Pre-compute axes fractions so we can restore them after clear()
    MARGIN_LEFT   = int(55 * scale)
    MARGIN_RIGHT  = int(175 * scale)
    MARGIN_BOTTOM = int(45 * scale)
    MARGIN_TOP    = int(45 * scale)
    if has_reward:
        MARGIN_BOTTOM += int(145 * scale)

    plot_w = width_px - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = height_px - MARGIN_BOTTOM - MARGIN_TOP

    ax_frac = dict(
        left   = MARGIN_LEFT   / width_px,
        bottom = MARGIN_BOTTOM / height_px,
        width  = plot_w        / width_px,
        height = plot_h        / height_px,
    )
    leg_left  = ax_frac["left"] + ax_frac["width"] + 20 / width_px

    ax     = fig.add_axes([ax_frac["left"], ax_frac["bottom"],
                           ax_frac["width"], ax_frac["height"]])
    leg_ax = fig.add_axes([leg_left, 0.20,
                           min(155/width_px, 0.98-leg_left), 0.60])

    rew_ax = None
    if has_reward:
        rew_bot = (45 * scale) / height_px
        rew_ht  = (100 * scale) / height_px
        rew_ax  = fig.add_axes([ax_frac["left"], rew_bot,
                                ax_frac["width"], rew_ht])

    return fig, canvas, ax, leg_ax, rew_ax


# ---------------------------------------------------------------------------
# Per-frame draw (reuses existing figure — axes are cleared, not recreated)
# ---------------------------------------------------------------------------
_THETA = np.linspace(0, 2 * np.pi, 96)


def _draw_frame(
    fig, ax, leg_ax, rew_ax,
    frame:          _FrameData,
    cfg:            DictConfig,
    rewards_so_far: np.ndarray | None,
    total_steps:    int,
) -> None:
    """Clear axes and redraw — no new Figure objects allocated."""

    # Explicitly pull dimensions or fallback to config to prevent AttributeErrors
    W = float(getattr(frame, "box_width", cfg.env.box_width))
    H = float(getattr(frame, "box_height", cfg.env.box_height))
    N = int(cfg.env.num_agents)
    vis_r  = float(cfg.env.visual_radius)
    comm_r = float(cfg.env.comm_radius)
    B      = int(cfg.env.radar_bins)
    v_cfg  = cfg.visualize

    # Clear only the dynamic axes
    ax.clear()
    leg_ax.clear()
    if rew_ax is not None:
        rew_ax.clear()

    # --- Axes base style ---
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(-2, H + 2)
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

    # ── Coverage ──────────────────────────────────────────────────────────
    if frame.coverage_grid.any():
        cell_size = float(cfg.env.grid_cell_size)
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

    # ── Adjacency ─────────────────────────────────────────────────────────
    adj, base_idx, target_idx, drone_start, ents = _build_adjacency(
        frame.pos, frame.base_pos, frame.target_pos, 
        comm_r, vis_r, frame.occ_grid, (W, H), cfg
    )
    
    base_comp   = _bfs(adj, source=base_idx) if base_idx >= 0 else set()
    target_comp = _bfs(adj, source=target_idx) if target_idx >= 0 else set()
    drone_cols  = _drone_colours(adj, N, base_comp, target_comp, drone_start)
    full_chain  = bool(base_idx >= 0 and target_idx >= 0 and (base_comp & target_comp - {base_idx, target_idx}))

    M = len(ents)
    for i in range(M):
        for j in range(i + 1, M):
            if adj[i, j]:
                xi, yi = ents[i]; xj, yj = ents[j]
                ib = i in base_comp;  jb = j in base_comp
                it = i in target_comp; jt = j in target_comp
                if ib and jb and it and jt: ec, lw = "#a855f7", 1.5
                elif ib and jb:              ec, lw = "#3b82f6", 1.2
                elif it and jt:              ec, lw = "#ef4444", 1.2
                else:                        ec, lw = "#94a3b8", 0.8
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
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx-2.5, by-2.5), 5, 5, boxstyle="square,pad=0.3",
            facecolor="#1d4ed8", edgecolor="#1e3a8a", lw=1.5, zorder=5,
        ))
        ax.text(bx, by, "B", color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=6)

    # --- Target ---
    if int(cfg.env.num_targets) > 0:
        tx, ty = frame.target_pos
        ax.plot(tx, ty, "*", color="#dc2626", ms=14,
                markeredgecolor="#7f1d1d", markeredgewidth=0.8, zorder=5)
        ax.text(tx+3, ty+3, "T", color="#dc2626", fontsize=7, fontweight="bold",
                ha="left", va="bottom", zorder=6)

    # --- Drones ---
    for i in range(N):
        x, y   = frame.pos[i]
        is_act = frame.active[i] if frame.active is not None else True
        alpha  = 1.0 if is_act else 0.3
        ax.scatter(x, y, s=60, color=drone_cols[i],
                   edgecolors="white", linewidths=0.8, zorder=7, alpha=alpha)
        ax.text(x+1.5, y+1.5, str(i), color=drone_cols[i],
                fontsize=6, ha="left", va="bottom", zorder=8, alpha=alpha)

    # --- Title ---
    cell_size = float(cfg.env.grid_cell_size)
    GW_active = int(W / cell_size)
    GH_active = int(H / cell_size)
    cov_cnt = frame.coverage_grid[:GW_active, :GH_active].sum()
    cov_pct = 100.0 * cov_cnt / (GW_active * GH_active)

    status  = "✓ CHAIN FORMED" if full_chain else ""
    ax.set_title(
        f"Step {frame.step:04d}   |   Coverage {cov_pct:.1f}%   {status}",
        fontsize=8, pad=4, color="#0f172a",
    )

    # --- Legend ---
    leg_ax.axis("off")
    leg_ax.set_title("Legend", fontsize=7, color="#0f172a", loc="left", pad=4)
    legend_items = []
    num_bases = int(cfg.env.num_bases)
    num_targets = int(cfg.env.num_targets)
    
    if num_bases > 0:   legend_items.append((mpatches.Patch(color="#1d4ed8"), "Base station"))
    if num_targets > 0: legend_items.append((mpatches.Patch(color="#dc2626"), "Target"))
    
    if num_bases > 0:   legend_items.append((mpatches.Patch(color="#3b82f6"), "Base-connected"))
    if num_targets > 0: legend_items.append((mpatches.Patch(color="#ef4444"), "Target-connected"))
    if num_bases > 0 and num_targets > 0:
        legend_items.append((mpatches.Patch(color="#a855f7"), "Both (bridge)"))
        
    legend_items.append((mpatches.Patch(color="#6b7280"),         "Isolated"))
    legend_items.append((mpatches.Patch(color="#0d9488", alpha=.3), "Coverage"))

    leg_ax.legend(
        handles=[p for p, _ in legend_items], labels=[l for _, l in legend_items],
        loc="upper left", fontsize=6.5, frameon=True,
        framealpha=0.8, edgecolor="#e2e8f0",
    )

    # --- Reward plot ---
    if rew_ax is not None and rewards_so_far is not None:
        rew_ax.clear()
        rew_ax.set_facecolor("#ffffff")
        rew_ax.plot(np.arange(len(rewards_so_far)), np.cumsum(rewards_so_far),
                    color="#10b981", lw=1.8)
        rew_ax.set_xlim(0, total_steps)
        rew_ax.set_title("Cumulative Team Reward", fontsize=7, color="#0f172a",
                          loc="left", pad=4)
        rew_ax.set_ylabel("Return", fontsize=7, color="#64748b")
        rew_ax.tick_params(labelsize=6, colors="#64748b")
        for sp in rew_ax.spines.values():
            sp.set_edgecolor("#cbd5e1")
        rew_ax.grid(True, linestyle=":", alpha=0.5, color="#94a3b8")


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
    if frame_stride is None: frame_stride = FRAME_STRIDE

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
    scale         = RENDER_DPI / 100.0
    MAX_LONG      = int(1100 * scale)
    MIN_SHORT     = int(200 * scale)
    aspect        = W / H
    if aspect >= 1.0:
        plot_w = MAX_LONG;  plot_h = max(int(MAX_LONG / aspect), MIN_SHORT)
    else:
        plot_h = MAX_LONG;  plot_w = max(int(MAX_LONG * aspect), MIN_SHORT)

    MARGIN_W = int(250 * scale)
    MARGIN_H = int(110 * scale)
    if rewards is not None:
        MARGIN_H += int(145 * scale)

    width_px  = plot_w + MARGIN_W
    height_px = plot_h + MARGIN_H

    # (Moved JAX/CPU extraction up to compute dynamic dimensions)

    T      = traj_cpu.pos.shape[0]
    frames = range(0, T, frame_stride)
    n_out  = len(frames)
    
    print(f"Frame size: {width_px}×{height_px} px  "
          f"(Matplotlib renderer, dpi={RENDER_DPI}, stride={frame_stride}, world {W:.0f}×{H:.0f} m)")
    print(f"Rendering {n_out} frames ({T} steps) → {filename}")

    # --- Create figure ONCE ---
    fig, canvas, ax, leg_ax, rew_ax = _make_figure(
        width_px, height_px, RENDER_DPI, has_reward=(rewards_cpu is not None)
    )

    crop_w = crop_h = None
    last_img = None

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
            occ_grid_static = np.array(map_def.occupancy_grid)

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
                    coverage_grid = np.array(traj_cpu.coverage_grid[t]),
                    step          = int(traj_cpu.step[t]),
                    active        = (np.array(traj_cpu.active[t]) if hasattr(traj_cpu, "active") else None),
                    occ_grid      = occ_grid_static,
                    box_width     = float(traj_cpu.box_width[t]),
                    box_height    = float(traj_cpu.box_height[t]),
                )
                rew_hist = rewards_cpu[:t+1] if rewards_cpu is not None else None

                _draw_frame(fig, ax, leg_ax, rew_ax, frame, cfg, rew_hist, T)

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


