"""
swarmecho/visualize/renderer.py
================================
Main entrypoint for video rendering.
Supports two backends:
1. "fast" (OpenCV): Extremely fast (15s per video), low memory overhead.
   Used automatically during training intermediate videos.
2. "slow" (Matplotlib): Beautiful, scientific layout, high resolution.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig

# Expose backend modules (they are lazily loaded to avoid heavy imports
# when possible, but the implementations are in their respective files)
from visualize.renderer_cv2 import render_video_cv2
from visualize.renderer_mpl import render_video as render_video_mpl


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def render_video(
    trajectory,
    cfg:          DictConfig,
    filename:     str | Path = "outputs/videos/rollout.mp4",
    fps:          int  = 20,
    frame_stride: Optional[int] = None,
    renderer:     Optional[str] = None,
    rewards:      Optional[np.ndarray] = None,
    extra_metrics: Optional[dict[str, np.ndarray]] = None,
) -> str:
    """
    Render a trajectory to an MP4 file.

    Parameters
    ----------
    trajectory   : stacked EnvState PyTree — each field shape (T, ...)
    cfg          : OmegaConf config
    filename     : output path
    fps          : frames per second in the output video
    frame_stride : render every N-th step
    renderer     : "fast" (OpenCV) or "slow" (Matplotlib). Overrides cfg.visualize.renderer if set.
    rewards      : optional (T,) reward array — adds a cumulative reward plot

    Memory usage and speed
    ----------------------
    fast: ~15s render, <100MB peak RAM overhead. 
    slow: ~90s render, <200MB peak RAM overhead.
    Both write frames directly to video stream. Constant DPIs are defined in backend files.
    """
    if renderer is None:
        renderer = getattr(cfg.visualize, "renderer", "fast")

    if renderer not in ("fast", "slow"):
        print(f"Warning: Unknown renderer '{renderer}', falling back to 'fast'.")
        renderer = "fast"

    filename = Path(filename).resolve()
    import re
    if not re.match(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_", filename.name):
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        filename = filename.with_name(f"{ts}_{filename.name}")
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV observation logs if requested and available
    obs_log = cfg.logging.get("obs_log", True)
    if obs_log and extra_metrics is not None and "obs" in extra_metrics:
        try:
            import csv
            csv_path = filename.with_name(f"{filename.stem}_obs_logs.csv")
            obs_arr = np.array(extra_metrics["obs"])
            if obs_arr.ndim == 3:
                T_steps, N_agents, obs_dim = obs_arr.shape
                B = int(cfg.env.radar_bins)
                headers = ["timestep"]
                for k in range(1, N_agents + 1):
                    headers.extend([
                        f"Agent_{k}_vel_x",
                        f"Agent_{k}_vel_y",
                        f"Agent_{k}_base_rel_pos_x",
                        f"Agent_{k}_base_rel_pos_y",
                        f"Agent_{k}_conn_base",
                        f"Agent_{k}_conn_target",
                        f"Agent_{k}_target_known",
                        f"Agent_{k}_target_rel_pos_x",
                        f"Agent_{k}_target_rel_pos_y",
                    ])
                    for c_idx in range(16):
                        headers.append(f"Agent_{k}_local_cov_{c_idx}")
                    for b_idx in range(B):
                        headers.extend([
                            f"Agent_{k}_radar_bin_{b_idx}_wall",
                            f"Agent_{k}_radar_bin_{b_idx}_drone",
                            f"Agent_{k}_radar_bin_{b_idx}_tgt_conn",
                            f"Agent_{k}_radar_bin_{b_idx}_base_conn",
                        ])

                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                    for t in range(T_steps):
                        row = [t]
                        for agent_idx in range(N_agents):
                            obs_i = obs_arr[t, agent_idx]
                            
                            vel = obs_i[0:2]
                            rel_base = obs_i[2:4]
                            conn_base = float(obs_i[4])
                            conn_tgt = float(obs_i[5])
                            tgt_known = float(obs_i[6])
                            rel_tgt = obs_i[7:9]
                            
                            local_cov = obs_i[9:25]
                            radar = obs_i[25:].reshape(B, 4)
                            
                            row.extend([vel[0], vel[1]])
                            row.extend([rel_base[0], rel_base[1]])
                            row.extend([conn_base, conn_tgt, tgt_known])
                            row.extend([rel_tgt[0], rel_tgt[1]])
                            row.extend(local_cov.tolist())
                            for b_idx in range(B):
                                row.extend(radar[b_idx].tolist())
                        
                        writer.writerow(row)
                print(f"  [obs_logs] Saved CSV -> {csv_path}")
        except Exception as csv_err:
            print(f"Warning: Failed to save obs logs: {csv_err}")

    # Filter out non-scalar metrics (like "obs") before passing to the renderers
    if extra_metrics is not None:
        extra_metrics = {k: v for k, v in extra_metrics.items() if k != "obs"}

    if renderer == "slow":
        return render_video_mpl(
            trajectory=trajectory,
            cfg=cfg,
            filename=filename,
            fps=fps,
            frame_stride=frame_stride,
            rewards=rewards,
            extra_metrics=extra_metrics,
        )
    else:
        return render_video_cv2(
            trajectory=trajectory,
            cfg=cfg,
            filename=filename,
            fps=fps,
            frame_stride=frame_stride,
            rewards=rewards,
            extra_metrics=extra_metrics,
        )


