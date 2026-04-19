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
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename.with_name(f"{filename.stem}_{ts}{filename.suffix}")
    filename.parent.mkdir(parents=True, exist_ok=True)

    if renderer == "slow":
        return render_video_mpl(
            trajectory=trajectory,
            cfg=cfg,
            filename=filename,
            fps=fps,
            frame_stride=frame_stride,
            rewards=rewards,
        )
    else:
        return render_video_cv2(
            trajectory=trajectory,
            cfg=cfg,
            filename=filename,
            fps=fps,
            frame_stride=frame_stride,
            rewards=rewards,
        )


