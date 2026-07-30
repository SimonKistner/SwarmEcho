"""
swarmecho/visualize/renderer.py
================================
OpenCV video rendering entrypoint.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from swarmecho.visualize.renderer_cv2 import render_video_cv2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_video(
    trajectory,
    cfg:          DictConfig,
    filename:     str | Path = "outputs/artifacts/eval/rollout.mp4",
    fps:          int  = 20,
    frame_stride: Optional[int] = None,
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
    rewards      : optional (T,) reward array — adds a cumulative reward plot

    Memory usage and speed
    ----------------------
    Frames are written directly to the video stream with low memory overhead.
    """
    filename = Path(filename).resolve()
    import re
    if not re.match(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_", filename.name):
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        filename = filename.with_name(f"{ts}_{filename.name}")
    filename.parent.mkdir(parents=True, exist_ok=True)

    return render_video_cv2(
        trajectory=trajectory,
        cfg=cfg,
        filename=filename,
        fps=fps,
        frame_stride=frame_stride,
        rewards=rewards,
        extra_metrics=extra_metrics,
    )


