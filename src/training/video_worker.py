"""
swarmecho/training/video_worker.py
====================================
Thin synchronous helper for rendering a single eval episode to disk.

Previously this module contained an async background-process worker
(VideoRenderWorker / RenderJob).  That architecture was removed because the
fast renderer is now quick enough that blocking for one frame never meaningfully
delays training, and the process-spawning overhead + multiprocessing complexity
was not worth it.

The sole public function is render_eval_video(), which wraps the renderer call
with trajectory stacking and path creation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def render_eval_video(
    ep_states: list,
    ep_rewards: list,
    ep_metrics: dict,
    cfg: Any,
    out_dir: Path,
    filename_stem: str,
) -> str:
    """
    Synchronously render one episode to ``<out_dir>/<filename_stem>.mp4``.

    Parameters
    ----------
    ep_states     : list of EnvState snapshots collected by the video episode collector
    ep_rewards    : list of per-step reward arrays
    ep_metrics    : dict of metric arrays (chain_pct, chain_gap, etc.)
    cfg           : OmegaConf DictConfig
    out_dir       : directory to write the video into (created if absent)
    filename_stem : base filename without extension, e.g.
                    ``"SUCCESS_eval_ckpt_000762_ep00"``
                    The renderer appends a timestamp + ``.mp4``.
    Returns
    ----------
    str : resolved path to the written .mp4 file
    """
    import jax

    from visualize.renderer import render_video

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stack list-of-structs → a single struct with leading time axis
    stacked = jax.tree.map(lambda *xs: np.array(np.stack(xs)), *ep_states)

    # Convert to a plain SimpleNamespace so there is no JAX / chex dependency
    # downstream in the renderer process.
    if dataclasses.is_dataclass(stacked):
        traj_ns = SimpleNamespace(
            **{f.name: getattr(stacked, f.name) for f in dataclasses.fields(stacked)}
        )
    else:
        traj_ns = stacked  # already a SimpleNamespace or compatible

    vid_path = str((out_dir / f"{filename_stem}.mp4").absolute())

    return render_video(
        traj_ns, cfg,
        filename      = vid_path,
        fps           = 20,
        rewards       = np.array(ep_rewards, dtype=np.float32),
        extra_metrics = ep_metrics,
    )
