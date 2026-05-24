"""
swarmecho/training/video_worker.py
====================================
Async video rendering via a dedicated background worker process.

Architecture
------------
Main process (GPU training)
  │
  │  put(RenderJob)   — non-blocking, returns immediately
  ▼
multiprocessing.Queue   (max_size cap to prevent memory runaway)
  │
  ▼
Worker process (pure CPU)
  ├── render_video()          — Matplotlib/OpenCV → .mp4
  └── wandb.log(Video())      — optional W&B upload

The worker receives a ``RenderJob`` (a plain dataclass of numpy arrays
and metadata) packed by the main loop, renders the video, and optionally
logs it to W&B — all without the main process ever blocking.

Usage
-----
    worker = VideoRenderWorker(cfg, wandb_run)
    worker.start()

    # Inside training loop (non-blocking):
    worker.submit(ep_states, ep_rewards, update, eval_ret)

    # At the end of training:
    worker.shutdown(wait=True)   # drains queue then kills process
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data container sent across the queue (numpy only — must be picklable)
# ---------------------------------------------------------------------------

@dataclass
class RenderJob:
    """All data needed to render and log one eval episode."""
    # Stacked trajectory: dict of leaf-name → numpy array with leading time axis
    traj_arrays:    Dict[str, np.ndarray]
    # Per-step scalar rewards, shape (T,)
    rewards:        np.ndarray
    # Metadata
    update:         int
    eval_ret:       float
    extra_metrics:  Dict[str, np.ndarray]
    video_path:     str          # Full absolute path for the .mp4
    wandb_step:     int
    renderer:       str          # "fast" or "slow"
    # W&B config (duplicated here so worker needs no reference to the run)
    use_wandb:      bool
    wandb_project:  Optional[str]
    wandb_run_id:   Optional[str]


# Sentinel: tells the worker to exit cleanly
_STOP = None


# ---------------------------------------------------------------------------
# Worker process target function
# ---------------------------------------------------------------------------

def _worker_fn(queue: mp.Queue, cfg_dict: dict) -> None:
    """
    Runs in the worker process. Reads RenderJobs from the queue and
    processes them sequentially.  Imports are deferred to here so the
    main process never pays the import cost twice.
    """
    # Deferred heavy imports (only paid once in the worker process)
    import sys
    from pathlib import Path as _Path
    root = str(_Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    from omegaconf import OmegaConf, DictConfig

    cfg: DictConfig = OmegaConf.create(cfg_dict)

    from visualize.renderer import render_video
    from env.state import EnvState

    # Keep a reference to the W&B API if needed
    wandb_api = None

    print("[video_worker] started ✓", flush=True)

    while True:
        job: Optional[RenderJob] = queue.get()   # blocks until item available
        if job is _STOP:
            print("[video_worker] received stop signal, exiting.", flush=True)
            break

        t0 = time.perf_counter()
        try:
            # traj_arrays is a dict of plain numpy arrays (converted in submit()).
            # Keep them as numpy — do NOT import jax here, as that would
            # initialise the CUDA runtime and inflate RAM by 50+ GB.
            from types import SimpleNamespace
            traj_ns     = SimpleNamespace(**job.traj_arrays)
            rewards_arr = job.rewards   # already np.ndarray

            vid_path = render_video(
                traj_ns, cfg,
                filename     = job.video_path,
                fps          = 20,
                renderer     = job.renderer,
                rewards      = rewards_arr,
                extra_metrics= job.extra_metrics,
            )

            elapsed = time.perf_counter() - t0
            print(
                f"[video_worker] update={job.update}  "
                f"render={elapsed:.1f}s  "
                f"video→{vid_path}",
                flush=True,
            )

            # Note: Child processes cannot easily log to the parent's W&B run.
            # We skip the wandb.log() call here to avoid the 'wandb.init() not called' error.
            # The video is still saved locally in the 'videos/' directory.
            pass

        except Exception as e:
            import traceback
            print(f"[video_worker] ERROR processing job: {e}", flush=True)
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Public API — used by the main training loop
# ---------------------------------------------------------------------------

class VideoRenderWorker:
    """
    Manages the background rendering process.

    Parameters
    ----------
    cfg         : OmegaConf DictConfig (serialised to dict for pickling)
    video_dir   : directory where .mp4 files will be saved
    wandb_run   : active W&B run object, or None
    max_queue   : max pending jobs before submit() blocks (memory safety)
    """

    def __init__(
        self,
        cfg,
        video_dir:  Path,
        wandb_run:  Any = None,
        max_queue:  int = 4,
    ) -> None:
        from omegaconf import OmegaConf
        self._cfg_dict    = OmegaConf.to_container(cfg, resolve=True)
        self._video_dir   = Path(video_dir)
        self._wandb_run   = wandb_run
        self._use_wandb   = wandb_run is not None
        self._wandb_proj  = cfg.logging.get("wandb_project", "swarmecho") if self._use_wandb else None
        self._wandb_id    = wandb_run.id if self._use_wandb else None

        # Use 'spawn' so the worker doesn't inherit JAX's CUDA context,
        # which would cause a deadlock (JAX is not fork-safe).
        ctx = mp.get_context("spawn")
        self._queue   = ctx.Queue(maxsize=max_queue)
        self._process = ctx.Process(
            target  = _worker_fn,
            args    = (self._queue, self._cfg_dict),
            daemon  = True,   # auto-killed if main process exits
            name    = "VideoRenderWorker",
        )

    def start(self) -> None:
        """Start the background worker process."""
        self._process.start()
        print(f"  [render] Worker process started (PID {self._process.pid})")

    def submit(
        self,
        ep_states:  List,     # list of EnvState objects (Python, from _evaluate)
        ep_rewards: List[float],
        update:     int,
        eval_ret:   float,
        extra_metrics: Dict[str, np.ndarray],
        renderer:   str = "fast",
    ) -> None:
        """
        Package the trajectory and queue a render job.
        Non-blocking unless the queue is full (max_queue reached),
        in which case it blocks just long enough for one slot to free up.

        All JAX arrays are eagerly converted to numpy here so the
        multiprocessing pickler can serialise them.
        """
        import jax

        # Flatten the list-of-structs into a dict of numpy arrays with
        # a leading time axis.  Works for any chex.dataclass / NamedTuple.
        stacked = jax.tree.map(
            lambda *xs: np.array(np.stack(xs)),
            *ep_states,
        )

        # jax.tree.map returns the same pytree type as the leaves.
        # Convert to a plain dict so it's safely picklable.
        if hasattr(stacked, '__dict__'):
            traj_arrays = {k: np.asarray(v) for k, v in vars(stacked).items()
                           if isinstance(v, np.ndarray)}
        else:
            # chex dataclass → use its fields
            import dataclasses
            traj_arrays = {
                f.name: np.asarray(getattr(stacked, f.name))
                for f in dataclasses.fields(stacked)
                if hasattr(stacked, f.name)
            }

        # Build the absolute video path (timestamped)
        from datetime import datetime
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_path = str((self._video_dir / f"eval_update_{update:06d}_{ts}.mp4").absolute())

        job = RenderJob(
            traj_arrays  = traj_arrays,
            rewards      = np.array(ep_rewards, dtype=np.float32),
            update       = update,
            eval_ret     = eval_ret,
            extra_metrics= {k: np.array(v) for k, v in extra_metrics.items()},
            video_path   = vid_path,
            wandb_step   = update,
            renderer     = renderer,
            use_wandb    = self._use_wandb,
            wandb_project= self._wandb_proj,
            wandb_run_id = self._wandb_id,
        )

        # put() with a timeout so a stalled worker doesn't freeze training forever
        self._queue.put(job, timeout=30)

    def shutdown(self, wait: bool = True) -> None:
        """
        Signal the worker to stop cleanly.

        Parameters
        ----------
        wait : If True, block until the worker drains the queue and exits.
               If False, return immediately (worker finishes in background).
        """
        self._queue.put(_STOP)
        if wait:
            print("  [render] Waiting for worker to finish remaining jobs...")
            self._process.join(timeout=300)   # max 5 min grace period
            if self._process.is_alive():
                print("  [render] Worker timeout — terminating forcefully.")
                self._process.terminate()
                self._process.join()
                self._queue.cancel_join_thread()
        else:
            print("  [render] Worker will finish remaining jobs in background.")


