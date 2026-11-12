"""Small, CPU-only bridge between Determined preemption and training checkpoints."""
from __future__ import annotations

import os
from pathlib import Path
import time


def checkpoint_metrics(metrics, training_started):
    result = dict(metrics or {})
    elapsed = time.perf_counter() - training_started
    result["training_seconds"] = elapsed
    result["cluster_training_seconds"] = float(os.environ.get(
        "SWARMECHO_CLUSTER_PRIOR_TRAINING_SECONDS", "0")) + elapsed
    return result


def checkpoint_on_preempt(save, model, checkpoint_dir, run_name, update, training,
                          resume, steps, adaptive_spawn, target_curriculum,
                          adaptive_maps, validation, destination, metrics, wandb_run):
    request = os.environ.get("SWARMECHO_PREEMPT_REQUEST")
    if not request or not Path(request).is_file():
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save(model, checkpoint_dir / f"ckpt_{update:06d}",
        run_name=run_name, update=update, num_envs=training.num_envs,
        num_steps=training.num_steps, prior_history=resume.prior_history,
        total_steps=steps, adaptive_spawn=adaptive_spawn,
        target_curriculum=target_curriculum, adaptive_maps=adaptive_maps,
        validation=validation, run_dir=destination,
        metrics=dict(metrics, stop_reason="preempted"), final=False)
    print(f"[CLUSTER PREEMPT] Published complete checkpoint {checkpoint}; yielding allocation.", flush=True)
    if wandb_run is not None:
        wandb_run.log({}, step=steps, commit=True)
        wandb_run.finish()
    return checkpoint
