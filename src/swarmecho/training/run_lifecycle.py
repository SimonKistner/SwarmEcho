"""Dimension-agnostic training lifecycle shared by environment adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def cfg_value(section: Any, key: str, default: Any = None) -> Any:
    """Read one option from DictConfig, mapping, or typed dataclass sections."""
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


@dataclass(frozen=True)
class RunLayout:
    run_name: str
    run_dir: Path
    checkpoint_dir: Path
    train_artifact_dir: Path
    train_media_dir: Path
    train_data_dir: Path
    train_manifest_dir: Path


def resolve_run_layout(logging: Any, evaluation: Any, *, media_dir: str) -> RunLayout:
    """Resolve the proven run/checkpoint/artifact hierarchy for any dimension."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_name = cfg_value(logging, "run_name")
    use_timestamp = bool(cfg_value(logging, "use_timestamp_postfix", False))
    run_name = (
        f"run_{timestamp}"
        if not configured_name
        else f"{configured_name}_{timestamp}" if use_timestamp else str(configured_name)
    )
    run_dir = Path(cfg_value(logging, "log_dir", "outputs")).absolute() / run_name
    configured_checkpoints = cfg_value(evaluation, "checkpoint_dir")
    checkpoint_dir = (
        Path(str(configured_checkpoints).replace("\\", "/")).absolute()
        if configured_checkpoints
        else run_dir / "checkpoints"
    )
    artifact_dir = run_dir / "artifacts" / "train"
    return RunLayout(
        run_name=run_name,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        train_artifact_dir=artifact_dir,
        train_media_dir=artifact_dir / media_dir,
        train_data_dir=artifact_dir / "data",
        train_manifest_dir=artifact_dir / "manifests",
    )


@dataclass(frozen=True)
class ResumeState:
    start_update: int
    step_offset: int
    prior_history: list[dict[str, Any]]


def resolve_resume_state(training: Any, run_name: str, num_envs: int, num_steps: int) -> ResumeState:
    """Apply the maintained branch/resume and cumulative-step-history semantics."""
    checkpoint_value = cfg_value(training, "checkpoint_path")
    if not checkpoint_value:
        manual = cfg_value(training, "checkpoint_step_offset")
        return ResumeState(0, int(manual or 0), [])
    checkpoint = Path(str(checkpoint_value).replace("\\", "/")).absolute()
    mode = str(cfg_value(training, "ckpt_loading_mode", "branch")).lower()
    match = re.search(r"ckpt_(?:early_|final_)?(\d+)", checkpoint.name, re.IGNORECASE)
    encoded_update = int(match.group(1)) if match else 0
    history: list[dict[str, Any]] = []
    total_steps = encoded_update * num_envs * num_steps
    history_path = checkpoint / "step_history.json"
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        history = list(payload.get("history", []))
        total_steps = int(payload.get("total_steps", total_steps))
    elif total_steps:
        parent_name = checkpoint.parents[1].name if len(checkpoint.parents) > 1 else "checkpoint"
        history = [{"run_name": parent_name, "steps": total_steps}]
    start_update = encoded_update if mode == "resume" else 0
    if mode == "resume" and history and history[-1].get("run_name") == run_name:
        prior_history = history[:-1]
    else:
        prior_history = history
    inferred_offset = max(0, total_steps - start_update * num_envs * num_steps)
    manual = cfg_value(training, "checkpoint_step_offset")
    return ResumeState(start_update, int(manual) if manual is not None else inferred_offset, prior_history)


def schedule_due(update: int, frequency: int, offset: int = 0) -> bool:
    """Return the maintained positive-update frequency/offset schedule decision."""
    return update > offset and (update - offset) % frequency == 0


def init_wandb(
    logging: Any,
    training: Any,
    run_name: str,
    run_dir: Path,
    config: Any,
) -> Any | None:
    """Initialize W&B with the maintained online/offline/resume behavior."""
    mode = str(cfg_value(logging, "wandb_mode", "disabled"))
    if mode == "disabled":
        return None
    import os

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    kwargs: dict[str, Any] = {
        "project": cfg_value(logging, "wandb_project", "SwarmEcho"),
        "entity": cfg_value(logging, "wandb_entity"),
        "name": run_name,
        "mode": mode,
        "dir": str(run_dir),
        "config": config,
    }
    group = cfg_value(logging, "wandb_group")
    if group:
        kwargs["group"] = str(group)
    if cfg_value(training, "checkpoint_path"):
        wandb_dir = run_dir / "wandb"
        candidates = (
            sorted(
                (path for path in wandb_dir.iterdir() if path.is_dir() and path.name.startswith("run-")),
                key=lambda path: path.stat().st_mtime,
            )
            if wandb_dir.exists()
            else []
        )
        if candidates:
            parts = candidates[-1].name.split("-")
            if len(parts) >= 3:
                kwargs.update(id=parts[-1], resume="allow")
    import wandb

    return wandb.init(**kwargs)


def broadcast_early_stop_metrics(
    wandb_run: Any,
    evaluation: Any,
    eval_logs: dict[str, float],
    trigger_update: int,
    num_updates: int,
    steps_per_update: int,
    total_timesteps: int,
) -> None:
    """Carry an early-stop evaluation across remaining W&B schedule points."""
    if wandb_run is None or not eval_logs:
        return
    if not bool(cfg_value(evaluation, "eval_broadcast_on_curriculum_early_stop", True)):
        return
    frequency = max(1, int(cfg_value(evaluation, "eval_freq", 50)))
    offset = int(cfg_value(evaluation, "eval_offset", 1))
    future_steps = [
        min(update * steps_per_update, total_timesteps)
        for update in range(trigger_update + 1, num_updates + 1)
        if schedule_due(update, frequency, offset) or update == num_updates
    ]
    if not future_steps or future_steps[-1] != total_timesteps:
        future_steps.append(total_timesteps)
    payload = {**eval_logs, "eval/curriculum_early_stop_broadcast": 1.0}
    for step in sorted(set(future_steps)):
        wandb_run.log(payload, step=step)
