"""Helpers for SwarmEcho-owned evaluation artifact names and locations."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_checkpoint_update(name_or_path: str | Path) -> int | None:
    """Return the update encoded in checkpoint-like names such as ckpt_000700."""
    name = Path(name_or_path).name
    match = re.search(r"(?:ckpt(?:_early)?|update)_(\d+)", name)
    return int(match.group(1)) if match else None


def steps_for_update(update: int, cfg: Any) -> int:
    """Convert a PPO update number to environment steps using the run config."""
    envs = int(cfg.training.get("num_envs", 4000))
    rollout_steps = int(cfg.training.get("num_steps", 100))
    return int(update) * envs * rollout_steps


def compact_steps(steps: int) -> str:
    """Format step counts for artifact filenames, e.g. 70M -> s00070M."""
    steps = int(steps)
    if steps >= 1_000_000:
        return f"s{round(steps / 1_000_000):05d}M"
    if steps >= 1_000:
        return f"s{round(steps / 1_000):05d}k"
    return f"s{steps:05d}"


def artifact_suffix(update: int, steps: int) -> str:
    """Return the canonical SwarmEcho artifact suffix: u000700_s00070M."""
    return f"u{int(update):06d}_{compact_steps(steps)}"


def checkpoint_artifact_suffix(checkpoint_path: str | Path, cfg: Any) -> str:
    """Build the artifact suffix for a checkpoint path without renaming Orbax dirs."""
    update = parse_checkpoint_update(checkpoint_path)
    if update is None:
        update = 0
    steps = steps_for_update(update, cfg)
    history_path = Path(checkpoint_path) / "step_history.json"
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text())
            steps = int(data.get("total_steps", steps))
        except Exception:
            pass
    return artifact_suffix(update, steps)


def train_artifact_root(run_dir: str | Path) -> Path:
    return Path(run_dir) / "artifacts" / "train"


def eval_checkpoint_artifact_root(run_dir: str | Path, checkpoint_path: str | Path, cfg: Any) -> Path:
    return Path(run_dir) / "artifacts" / "eval" / f"ckpt_{checkpoint_artifact_suffix(checkpoint_path, cfg)}"


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_eval_info_csv(
    path: str | Path,
    target_positions: Any,
    base_positions: Any,
    successes: Any,
) -> Path:
    """Save compact, per-episode information from a parallel evaluation."""
    targets = np.asarray(target_positions, dtype=np.float32).reshape((-1, 2))
    bases = np.asarray(base_positions, dtype=np.float32)
    if bases.shape == (2,):
        bases = np.broadcast_to(bases, targets.shape)
    else:
        bases = bases.reshape((-1, 2))
    outcomes = np.asarray(successes, dtype=bool).reshape((-1,))

    if len(targets) != len(bases) or len(targets) != len(outcomes):
        raise ValueError(
            "Evaluation CSV arrays must contain the same number of targets, bases, and outcomes."
        )

    distances = np.linalg.norm(targets - bases, axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "success", "distance_to_base"])
        writer.writerows(
            (
                f"{target[0]:.6f}",
                f"{target[1]:.6f}",
                "true" if success else "false",
                f"{distance:.6f}",
            )
            for target, success, distance in zip(targets, outcomes, distances)
        )
    return path
