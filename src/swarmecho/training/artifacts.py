"""Helpers for SwarmEcho-owned evaluation artifact names and locations."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


EVAL_STAGES = (
    "not_found",
    "visually_found",
    "found_and_delivered",
    "chain_success",
)


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
    delivered: Any,
    visually_found: Any,
) -> Path:
    """Save the canonical per-episode result of a parallel evaluation."""
    targets = np.asarray(target_positions, dtype=np.float32).reshape((-1, 2))
    bases = np.asarray(base_positions, dtype=np.float32)
    if bases.shape == (2,):
        bases = np.broadcast_to(bases, targets.shape)
    else:
        bases = bases.reshape((-1, 2))
    successes = np.asarray(successes, dtype=bool).reshape((-1,))
    delivered = np.asarray(delivered, dtype=bool).reshape((-1,))
    visually_found = np.asarray(visually_found, dtype=bool).reshape((-1,))

    if not (
        len(targets)
        == len(bases)
        == len(successes)
        == len(delivered)
        == len(visually_found)
    ):
        raise ValueError(
            "Evaluation CSV arrays must contain one target, base, and outcome "
            "for every evaluated episode."
        )

    stages = np.full(len(successes), "not_found", dtype="<U21")
    stages[visually_found] = "visually_found"
    stages[delivered] = "found_and_delivered"
    stages[successes] = "chain_success"
    distances = np.linalg.norm(targets - bases, axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "stage", "distance_to_base"])
        writer.writerows(
            (
                f"{target[0]:.6f}",
                f"{target[1]:.6f}",
                stage,
                f"{distance:.6f}",
            )
            for target, stage, distance in zip(targets, stages, distances)
        )
    return path


def load_eval_info_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load one canonical evaluation CSV for artifact filtering or analysis."""
    path = Path(path)
    positions: list[tuple[float, float]] = []
    stages: list[str] = []
    distances: list[float] = []
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"x", "y", "stage", "distance_to_base"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Evaluation CSV must contain {sorted(required)}: {path}"
            )
        for row in reader:
            stage = str(row["stage"])
            if stage not in EVAL_STAGES:
                raise ValueError(
                    f"Unknown evaluation stage {stage!r} in {path}"
                )
            positions.append((float(row["x"]), float(row["y"])))
            stages.append(stage)
            distances.append(float(row["distance_to_base"]))

    return {
        "positions": np.asarray(positions, dtype=np.float32).reshape((-1, 2)),
        "stages": np.asarray(stages, dtype="<U21"),
        "distance_to_base": np.asarray(distances, dtype=np.float32),
    }
