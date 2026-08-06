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


def evaluation_stage(
    *, success: bool, delivered: bool, visually_found: bool
) -> str:
    """Return the canonical CSV stage for one evaluation result."""
    if success:
        return "chain_success"
    if delivered:
        return "found_and_delivered"
    if visually_found:
        return "visually_found"
    return "not_found"


def parse_checkpoint_update(name_or_path: str | Path) -> int | None:
    """Return the update encoded in checkpoint-like names such as ckpt_000700."""
    name = Path(name_or_path).name
    match = re.search(r"(?:ckpt(?:_early)?|update)_(\d+)", name)
    return int(match.group(1)) if match else None


def steps_for_update(update: int, cfg: Any) -> int:
    """Convert a PPO update number to environment steps using the run config."""
    training = cfg.training
    if hasattr(training, "get"):
        envs = int(training.get("num_envs", 4000))
        rollout_steps = int(training.get("num_steps", 100))
    else:
        envs = int(getattr(training, "num_envs", 4000))
        rollout_steps = int(getattr(training, "num_steps", 100))
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


def train_replay_root(run_dir: str | Path) -> Path:
    """Return the 3D equivalent of the maintained training-video directory."""
    return train_artifact_root(run_dir) / "replays"


def eval_checkpoint_artifact_root(run_dir: str | Path, checkpoint_path: str | Path, cfg: Any) -> Path:
    return Path(run_dir) / "artifacts" / "eval" / f"ckpt_{checkpoint_artifact_suffix(checkpoint_path, cfg)}"


def eval_checkpoint_replay_root(
    run_dir: str | Path, checkpoint_path: str | Path, cfg: Any
) -> Path:
    """Return the replay directory for one checkpoint-scoped 3D evaluation."""
    return eval_checkpoint_artifact_root(run_dir, checkpoint_path, cfg) / "replays"


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
    targets = np.asarray(target_positions, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[-1] not in {2, 3}:
        raise ValueError("Evaluation targets must have shape (episodes, 2) or (episodes, 3).")
    dimensions = targets.shape[-1]
    bases = np.asarray(base_positions, dtype=np.float32)
    if bases.shape == (dimensions,):
        bases = np.broadcast_to(bases, targets.shape)
    else:
        bases = bases.reshape((-1, dimensions))
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

    stages = np.asarray(
        [
            evaluation_stage(
                success=bool(success),
                delivered=bool(is_delivered),
                visually_found=bool(was_visually_found),
            )
            for success, is_delivered, was_visually_found in zip(
                successes, delivered, visually_found, strict=True
            )
        ],
        dtype="<U21",
    )
    distances = np.linalg.norm(targets - bases, axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        coordinates = ["x", "y"] + (["z"] if dimensions == 3 else [])
        writer.writerow([*coordinates, "stage", "distance_to_base"])
        writer.writerows(
            # Nine significant digits round-trip float32 coordinates while
            # avoiding the precision loss that made manual replays diverge.
            tuple(f"{float(coordinate):.9g}" for coordinate in target)
            + (stage, f"{distance:.6f}")
            for target, stage, distance in zip(targets, stages, distances)
        )
    return path


def load_eval_info_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load one canonical evaluation CSV for artifact filtering or analysis."""
    path = Path(path)
    positions: list[tuple[float, ...]] = []
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
            point = (float(row["x"]), float(row["y"]))
            if "z" in (reader.fieldnames or []):
                point += (float(row["z"]),)
            positions.append(point)
            stages.append(stage)
            distances.append(float(row["distance_to_base"]))

    return {
        "positions": np.asarray(positions, dtype=np.float32).reshape(
            (-1, 3 if positions and len(positions[0]) == 3 else 2)
        ),
        "stages": np.asarray(stages, dtype="<U21"),
        "distance_to_base": np.asarray(distances, dtype=np.float32),
    }
