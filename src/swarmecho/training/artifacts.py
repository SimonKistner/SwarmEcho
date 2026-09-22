"""Helpers for SwarmEcho-owned evaluation artifact names and locations."""

from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from swarmecho.core.compatibility import canonical_marker


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


def evaluation_stage_from_rates(
    *,
    chain_success_rate: float,
    found_and_delivered_rate: float,
    visually_found_rate: float,
    confidence: float = 1.0,
) -> str:
    """Return the highest cumulative stage meeting ``confidence``."""
    if chain_success_rate >= confidence:
        return "chain_success"
    if found_and_delivered_rate >= confidence:
        return "found_and_delivered"
    if visually_found_rate >= confidence:
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
    """Return the equivalent of the maintained training-video directory."""
    return train_artifact_root(run_dir) / "replays"


def eval_checkpoint_artifact_root(run_dir: str | Path, checkpoint_path: str | Path, cfg: Any) -> Path:
    return Path(run_dir) / "artifacts" / "eval" / f"ckpt_{checkpoint_artifact_suffix(checkpoint_path, cfg)}"


def eval_checkpoint_replay_root(
    run_dir: str | Path, checkpoint_path: str | Path, cfg: Any
) -> Path:
    """Return the replay directory for one checkpoint-scoped evaluation."""
    return eval_checkpoint_artifact_root(run_dir, checkpoint_path, cfg) / "replays"


_EVAL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def create_eval_run_root(
    run_dir: str | Path,
    checkpoint_path: str | Path,
    cfg: Any,
    *,
    eval_name: str | None = None,
    timestamp_ns: int | None = None,
) -> Path:
    """Create one isolated manual-evaluation folder for a checkpoint.

    The normal layout is ``eval_<UTC timestamp>``. A caller-provided label
    produces ``eval_<label>_<UTC timestamp>``. The directory is created with
    exclusive semantics so every new evaluation has a distinct destination.
    ``timestamp_ns`` is only an injection point for deterministic tests.
    """
    if eval_name is not None and not _EVAL_NAME_PATTERN.fullmatch(eval_name):
        raise ValueError(
            "eval_name must contain only letters, numbers, underscores, and hyphens."
        )

    value_ns = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y%m%dT%H%M%S")
    timestamp = f"{stamp}_{nanoseconds:09d}Z"
    prefix = "eval" if eval_name is None else f"eval_{eval_name}"
    parent = eval_checkpoint_artifact_root(run_dir, checkpoint_path, cfg)

    for suffix in range(1_000):
        postfix = "" if suffix == 0 else f"_{suffix}"
        candidate = parent / f"{prefix}_{timestamp}{postfix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique evaluation-run directory.")


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_eval_layout(path: str | Path, obstacle_min: Any, obstacle_max: Any) -> Path:
    """Store one evaluation layout once instead of repeating it in every CSV row."""
    lower = np.asarray(obstacle_min, dtype=np.float32)
    upper = np.asarray(obstacle_max, dtype=np.float32)
    if lower.ndim == 3:
        if not (np.all(lower == lower[:1]) and np.all(upper == upper[:1])):
            raise ValueError("Evaluation layout must be identical for every episode.")
        lower, upper = lower[0], upper[0]
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[-1] != 3:
        raise ValueError("Evaluation layout bounds must have shape (obstacles, 3).")
    output = Path(path)
    write_manifest(output, {
        "format": "swarmecho-obstacle-layout/v1",
        "obstacle_min": lower.tolist(),
        "obstacle_max": upper.tolist(),
    })
    return output


def load_eval_layout(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if canonical_marker(payload.get("format")) != "swarmecho-obstacle-layout/v1":
        raise ValueError(f"Unsupported obstacle layout: {path}")
    return (
        np.asarray(payload["obstacle_min"], dtype=np.float32),
        np.asarray(payload["obstacle_max"], dtype=np.float32),
    )


def save_eval_info_csv(
    path: str | Path,
    target_positions: Any,
    base_positions: Any,
    successes: Any = None,
    delivered: Any = None,
    visually_found: Any = None,
    *,
    stage_rates: dict[str, Any] | None = None,
    final_chain_lengths: Any = None,
    obstacle_min: Any = None,
    obstacle_max: Any = None,
) -> Path:
    """Save the canonical per-episode result of a parallel evaluation."""
    targets = np.asarray(target_positions, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[-1] != 3:
        raise ValueError("Evaluation targets must have shape (episodes, 3).")
    dimensions = targets.shape[-1]
    bases = np.asarray(base_positions, dtype=np.float32)
    if bases.shape == (dimensions,):
        bases = np.broadcast_to(bases, targets.shape)
    else:
        bases = bases.reshape((-1, dimensions))
    robust_rates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if stage_rates is not None:
        required_rates = {
            "chain_success",
            "found_and_delivered",
            "visually_found",
        }
        if set(stage_rates) != required_rates:
            raise ValueError(
                f"stage_rates must contain exactly {sorted(required_rates)}."
            )
        chain_rates = np.asarray(stage_rates["chain_success"], dtype=np.float32).reshape((-1,))
        delivered_rates = np.asarray(
            stage_rates["found_and_delivered"], dtype=np.float32
        ).reshape((-1,))
        visually_found_rates = np.asarray(
            stage_rates["visually_found"], dtype=np.float32
        ).reshape((-1,))
        if not (
            len(targets)
            == len(bases)
            == len(chain_rates)
            == len(delivered_rates)
            == len(visually_found_rates)
        ):
            raise ValueError(
                "Evaluation CSV arrays must contain one target, base, and rate "
                "for every evaluated episode."
            )
        if (
            np.any(visually_found_rates < 0.0)
            or np.any(visually_found_rates > 1.0)
            or np.any(delivered_rates < 0.0)
            or np.any(delivered_rates > visually_found_rates)
            or np.any(chain_rates < 0.0)
            or np.any(chain_rates > delivered_rates)
        ):
            raise ValueError(
                "Stage rates must satisfy 0 <= chain <= delivered <= visually_found <= 1."
            )
        robust_rates = chain_rates, delivered_rates, visually_found_rates
    else:
        if successes is None or delivered is None or visually_found is None:
            raise ValueError(
                "Legacy evaluation CSV output requires successes, delivered, "
                "and visually_found arrays."
            )
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
    chain_lengths = (
        np.zeros(len(targets), dtype=np.float32)
        if final_chain_lengths is None
        else np.asarray(final_chain_lengths, dtype=np.float32).reshape((-1,))
    )
    if len(chain_lengths) != len(targets):
        raise ValueError("final_chain_lengths must contain one value per episode.")
    obstacle_mins = np.asarray(obstacle_min if obstacle_min is not None else np.empty((len(targets), 0, 3)), dtype=np.float32)
    obstacle_maxs = np.asarray(obstacle_max if obstacle_max is not None else np.empty((len(targets), 0, 3)), dtype=np.float32)
    if obstacle_mins.ndim == 2:
        obstacle_mins = np.broadcast_to(obstacle_mins, (len(targets), *obstacle_mins.shape))
        obstacle_maxs = np.broadcast_to(obstacle_maxs, (len(targets), *obstacle_maxs.shape))
    if obstacle_mins.shape != obstacle_maxs.shape or obstacle_mins.shape[:1] != (len(targets),) or obstacle_mins.shape[-1:] != (3,):
        raise ValueError("Obstacle bounds must have shape (episodes, obstacles, 3).")
    obstacle_columns = [
        f"obstacle_{index}_{bound}_{axis}"
        for index in range(obstacle_mins.shape[1])
        for bound in ("min", "max")
        for axis in "xyz"
    ]

    def obstacle_values(index):
        return tuple(
            f"{float(value):.9g}"
            for obstacle_index in range(obstacle_mins.shape[1])
            for values in (obstacle_mins[index, obstacle_index], obstacle_maxs[index, obstacle_index])
            for value in values
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        coordinates = ["x", "y", "z"]
        if robust_rates is None:
            writer.writerow([*coordinates, "stage", "distance_to_base", "final_chain_length", *obstacle_columns])
            writer.writerows(
                # Nine significant digits round-trip float32 coordinates while
                # avoiding the precision loss that made manual replays diverge.
                tuple(f"{float(coordinate):.9g}" for coordinate in target)
                + (stage, f"{distance:.6f}", f"{chain_lengths[index]:.6f}")
                + obstacle_values(index)
                for index, (target, stage, distance) in enumerate(zip(targets, stages, distances))
            )
        else:
            chain_rates, delivered_rates, visually_found_rates = robust_rates
            writer.writerow(
                [
                    *coordinates,
                    "visually_found_rate",
                    "found_and_delivered_rate",
                    "chain_success_rate",
                    "distance_to_base",
                    "final_chain_length",
                    *obstacle_columns,
                ]
            )
            writer.writerows(
                tuple(f"{float(coordinate):.9g}" for coordinate in target)
                + (
                    f"{float(visual_rate):.6f}",
                    f"{float(delivered_rate):.6f}",
                    f"{float(chain_rate):.6f}",
                    f"{distance:.6f}",
                    f"{chain_lengths[index]:.6f}",
                )
                + obstacle_values(index)
                for index, (target, visual_rate, delivered_rate, chain_rate, distance) in enumerate(zip(
                    targets,
                    visually_found_rates,
                    delivered_rates,
                    chain_rates,
                    distances,
                    strict=True,
                ))
            )
    return path


def load_eval_info_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load one canonical evaluation CSV for artifact filtering or analysis."""
    path = Path(path)
    positions: list[tuple[float, ...]] = []
    stages: list[str] = []
    chain_success_rates: list[float] = []
    found_and_delivered_rates: list[float] = []
    visually_found_rates: list[float] = []
    distances: list[float] = []
    final_chain_lengths: list[float] = []
    obstacle_mins: list[list[list[float]]] = []
    obstacle_maxs: list[list[list[float]]] = []
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or [])
        base_required = {"x", "y", "z", "distance_to_base"}
        rate_required = {
            "visually_found_rate",
            "found_and_delivered_rate",
            "chain_success_rate",
        }
        has_legacy_stage = "stage" in fields
        has_rates = rate_required.issubset(fields)
        obstacle_indices = sorted({
            int(match.group(1))
            for field in fields
            if (match := re.fullmatch(r"obstacle_(\d+)_min_x", field))
        })
        if not base_required.issubset(fields) or not (has_legacy_stage or has_rates):
            raise ValueError(
                "Evaluation CSV must contain coordinates, distance, and either "
                f"stage or {sorted(rate_required)}: {path}"
            )
        for row in reader:
            if has_rates:
                visual_rate = float(row["visually_found_rate"])
                delivered_rate = float(row["found_and_delivered_rate"])
                chain_rate = float(row["chain_success_rate"])
                stage = evaluation_stage_from_rates(
                    chain_success_rate=chain_rate,
                    found_and_delivered_rate=delivered_rate,
                    visually_found_rate=visual_rate,
                )
            else:
                stage = str(row["stage"])
                if stage not in EVAL_STAGES:
                    raise ValueError(
                        f"Unknown evaluation stage {stage!r} in {path}"
                    )
                chain_rate = float(stage == "chain_success")
                delivered_rate = float(
                    stage in {"found_and_delivered", "chain_success"}
                )
                visual_rate = float(stage != "not_found")
            point = (float(row["x"]), float(row["y"]), float(row["z"]))
            positions.append(point)
            stages.append(stage)
            chain_success_rates.append(chain_rate)
            found_and_delivered_rates.append(delivered_rate)
            visually_found_rates.append(visual_rate)
            distances.append(float(row["distance_to_base"]))
            final_chain_lengths.append(float(row.get("final_chain_length") or 0.0))
            obstacle_mins.append([
                [float(row[f"obstacle_{index}_min_{axis}"]) for axis in "xyz"]
                for index in obstacle_indices
            ])
            obstacle_maxs.append([
                [float(row[f"obstacle_{index}_max_{axis}"]) for axis in "xyz"]
                for index in obstacle_indices
            ])

    if obstacle_indices:
        obstacle_min_array = np.asarray(obstacle_mins, dtype=np.float32).reshape(
            (len(positions), len(obstacle_indices), 3)
        )
        obstacle_max_array = np.asarray(obstacle_maxs, dtype=np.float32).reshape(
            (len(positions), len(obstacle_indices), 3)
        )
    else:
        # ``reshape((-1, 0, 3))`` cannot infer the leading dimension from an
        # empty array. Preserve one empty obstacle axis for every CSV episode.
        obstacle_min_array = np.empty((len(positions), 0, 3), dtype=np.float32)
        obstacle_max_array = np.empty((len(positions), 0, 3), dtype=np.float32)

    return {
        "positions": np.asarray(positions, dtype=np.float32).reshape(
            (-1, 3)
        ),
        "stages": np.asarray(stages, dtype="<U21"),
        "chain_success_rate": np.asarray(chain_success_rates, dtype=np.float32),
        "found_and_delivered_rate": np.asarray(
            found_and_delivered_rates, dtype=np.float32
        ),
        "visually_found_rate": np.asarray(visually_found_rates, dtype=np.float32),
        "distance_to_base": np.asarray(distances, dtype=np.float32),
        "final_chain_length": np.asarray(final_chain_lengths, dtype=np.float32),
        "obstacle_min": obstacle_min_array,
        "obstacle_max": obstacle_max_array,
    }
