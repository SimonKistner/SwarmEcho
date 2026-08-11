"""Small, dependency-light helpers for training orchestration commands."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


def pop_arg(args: list[str], key: str, default):
    """Remove one ``key=value`` argument and return its typed value."""
    value = default
    remaining: list[str] = []
    for arg in args:
        if arg.startswith(f"{key}="):
            raw_value = arg.split("=", 1)[1]
            try:
                value = type(default)(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a valid {type(default).__name__}.") from exc
        else:
            remaining.append(arg)
    return value, remaining


def generate_unique_seeds(count: int, base_seed: int) -> list[int]:
    """Return a reproducible, sorted batch of unique positive training seeds."""
    if count < 1:
        raise ValueError("seeds must be at least 1.")
    rng = random.Random(base_seed)
    seeds: set[int] = set()
    while len(seeds) < count:
        seeds.add(rng.randint(1, 2**31 - 1))
    return sorted(seeds)


def split_levels(
    args: list[str], default: str
) -> tuple[list[str], list[str]]:
    """Extract a comma-separated ``levels=`` list from CLI overrides."""
    levels = [default]
    remaining: list[str] = []
    for arg in args:
        if arg.startswith("levels="):
            levels = [item.strip() for item in arg.split("=", 1)[1].split(",")]
            levels = [item for item in levels if item]
            if not levels:
                raise ValueError("levels must contain at least one level name.")
        else:
            remaining.append(arg)
    return levels, remaining


def explicit_override(args: list[str], key: str) -> str | None:
    """Return the last value supplied for a dotlist key, if present."""
    prefix = f"{key}="
    values = [arg[len(prefix):] for arg in args if arg.startswith(prefix)]
    return values[-1] if values else None


def checkpoint_total_steps(
    checkpoint: str | Path | None,
    *,
    num_envs: int,
    num_steps: int,
) -> int:
    """Read cumulative steps from a checkpoint, with a name-based fallback."""
    if not checkpoint:
        return 0
    path = Path(str(checkpoint).replace("\\", "/"))
    history_path = path / "step_history.json"
    if history_path.exists():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            return max(0, int(payload.get("total_steps", 0)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    match = re.search(r"ckpt_(?:early_|final_)?(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) * num_envs * num_steps if match else 0
