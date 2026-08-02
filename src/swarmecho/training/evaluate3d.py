"""Checkpoint evaluation entry point for the deterministic 3D policy."""

from __future__ import annotations

import sys
from pathlib import Path

from swarmecho.core.config3d import load_level_3d_cli
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_replay_root,
)
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train3d import build_model_3d, evaluate_model_3d
from swarmecho.visualize.replay3d import write_replay


def main() -> None:
    checkpoint: Path | None = None
    output: Path | None = None
    max_steps: int | None = None
    config_arguments: list[str] = []
    for argument in sys.argv[1:]:
        if "=" not in argument:
            raise ValueError("Use checkpoint=<path> and key=value overrides.")
        key, value = argument.split("=", 1)
        if key == "checkpoint":
            checkpoint = Path(value.replace("\\", "/")).absolute()
        elif key == "output":
            output = Path(value)
        elif key == "max_steps":
            max_steps = int(value)
        else:
            config_arguments.append(argument)
    if checkpoint is None:
        raise ValueError("Specify checkpoint=<path>.")
    level = load_level_3d_cli(config_arguments)
    run_dir = checkpoint.parents[1]
    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    output = output or (
        eval_checkpoint_replay_root(run_dir, checkpoint, level) / f"eval_{artifact_tag}"
    )
    model = build_model_3d(level)
    restore_model_checkpoint(model, checkpoint)
    states, rewards = evaluate_model_3d(model, level, max_steps=max_steps)
    _, manifest = write_replay(
        output,
        states,
        map_name=level.building_name,
        dt=level.env.dt,
        reward_terms=rewards,
        metadata={
            "world_size_m": level.building.world_size_m.tolist(),
            "cell_size_m": level.building.cell_size_m,
            "comm_radius_m": level.env.comm_radius,
            "comm_radius_base_m": level.env.comm_radius_base,
            "visual_radius_m": level.env.visual_radius,
            "checkpoint": str(checkpoint),
            "artifact_scope": "eval",
        },
    )
    print(f"3D evaluation replay: {manifest}")


if __name__ == "__main__":
    main()
