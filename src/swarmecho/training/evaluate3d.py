"""Checkpoint evaluation entry point for the deterministic 3D policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmecho.core.config3d import load_level_3d
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train3d import build_model_3d, evaluate_model_3d
from swarmecho.visualize.replay3d import write_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--level", default="M00_no_maze_open_cuboid_3D")
    parser.add_argument("--output", type=Path, default=Path("outputs/3d_evaluation/latest"))
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    level = load_level_3d(args.level)
    model = build_model_3d(level)
    restore_model_checkpoint(model, args.checkpoint)
    states, rewards = evaluate_model_3d(model, level, max_steps=args.max_steps)
    _, manifest = write_replay(
        args.output,
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
        },
    )
    print(f"3D evaluation replay: {manifest}")


if __name__ == "__main__":
    main()
