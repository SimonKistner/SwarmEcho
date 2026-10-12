"""Create short inspector replays across C maps using one existing checkpoint."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

from swarmecho.core.config import load_level, resolve_evaluation_level
from swarmecho.core.paths import normalize_wsl_path
from swarmecho.core.terminal import terminal_print
from swarmecho.env.buildings import load_building
from swarmecho.training.artifacts import (
    create_eval_run_root, parse_checkpoint_update, steps_for_checkpoint, write_manifest,
)
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.orchestration import representative_map_indices
from swarmecho.training.train import build_model, evaluate_model
from swarmecho.visualize.replay import building_snapshot, planner_metadata, write_replay


C_LEVELS = ("C01_linear_hallway", "C02_corner_hallway", "C03_space_station",
            "C05b_cellar")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=2, help="Maximum maps per level (default: 2).")
    parser.add_argument("--all-maps", action="store_true", help="Replay every alternative instead of sampling.")
    parser.add_argument("--max-steps", type=int, default=100, help="Short geometry inspection horizon (default: 100).")
    parser.add_argument("--levels", default=",".join(C_LEVELS))
    parser.add_argument("--seed", type=int, default=35461354)
    parser.add_argument("--wall-sensing-layout", choices=("fibonacci", "xyz"), default=None,
                        help="Override the levels' wall layout to match the checkpoint.")
    parser.add_argument("--wall-sensing-rays", type=int, default=None,
                        help="Override the wall ray count to match the checkpoint (XYZ: 14, 18, 26).")
    parser.add_argument("--wall-sensing-ray-max-distance", type=float, default=None,
                        help="Override wall range in metres to match the checkpoint.")
    args = parser.parse_args()
    if args.samples < 1 or args.max_steps < 1:
        parser.error("--samples and --max-steps must be positive.")
    names = [name.strip() for name in args.levels.split(",") if name.strip()]
    if not names:
        parser.error("--levels must contain at least one level.")
    checkpoint = normalize_wsl_path(args.checkpoint).absolute()
    if not checkpoint.is_dir():
        parser.error("Checkpoint directory does not exist: %s" % checkpoint)
    wall_overrides = [f"env.{name}={value}" for name, value in (
        ("wall_sensing_layout", args.wall_sensing_layout),
        ("wall_sensing_rays", args.wall_sensing_rays),
        ("wall_sensing_ray_max_distance", args.wall_sensing_ray_max_distance),
    ) if value is not None]
    levels = [load_level(name, wall_overrides) for name in names]
    terminal_print(f"[PREVIEW] restoring {checkpoint}; generating sampled C geometry replays.", flush=True)
    model = build_model(levels[0])
    restore_model_checkpoint(model, checkpoint)
    run_dir = checkpoint.parents[1]
    root = create_eval_run_root(run_dir, checkpoint, levels[0], eval_name="C_geometry")
    selections = [(level, representative_map_indices(
        len(level.map_paths), None if args.all_maps else args.samples)) for level in levels]
    suite_metadata = {"map_suite_group": str(root),
                      "map_suite_maps": sum(len(selected) for _, selected in selections)}
    index = {"type": "curriculum_geometry_preview", "checkpoint": str(checkpoint),
             "max_steps": args.max_steps, **suite_metadata, "replays": []}
    for level, selected in selections:
        for map_index in selected:
            source = level.map_paths[map_index]
            single = resolve_evaluation_level(replace(
                level, map_names=[source.stem], map_paths=(source,), building=load_building(source),
            ))
            seed = args.seed + len(index["replays"])
            terminal_print(f"[PREVIEW] {level.name}: {source.stem}; {single.env.num_agents} agent cap", flush=True)
            states, rewards = evaluate_model(model, single, max_steps=args.max_steps, seed=seed)
            output = root / "replays" / (level.name + "__" + source.stem)
            write_replay(output, states, map_name=source.stem, building=single.building,
                dt=single.env.dt, reward_terms=rewards, progress=False, metadata={
                    **planner_metadata(single.env), "building_snapshot": building_snapshot(single.building),
                    "world_size_m": single.building.world_size_m.tolist(),
                    "cell_size_m": single.building.cell_size_m,
                    "coverage_voxel_size_m": single.env.coverage_voxel_size or single.building.cell_size_m,
                    "comm_radius_m": single.env.comm_radius, "comm_radius_base_m": single.env.comm_radius_base,
                    "visual_radius_m": single.env.visual_radius, "num_agents": single.env.num_agents,
                    "allow_redundancy_reward": single.reward.allow_redundancy_reward,
                    "success_condition": single.env.success_condition,
                    "checkpoint": str(checkpoint), "level_name": level.name,
                    "source_map_name": source.stem, "source_map_path": str(source),
                    "training_update": parse_checkpoint_update(checkpoint) or 0,
                    "environment_steps": steps_for_checkpoint(checkpoint, single),
                    "artifact_scope": "eval", "eval_name": "C_geometry", "seed": seed,
                    **suite_metadata, "map_suite_map_id": output.stem,
                    "geometry_preview": True,
                })
            index["replays"].append({"level": level.name, "map": source.stem,
                                     "path": str(output.with_suffix(".json").relative_to(root)), "seed": seed})
            write_manifest(root / "manifests" / "C_geometry.json", index)
    terminal_print(f"[PREVIEW] {len(index['replays'])} inspector replays saved to {root}", flush=True)


if __name__ == "__main__":
    main()
