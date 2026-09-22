"""Persistent, sequential, noise-free evaluation across a fixed map suite."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import yaml

from swarmecho.core.config import MAP_DIR
from swarmecho.core.compatibility import resolve_map_file
from swarmecho.env.buildings import compile_building
from swarmecho.env.random_buildings import generate_building, map_seed, save_generated_map
from swarmecho.training.artifacts import artifact_suffix, save_eval_info_csv, write_manifest
from swarmecho.visualize.replay import building_snapshot, write_replay


def prepare_evaluation_maps(level, directory, *, log=lambda message: None):
    """Persist the small eval suite even when saving training maps is disabled.

    Reopening the same run loads its copies, independent of source-map edits.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for index in range(level.evaluation.random_eval_envs):
        name = f"random_eval_{index:04d}"
        target = directory / f"{name}.yaml"
        if target.exists():
            data = yaml.safe_load(target.read_text(encoding="utf-8"))
        elif level.evaluation.random_eval_maps is not None:
            source = resolve_map_file(level.evaluation.random_eval_maps[index], MAP_DIR)
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        else:
            data, _ = generate_building(level.random_buildings, map_seed(level.training.seed, index, evaluation=True),
                name=name, drone_clearance=level.env.drone_radius + level.env.obstacle_planning_clearance_m)
        building = compile_building(data)
        if not target.exists():
            save_generated_map(data, target)
        result.append(replace(level, map_names=[name], building=building,
            random_buildings=replace(level.random_buildings, enabled=False),
            evaluation=replace(level.evaluation, random_eval=False, training_robustness=False,
                               eval_differes_from_training_map=False, eval_map=None)))
        log(f"Persistent evaluation map {index + 1}/{level.evaluation.random_eval_envs}: {target.name}")
    return result


def evaluate_random_maps(model, levels, destination, *, update, steps, eval_due=True, replay_due=False, on_run=None,
                         artifact_root=None, scope="train", checkpoint=None, eval_name=None):
    from swarmecho.training.train import evaluate_suite, evaluate_model
    destination = Path(destination)
    artifact_root = Path(artifact_root) if artifact_root is not None else destination / "artifacts" / "train"
    suffix = artifact_suffix(update, steps)
    results = []
    for index, level in enumerate(levels):
        cfg, evaluation = level.env, level.evaluation
        name = level.building_name
        common = {"map_name": name, "random_eval_map_id": name,
                  "random_eval_group": f"{artifact_root}:{suffix}", "random_eval_maps": len(levels),
                  "world_size_m": level.building.world_size_m.tolist(),
                  "training_update": update, "environment_steps": steps, "artifact_scope": scope,
                  "action_noise_max": 0., "building_snapshot": building_snapshot(level.building),
                  **({"checkpoint": str(checkpoint)} if checkpoint is not None else {}),
                  **({"eval_name": eval_name} if eval_name else {})}
        if eval_due:
            metrics, info = evaluate_suite(model, level, episodes=evaluation.eval_parallel_envs,
                                          return_episode_info=True, action_noise_max=0.)
            results.append(metrics)
            if on_run:
                on_run(metrics, index + 1, len(levels))
            if evaluation.training_heatmap_creation:
                success = np.asarray(info["successes"], dtype=bool)
                delivered = np.asarray(info["delivered"], dtype=bool) | success
                visual = np.asarray(info["visually_found"], dtype=bool) | delivered
                path = save_eval_info_csv(artifact_root / "data" / f"eval_info_{suffix}_{name}.csv",
                    target_positions=info["target_positions"], base_positions=info["base_positions"],
                    final_chain_lengths=info["final_chain_lengths"],
                    stage_rates={"chain_success": success.astype(float), "found_and_delivered": delivered.astype(float),
                                 "visually_found": visual.astype(float)})
                write_manifest(path.with_suffix(".heatmap.json"), {**common,
                    "format": "swarmecho-eval-heatmap/v1", "data_file": path.name, "robustness_runs": 1})
        if replay_due:
            states, rewards = evaluate_model(model, level, max_steps=cfg.max_steps)
            write_replay(artifact_root / "replays" / f"eval_{suffix}_{name}",
                states, map_name=name, building=level.building, dt=cfg.dt, reward_terms=rewards, progress=False,
                metadata={**common, "cell_size_m": level.building.cell_size_m,
                    "coverage_voxel_size_m": cfg.coverage_voxel_size or level.building.cell_size_m,
                    "comm_radius_m": cfg.comm_radius, "comm_radius_base_m": cfg.comm_radius_base,
                    "visual_radius_m": cfg.visual_radius, "allow_redundancy_reward": level.reward.allow_redundancy_reward})
    if not results:
        return None
    aggregate = {key: float(np.mean([m[key] for m in results])) for key in results[0]}
    aggregate.update(eval_random_maps=len(levels), eval_robustness_runs=1)
    write_manifest(artifact_root / "manifests" / f"random_eval_{suffix}.json",
                   {"type": "random_map_evaluation", "update": update, "steps": steps,
                    "maps": [{"id": level.building_name, **metrics} for level, metrics in zip(levels, results)], **aggregate})
    return aggregate
