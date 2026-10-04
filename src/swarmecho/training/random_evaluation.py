"""Persistent, sequential, noise-free evaluation across a fixed map suite."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import yaml

from swarmecho.core.config import MAP_DIR, resolve_evaluation_level
from swarmecho.core.compatibility import resolve_map_file
from swarmecho.core.terminal import terminal_print
from swarmecho.env.buildings import compile_building, target_spawn_boxes
from swarmecho.env.random_buildings import generate_building, map_seed, save_generated_map
from swarmecho.training.artifacts import artifact_suffix, save_eval_info_csv, write_manifest
from swarmecho.visualize.replay import building_snapshot, planner_metadata, write_replay


def prepare_evaluation_maps(level, directory, *, bank_maps=None, static_maps=None, log=lambda message: None):
    """Persist the small eval suite even when saving training maps is disabled.

    Reopening the same run loads its copies, independent of source-map edits.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if static_maps is not None and bank_maps is not None:
        raise ValueError("An evaluation suite cannot use both static and random bank maps.")
    count = len(static_maps) if static_maps is not None else level.evaluation.random_eval_envs
    prefix = "static_eval" if static_maps is not None else "random_eval"
    if (bank_maps is None and level.random_buildings.enabled
            and getattr(level.random_buildings, "load_maps_from_bank", False)
            and level.evaluation.random_eval_maps is None
            and any(not (directory / f"random_eval_{index:04d}.yaml").exists()
                    for index in range(level.evaluation.random_eval_envs))):
        from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
            DEFAULT_POOL_SIZE, build_pool, ensure_pool, select_maps)
        count = level.evaluation.random_eval_envs
        pool, metadata = ensure_pool(level, level.training.num_envs + count, log=log)
        _, selection = select_maps(pool, metadata, level.training.seed, level.training.num_envs, count)
        bank_maps = [pool / "maps" / f"{item['id']}.yaml" for item in selection]
        if any(not source.is_file() for source in bank_maps):
            terminal_print(f"[BANK WARNING] Selected evaluation map is missing from {pool}; generating a new pool.")
            pool, metadata = build_pool(level, count=max(DEFAULT_POOL_SIZE, level.training.num_envs + count),
                                        avoid_pool=pool, log=log)
            _, selection = select_maps(pool, metadata, level.training.seed, level.training.num_envs, count)
            bank_maps = [pool / "maps" / f"{item['id']}.yaml" for item in selection]
    if bank_maps is not None and len(bank_maps) != level.evaluation.random_eval_envs:
        raise ValueError("bank_maps must contain exactly random_eval_envs maps.")
    result = []
    for index in range(count):
        name = f"{prefix}_{index:04d}"
        target = directory / f"{name}.yaml"
        if target.exists():
            data = yaml.safe_load(target.read_text(encoding="utf-8"))
        elif static_maps is not None:
            data = yaml.safe_load(Path(static_maps[index]).read_text(encoding="utf-8"))
            data["name"] = name
        elif level.evaluation.random_eval_maps is not None:
            source = resolve_map_file(level.evaluation.random_eval_maps[index], MAP_DIR)
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        elif bank_maps is not None:
            data = yaml.safe_load(Path(bank_maps[index]).read_text(encoding="utf-8"))
            data["name"] = name
        else:
            data, _ = generate_building(level.random_buildings, map_seed(level.training.seed, index, evaluation=True),
                name=name, drone_clearance=level.env.drone_radius + level.env.obstacle_planning_clearance_m)
        building = compile_building(data)
        if not target.exists():
            save_generated_map(data, target)
        result.append(replace(level, map_names=[name], map_paths=(target,), building=building,
            map_source_name=level.map_names[index] if static_maps is not None else None,
            random_buildings=replace(level.random_buildings, enabled=False),
            evaluation=replace(level.evaluation, random_eval=False, training_robustness=False,
                               eval_differes_from_training_map=False, eval_map=None)))
        log(f"Persistent evaluation map {index + 1}/{count}: {target.name}")
    return result


def _replay_level_for_storey(level):
    """Limit only replay target sampling to one 1-based building storey."""
    storey = level.evaluation.replay_targets_from_storey
    if storey is None:
        return level
    building = level.building
    layers = building.target_exclusion.shape[2]
    if storey > layers:
        raise ValueError(
            f"evaluation.replay_targets_from_storey={storey} exceeds the "
            f"{layers} storeys in evaluation map {level.building_name!r}."
        )
    excluded = np.ones_like(building.target_exclusion, dtype=np.bool_)
    excluded[:, :, storey - 1] = building.target_exclusion[:, :, storey - 1]
    replay_building = replace(building, target_exclusion=excluded)
    try:
        target_spawn_boxes(replay_building, level.env)
    except ValueError as error:
        raise ValueError(
            f"Evaluation map {level.building_name!r} has no valid target spawn "
            f"volume on storey {storey}."
        ) from error
    return replace(level, building=replay_building)


def evaluate_map_suite(model, levels, destination, *, update, steps, eval_due=True, replay_due=False, on_run=None,
                       artifact_root=None, scope="train", checkpoint=None, eval_name=None):
    from swarmecho.training.train import evaluate_suite, evaluate_model
    destination = Path(destination)
    artifact_root = Path(artifact_root) if artifact_root is not None else destination / "artifacts" / "train"
    suffix = artifact_suffix(update, steps)
    results = []
    static = levels[0].building_name.startswith("static_eval_")
    for index, level in enumerate(levels):
        level = resolve_evaluation_level(level)
        cfg, evaluation = level.env, level.evaluation
        name = level.building_name
        suite_metadata = ({"map_suite_map_id": name, "map_suite_group": f"{artifact_root}:{suffix}",
                           "map_suite_maps": len(levels)} if static else
                          {"random_eval_map_id": name, "random_eval_group": f"{artifact_root}:{suffix}",
                           "random_eval_maps": len(levels)})
        common = {"map_name": name, "num_agents": cfg.num_agents, **suite_metadata,
                  **({"source_map_name": level.map_source_name} if level.map_source_name else {}),
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
                chain_success = success & (resolve_evaluation_level(level).env.success_condition == "chain_held")
                delivered = np.asarray(info["delivered"], dtype=bool) | chain_success
                visual = np.asarray(info["visually_found"], dtype=bool) | delivered
                path = save_eval_info_csv(artifact_root / "data" / f"eval_info_{suffix}_{name}.csv",
                    target_positions=info["target_positions"], base_positions=info["base_positions"],
                    final_chain_lengths=info["final_chain_lengths"],
                    stage_rates={"chain_success": chain_success.astype(float), "found_and_delivered": delivered.astype(float),
                                 "visually_found": visual.astype(float)})
                write_manifest(path.with_suffix(".heatmap.json"), {**common,
                    "format": "swarmecho-eval-heatmap/v1", "data_file": path.name, "robustness_runs": 1,
                    "success_condition": resolve_evaluation_level(level).env.success_condition})
        if replay_due:
            replay_level = _replay_level_for_storey(level)
            states, rewards = evaluate_model(model, replay_level, max_steps=cfg.max_steps)
            write_replay(artifact_root / "replays" / f"eval_{suffix}_{name}",
                states, map_name=name, building=level.building, dt=cfg.dt, reward_terms=rewards, progress=False,
                metadata={**common, **planner_metadata(level.env), "cell_size_m": level.building.cell_size_m,
                    "replay_targets_from_storey": evaluation.replay_targets_from_storey,
                    "coverage_voxel_size_m": cfg.coverage_voxel_size or level.building.cell_size_m,
                    "comm_radius_m": cfg.comm_radius, "comm_radius_base_m": cfg.comm_radius_base,
                    "visual_radius_m": cfg.visual_radius, "allow_redundancy_reward": level.reward.allow_redundancy_reward,
                    "success_condition": resolve_evaluation_level(level).env.success_condition})
    if not results:
        return None
    aggregate = {key: float(np.mean([m[key] for m in results])) for key in results[0]}
    successful_count = sum(m["eval_success_episode_count"] for m in results)
    successful_length = (sum(m["eval_success_episode_length"] * m["eval_success_episode_count"]
                             for m in results) / successful_count if successful_count
                         else float(levels[0].env.max_steps))
    aggregate.update(eval_success_episode_count=successful_count,
                     eval_success_episode_length=successful_length,
                     eval_success_length_reduction=1.0 - successful_length / levels[0].env.max_steps)
    aggregate.update({"eval_static_maps" if static else "eval_random_maps": len(levels),
                      "eval_robustness_runs": 1})
    write_manifest(artifact_root / "manifests" / f"{'static_eval' if static else 'random_eval'}_{suffix}.json",
                   {"type": "static_map_evaluation" if static else "random_map_evaluation",
                    "update": update, "steps": steps,
                    "maps": [{"id": level.building_name, **metrics} for level, metrics in zip(levels, results)], **aggregate})
    return aggregate


evaluate_random_maps = evaluate_map_suite
