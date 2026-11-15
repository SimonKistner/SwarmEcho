"""Persistent, sequential, noise-free evaluation across a fixed map suite."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import yaml

from swarmecho.core.config import CHAIN_SUCCESS_CONDITIONS, MAP_DIR, resolve_evaluation_level
from swarmecho.core.compatibility import resolve_map_file
from swarmecho.core.terminal import terminal_print
from swarmecho.env.buildings import compile_building, target_spawn_boxes
from swarmecho.env.random_buildings import generate_building, map_seed, save_generated_map
from swarmecho.training.artifacts import (
    artifact_suffix, parse_checkpoint_update, save_eval_info_csv, steps_for_checkpoint, write_manifest,
)
from swarmecho.visualize.replay import building_snapshot, planner_metadata, write_replay
from swarmecho.training.orchestration import representative_map_indices
from swarmecho.training.information_metrics import (
    EVALUATION_INFORMATION_KEYS, pool_evaluation_information,
)


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
    if bank_maps is None and static_maps is None and level.random_buildings.enabled:
        from swarmecho.training.map_selection import load_selection
        selection = load_selection(directory.parent, level.training.checkpoint_path)
        if selection is not None and selection.get("kind") == "bank":
            pool = Path(selection["pool"])
            bank_maps = [pool / "maps" / f"{item['id']}.yaml"
                         for item in selection["evaluation_maps"][:count]]
        elif level.validation.enabled and selection is not None and selection.get("kind") == "generated":
            from swarmecho.training.validation import prepare_generated_validation_maps
            bank_maps = prepare_generated_validation_maps(level, directory.parent, log)[:count]
        elif level.validation.enabled:
            raise ValueError("Inspection requires this run's frozen validation selection; refusing a new split.")
    if (bank_maps is None and level.random_buildings.enabled
            and getattr(level.random_buildings, "load_maps_from_bank", False)
            and level.evaluation.random_eval_maps is None
            and any(not (directory / f"random_eval_{index:04d}.yaml").exists()
                    for index in range(level.evaluation.random_eval_envs))):
        from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
            DEFAULT_POOL_SIZE, build_pool, ensure_pool, select_maps)
        count = level.evaluation.random_eval_envs
        pool, metadata = ensure_pool(level, level.training_map_count + count, log=log)
        _, selection = select_maps(pool, metadata, level.training.bank_selection_seed, level.training_map_count, count)
        bank_maps = [pool / "maps" / f"{item['id']}.yaml" for item in selection]
        if any(not source.is_file() for source in bank_maps):
            if level.training.map_pool_path is not None:
                raise ValueError("Pinned inspection map is missing; refusing to replace the dataset.")
            terminal_print(f"[BANK WARNING] Selected evaluation map is missing from {pool}; generating a new pool.")
            pool, metadata = build_pool(level, count=max(DEFAULT_POOL_SIZE, level.training_map_count + count),
                                        avoid_pool=pool, log=log)
            _, selection = select_maps(pool, metadata, level.training.bank_selection_seed, level.training_map_count, count)
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
                name=name, drone_clearance=level.env.drone_radius + level.env.obstacle_planning_clearance_m,
                env_cfg=level.env)
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
                       artifact_root=None, scope="train", checkpoint=None, eval_name=None,
                       sampling_kwargs=None, spawn_sampler=None, replay_by_required_agents=False,
                       eval_samples=None, authored_single_map=False):
    from swarmecho.training.train import evaluate_suite, evaluate_model
    destination = Path(destination)
    artifact_root = Path(artifact_root) if artifact_root is not None else destination / "artifacts" / "train"
    suffix = artifact_suffix(update, steps)
    results = []
    sampling_kwargs = {} if sampling_kwargs is None else dict(sampling_kwargs)
    if replay_by_required_agents and not eval_due:
        raise ValueError("Required-agent replays need a fresh evaluation batch.")
    score_spawn = getattr(spawn_sampler, "curriculum_source", None) == "evaluation"
    curriculum_outcomes = []
    static = authored_single_map or levels[0].building_name.startswith("static_eval_")
    replay_indices = set(representative_map_indices(len(levels), levels[0].evaluation.map_suite_replay_limit))
    for index, level in enumerate(levels):
        level = resolve_evaluation_level(level)
        cfg, evaluation = level.env, level.evaluation
        name = level.building_name
        suite_metadata = ({"map_suite_map_id": name, "map_suite_group": f"{artifact_root}:{suffix}",
                           "map_suite_maps": len(levels)} if static else
                          {"random_eval_map_id": name, "random_eval_group": f"{artifact_root}:{suffix}",
                           "random_eval_maps": len(levels)})
        common = {"map_name": name, "num_agents": cfg.num_agents, **suite_metadata,
                  "target_sampling": evaluation.target_sampling,
                  **({"target_budget": int(sampling_kwargs["target_agent_budget"])}
                     if "target_agent_budget" in sampling_kwargs else {}),
                  **({"source_map_name": level.map_source_name} if level.map_source_name else {}),
                  "world_size_m": level.building.world_size_m.tolist(),
                  "training_update": update, "environment_steps": steps, "artifact_scope": scope,
                  "action_noise_max": 0., "building_snapshot": building_snapshot(level.building),
                  **({"checkpoint": str(checkpoint)} if checkpoint is not None else {}),
                  **({"eval_name": eval_name} if eval_name else {})}
        if eval_due:
            batch_size = level.effective_eval_parallel_envs
            batching = {"batch_size": batch_size, "show_progress": True} if eval_samples is not None else {}
            metrics, info = evaluate_suite(model, level, episodes=eval_samples or batch_size,
                                          return_episode_info=True,
                                          return_required_agents=score_spawn or replay_by_required_agents,
                                          action_noise_max=0., **batching, **sampling_kwargs)
            results.append(metrics)
            if score_spawn:
                curriculum_outcomes.append(info)
            if on_run:
                on_run(metrics, index + 1, len(levels))
            if evaluation.training_heatmap_creation:
                success = np.asarray(info["successes"], dtype=bool)
                chain_success = success & (resolve_evaluation_level(level).env.success_condition in CHAIN_SUCCESS_CONDITIONS)
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
        if replay_due and index in replay_indices:
            if replay_by_required_agents:
                if eval_samples is not None:
                    terminal_print(f"[EVAL] {name}: sweep complete. Generating required-agent replays.")
                _write_required_agent_replays(model, level, info, artifact_root, suffix, common)
                continue
            replay_level = _replay_level_for_storey(level)
            states, rewards = evaluate_model(model, replay_level, max_steps=cfg.max_steps, **sampling_kwargs)
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
    aggregate = {key: float(np.mean([m[key] for m in results])) for key in results[0]
                 if key != "eval_useless_agents_at_success" and key not in EVALUATION_INFORMATION_KEYS}
    aggregate.update(pool_evaluation_information(results))
    successful_count = sum(m["eval_success_episode_count"] for m in results)
    successful_length = (sum(m["eval_success_episode_length"] * m["eval_success_episode_count"]
                             for m in results) / successful_count if successful_count
                         else float(levels[0].env.max_steps))
    aggregate.update(eval_success_episode_count=successful_count,
                     eval_success_episode_length=successful_length,
                     eval_success_length_reduction=1.0 - successful_length / levels[0].env.max_steps)
    if successful_count and any("eval_useless_agents_at_success" in m for m in results):
        aggregate["eval_useless_agents_at_success"] = sum(
            m.get("eval_useless_agents_at_success", 0.0) * m["eval_success_episode_count"]
            for m in results) / successful_count
    aggregate.update({"eval_static_maps" if static else "eval_random_maps": len(levels),
                      "eval_robustness_runs": 1})
    if curriculum_outcomes:
        scores = spawn_sampler.observe_curriculum("evaluation",
            np.concatenate([info["required_agents"] for info in curriculum_outcomes]),
            np.concatenate([info["successes"] for info in curriculum_outcomes]),
            np.concatenate([info["lengths"] for info in curriculum_outcomes]), levels[0].env.max_steps)
        aggregate.update({"eval_spawn_curriculum_" + key: value for key, value in scores.items()})
    rates = np.asarray([metrics["eval_success"] for metrics in results])
    aggregate["eval_worst_map_success"] = float(rates.min())
    individual_diagnostics = levels[0].logging.individual_map_diagnostics
    if individual_diagnostics:
        aggregate.update({f"eval_map/{level.map_source_name or level.building_name}/success_rate": metrics["eval_success"]
                          for level, metrics in zip(levels, results)})
    else:
        aggregate.update(eval_best_map_success=float(rates.max()),
                         eval_median_map_success=float(np.median(rates)),
                         eval_p10_map_success=float(np.quantile(rates, .10)))
    write_manifest(artifact_root / "manifests" / f"{'static_eval' if static else 'random_eval'}_{suffix}.json",
                   {"type": "static_map_evaluation" if static else "random_map_evaluation",
                    "update": update, "steps": steps,
                    "maps": [{"id": level.building_name, **(metrics if individual_diagnostics else {})}
                             for level, metrics in zip(levels, results)], **aggregate})
    return aggregate


def write_geometry_selected_replays(model, level, artifact_root, *, checkpoint, eval_name):
    """Classify spawn candidates with the environment reset, then run only the selected replays."""
    import jax
    from swarmecho.training.train import _evaluation_env_fns

    level = resolve_evaluation_level(level)
    name = level.building_name
    candidate_count = 256
    terminal_print(f"[REPLAY] {name}: classifying {candidate_count} target candidates.")
    reset, _, _, _ = _evaluation_env_fns(level)

    def classify(key):
        state = reset(key)
        return state.target_pos, state.target_required_agents, state.spawn_pair_attempts > 0

    keys = jax.random.split(jax.random.PRNGKey(level.training.seed + 10_000), candidate_count)
    positions, required, valid = jax.device_get(jax.jit(jax.vmap(classify))(keys))
    if not np.any(valid):
        raise ValueError(f"No certified replay targets sampled on {name}.")
    info = {"target_positions": positions[valid], "required_agents": required[valid]}
    classes = np.unique(info["required_agents"])
    terminal_print(f"[REPLAY] {name}: selected one target for each observed count {classes.tolist()}.")
    update = parse_checkpoint_update(checkpoint) or 0
    steps = steps_for_checkpoint(checkpoint, level)
    _write_required_agent_replays(model, level, info, artifact_root, artifact_suffix(update, steps), {
        "map_name": name, "num_agents": level.env.num_agents,
        "map_suite_map_id": name, "map_suite_group": str(artifact_root), "map_suite_maps": 1,
        "training_update": update, "environment_steps": steps, "artifact_scope": "eval",
        "checkpoint": str(checkpoint), **({"eval_name": eval_name} if eval_name else {}),
    })


def _write_required_agent_replays(model, level, info, artifact_root, suffix, common):
    """Replay one evaluated target per observed estimator class on this map."""
    from swarmecho.training.train import evaluate_model

    required = np.asarray(info["required_agents"])
    cfg = level.env
    expected = int(common.get("target_budget", cfg.geometric_agent_budget))
    if np.any((required < 1) | (required > expected)):
        raise ValueError("Required-agent replays need classified targets; enable inspection_evaluation.dynamic_max_geo_distance.")
    for count in range(1, expected + 1):
        lanes = np.flatnonzero(required == count)
        if not len(lanes):
            terminal_print(f"[REPLAY] {level.building_name}: no sampled target requiring {count} agent(s).", flush=True)
            continue
        lane = int(lanes[0])
        target = np.asarray(info["target_positions"][lane])
        terminal_print(f"[REPLAY] {level.building_name}: target requiring {count} agent(s).", flush=True)
        states, rewards = evaluate_model(model, level, max_steps=cfg.max_steps, target_position=target)
        actual = int(np.asarray(states[0].target_required_agents))
        if actual != count:
            raise ValueError(f"Replay target required-agent class changed from {count} to {actual}.")
        write_replay(artifact_root / "replays" / f"eval_{suffix}_{level.building_name}_required_agents_{count}",
            states, map_name=level.building_name, building=level.building, dt=cfg.dt,
            reward_terms=rewards, progress=False,
            metadata={**common, **planner_metadata(cfg), "cell_size_m": level.building.cell_size_m,
                "coverage_voxel_size_m": cfg.coverage_voxel_size or level.building.cell_size_m,
                "comm_radius_m": cfg.comm_radius, "comm_radius_base_m": cfg.comm_radius_base,
                "visual_radius_m": cfg.visual_radius,
                "allow_redundancy_reward": level.reward.allow_redundancy_reward,
                "success_condition": cfg.success_condition, "required_agents": count,
                "replay_source_lane": lane, "selected_target_position": target.tolist(),
                "target_selection": f"required_agents_{count}", "replay_execution": "single"})


evaluate_random_maps = evaluate_map_suite
