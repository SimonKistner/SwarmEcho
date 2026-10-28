"""Config-driven recurrent MAPPO training runner for the baseline."""

from __future__ import annotations

import os
import sys

# XLA/absl verbosity must be configured before importing JAX. Doing this in the
# training function is too late because plugin discovery happens at import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

if __name__ == "__main__":
    from swarmecho.training.launcher import main as launch
    launch()
    raise SystemExit(0)

import functools
import json
import time
from collections import OrderedDict, deque
from dataclasses import asdict, fields, replace
from swarmecho.models.actions import physical_actions, deterministic_actions
from swarmecho.env.adapt_size import policy_resets
from swarmecho.training.adapt_size_metrics import (
    AdaptiveWindow, RequiredAgentsDeploymentWindow,
    REWARD_NAMES, VOTE_SHAPING_REWARD_NAMES, REFERENCE_DIAGNOSTIC_NAMES, INFO_NAMES, validate_zones,
)
from pathlib import Path
from swarmecho.core.terminal import terminal_print, display_path, evaluation_row, init_logging

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx

from swarmecho.core.config import (
    DEPLOYED_AGENTS_OBS_REF,
    Level,
    evaluation_minimum_geodesic_separation,
    evaluation_dynamic_max_geo_distance,
    load_level_cli,
    resolve_evaluation_level,
    inspection_config_dict,
)
from swarmecho.env.environment import (
    chain_diagnostics,
    coverage_grid_geometry,
    final_chain_length,
    make_autoreset_fns,
    make_env_fns,
    observation_dim,
    obstacle_chain_diagnostics,
    compute_rewards,
)
from swarmecho.env.obstacles import segments_blocked
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.artifacts import (
    artifact_suffix,
    save_eval_info_csv,
    save_eval_layout,
    train_replay_root,
    write_manifest,
)
from swarmecho.training.checkpoints import restore_model_checkpoint, save_training_checkpoint
from swarmecho.training.launcher import is_out_of_memory
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.ppo import PPOTrainer, RunningValueNormalizer
from swarmecho.training.run_lifecycle import (
    EvaluationHold,
    TrainingSuccessHold,
    broadcast_early_stop_metrics,
    init_wandb,
    resolve_resume_state,
    resolve_run_layout,
    schedule_due,
)
from swarmecho.visualize.replay import ReplayRewards, building_snapshot, planner_metadata, write_replay


def communication_due(episode_step, every_k_steps):
    """Shared episode clock: independent of rollout boundaries and auto-reset lanes."""
    return (episode_step % every_k_steps) == 0


def build_model(level: Level, hidden_dim: int | None = None) -> MAPPOModel:
    """Construct the canonical recurrent MAPPO/TarMAC model."""
    cfg = level.env
    if not level.network.actor_memory or not level.network.critic_memory:
        raise ValueError("The runtime requires recurrent actor and critic memory.")
    if level.network.critic_type != "observation":
        raise ValueError("network.critic_type currently supports only observation.")
    model = MAPPOModel(
        obs_dim=observation_dim(cfg),
        act_dim=3,
        vote_mode=cfg.adapt_size.vote_mode if cfg.adapt_size.enabled else None,
        num_agents=cfg.num_agents,
        hidden_dim=hidden_dim or level.network.hidden_dim,
        num_layers=level.network.num_layers,
        actor_num_layers=level.network.actor_num_layers,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
        memory_comm_enabled=level.network.memory_comm_enabled,
        memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
        tarmac_sig_dim=level.network.tarmac_sig_dim,
        tarmac_val_dim=level.network.tarmac_val_dim,
        tarmac_include_self=level.network.tarmac_include_self,
        rngs=nnx.Rngs(level.training.seed),
    )
    model.mask_inactive = True
    model.actor.mask_inactive = True
    if level.training.value_normalization == "running":
        model.value_normalizer = RunningValueNormalizer()
    # Static metadata; the numerical normalizer state is saved inside the model.
    model.checkpoint_contract = tuple(sorted({
        "format": 1,
        "obs_dim": observation_dim(cfg),
        "hidden_dim": hidden_dim or level.network.hidden_dim,
        "num_layers": level.network.num_layers,
        "actor_num_layers": level.network.actor_num_layers,
        "tarmac_sig_dim": level.network.tarmac_sig_dim,
        "tarmac_val_dim": level.network.tarmac_val_dim,
        "memory_comm_enabled": level.network.memory_comm_enabled,
        "memory_comm_every_k_steps": level.network.memory_comm_every_k_steps,
        "tarmac_include_self": level.network.tarmac_include_self,
        "value_normalization": level.training.value_normalization,
        "entropy_mode": level.training.entropy_mode,
        "actor_clip_eps": level.training.actor_clip_eps,
        "value_clip_eps": level.training.value_clip_eps,
        "gamma": level.training.gamma,
        "gae_lambda": level.training.gae_lambda,
        "progress_based_cluster_tiebreaker_soft": level.reward.progress_based_cluster_tiebreaker_soft,
        "reward_system": "signed_individual_v1",
        "communication_clock": "episode",
        "inactive_agents": "excluded",
        **({"relay_estimator": json.dumps(_global_relay_identity(level.training.dynamic_max_geo_distance), sort_keys=True)}
           if level.training.dynamic_max_geo_distance.enabled
           and level.training.dynamic_max_geo_distance.mode == "global_candidates" else {}),
        "visually_gated_target_displacement_scale_m": (
            cfg.visual_radius if cfg.observe_visually_gated_target_displacement else None
        ),
        "adaptive_size": cfg.adapt_size.enabled,
        "vote_mode": cfg.adapt_size.vote_mode if cfg.adapt_size.enabled else None,
        "vote_protocol": cfg.adapt_size.vote_protocol if cfg.adapt_size.enabled else None,
        **({"priveleged_vote_reward_enabled": True}
           if level.reward.priveleged_vote_reward.enabled else {}),
        **{field.name: getattr(cfg, field.name) for field in fields(cfg)
           if field.name.startswith("observe_") or field.name in {
               "wall_sensing_rays", "wall_sensing_layout", "wall_sensing_ray_max_distance",
               "drone_radar_bins", "radar_mode",
               "distance_bands", "radar_count_cap",
           }},
        "radar_layout": ("band_counts_v2" if cfg.radar_mode == "band_sep_counting"
                         else "legacy_interleaved" if cfg.wall_sensing_rays == cfg.drone_radar_bins
                         else "radar_blocks_v1"),
        "observe_base_vote_ratio": cfg.adapt_size.enabled and cfg.observe_base_vote_ratio,
        "observe_base_population": cfg.adapt_size.enabled and cfg.observe_base_population,
        "base_population_observation_reference": (
            DEPLOYED_AGENTS_OBS_REF if cfg.adapt_size.enabled and cfg.observe_base_population else None
        ),
    }.items()))
    return model


_EVALUATION_ENV_CACHE = OrderedDict()


def _global_relay_identity(options):
    from swarmecho.env.relay_candidates import settings
    return settings(options)


def _validate_relay_resume(training, checkpoint):
    if not checkpoint or training.ckpt_loading_mode != "resume":
        return
    source = Path(str(checkpoint).replace("\\", "/")) / "training_contract.json"
    actual = json.loads(source.read_text()).get("relay_estimator") if source.is_file() else None
    options = training.dynamic_max_geo_distance
    expected = (json.dumps(_global_relay_identity(options), sort_keys=True)
                if options.enabled and options.mode == "global_candidates" else None)
    if actual != expected:
        raise ValueError("Relay estimator mode, resolution or implementation changed. Use ckpt_loading_mode=branch "
                         "and a new run name to retain weights with fresh adaptive support and statistics.")


def _evaluation_env_fns(level: Level):
    """Keep static JIT function arguments stable across evaluation calls.

    Include all geometry contents, not just a map name or array identity, so
    another map/config cannot accidentally reuse stale captured geometry.
    """
    geometry_key = []
    for field in fields(level.building):
        value = getattr(level.building, field.name)
        if isinstance(value, np.ndarray):
            value = (value.dtype.str, value.shape, value.tobytes())
        geometry_key.append((field.name, value))
    plan_geodesic = level.reward.chain_reward_system == "obstacle_geodesic" or level.reward.enable_chain_efficiency_reward
    apply_separation = evaluation_minimum_geodesic_separation(level)
    apply_dynamic = evaluation_dynamic_max_geo_distance(level)
    key = (
        tuple(geometry_key),
        level.env,
        plan_geodesic,
        level.reward.chain_reward_system,
        level.reward.progress_based_cluster_tiebreaker_soft,
        level.reward.allow_redundancy_reward,
        level.reward.target_found_requires_delivery,
        level.reward.delivery_gap_vesting_enabled,
        level.reward.priveleged_vote_reward,
        level.network.memory_comm_every_k_steps,
        apply_separation,
        apply_dynamic,
        level.training.randomize_base if apply_dynamic else False,
        level.training.minimum_geodesic_separation_multiplier,
        level.training.spawn_pair_max_attempts,
    )
    if apply_dynamic and level.training.dynamic_max_geo_distance.mode == "global_candidates":
        key += (level.training.dynamic_max_geo_distance,)
    if key not in _EVALUATION_ENV_CACHE:
        with init_logging(level.logging.terminal_log_init):
            _EVALUATION_ENV_CACHE[key] = make_env_fns(
                level.building, level.env, plan_geodesic=plan_geodesic,
                allow_redundancy_reward=level.reward.allow_redundancy_reward,
                target_found_requires_delivery=level.reward.target_found_requires_delivery,
                minimum_geodesic_separation=apply_separation,
                dynamic_max_geo_distance=apply_dynamic,
                relay_options=level.training.dynamic_max_geo_distance,
                randomize_base=level.training.randomize_base if apply_dynamic else False,
                minimum_geodesic_separation_multiplier=level.training.minimum_geodesic_separation_multiplier,
                spawn_pair_max_attempts=level.training.spawn_pair_max_attempts,
                memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
                chain_reward_system=level.reward.chain_reward_system,
                progress_based_cluster_tiebreaker_soft=level.reward.progress_based_cluster_tiebreaker_soft,
                delivery_gap_vesting_enabled=level.reward.delivery_gap_vesting_enabled,
                priveleged_vote_reward=level.reward.priveleged_vote_reward,
            )
        if len(_EVALUATION_ENV_CACHE) > max(8, level.evaluation.random_eval_envs):
            _EVALUATION_ENV_CACHE.popitem(last=False)
    _EVALUATION_ENV_CACHE.move_to_end(key)
    return _EVALUATION_ENV_CACHE[key]


def _communication_inputs(states, cfg, base_valid, base_signature, base_value):
    delta = states.pos[:, :, None, :] - states.pos[:, None, :, :]
    comm_masks = (
        (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
        & states.active[:, :, None]
        & states.active[:, None, :]
        & ~jnp.eye(cfg.num_agents, dtype=jnp.bool_)[None]
    )
    in_base_range = (
        jnp.linalg.norm(states.pos - states.base_pos[:, None, :], axis=-1)
        <= cfg.comm_radius_base
    ) & states.active
    if states.solid_min.shape[-2]:
        comm_masks &= ~jax.vmap(
            lambda pos, lower, upper: segments_blocked(
                pos[:, None, :], pos[None, :, :], lower, upper
            )
        )(states.pos, states.solid_min, states.solid_max)
        in_base_range &= ~jax.vmap(
            lambda pos, base, lower, upper: segments_blocked(
                pos, base, lower, upper
            )
        )(states.pos, states.base_pos, states.solid_min, states.solid_max)
    receivers = jnp.ones_like(states.target_known) if cfg.base_keeps_informing else ~states.target_known
    base_memory_masks = base_valid[:, None] & in_base_range & receivers
    return comm_masks, in_base_range, base_memory_masks, base_signature, base_value


def _update_base_memory(
    states,
    in_base_range,
    emitted_signature,
    emitted_value,
    base_valid,
    base_signature,
    base_value,
    dones,
    share_now=None,
):
    reporters = states.target_known & in_base_range
    has_reporter = jnp.any(reporters, axis=-1)
    reporter_index = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
    env_index = jnp.arange(states.pos.shape[0])
    if share_now is None:
        share_now = jnp.ones_like(base_valid, dtype=jnp.bool_)
    should_store = ~base_valid & has_reporter & share_now
    base_signature = jnp.where(
        should_store[:, None], emitted_signature[env_index, reporter_index], base_signature
    )
    base_value = jnp.where(
        should_store[:, None], emitted_value[env_index, reporter_index], base_value
    )
    base_valid = base_valid | should_store
    base_valid = jnp.where(dones, False, base_valid)
    base_signature = jnp.where(dones[:, None], 0.0, base_signature)
    base_value = jnp.where(dones[:, None], 0.0, base_value)
    return base_valid, base_signature, base_value


def _evaluate_training_metrics(model, level, hold, *, eval_due, on_run=None):
    """Measure the configured training metric and advance the stop hold once."""
    if not eval_due:
        return None, None, False
    from swarmecho.training.evaluate import collect_robust_evaluation

    evaluation = level.evaluation
    evaluation_result = None
    if evaluation.training_robustness:
        evaluation_result = collect_robust_evaluation(model, level, on_run=on_run)
        metrics, _ = evaluation_result
    else:
        result = evaluate_suite(
            model, level, episodes=level.effective_eval_parallel_envs,
            return_episode_info=evaluation.training_heatmap_creation,
        )
        if evaluation.training_heatmap_creation:
            metrics, episode_info = result
            successes = np.asarray(episode_info["successes"], dtype=bool)
            chain_success = successes & (resolve_evaluation_level(level).env.success_condition == "chain_held")
            delivered = np.asarray(episode_info["delivered"], dtype=bool) | chain_success
            visually_found = np.asarray(episode_info["visually_found"], dtype=bool) | delivered
            episode_info["stage_rates"] = {
                "chain_success": chain_success.astype(float),
                "found_and_delivered": delivered.astype(float),
                "visually_found": visually_found.astype(float),
            }
        else:
            metrics = result
        metrics = {**metrics, "eval_robustness_runs": 1}
        if evaluation.training_heatmap_creation:
            evaluation_result = metrics, episode_info
    stop = evaluation.early_exit and evaluation.early_exit_source == "evaluation" and hold.observe(
        metrics["eval_success"], metrics.get("eval_success_length_reduction"))
    return metrics, evaluation_result, stop


def _report_evaluation(metrics, level, *, update, total, steps, phase, wandb_run, history_path, hold):
    """Publish one measurement with explicit provenance to all logging sinks."""
    if not level.logging.individual_map_diagnostics:
        metrics = {name: value for name, value in metrics.items() if not name.startswith("eval_map/")}
    runs = int(metrics.get("eval_robustness_runs", 1))
    prefix = "FINAL" if phase == "final_eval" else "EVAL"
    tail = (
        f"{f'est={level.evaluation.early_exit_success_rate:.0%}':<10}  "
        f"hold={hold.consecutive}/{max(1, level.evaluation.early_exit_hold_evals)}"
        if level.evaluation.early_exit else "est=off"
    )
    if level.evaluation.early_exit and level.evaluation.early_exit_source == "training":
        tail = "early-exit source=training (diagnostic eval)"
    if level.evaluation.early_exit and level.evaluation.early_exit_source == "validation":
        tail = "early-exit source=validation (inspection only)"
    if phase == "final_eval" and level.evaluation.early_exit:
        tail += " (unchanged)"
    evaluation_row(
        metrics, label=f"[{prefix} {'UNAN' if runs > 1 else '1/1'}]",
        update=update, total=total, episodes=level.effective_eval_parallel_envs, tail=tail,
    )
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "update": update, "global_step": steps, "phase": phase,
            "mode": "robust" if runs > 1 else "single_pass", **metrics,
            "early_exit_source": level.evaluation.early_exit_source,
        }) + "\n")
    values = {
        "ep_return": metrics["eval_return"],
        "ep_length": metrics["eval_episode_length"],
        "chain_progress_pct": metrics["eval_chain_progress_pct"],
        "ep_length_reduction": (1.0 - metrics["eval_episode_length"] / level.env.max_steps) * 100.0,
        "success_ep_length_reduction": metrics["eval_success_length_reduction"] * 100.0,
        "success_rate": metrics["eval_success"],
        "target_found_rate": metrics["eval_target_found_rate"],
        "visually_found_rate": metrics["eval_visually_found_rate"],
        "map_coverage_pct": metrics["eval_coverage"] * 100.0,
        "robustness_runs": runs,
    }
    logs = {f"{phase}/{key}": value for key, value in values.items()}
    for statistic in ("worst", "best", "median", "p10"):
        key = f"eval_{statistic}_map_success"
        if key in metrics:
            logs[f"{phase}/{statistic}_map_success_rate"] = metrics[key]
    logs.update({f"{phase}/{name.removeprefix('eval_')}": value
                 for name, value in metrics.items() if name.startswith("eval_map/")})
    if wandb_run is not None and phase != "final_eval":
        import wandb
        wandb.log(logs, step=steps, commit=False)
    return logs


def _create_final_checkpoint_evaluation(
    model: MAPPOModel, level: Level, checkpoint: Path, run_dir: Path,
    *, result=None,
) -> None:
    """Run the standalone checkpoint evaluation contract without risking training."""
    try:
        # Local import avoids the evaluate -> train import cycle.
        from swarmecho.training.evaluate import (
            render_csv_replays,
            run_parallel_evaluation,
        )
        from swarmecho.training.artifacts import create_eval_run_root

        eval_run_root = create_eval_run_root(run_dir, checkpoint, level)
        info_path = run_parallel_evaluation(
            model,
            level,
            checkpoint,
            run_dir,
            eval_run_root=eval_run_root,
            result=result,
        )
    except Exception as exc:
        if level.training.oom_recovery.enabled and is_out_of_memory(exc):
            raise
        terminal_print(
            f"[WARNING] Final checkpoint evaluation failed; replay creation skipped: {exc}",
            flush=True,
        )
        return

    if not level.evaluation.eval_video:
        return
    for result in ("success", "fail"):
        try:
            render_csv_replays(
                model,
                level,
                checkpoint,
                run_dir,
                result=result,
                replay_count=1,
                start_offset=0,
                source_csv=info_path,
                eval_run_root=eval_run_root,
            )
            if result == "fail":
                terminal_print(
                    "[WARNING] No successful final-evaluation lane was available; "
                    "created a failure replay instead.",
                    flush=True,
                )
            return
        except Exception as exc:
            if level.training.oom_recovery.enabled and is_out_of_memory(exc):
                raise
            terminal_print(
                f"[WARNING] Final {result} replay selection failed: {exc}",
                flush=True,
            )

    terminal_print(
        "[WARNING] Final success and failure replay creation both failed; "
        "replay creation skipped.",
        flush=True,
    )


def train(
    level: Level,
    *,
    output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[Path | None, dict[str, float]]:
    """Train the strict level and return its final checkpoint and metrics."""
    if level.training.agent_count_curriculum.enabled:
        from swarmecho.training.curriculum import train_agent_counts
        return train_agent_counts(level, output_dir=output_dir, checkpoint_path=checkpoint_path)
    init_started = init_previous = time.perf_counter()

    def init_log(message):
        nonlocal init_previous
        if not level.logging.terminal_log_init:
            return
        now = time.perf_counter()
        terminal_print(
            f"[INIT +{now - init_started:.1f}s; "
            f"stage {now - init_previous:.1f}s] {message}",
            flush=True,
        )
        init_previous = now

    init_log(f"Starting training setup: {level.name} / {level.building_name}")
    training = level.training
    network = level.network
    evaluation = level.evaluation
    if level.random_buildings.enabled:
        inspection_count = evaluation.inspection_maps
        if inspection_count is None:
            inspection_count = (training.adaptive_map_sampling.held_out_maps
                                if training.adaptive_map_sampling.enabled else evaluation.random_eval_envs)
        evaluation = replace(evaluation, random_eval=True,
                             random_eval_envs=inspection_count,
                             random_eval_maps=(None if level.validation.enabled or training.adaptive_map_sampling.enabled
                                               else evaluation.random_eval_maps))
        level = replace(level, evaluation=evaluation)
    evaluation_level = resolve_evaluation_level(level)
    logging = level.logging
    num_updates = level.num_updates
    layout = resolve_run_layout(logging, evaluation, media_dir="replays")
    run_name = layout.run_name
    destination = Path(output_dir).absolute() if output_dir is not None else layout.run_dir
    destination.mkdir(parents=True, exist_ok=True)
    if not level.static_map_suite and (destination / "static_maps.json").exists():
        raise ValueError("This run directory contains a multi-map static run; use a new run name for a single-map or random run.")
    def record_timing(update, phase, **values):
        if not training.profile_timing:
            return
        row = {"update": update, "phase": phase, **values}
        with (destination / "timings.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        terminal_print("[TIMING] " + json.dumps(row), flush=True)
    checkpoint_dir = layout.checkpoint_dir if output_dir is None else destination / "checkpoints"
    replay_dir = train_replay_root(destination)
    cfg = level.env
    building_bank = None
    adaptive_spawn = None
    adaptive_spawn_journal = None
    target_spawn_weights = None
    adaptive_maps = None
    adaptive_map_journal = None
    map_sampling_cdf = None
    map_metrics = {}
    map_identities = None
    map_names = level.map_names
    # Random datasets always choose a map at episode boundaries. Adaptation
    # changes only their probabilities; authored suites retain legacy opt-out.
    resample_maps = level.random_buildings.enabled or training.adaptive_map_sampling.enabled
    if resample_maps and evaluation.early_exit and evaluation.early_exit_source == "training":
        raise ValueError("Episode map resampling requires inspection- or validation-based early exit.")
    frozen_selection = None
    if level.random_buildings.enabled or training.adaptive_map_sampling.enabled:
        from swarmecho.training.map_selection import (
            load_selection, save_selection, building_identity, verify_selection_geometry)
        frozen_selection = load_selection(destination, checkpoint_path or training.checkpoint_path)
    if training.adaptive_map_sampling.enabled:
        from swarmecho.core.config import validate_adaptive_map_sampling
        validate_adaptive_map_sampling(training.adaptive_map_sampling)
        if evaluation.early_exit_source not in {"evaluation", "validation"}:
            raise ValueError("Adaptive map sampling requires evaluation-based early exit.")
    if training.adaptive_target_spawn.enabled and (
        cfg.num_obstacles or training.randomize_base or training.minimum_geodesic_separation
        or cfg.success_condition == "coverage"
    ):
        raise ValueError("Adaptive target spawning requires static target maps and fixed bases, "
                         "without generated obstacles or minimum geodesic separation.")
    bank_eval_maps = None
    validation_map_paths = None
    adaptive_spawn_buildings = None
    bank_map_count = level.training_map_count
    if level.static_map_suite:
        import hashlib
        from swarmecho.env.buildings import load_building
        from swarmecho.env.building_bank import balanced_map_ids, prepare_bank

        sources = level.map_paths
        lane_ids = (np.arange(training.num_envs, dtype=np.int32) % len(sources)
                    if training.adaptive_map_sampling.enabled else balanced_map_ids(training.num_envs, len(sources)))
        static_manifest = {
            "type": "static_map_suite", "maps": [
                {"name": name, "path": str(path),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "training_lanes": int(np.count_nonzero(lane_ids == index))}
                for index, (name, path) in enumerate(zip(level.map_names, sources))
            ],
        }
        init_log("Static map lanes: " + ", ".join(
            f"{item['name']}={item['training_lanes']}" for item in static_manifest["maps"]))
        manifest_path = destination / "static_maps.json"
        if manifest_path.exists():
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            def map_identity(manifest):
                return {**manifest, "maps": [{key: value for key, value in item.items()
                                              if key != "training_lanes"} for item in manifest["maps"]]}
            resizing = training.oom_recovery.enabled and (
                training.ckpt_loading_mode == "resume" or os.environ.get("SWARMECHO_RECOVERY_RETRY") == "1")
            if (previous_manifest != static_manifest and
                    (not resizing or map_identity(previous_manifest) != map_identity(static_manifest))):
                raise ValueError("The static map selection or map content changed in this run; use a new run directory.")
        buildings = [load_building(path) for path in sources]
        if training.adaptive_map_sampling.enabled:
            map_identities = [building_identity(building, name) for building, name in zip(buildings, level.map_names)]
        init_log(f"Compiling {len(buildings)} static maps for {training.num_envs} training lanes...")
        building_bank = prepare_bank(buildings, cfg,
            dynamic_max_geo_distance=training.dynamic_max_geo_distance.enabled,
            relay_options=training.dynamic_max_geo_distance, randomize_base=training.randomize_base,
            plan=cfg.success_condition != "coverage" and (level.reward.chain_reward_system == "obstacle_geodesic" or level.reward.enable_chain_efficiency_reward),
            corner_bonus_m=cfg.roadmap_corner_bonus_m if training.minimum_geodesic_separation else 0., log=init_log)
        write_manifest(manifest_path, static_manifest)
        bank_map_count = len(buildings)
        if training.adaptive_target_spawn.enabled:
            adaptive_spawn_buildings = buildings
        del buildings
    if level.random_buildings.enabled:
        if level.random_buildings.load_maps_from_bank:
            import shutil
            from zipfile import BadZipFile
            from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
                DEFAULT_POOL_SIZE, build_pool, ensure_pool, load_records, select_maps, required_settings)
            from swarmecho.env.building_bank import prepare_bank_from_records

            eval_count = (level.validation.num_maps if level.validation.enabled else
                          evaluation.random_eval_envs if evaluation.random_eval
                          and evaluation.random_eval_maps is None else 0)
            required = level.training_map_count + eval_count
            if frozen_selection is not None:
                if (frozen_selection.get("kind") != "bank" or frozen_selection.get("seed") != training.seed
                        or frozen_selection.get("settings") != required_settings(level)):
                    raise ValueError("Frozen training bank settings changed; use a new run without the old selection.")
                pool = Path(frozen_selection["pool"])
                metadata = json.loads((pool / "metadata.json").read_text(encoding="utf-8"))
                if metadata["settings"] != frozen_selection["settings"]:
                    raise ValueError("Frozen map bank metadata changed.")
                train_maps, eval_maps = frozen_selection["maps"], frozen_selection["evaluation_maps"]
                if training.num_maps is not None and len(train_maps) != training.num_maps:
                    raise ValueError("Frozen training map count differs from training.num_maps; use a new run.")
                if len(eval_maps) != eval_count:
                    raise ValueError("Held-out map count changed; choose a new bank selection.")
            else:
                pool, metadata = ensure_pool(level, required, log=init_log)
                train_maps, eval_maps = select_maps(pool, metadata, training.seed, level.training_map_count, eval_count)
            selected_ids = [item["id"] for item in (*train_maps, *eval_maps)]
            if len(set(selected_ids)) != len(selected_ids):
                raise ValueError("Training and held-out map selections must contain distinct, disjoint map IDs.")
            try:
                records = load_records(pool, train_maps,
                    require_corner=bool(metadata["settings"]["roadmap"]["corner_bonus_m"]) and cfg.success_condition != "coverage",
                    include_target=cfg.success_condition != "coverage")
                if any(not (pool / "maps" / f"{item['id']}.yaml").is_file()
                       for item in (*train_maps, *eval_maps)):
                    raise FileNotFoundError("A selected building map is missing from the pool.")
            except (OSError, ValueError, EOFError, BadZipFile) as exc:
                if frozen_selection is not None:
                    raise RuntimeError("Frozen map bank cannot be loaded; refusing to replace its map identities.") from exc
                message = f"Map pool {pool.name} cannot be loaded ({exc}); generating a new pool."
                terminal_print(f"[BANK WARNING] {message}")
                pool, metadata = build_pool(level, count=max(DEFAULT_POOL_SIZE, required),
                                            avoid_pool=pool, log=init_log)
                train_maps, eval_maps = select_maps(pool, metadata, training.seed, level.training_map_count, eval_count)
                records = load_records(pool, train_maps,
                    require_corner=bool(metadata["settings"]["roadmap"]["corner_bonus_m"]) and cfg.success_condition != "coverage",
                    include_target=cfg.success_condition != "coverage")
                if any(not (pool / "maps" / f"{item['id']}.yaml").is_file()
                       for item in (*train_maps, *eval_maps)):
                    raise FileNotFoundError("A selected building map is missing from the replacement pool.")
            save_selection(destination, {**(frozen_selection or {}),
                "kind": "bank", "seed": training.seed, "pool": str(pool.resolve()),
                "settings": metadata["settings"], "maps": train_maps, "evaluation_maps": eval_maps})
            init_log(f"Loaded {len(records)} precompiled random buildings from {pool}")
            bank_map_count = len(records)
            import hashlib
            map_names = [item["id"] for item in train_maps]
            paths = [pool / "maps" / f"{name}.yaml" for name in map_names]
            map_identities = [{"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for name, path in zip(map_names, paths)]
            validation_identities = [
                {"name": item["id"], "sha256": hashlib.sha256(
                    (pool / "maps" / f"{item['id']}.yaml").read_bytes()).hexdigest()}
                for item in eval_maps]
            verify_selection_geometry(destination, map_identities)
            selection = load_selection(destination, None)
            if selection.get("held_out_identities", validation_identities) != validation_identities:
                raise ValueError("Frozen held-out bank map geometry changed.")
            save_selection(destination, {**selection, "held_out_identities": validation_identities})
            if training.adaptive_target_spawn.enabled:
                from swarmecho.env.buildings import load_building
                adaptive_spawn_buildings = [load_building(path) for path in paths]
            if training.dynamic_max_geo_distance.enabled and training.dynamic_max_geo_distance.mode == "simplifier":
                from swarmecho.env.buildings import load_building
                from swarmecho.env.relay import opening_arrays
                for record, item in zip(records, train_maps):
                    record["relay_openings"] = opening_arrays(load_building(pool / "maps" / f"{item['id']}.yaml"))
            building_bank = prepare_bank_from_records(records, level.building, cfg,
                dynamic_max_geo_distance=training.dynamic_max_geo_distance.enabled,
                relay_options=training.dynamic_max_geo_distance, randomize_base=training.randomize_base,
                relay_cache_dir=pool / "relay_cache", log=init_log)
            del records
            bank_eval_maps = ([pool / "maps" / f"{item['id']}.yaml" for item in eval_maps]
                              if eval_maps else None)
            if level.validation.enabled:
                validation_map_paths = bank_eval_maps
                bank_eval_maps = validation_map_paths[:evaluation.random_eval_envs]
            if level.random_buildings.save_training_maps:
                saved = destination / "generated_maps"
                saved.mkdir(parents=True, exist_ok=True)
                for item in train_maps:
                    shutil.copy2(pool / "maps" / f"{item['id']}.yaml", saved / f"{item['id']}.yaml")
            write_manifest(destination / "random_buildings.json", {
                "version": metadata["settings"]["generator_version"],
                "settings": metadata["settings"], "pool": str(pool.resolve()),
                "maps": train_maps, "evaluation_maps": eval_maps,
            })
        else:
            from swarmecho.env.random_buildings import generate_maps
            from swarmecho.env.building_bank import prepare_bank
            generated_count = level.training_map_count
            if frozen_selection is not None:
                if (frozen_selection.get("kind") != "generated" or frozen_selection.get("seed") != training.seed
                        or frozen_selection.get("settings") != asdict(level.random_buildings)):
                    raise ValueError("Frozen generated-map settings changed.")
                generated_count = frozen_selection["count"]
                if training.num_maps is not None and generated_count != training.num_maps:
                    raise ValueError("Frozen generated-map count changed; use a new run.")
            save_selection(destination, {**(frozen_selection or {}), "kind": "generated", "seed": training.seed,
                "settings": asdict(level.random_buildings), "count": generated_count})
            init_log(f"Generating {generated_count} persistent random buildings...")
            buildings, generation_manifest = generate_maps(
                level.random_buildings, training.seed, generated_count,
                directory=destination / "generated_maps" if level.random_buildings.save_training_maps else None,
                log=init_log, drone_clearance=cfg.drone_radius + cfg.obstacle_planning_clearance_m)
            write_manifest(destination / "random_buildings.json", generation_manifest)
            bank_map_count = len(buildings)
            map_names = [item["id"] for item in generation_manifest["maps"]]
            map_identities = [building_identity(building, name) for building, name in zip(buildings, map_names)]
            verify_selection_geometry(destination, map_identities, version=generation_manifest["version"])
            if training.adaptive_target_spawn.enabled:
                adaptive_spawn_buildings = buildings
            building_bank = prepare_bank(buildings, cfg,
                dynamic_max_geo_distance=training.dynamic_max_geo_distance.enabled,
                relay_options=training.dynamic_max_geo_distance, randomize_base=training.randomize_base,
                plan=cfg.success_condition != "coverage" and (level.reward.chain_reward_system == "obstacle_geodesic" or level.reward.enable_chain_efficiency_reward),
                corner_bonus_m=cfg.roadmap_corner_bonus_m if training.minimum_geodesic_separation else 0., log=init_log)
            if level.validation.enabled:
                from swarmecho.training.validation import prepare_generated_validation_maps
                validation_map_paths = prepare_generated_validation_maps(level, destination, init_log,
                                                                         training_buildings=buildings)
                bank_eval_maps = validation_map_paths[:evaluation.random_eval_envs]
            del buildings
    validation_runner = None
    if level.validation.enabled:
        from swarmecho.training.validation import ValidationRunner
        validation_runner = ValidationRunner(level, destination, validation_map_paths,
            checkpoint_path=checkpoint_path or training.checkpoint_path, checkpoint_dir=checkpoint_dir, log=init_log)
    map_suite_levels = None
    if evaluation.random_eval or level.static_map_suite:
        from swarmecho.training.random_evaluation import prepare_evaluation_maps
        suite_dir = destination / ("static_eval_maps" if level.static_map_suite else "random_eval_maps")
        map_suite_levels = prepare_evaluation_maps(level, suite_dir,
            bank_maps=bank_eval_maps, static_maps=level.map_paths if level.static_map_suite else None,
            log=init_log)
        if level.random_buildings.enabled:
            verify_selection_geometry(destination, map_identities, evaluation_identities=[
                building_identity(item.building, item.building_name) for item in map_suite_levels])
    init_log("Initializing JAX devices...")
    devices = jax.devices()
    init_log(f"JAX devices ready: {devices}")
    voxel_size, coverage_shape = coverage_grid_geometry(
        level.building if building_bank is None else building_bank.envelope, cfg)
    init_log(
        f"Geometry: {len(level.building.solid_min_m):,} authored solids; "
        f"{cfg.num_obstacles} generated obstacles; coverage {coverage_shape} = "
        f"{int(np.prod(coverage_shape)):,} voxels at {voxel_size:g} m; "
        f"{training.num_envs:,} environments. Constructing environment functions..."
    )
    with init_logging(logging.terminal_log_init):
        reset, env_step, observations, _ = make_autoreset_fns(
            level.building, cfg, level.reward,
            randomize_base=training.randomize_base,
            minimum_geodesic_separation=training.minimum_geodesic_separation,
            dynamic_max_geo_distance=training.dynamic_max_geo_distance.enabled,
            relay_options=training.dynamic_max_geo_distance,
            minimum_geodesic_separation_multiplier=training.minimum_geodesic_separation_multiplier,
            skip_on_no_pair_found=training.skip_on_no_pair_found,
            spawn_pair_max_attempts=training.spawn_pair_max_attempts,
            building_bank=building_bank,
            memory_comm_every_k_steps=network.memory_comm_every_k_steps,
            **({"resample_maps": True} if resample_maps else {}),
        )
    obs_dim = observation_dim(cfg)
    sig_dim = network.tarmac_sig_dim
    val_dim = network.tarmac_val_dim
    init_log("Environment functions constructed. Initializing model...")
    model = build_model(level)
    init_log("Model constructed. Preparing checkpoint, optimizer and rollout buffer...")
    resume_from = checkpoint_path or training.checkpoint_path
    _validate_relay_resume(training, resume_from)
    resume = resolve_resume_state(training, run_name, training.num_envs, training.num_steps)
    completed_checkpoint = {}
    if resume_from and training.ckpt_loading_mode == "resume":
        ready = Path(resume_from) / "recovery_ready.json"
        if ready.is_file():
            completed_checkpoint = json.loads(ready.read_text(encoding="utf-8"))
        remaining_steps = max(0, training.total_timesteps - resume.completed_run_steps)
        num_updates = resume.start_update + remaining_steps // (training.num_envs * training.num_steps)
        if (completed_checkpoint.get("final") and os.environ.get("SWARMECHO_RECOVERY_ID")
                and completed_checkpoint.get("recovery_id") == os.environ["SWARMECHO_RECOVERY_ID"]):
            num_updates = resume.start_update
    if resume_from:
        restored = restore_model_checkpoint(model, resume_from)
        terminal_print(f"  Restored weights  ✓  {display_path(restored)}")
    if training.adaptive_target_spawn.enabled:
        from swarmecho.training.adaptive_spawn import (
            AdaptiveTargetSpawnController, AdaptiveSpawnJournal,
            MapAdaptiveTargetSpawnController, AdaptiveMapSpawnJournal,
        )
        feasibility_budget = cfg.geometric_agent_budget if training.dynamic_max_geo_distance.enabled else None
        if building_bank is not None:
            adaptive_spawn = MapAdaptiveTargetSpawnController(
                adaptive_spawn_buildings, map_names, cfg, training.adaptive_target_spawn,
                compact=resample_maps,
                feasibility_budget=feasibility_budget, relay_options=training.dynamic_max_geo_distance)
        else:
            adaptive_spawn = AdaptiveTargetSpawnController(level.building, cfg, training.adaptive_target_spawn,
                feasibility_budget=feasibility_budget, relay_options=training.dynamic_max_geo_distance)
        resume_controller = bool(resume_from and training.ckpt_loading_mode == "resume")
        spawn_history_path = destination / "adaptive_spawn" / "history.jsonl"
        spawn_history_paths = [
            spawn_history_path,
            destination / "adaptive_spawn_maps" / "history.jsonl",
            *destination.glob("adaptive_spawn_maps/*/adaptive_spawn/history.jsonl"),
        ]
        if (not resume_controller and os.environ.get("SWARMECHO_RECOVERY_RETRY") != "1"
                and any(path.is_file() and path.stat().st_size for path in spawn_history_paths)):
            raise ValueError("Adaptive spawn history already exists; resume explicitly or choose a new run name.")
        if resume_controller:
            if not adaptive_spawn.restore(resume_from):
                terminal_print("[WARNING] Checkpoint has no adaptive spawn state; starting at baseline probabilities.")
            elif adaptive_spawn.update != resume.start_update:
                raise ValueError("Adaptive spawn state update does not match the model checkpoint.")
        if (not adaptive_spawn.eligibility_ready and
                (resume_controller or os.environ.get("SWARMECHO_RECOVERY_RETRY") == "1")):
            adaptive_spawn.restore_eligibility(destination / "adaptive_spawn" / "eligibility.json")
        if adaptive_spawn.eligibility_ready:
            target_spawn_weights = jnp.asarray(adaptive_spawn.box_weights)
        else:
            init_log("Adaptive target spawning: discovering eligible cells from the uniform accepted startup batch...")
    trainer = PPOTrainer(model, training)
    trainer.updates = resume.start_update
    buffer = MAPPORolloutBuffer(
        num_steps=training.num_steps,
        num_envs=training.num_envs,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        act_dim=model.act_dim,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        recurrent=True,
        hidden_dim=network.hidden_dim,
        tarmac_sig_dim=sig_dim,
        tarmac_val_dim=val_dim,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
        mask_inactive=True,
    )
    init_log("Optimizer and buffer prepared. Initial batched environment reset...")
    keys = jax.random.split(jax.random.PRNGKey(training.seed + 1), training.num_envs)
    if training.randomize_base or training.minimum_geodesic_separation or training.dynamic_max_geo_distance.enabled or building_bank is not None:
        reset_batch_size = min(32, training.num_envs)
        reset_batch_count = (training.num_envs + reset_batch_size - 1) // reset_batch_size
        reset_padded_count = reset_batch_count * reset_batch_size
        reset_keys = jnp.concatenate((keys, jnp.repeat(keys[-1:], reset_padded_count - training.num_envs, axis=0)))
        reset_keys = reset_keys.reshape((reset_batch_count, reset_batch_size) + keys.shape[1:])
        if building_bank is not None:
            from swarmecho.env.building_bank import balanced_map_ids
            lane_ids = (np.arange(training.num_envs, dtype=np.int32) % bank_map_count
                        if resample_maps else balanced_map_ids(training.num_envs, bank_map_count))
            padded_ids = np.pad(lane_ids, (0, reset_padded_count - training.num_envs), mode="edge")
            ids = jnp.asarray(padded_ids).reshape(reset_batch_count, reset_batch_size)
            reset_batches = jax.lax.map(lambda pair: jax.vmap(lambda key, index: reset(
                key, map_id=index, **({"target_spawn_weights": target_spawn_weights}
                                      if adaptive_spawn is not None else {})))(*pair), (reset_keys, ids))
        elif adaptive_spawn is not None:
            reset_batches = jax.lax.map(jax.vmap(lambda key: reset(key, target_spawn_weights=target_spawn_weights)), reset_keys)
        else:
            reset_batches = jax.lax.map(jax.vmap(reset), reset_keys)
        # Explicit dimensions avoid lax.map(batch_size=...)'s inferred -1
        # reshape, which fails for legitimate zero-sized geometry leaves.
        states = jax.tree_util.tree_map(
            lambda value: value.reshape((reset_padded_count,) + value.shape[2:])[:training.num_envs],
            reset_batches,
        )
    elif adaptive_spawn is not None:
        states = jax.vmap(lambda key: reset(key, target_spawn_weights=target_spawn_weights))(keys)
    else:
        states = jax.vmap(reset)(keys)
    # Synchronize once at startup so this milestone reports completion, not dispatch.
    jax.block_until_ready(states)
    def check_spawn_pairs(attempts):
        if np.any(np.asarray(attempts) == 0):
            if training.dynamic_max_geo_distance.enabled:
                raise ValueError(f"No certified spawn pair within geometric budget {cfg.geometric_agent_budget} after "
                                 f"{training.spawn_pair_max_attempts} proposals. Check map eligibility, "
                                 "minimum separation and the stage budget; no unfiltered fallback is allowed.")
            raise ValueError(
                f"No valid {'base spawn' if cfg.success_condition == 'coverage' else 'base/target pair'} "
                f"in {training.spawn_pair_max_attempts} attempts per environment. "
                f"Required roadmap separation: {training.minimum_geodesic_separation_multiplier * cfg.comm_radius if training.minimum_geodesic_separation else 0:g} m. "
                "Randomized bases keep the target fixed; it may have no qualifying base location. "
                "Reduce the separation multiplier or check spawn geometry; increasing the attempt limit only helps feasible targets."
            )

    if training.randomize_base or training.minimum_geodesic_separation or training.dynamic_max_geo_distance.enabled:
        check_spawn_pairs(states.spawn_pair_attempts)
    if adaptive_spawn is not None:
        if not adaptive_spawn.eligibility_ready:
            if resample_maps:
                from swarmecho.training.adaptive_maps import discover_cell_support
                discover_cell_support(reset, adaptive_spawn, states,
                    training.adaptive_map_sampling.eligibility_samples_per_map,
                    training.seed + cfg.geometric_agent_budget * 1009, init_log)
            else:
                adaptive_spawn.initialize_eligibility(jax.device_get(states.target_pos),
                                                      map_ids=jax.device_get(states.map_id))
        target_spawn_weights = jnp.asarray(adaptive_spawn.box_weights)
        eligibility_path = destination / "adaptive_spawn" / "eligibility.json"
        eligibility_path.parent.mkdir(parents=True, exist_ok=True)
        adaptive_spawn.save_eligibility(eligibility_path)
        spawn_controllers = getattr(adaptive_spawn, "controllers", (adaptive_spawn,))
        if logging.adaptive_cell_diagnostics:
            # Authored suites and persistent random banks use the same map controller.
            map_spawn_controller = isinstance(adaptive_spawn, MapAdaptiveTargetSpawnController)
            journal_type = AdaptiveMapSpawnJournal if map_spawn_controller else AdaptiveSpawnJournal
            adaptive_spawn_journal = journal_type(
                destination, adaptive_spawn, run_name,
                resume=resume_controller or os.environ.get("SWARMECHO_RECOVERY_RETRY") == "1",
                steps=resume.step_offset + resume.start_update * training.num_envs * training.num_steps,
                update=resume.start_update,
                **({"compact": adaptive_spawn.compact} if map_spawn_controller else {}),
            )
            for map_name, controller in zip(map_names, spawn_controllers):
                if controller.eligibility is not None:
                    discovery = controller.eligibility
                    init_log(f"Adaptive target spawning [{map_name}]: {np.count_nonzero(controller.eligible)}/{len(controller.cells)} cells retained; "
                             f"at least {discovery['minimum_count']}/{discovery['sample_count']} accepted startup targets (1%); "
                             "excluded cells have zero proposal probability.")
            init_log(f"Adaptive target spawning history: {display_path(adaptive_spawn_journal.directory)} (no W&B metrics).")
        else:
            # Journal constructors normally align fresh controller counters on resume/branch.
            for controller in spawn_controllers:
                if controller.update == 0 and not np.any(controller.total_episodes):
                    controller.update = resume.start_update
                    controller.steps = resume.step_offset + resume.start_update * training.num_envs * training.num_steps
                    controller.warning_history[0]["update"] = controller.update
            adaptive_spawn.update, adaptive_spawn.steps = spawn_controllers[0].update, spawn_controllers[0].steps
            retained = sum(np.count_nonzero(controller.eligible) for controller in spawn_controllers)
            total = sum(len(controller.cells) for controller in spawn_controllers)
            init_log(f"Adaptive target spawning: {retained}/{total} cells retained across {len(spawn_controllers)} maps; "
                     "individual-cell diagnostics disabled.")
    if training.adaptive_map_sampling.enabled:
        from swarmecho.training.adaptive_maps import AdaptiveMapSampler, AdaptiveMapJournal
        if building_bank is None or map_identities is None:
            raise ValueError("Adaptive map sampling requires a persistent prepared map bank.")
        adaptive_maps = AdaptiveMapSampler(map_identities, training.adaptive_map_sampling,
            stage_identity={"env": asdict(cfg), "reward": asdict(level.reward),
                            "seed": training.seed, "filtered": training.dynamic_max_geo_distance.enabled,
                            "randomize_base": training.randomize_base,
                            "minimum_geodesic_separation": training.minimum_geodesic_separation,
                            "minimum_geodesic_separation_multiplier": training.minimum_geodesic_separation_multiplier,
                            "adaptive_cells": asdict(training.adaptive_target_spawn)
                                if training.adaptive_target_spawn.enabled else None,
                            **({"relay_estimator": _global_relay_identity(training.dynamic_max_geo_distance)}
                               if training.dynamic_max_geo_distance.enabled
                               and training.dynamic_max_geo_distance.mode == "global_candidates" else {})})
        resume_maps = bool(resume_from and training.ckpt_loading_mode == "resume")
        if resume_maps:
            if not adaptive_maps.restore(resume_from):
                terminal_print("[WARNING] No adaptive map state in checkpoint; starting fresh map warmup.")
            elif adaptive_maps.update != resume.start_update:
                raise ValueError("Adaptive map state update does not match the model checkpoint.")
        adaptive_maps.update = resume.start_update
        adaptive_maps.steps = resume.step_offset + resume.start_update * training.num_envs * training.num_steps
        adaptive_maps.episode_starts += np.bincount(jax.device_get(states.map_id), minlength=bank_map_count)
        map_sampling_cdf = jnp.asarray(adaptive_maps.cdf)
        adaptive_map_journal = AdaptiveMapJournal(destination, adaptive_maps,
            resume=resume_maps or os.environ.get("SWARMECHO_RECOVERY_RETRY") == "1")
        init_log(f"Adaptive map sampling: {bank_map_count} maps, {training.adaptive_map_sampling.min_probability_fraction:.0%} "
                 "uniform floor; independent success windows and episode-boundary map selection.")
    elif resample_maps:
        init_log(f"Uniform map sampling: all {bank_map_count} training maps available at each episode reset; "
                 "adaptive weighting and map warmup disabled.")
    init_log("Initial reset complete on device. Preparing recurrent state and logging...")
    if cfg.adapt_size.enabled:
        validate_zones(jax.device_get(states.adaptive.zone_valid))
    actor_hidden = model.initial_actor_hidden((training.num_envs,))
    actor_signature = model.initial_actor_signature((training.num_envs,))
    actor_value = model.initial_actor_value((training.num_envs,))
    critic_hidden = model.initial_critic_hidden((training.num_envs,))
    base_valid = jnp.zeros(training.num_envs, dtype=jnp.bool_)
    base_signature = jnp.zeros((training.num_envs, sig_dim), dtype=jnp.float32)
    base_value = jnp.zeros((training.num_envs, val_dim), dtype=jnp.float32)
    resets = jnp.zeros((training.num_envs, cfg.num_agents), dtype=jnp.bool_)
    latest_stats: dict[str, float] = {}
    eval_hold = EvaluationHold(
        evaluation.early_exit_success_rate,
        evaluation.early_exit_hold_evals,
        evaluation.early_exit_min_success_length_reduction,
    )
    training_hold = None
    if evaluation.early_exit and evaluation.early_exit_source == "training":
        training_hold = TrainingSuccessHold(
            lane_ids if level.static_map_suite else [0] * training.num_envs,
            cfg.max_steps, evaluation.early_exit_success_rate,
            evaluation.early_exit_hold_evals,
            evaluation.early_exit_min_success_length_reduction,
        )
    final_evaluation_result = None
    episode_returns = np.zeros(training.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(training.num_envs, dtype=np.int32)
    episode_success = np.zeros(training.num_envs, dtype=np.float32)
    episode_found = np.zeros(training.num_envs, dtype=np.float32)
    episode_visually_found = np.zeros(training.num_envs, dtype=np.float32)
    episode_idle = np.zeros(training.num_envs, dtype=np.float32)
    episode_gap = np.zeros(training.num_envs, dtype=np.float32)
    episode_progress_pct = np.zeros(training.num_envs, dtype=np.float32)
    episode_coverage = np.zeros(training.num_envs, dtype=np.float32)
    episode_peer_informed = np.zeros(training.num_envs, dtype=np.float32)
    episode_gap_gain = np.zeros(training.num_envs, dtype=np.float64)
    episode_gap_loss = np.zeros(training.num_envs, dtype=np.float64)
    episode_rewards = {
        name: np.zeros(training.num_envs, dtype=np.float64)
        for name in ("exploration_reward", "collision_reward", "finder_reward", "time_presure_reward", "gap_reward",
                     "peer_informing_reward", "target_found_reward", "success_reward",
                     "useless_agent_at_success_reward",
                     "useless_agent_at_success_all_agents_reward",
                     "no_movement_termination_penalty")
    }
    if level.reward.enable_chain_efficiency_reward:
        episode_rewards["chain_efficiency_reward"] = np.zeros(training.num_envs, dtype=np.float64)
    # Collect one completed episode per parallel environment,
    # rather than a fixed-size history of partial/live rollouts.
    window_size = training.num_envs
    adaptive_window = (AdaptiveWindow(
        window_size, target_agents=cfg.geometric_agent_budget,
        max_agents=cfg.num_agents, initial_agents=cfg.adapt_size.initial_agents,
    ) if cfg.adapt_size.enabled else None)
    required_agents_window = (RequiredAgentsDeploymentWindow(
        logging.required_agents_metrics_window_episodes,
        episode_window_size=window_size, max_required_agents=cfg.geometric_agent_budget,
    ) if cfg.adapt_size.enabled and training.dynamic_max_geo_distance.enabled else None)
    adaptive_reward_names = REWARD_NAMES + (
        VOTE_SHAPING_REWARD_NAMES if level.reward.priveleged_vote_reward.enabled else ()
    )
    if cfg.adapt_size.enabled:
        episode_rewards.update({name: np.zeros(training.num_envs)
                                for name in (*adaptive_reward_names, *REFERENCE_DIAGNOSTIC_NAMES)})
    window_ret: deque[float] = deque(maxlen=window_size)
    window_len: deque[int] = deque(maxlen=window_size)
    window_succ: deque[float] = deque(maxlen=window_size)
    window_fnd: deque[float] = deque(maxlen=window_size)
    window_visually_found: deque[float] = deque(maxlen=window_size)
    window_idle: deque[float] = deque(maxlen=window_size)
    window_gap: deque[float] = deque(maxlen=window_size)
    window_prog_pct: deque[float] = deque(maxlen=window_size)
    window_cov: deque[float] = deque(maxlen=window_size)
    window_best_hold: deque[float] = deque(maxlen=window_size)
    window_peer_informed: deque[float] = deque(maxlen=window_size)
    window_gap_gain: deque[float] = deque(maxlen=window_size)
    window_gap_loss: deque[float] = deque(maxlen=window_size)
    window_pair_attempts: deque[int] = deque(maxlen=window_size)
    window_success_target_distance: deque[float] = deque(maxlen=window_size)
    window_rewards = {name: deque(maxlen=window_size) for name in episode_rewards}
    path_stat_names = (["number_of_valid_paths"] if level.reward.allow_redundancy_reward or level.reward.enable_chain_efficiency_reward else [])
    if level.reward.enable_chain_efficiency_reward:
        path_stat_names.append("chain_efficiency")
    window_path_stats = {name: deque(maxlen=window_size) for name in path_stat_names}
    completed_eps_count = 0
    history_path = destination / "training_history.jsonl"
    training_snapshot = asdict(training)
    if not training.adaptive_target_spawn.enabled:
        training_snapshot.pop("adaptive_target_spawn")
    if not training.adaptive_map_sampling.enabled and not level.random_buildings.enabled:
        training_snapshot.pop("adaptive_map_sampling")
    elif level.validation.enabled:
        training_snapshot["adaptive_map_sampling"].pop("held_out_maps", None)
    config_snapshot = {
        "name": level.name,
        "building": level.building_name,
        "env": {**{k: v for k, v in asdict(level.env).items() if k != "adapt_size"}, "map_names": level.map_names},
        "adapt_size": asdict(level.adapt_size),
        "reward": asdict(level.reward),
        "training": training_snapshot,
        "network": asdict(network),
        "inspection_evaluation": inspection_config_dict(evaluation),
        "validation": asdict(level.validation),
        "random_buildings": asdict(level.random_buildings),
        "logging": asdict(logging),
    }
    (destination / "config.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8"
    )
    init_log("State and configuration prepared. Initializing W&B...")
    wandb_run = init_wandb(logging, training, run_name, destination, config_snapshot)
    init_log("W&B initialization returned. Preparing first rollout...")

    def perform_validation(update, steps):
        metrics = validation_runner.run(model, update=update, steps=steps)
        stop = validation_runner.observe_gate(metrics)
        values = {f"validation/{key}": value for key, value in metrics.items()}
        latest_stats.update(values)
        terminal_print(f"[VALIDATION update={update}] success={metrics['success_rate']:.2%}; "
                       f"maps={metrics['maps_evaluated']}/{metrics['maps_total']}; "
                       f"episodes={metrics['episodes']}; hold={validation_runner.hold_count}", flush=True)
        if validation_runner.config.save_best_checkpoint and (
                validation_runner.best_success_rate is None
                or metrics["success_rate"] > validation_runner.best_success_rate):
            previous_best = validation_runner.best_success_rate
            validation_runner.best_success_rate = metrics["success_rate"]
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_training_checkpoint(model, checkpoint_dir / "best_val_checkpoint",
                    run_name=run_name, update=update, num_envs=training.num_envs,
                    num_steps=training.num_steps, prior_history=resume.prior_history,
                    total_steps=steps, adaptive_spawn=adaptive_spawn, adaptive_maps=adaptive_maps,
                    validation=validation_runner, run_dir=destination, metrics=latest_stats)
            except BaseException:
                validation_runner.best_success_rate = previous_best
                raise
            terminal_print(f"[VALIDATION] saved best actor and critic: {metrics['success_rate']:.2%}", flush=True)
        if wandb_run is not None:
            import wandb
            wandb.log(values, step=steps, commit=False)
        return stop

    _, params = nnx.split(model)
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
    total_steps = resume.completed_run_steps + (num_updates - resume.start_update) * training.num_envs * training.num_steps
    terminal_print("\n══════════════════════════════════════════════════════")
    terminal_print("  SwarmEcho — Recurrent MAPPO + TarMAC")
    terminal_print("══════════════════════════════════════════════════════")
    terminal_print(f"  level / building : {level.name} / {level.building_name}")
    terminal_print(f"  devices          : {jax.devices()}")
    terminal_print(f"  agents           : {cfg.num_agents}")
    terminal_print(f"  observation      : {obs_dim}  (wall rays: {cfg.wall_sensing_rays}, {cfg.wall_sensing_layout}, {cfg.wall_sensing_ray_max_distance:g} m; drone bins: {cfg.drone_radar_bins})")
    terminal_print(f"  radar mode       : {cfg.radar_mode}")
    if cfg.radar_mode == "band_sep_counting":
        terminal_print(f"  drone counting   : {max(1, cfg.distance_bands)} distance bands; cap {cfg.radar_count_cap}")
    terminal_print("  action           : XYZ continuous force")
    terminal_print(f"  model parameters : {parameter_count:,}")
    terminal_print(f"  environments     : {training.num_envs:,}")
    terminal_print(f"  rollout / updates: {training.num_steps} / {num_updates}")
    terminal_print(f"  eval map         : {evaluation_level.building_name}")
    terminal_print(f"  eval agents      : {evaluation_level.env.num_agents}")
    if level.static_map_suite:
        terminal_print(f"  eval mode        : {len(level.map_paths)} training maps, sequential, no action noise")
    elif evaluation.random_eval:
        terminal_print(f"  eval mode        : {evaluation.random_eval_envs} persistent maps, sequential, no action noise")
    else:
        final_mode = (f"robust {evaluation.eval_robustness_runs} runs; noise +/-{evaluation.eval_action_noise_max:g}"
                      if evaluation.final_robustness else "single-pass, no action noise")
        terminal_print(f"  eval mode        : {'robust' if evaluation.training_robustness else 'single-pass'}; final {final_mode}")
    terminal_print(f"  terminal logging : every {logging.terminal_logging_frequency}; warmup={logging.terminal_log_warmup}; init={logging.terminal_log_init}")
    terminal_print(f"  PPO critic units : {training.value_normalization}; actor/value clip "
          f"{training.actor_clip_eps}/{training.value_clip_eps}; entropy {training.entropy_mode}")
    terminal_print(f"  idle penalty     : {level.reward.no_movement_termination_penalty} (effective reward config)")
    if training.training_noise:
        terminal_print(f"  training noise   : uniform pre-tanh +/-{training.noise_level:g}")
    terminal_print(f"  total env steps  : {total_steps:,}")
    terminal_print(f"  output           : {display_path(destination)}")
    if wandb_run is not None:
        terminal_print(f"  W&B              : {wandb_run.url}")
    terminal_print("──────────────────────────────────────────────────────")
    log_every = logging.terminal_logging_frequency
    start_time = time.perf_counter()

    # A complete rollout is one compiled device program.  In particular, do
    # not materialise any transition inside the scan: the host receives the
    # stacked rollout once per update for the existing CPU replay buffer.
    comm_cadence = int(network.memory_comm_every_k_steps)

    def rollout_impl(
        model,
        states,
        keys,
        actor_hidden,
        actor_signature,
        actor_value,
        critic_hidden,
        base_valid,
        base_signature,
        base_value,
        resets,
        target_spawn_weights=None,
        map_sampling_cdf=None,
    ):
        def rollout_step(carry, step_index):
            (
                states,
                keys,
                actor_hidden,
                actor_signature,
                actor_value,
                critic_hidden,
                base_valid,
                base_signature,
                base_value,
                resets,
            ) = carry
            obs = jax.vmap(observations)(states)
            split = jax.vmap(lambda key: jax.random.split(key))(keys)
            next_keys, action_roots = split[:, 0], split[:, 1]
            agent_keys = jax.vmap(lambda key: jax.random.split(key, cfg.num_agents))(
                action_roots
            )
            comm_masks, in_base_range, base_masks, base_sig_in, base_val_in = (
                _communication_inputs(states, cfg, base_valid, base_signature, base_value)
            )
            share_step = communication_due(states.step, comm_cadence)
            comm_masks = comm_masks & share_step[:, None, None]
            base_masks = base_masks & share_step[:, None]

            def policy_one(obs_e, keys_e, ah, sig, val, ch, cm, active, bs, bv, bm, rst):
                return model.rollout_step_recurrent(
                    obs_e,
                    keys_e,
                    ah,
                    ch,
                    rst,
                    actor_signature=sig,
                    actor_value=val,
                    comm_mask=cm,
                    active=active,
                    base_signature=bs,
                    base_value=bv,
                    base_memory_mask=bm,
                )

            (
                next_actor_hidden,
                emitted_signature,
                emitted_value,
                next_critic_hidden,
                actions,
                log_probs,
                values,
            ) = jax.vmap(policy_one)(
                obs,
                agent_keys,
                actor_hidden,
                actor_signature,
                actor_value,
                critic_hidden,
                comm_masks,
                states.active,
                base_sig_in,
                base_val_in,
                base_masks,
                resets | policy_resets(states),
            )
            # Mirror robust evaluation: perturb pre-tanh actions with bounded
            # uniform noise. Keep the sampled policy actions and their
            # log-probabilities in the rollout, since the perturbation belongs
            # to the environment transition rather than to the policy density.
            if training.training_noise:
                noise_roots = jax.vmap(lambda key: jax.random.fold_in(key, 1))(
                    action_roots
                )
                action_noise = jax.vmap(
                    lambda key: jax.random.uniform(
                        key,
                        (cfg.num_agents, 3),
                        dtype=actions.dtype,
                        minval=-training.noise_level,
                        maxval=training.noise_level,
                    )
                )(noise_roots)
                executed_actions = actions.at[..., :3].add(action_noise)
            else:
                executed_actions = actions
            executed_actions = jnp.where(states.active[..., None], executed_actions, 0.0)
            if resample_maps:
                next_states, rewards, dones, info = env_step.batched(
                    states, physical_actions(executed_actions), target_spawn_weights,
                    map_sampling_cdf=map_sampling_cdf)
            elif target_spawn_weights is None:
                next_states, rewards, dones, info = env_step.batched(
                    states, physical_actions(executed_actions)
                )
            else:
                next_states, rewards, dones, info = env_step.batched(
                    states, physical_actions(executed_actions), target_spawn_weights
                )
            next_base_valid, next_base_signature, next_base_value = _update_base_memory(
                states,
                in_base_range,
                emitted_signature,
                emitted_value,
                base_valid,
                base_signature,
                base_value,
                dones,
                share_step,
            )
            next_resets = jnp.broadcast_to(dones[:, None], resets.shape) | policy_resets(next_states)
            transition = {
                "obs": obs,
                "actions": actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": rewards,
                "dones": dones,
                "rnn_resets": resets | policy_resets(states),
                "comm_masks": comm_masks,
                "active_masks": states.active,
                "agent_terminated": info["agent_terminated"],
                **({name: info[name] for name in (*INFO_NAMES, *adaptive_reward_names, *REFERENCE_DIAGNOSTIC_NAMES)}
                   if cfg.adapt_size.enabled else {}),
                "base_signatures": base_sig_in,
                "base_values": base_val_in,
                "base_memory_masks": base_masks,
                "exploration_reward": info["exploration_reward"],
                "collision_reward": info["collision_reward"],
                "finder_reward": info["finder_reward"],
                "time_presure_reward": info["time_presure_reward"],
                "gap_reward": info["gap_reward"],
                "gap_gain": info["gap_gain"],
                "gap_loss": info["gap_loss"],
                "peer_informing_reward": info["peer_informing_reward"],
                "peer_informed_count": info["peer_informed_count"],
                "best_chain_held_steps": info["best_chain_held_steps"],
                "target_found_reward": info["target_found_reward"],
                "success_reward": info["success_reward"],
                "useless_agent_at_success_reward": info["useless_agent_at_success_reward"],
                "useless_agent_at_success_all_agents_reward": info["useless_agent_at_success_all_agents_reward"],
                "no_movement_termination_penalty": info["no_movement_termination_penalty"],
                **({"chain_efficiency_reward": info["chain_efficiency_reward"]}
                   if level.reward.enable_chain_efficiency_reward else {}),
                "success": info["success"],
                "global_target_found": info["global_target_found"],
                "global_visually_found": info["global_visually_found"],
                "idle_terminated": info["idle_terminated"],
                "chain_gap_dist": info["chain_gap_dist"],
                "chain_progress_pct": info["chain_progress_pct"],
                "global_coverage": info["global_coverage"],
                "terminal_target_pos": info["terminal_target_pos"],
                **({"terminal_required_agents": info["terminal_required_agents"]}
                   if required_agents_window is not None else {}),
                **({"terminal_map_id": states.map_id}
                   if training.adaptive_map_sampling.enabled or
                      (training.adaptive_target_spawn.enabled and building_bank is not None) else {}),
                **({"started_map_id": next_states.map_id} if training.adaptive_map_sampling.enabled else {}),
                "terminal_base_pos": states.base_pos,
                "spawn_pair_attempts": states.spawn_pair_attempts,
                **{name: info[name] for name in path_stat_names},
            }
            next_carry = (
                next_states,
                next_keys,
                next_actor_hidden,
                emitted_signature,
                emitted_value,
                next_critic_hidden,
                next_base_valid,
                next_base_signature,
                next_base_value,
                next_resets,
            )
            return next_carry, transition

        carry, rollout = jax.lax.scan(
            rollout_step,
            (
                states,
                keys,
                actor_hidden,
                actor_signature,
                actor_value,
                critic_hidden,
                base_valid,
                base_signature,
                base_value,
                resets,
            ),
            jnp.arange(training.num_steps, dtype=jnp.int32),
        )
        final_states, _, _, _, _, final_critic_hidden, _, _, _, final_resets = carry
        final_obs = jax.vmap(observations)(final_states)

        def value_one(obs_e, hidden_e, active_e, reset_e):
            return model.get_value_recurrent(
                obs_e, hidden_e, reset_e, active=active_e,
            )[1]

        final_values = jax.vmap(value_one)(
            final_obs, final_critic_hidden, final_states.active, final_resets | policy_resets(final_states),
        )
        return (*carry, final_values, rollout)

    rollout_jit = nnx.jit(rollout_impl)

    for update in range(resume.start_update + 1, num_updates + 1):
        update_started = time.perf_counter()
        initial_update = update == resume.start_update + 1
        if initial_update:
            init_log("First rollout: entering JIT tracing/compilation and dispatch...")
        start_actor_hidden = actor_hidden
        start_actor_signature = actor_signature
        start_actor_value = actor_value
        start_critic_hidden = critic_hidden
        (
            states,
            keys,
            actor_hidden,
            actor_signature,
            actor_value,
            critic_hidden,
            base_valid,
            base_signature,
            base_value,
            resets,
            final_values,
            rollout_device,
        ) = rollout_jit(
            model,
            states,
            keys,
            actor_hidden,
            actor_signature,
            actor_value,
            critic_hidden,
            base_valid,
            base_signature,
            base_value,
            resets,
            **({"target_spawn_weights": target_spawn_weights} if adaptive_spawn is not None else {}),
            **({"map_sampling_cdf": map_sampling_cdf} if adaptive_maps is not None else {}),
        )
        if initial_update:
            init_log("First rollout call returned. Waiting for device results and host transfer...")
        rollout_host, final_values_host, initial_states = jax.device_get(
            (
                rollout_device,
                final_values,
                (start_actor_hidden, start_critic_hidden, start_actor_signature, start_actor_value),
            )
        )
        rollout_finished = time.perf_counter()
        if cfg.adapt_size.enabled:
            validate_zones(rollout_host["adapt_zone_valid"])
        if training.randomize_base or training.minimum_geodesic_separation or training.dynamic_max_geo_distance.enabled:
            check_spawn_pairs(rollout_host["spawn_pair_attempts"])
            check_spawn_pairs(states.spawn_pair_attempts)
        if initial_update:
            init_log("First rollout results received. Building host buffer and advantages...")
        buffer.reset(
            initial_states[0], initial_states[1], initial_states[2], initial_states[3]
        )
        for step_index in range(training.num_steps):
            rewards_host = rollout_host["rewards"][step_index]
            dones_host = rollout_host["dones"][step_index].astype(bool)
            episode_returns += rewards_host.sum(axis=-1)
            episode_lengths += 1
            episode_success = np.maximum(episode_success, rollout_host["success"][step_index])
            episode_found = np.maximum(
                episode_found, rollout_host["global_target_found"][step_index]
            )
            episode_visually_found = np.maximum(
                episode_visually_found,
                rollout_host["global_visually_found"][step_index],
            )
            episode_idle = np.maximum(
                episode_idle, rollout_host["idle_terminated"][step_index]
            )
            episode_gap = rollout_host["chain_gap_dist"][step_index]
            episode_progress_pct = rollout_host["chain_progress_pct"][step_index]
            episode_coverage = rollout_host["global_coverage"][step_index]
            episode_peer_informed += rollout_host["peer_informed_count"][step_index]
            episode_gap_gain += rollout_host["gap_gain"][step_index].sum(axis=-1)
            episode_gap_loss += rollout_host["gap_loss"][step_index].sum(axis=-1)
            for reward_name, accumulator in episode_rewards.items():
                accumulator += rollout_host[reward_name][step_index].sum(axis=-1)
            buffer.add(
                MAPPOTransition(
                    obs=rollout_host["obs"][step_index],
                    actions=rollout_host["actions"][step_index],
                    log_probs=rollout_host["log_probs"][step_index],
                    values=rollout_host["values"][step_index],
                    rewards=rewards_host,
                    dones=dones_host,
                    rnn_resets=rollout_host["rnn_resets"][step_index],
                    comm_masks=rollout_host["comm_masks"][step_index],
                    active_masks=rollout_host["active_masks"][step_index],
                    agent_terminated=rollout_host["agent_terminated"][step_index],
                    base_signatures=rollout_host["base_signatures"][step_index],
                    base_values=rollout_host["base_values"][step_index],
                    base_memory_masks=rollout_host["base_memory_masks"][step_index],
                )
            )
            completed_indices = np.flatnonzero(dones_host)
            for index in completed_indices:
                if training_hold is not None:
                    training_hold.append(index, episode_success[index] > .5, episode_lengths[index])
                if adaptive_window is not None:
                    adaptive_window.append(
                        {name: rollout_host[name][step_index, index] for name in INFO_NAMES},
                        success=bool(episode_success[index]),
                    )
                if required_agents_window is not None:
                    required_agents_window.append(
                        rollout_host["terminal_required_agents"][step_index, index],
                        rollout_host["adapt_end_agents"][step_index, index],
                        success=bool(episode_success[index]),
                    )
                window_ret.append(float(episode_returns[index]))
                window_len.append(int(episode_lengths[index]))
                window_succ.append(float(episode_success[index]))
                window_fnd.append(float(episode_found[index]))
                window_visually_found.append(float(episode_visually_found[index]))
                window_idle.append(float(episode_idle[index]))
                window_gap.append(float(episode_gap[index]))
                window_prog_pct.append(float(episode_progress_pct[index]))
                window_cov.append(float(episode_coverage[index]))
                window_best_hold.append(float(rollout_host["best_chain_held_steps"][step_index, index]))
                window_peer_informed.append(float(episode_peer_informed[index]))
                window_gap_gain.append(float(episode_gap_gain[index]))
                window_gap_loss.append(float(episode_gap_loss[index]))
                window_pair_attempts.append(int(rollout_host["spawn_pair_attempts"][step_index, index]))
                if episode_success[index] > 0.5:
                    for name, window in window_path_stats.items():
                        window.append(float(rollout_host[name][step_index, index]))
                    target_distance = np.linalg.norm(
                        rollout_host["terminal_target_pos"][step_index, index]
                        - rollout_host["terminal_base_pos"][step_index, index]
                    ) / max(level.building.max_base_to_top_corner_m, 1e-6)
                    window_success_target_distance.append(float(target_distance))
                else:
                    window_success_target_distance.append(float("nan"))
                for reward_name, accumulator in episode_rewards.items():
                    window_rewards[reward_name].append(float(accumulator[index]))
            completed_eps_count += len(completed_indices)
            episode_returns[completed_indices] = 0.0
            episode_lengths[completed_indices] = 0
            episode_success[completed_indices] = 0.0
            episode_found[completed_indices] = 0.0
            episode_visually_found[completed_indices] = 0.0
            episode_idle[completed_indices] = 0.0
            episode_peer_informed[completed_indices] = 0.0
            episode_gap_gain[completed_indices] = 0.0
            episode_gap_loss[completed_indices] = 0.0
            # These are read-only host views of per-step rollout data, and
            # are overwritten on the next step rather than accumulated across
            # an episode.  They therefore do not need resetting here.
            for accumulator in episode_rewards.values():
                accumulator[completed_indices] = 0.0
        advantages, returns = buffer.compute_gae(final_values_host, None)
        minibatches = buffer.get_minibatches(
            advantages,
            returns,
            training.num_minibatches,
            jax.random.fold_in(jax.random.PRNGKey(training.seed), update),
        )
        if initial_update:
            init_log("First buffer and advantages ready. Entering first optimizer update...")
        optimizer_started = time.perf_counter()
        latest_stats = trainer.update(minibatches)
        if initial_update or training.profile_timing:
            jax.block_until_ready(latest_stats)
        if initial_update:
            init_log("First optimizer update complete. Initialization diagnostics finished.")
        optimizer_finished = time.perf_counter()
        record_timing(
            update, "training", first_update=initial_update,
            rollout_and_transfer_s=rollout_finished - update_started,
            buffer_and_advantages_s=optimizer_started - rollout_finished,
            optimizer_s=optimizer_finished - optimizer_started,
            total_s=optimizer_finished - update_started,
            training_sps=training.num_envs * training.num_steps / max(optimizer_finished - update_started, 1e-6),
        )
        elapsed = time.perf_counter() - start_time
        session_steps = update * training.num_envs * training.num_steps
        steps_done = resume.step_offset + session_steps
        if adaptive_maps is not None:
            completed = rollout_host["dones"].astype(bool)
            adaptive_maps.observe(rollout_host["terminal_map_id"][completed], rollout_host["success"][completed],
                step_map_ids=rollout_host["terminal_map_id"],
                started_map_ids=rollout_host["started_map_id"][completed])
            map_row, map_metrics = adaptive_maps.advance(update, steps_done, jax.device_get(states.map_id),
                individual_diagnostics=logging.individual_map_diagnostics)
            adaptive_map_journal.append(map_row)
            map_sampling_cdf = jnp.asarray(adaptive_maps.cdf)
        if adaptive_spawn is not None:
            completed = rollout_host["dones"].astype(bool)
            adaptive_spawn.observe(rollout_host["terminal_target_pos"][completed],
                                   rollout_host["success"][completed],
                                   map_ids=(rollout_host["terminal_map_id"][completed]
                                            if "terminal_map_id" in rollout_host else None))
            spawn_row, spawn_warnings = adaptive_spawn.advance(update, steps_done,
                diagnostics=logging.adaptive_cell_diagnostics)
            if adaptive_spawn_journal is not None:
                adaptive_spawn_journal.append(spawn_row)
            for message in spawn_warnings:
                terminal_print(message)
            target_spawn_weights = jnp.asarray(adaptive_spawn.box_weights)
        completed_session_steps = (update - resume.start_update) * training.num_envs * training.num_steps
        sps = completed_session_steps / max(elapsed, 1e-6)
        latest_stats.update(map_metrics)
        eta_seconds = (num_updates - update) * elapsed / max(1, update - resume.start_update)
        latest_stats.update(
            {
                "global_step": float(steps_done),
                "sps": float(sps),
                "completed_episodes": float(completed_eps_count),
                "mean_episode_return": float(np.mean(window_ret)) if window_ret else 0.0,
                "mean_episode_length": float(np.mean(window_len)) if window_len else 0.0,
                "rolling_success_rate": float(np.mean(window_succ)) if window_succ else 0.0,
                "target_found_rate": float(np.mean(window_fnd)) if window_fnd else 0.0,
                "visually_found_rate": float(np.mean(window_visually_found)) if window_visually_found else 0.0,
                "chain_gap": float(np.mean(window_gap)) if window_gap else 0.0,
                "chain_progress_pct": float(np.mean(window_prog_pct)) if window_prog_pct else 0.0,
                "mean_terminal_coverage": float(np.mean(window_cov)) if window_cov else 0.0,
                "mean_best_chain_hold": float(np.mean(window_best_hold)) if window_best_hold else 0.0,
                "mean_peer_informed": float(np.mean(window_peer_informed)) if window_peer_informed else 0.0,
                "mean_gap_gain": float(np.mean(window_gap_gain)) if window_gap_gain else 0.0,
                "mean_gap_loss": float(np.mean(window_gap_loss)) if window_gap_loss else 0.0,
            }
        )
        successful_distances = np.asarray(window_success_target_distance, dtype=np.float64)
        successful_distances = successful_distances[np.isfinite(successful_distances)]
        latest_stats.update({
            "success_target_distance_mean_norm": float(np.mean(successful_distances)) if successful_distances.size else 0.0,
            "success_target_distance_max_norm": float(np.max(successful_distances)) if successful_distances.size else 0.0,
        })
        latest_stats.update(
            {
                f"reward_{name}": float(np.mean(values)) if values else 0.0
                for name, values in window_rewards.items()
            }
        )
        window_full = len(window_ret) == window_ret.maxlen
        train_stop = False
        if training_hold is not None:
            train_stop, gate_metrics = training_hold.observe()
            if gate_metrics is not None:
                latest_stats.update(gate_metrics)
                terminal_print(
                    f"[TRAIN GATE] worst-map success={gate_metrics['train_early_exit_min_map_success']:.1%}; "
                    f"worst-map speed reduction={gate_metrics['train_early_exit_min_map_length_reduction']:.1%}; "
                    f"hold={training_hold.hold.consecutive}/{max(1, evaluation.early_exit_hold_evals)}",
                    flush=True,
                )
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps({"update": update, **latest_stats}) + "\n")
        if update % log_every == 0 and (window_full or logging.terminal_log_warmup):
            eta_m, eta_s = divmod(int(eta_seconds), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"
            display_coverage = f"{np.mean(window_cov):>5.1%}" if window_full else " ----"
            display_length = f"{np.mean(window_len):>6.0f}" if window_full else "  ----"
            display_found = f"{np.mean(window_fnd):>5.1%}" if window_full else " ----"
            display_progress = f"{np.mean(window_prog_pct):>5.1f}%" if window_full else "  ---%"
            display_success = f"{np.mean(window_succ):>5.1%}" if window_full else " ----"
            pair_stat = ""
            if training.minimum_geodesic_separation or training.dynamic_max_geo_distance.enabled:
                rejected = (1.0 - len(window_pair_attempts) / sum(window_pair_attempts)) if window_full else 0.0
                display_rejected = f"{rejected:>5.1%}" if window_full else " ----"
                pair_stat = f"  pair_reject={display_rejected}"
            terminal_print(
                f" [{update:>4}/{num_updates}]  "
                f"steps={steps_done:>12,}  sps={sps:>6,.0f}  "
                f"ep_len={display_length}  cov={display_coverage}  "
                f"found={display_found}  chain={display_progress}  "
                f"succ={display_success}  eta={eta}{pair_stat}"
            )
            if not window_full:
                current_steps = np.asarray(states.step)
                terminal_print(
                    f"         [warmup] completed_episodes={len(window_ret)}/{window_ret.maxlen} | "
                    f"env_steps: min={int(current_steps.min())} "
                    f"mean={float(current_steps.mean()):.1f} max={int(current_steps.max())}"
                )
        if wandb_run is not None:
            import wandb
            wandb_logs = {
                    "ppo/policy_loss": latest_stats["policy_loss"],
                    "ppo/value_loss": latest_stats["value_loss"],
                    "ppo/entropy": latest_stats["entropy"],
                    "ppo/approx_kl": latest_stats["approx_kl"],
                    "ppo/clip_fraction": latest_stats["clip_fraction"],
                    **{name: value for name, value in latest_stats.items()
                       if name.startswith(("critic/", "policy/"))},
                    "perf/sps": sps,
                    "perf/ppo_updates": update,
                    "perf/global_step": steps_done,
            }
            if window_full:
                wandb_logs.update({
                    "train/ep_return": latest_stats["mean_episode_return"],
                    "train/ep_length": latest_stats["mean_episode_length"],
                    "train/success_rate": latest_stats["rolling_success_rate"],
                    "train/no_movement_termination_rate": float(np.mean(window_idle)) * 100.0,
                    "train/target_found_rate": latest_stats["target_found_rate"],
                    "train/visually_found_rate": latest_stats["visually_found_rate"],
                    "train/chain_progress_pct": latest_stats["chain_progress_pct"],
                    "train/ep_length_reduction": (1.0 - latest_stats["mean_episode_length"] / cfg.max_steps) * 100.0,
                    "train/map_coverage_pct": latest_stats["mean_terminal_coverage"] * 100.0,
                    "train/best_chain_hold": latest_stats["mean_best_chain_hold"],
                    "train/peer_informed": latest_stats["mean_peer_informed"],
                    "train/gap_gain": latest_stats["mean_gap_gain"],
                    "train/gap_loss": latest_stats["mean_gap_loss"],
                    "train/episodes_completed": completed_eps_count,
                    "train/success_target_distance_mean_norm": latest_stats["success_target_distance_mean_norm"],
                    "train/success_target_distance_max_norm": latest_stats["success_target_distance_max_norm"],
                    **{f"rewards/{name}": latest_stats[f"reward_{name}"] for name in window_rewards},
                })
                if adaptive_window is not None:
                    wandb_logs.update(adaptive_window.metrics())
            if required_agents_window is not None:
                wandb_logs.update(required_agents_window.metrics())
            for name, window in window_path_stats.items():
                if len(window) == window.maxlen:
                    wandb_logs[f"train/{name}"] = float(np.mean(window))
            wandb_logs.update({f"train/{name.removeprefix('train_')}": value
                               for name, value in latest_stats.items() if name.startswith("train_early_exit_")})
            wandb_logs.update(map_metrics)
            # Keep this step open for evaluation metrics from the same update.
            wandb.log(wandb_logs, step=steps_done, commit=False)
        validation_stop = False
        if validation_runner is not None and (
                update % level.validation.val_freq == 0
                or (update == num_updates and level.validation.val_on_finalization)):
            validation_stop = perform_validation(update, steps_done)
        checkpoint_due = evaluation.save_model and schedule_due(
            update, evaluation.checkpoint_freq, evaluation.checkpoint_offset
        )
        if checkpoint_due and update < num_updates:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(
                model,
                checkpoint_dir / f"ckpt_{update:06d}",
                run_name=run_name,
                update=update,
                num_envs=training.num_envs,
                num_steps=training.num_steps,
                prior_history=resume.prior_history,
                total_steps=steps_done,
                adaptive_spawn=adaptive_spawn,
                adaptive_maps=adaptive_maps,
                validation=validation_runner,
                run_dir=destination,
                metrics=latest_stats,
            )
        # The final update bypasses the gate. Its training metric uses the same
        # mode as earlier evaluations; optional final robustness is a separate phase.
        is_final_update = update == num_updates
        if train_stop:
            terminal_print("[EARLY-STOP] every map passed fresh training episode windows; collecting final diagnostics.")
        threshold_met = (
            latest_stats["rolling_success_rate"] >= evaluation.eval_min_train_success
        )
        evaluation_enabled = threshold_met or is_final_update or train_stop
        if schedule_due(update, evaluation.eval_freq, evaluation.eval_offset) and not evaluation_enabled:
            terminal_print(
                f"[EVAL update={update}] skipped: training success "
                f"{latest_stats['rolling_success_rate']:.1%} < "
                f"{evaluation.eval_min_train_success:.1%} required.",
                flush=True,
            )
        eval_due = is_final_update or train_stop or validation_stop or (
            schedule_due(update, evaluation.eval_freq, evaluation.eval_offset)
            and evaluation_enabled
        )
        replay_due = bool(evaluation.eval_video) and (
            not is_final_update
            and (
                schedule_due(
                    update, evaluation.eval_video_freq, evaluation.eval_video_offset
                )
                and evaluation_enabled
            )
        )
        if eval_due or replay_due:
            evaluation_started = time.perf_counter()
            def report_run(metrics, index, runs, *, prefix="EVAL "):
                evaluation_row(
                    metrics, label=f"[{prefix} {index}/{runs}]", update=update,
                    total=num_updates, episodes=level.effective_eval_parallel_envs,
                )
            # Replay-only updates do not create metrics or advance the hold.
            if map_suite_levels is not None:
                from swarmecho.training.random_evaluation import evaluate_map_suite
                eval_metrics = evaluate_map_suite(model, map_suite_levels, destination,
                    update=update, steps=steps_done, eval_due=eval_due, replay_due=replay_due, on_run=report_run)
                evaluation_result = None
                stop_training = bool(
                    eval_metrics is not None and evaluation.early_exit and evaluation.early_exit_source == "evaluation"
                    and eval_hold.observe(eval_metrics["eval_success"],
                                          eval_metrics.get("eval_success_length_reduction")))
            else:
                eval_metrics, evaluation_result, stop_training = _evaluate_training_metrics(
                    model, evaluation_level, eval_hold,
                    eval_due=eval_due, on_run=report_run,
                )
            stop_training = stop_training or train_stop or validation_stop
            if (
                stop_training
                and map_suite_levels is not None
                and not replay_due
                and evaluation.eval_video
            ):
                # Early exit can happen between scheduled replay updates. Make
                # sure the stopping checkpoint still has representative suite
                # replays, without repeating its metric/heatmap evaluation.
                from swarmecho.training.random_evaluation import evaluate_map_suite
                evaluate_map_suite(
                    model, map_suite_levels, destination,
                    update=update, steps=steps_done, eval_due=False, replay_due=True,
                )
            eval_wandb_logs = {}
            if eval_metrics is not None:
                latest_stats.update(eval_metrics)
                eval_wandb_logs = _report_evaluation(
                    eval_metrics, evaluation_level, update=update, total=num_updates, steps=steps_done,
                    phase="eval", wandb_run=wandb_run,
                    history_path=destination / "evaluation_history.jsonl",
                    hold=eval_hold,
                )
                if stop_training:
                    terminal_print("[EARLY-STOP]")
            if (eval_metrics is not None and (is_final_update or stop_training)
                    and map_suite_levels is None):
                if evaluation.final_robustness and (evaluation_result is None
                        or int(evaluation_result[0]["eval_robustness_runs"]) == 1):
                    from swarmecho.training.evaluate import collect_robust_evaluation
                    evaluation_result = collect_robust_evaluation(
                        model, evaluation_level,
                        on_run=functools.partial(report_run, prefix="FINAL"),
                    )
                elif evaluation.final_robustness:
                    terminal_print("[FINAL] reusing this update's robust runs.")
                else:
                    terminal_print("[FINAL] reusing this update's noise-free metric pass.")
                final_evaluation_result = evaluation_result
                final_metrics = evaluation_result[0] if evaluation_result is not None else eval_metrics
                latest_stats.update({f"final_{key}": value for key, value in final_metrics.items()})
                _report_evaluation(
                    final_metrics, evaluation_level, update=update, total=num_updates, steps=steps_done,
                    phase="final_eval", wandb_run=wandb_run,
                    history_path=destination / "evaluation_history.jsonl",
                    hold=eval_hold,
                )
            suffix = artifact_suffix(update, steps_done)
            if evaluation_result is not None and (evaluation.training_heatmap_creation or is_final_update or stop_training):
                artifact_metrics, episode_info = evaluation_result
                artifact_runs = int(artifact_metrics["eval_robustness_runs"])
                info_path = save_eval_info_csv(
                    destination / "artifacts" / "train" / "data" / f"eval_info_{suffix}.csv",
                    target_positions=episode_info["target_positions"],
                    base_positions=episode_info["base_positions"],
                    stage_rates=episode_info["stage_rates"],
                    final_chain_lengths=episode_info["final_chain_lengths"],
                )
                layout_path = None
                if cfg.num_obstacles:
                    layout_path = save_eval_layout(
                        info_path.with_suffix(".layout.json"),
                        episode_info["obstacle_min"],
                        episode_info["obstacle_max"],
                    )
                write_manifest(
                    info_path.with_suffix(".heatmap.json"),
                    {
                        "format": "swarmecho-eval-heatmap/v1",
                        "data_file": info_path.name,
                        "map_name": evaluation_level.building_name, "building_snapshot": building_snapshot(evaluation_level.building),
                        "world_size_m": evaluation_level.building.world_size_m.tolist(),
                        "success_condition": evaluation_level.env.success_condition,
                        "num_agents": evaluation_level.env.num_agents,
                        "training_update": update,
                        "environment_steps": steps_done,
                        "artifact_scope": "train",
                        "robustness_runs": artifact_runs,
                        "action_noise_max": evaluation.eval_action_noise_max if artifact_runs > 1 else 0.0,
                        "obstacle_layout_mode": "fixed" if cfg.num_obstacles else "free_space",
                        **({"layout_file": layout_path.name} if layout_path else {}),
                    },
                )
                write_manifest(
                    destination / "artifacts" / "train" / "manifests" / f"eval_{suffix}.json",
                    {
                        "type": "evaluation",
                        "update": update,
                        "steps": steps_done,
                        "episodes": level.effective_eval_parallel_envs,
                        "data_path": str(info_path),
                        **artifact_metrics,
                    },
                )
            if (
                ((evaluation.eval_video and replay_due) or stop_training)
                and map_suite_levels is None
            ):
                replay_started = time.perf_counter()
                replay_bounds = evaluation.eval_fixed_obstacle_bounds
                eval_states, eval_rewards = evaluate_model(
                    model, evaluation_level, max_steps=cfg.max_steps,
                    obstacle_min=(
                        None if not replay_bounds else jnp.asarray(replay_bounds)[:, :3]
                    ),
                    obstacle_max=(
                        None if not replay_bounds else jnp.asarray(replay_bounds)[:, 3:]
                    ),
                )
                write_replay(
                    replay_dir / f"eval_{suffix}",
                    eval_states,
                    map_name=evaluation_level.building_name, building=evaluation_level.building,
                    dt=cfg.dt,
                    reward_terms=eval_rewards,
                    metadata={
                        **planner_metadata(evaluation_level.env, reward_cfg=evaluation_level.reward),
                        "world_size_m": evaluation_level.building.world_size_m.tolist(),
                        "cell_size_m": evaluation_level.building.cell_size_m,
                        "coverage_voxel_size_m": (
                            evaluation_level.building.cell_size_m
                            if cfg.coverage_voxel_size is None
                            else cfg.coverage_voxel_size
                        ),
                        "comm_radius_m": cfg.comm_radius,
                        "allow_redundancy_reward": level.reward.allow_redundancy_reward,
                        "comm_radius_base_m": cfg.comm_radius_base,
                        "visual_radius_m": cfg.visual_radius,
                        "success_condition": evaluation_level.env.success_condition,
                        "training_update": update,
                        "environment_steps": steps_done,
                        "artifact_scope": "train",
                    },
                    progress=False,
                )
                terminal_print(
                    f"[REPLAY] collected {len(eval_states) - 1} steps and created replay in "
                    f"{time.perf_counter() - replay_started:.1f}s.",
                    flush=True,
                )
            record_timing(update, "evaluation_and_artifacts", total_s=time.perf_counter() - evaluation_started)
            if stop_training:
                if training.dynamic_max_geo_distance.enabled:
                    latest_stats["agent_curriculum_mastered"] = True
                if wandb_run is not None:
                    import wandb
                    wandb.log({}, step=steps_done, commit=True)
                broadcast_early_stop_metrics(
                    wandb_run,
                    evaluation,
                    eval_wandb_logs,
                    update,
                    num_updates,
                    training.num_envs * training.num_steps,
                    training.total_timesteps,
                )
                num_updates = update
                break
        if wandb_run is not None:
            import wandb
            wandb.log({}, step=steps_done, commit=True)

    checkpoint = None
    if num_updates == resume.start_update:
        latest_stats = dict(completed_checkpoint.get("metrics", {}))
    if (validation_runner is not None and level.validation.val_on_finalization
            and validation_runner.last_update != num_updates):
        perform_validation(num_updates, resume.step_offset + num_updates * training.num_envs * training.num_steps)
    latest_stats["effective_num_envs"] = training.num_envs
    if evaluation.save_model:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = save_training_checkpoint(
            model,
            checkpoint_dir / f"ckpt_{num_updates:06d}",
            run_name=run_name,
            update=num_updates,
            num_envs=training.num_envs,
            num_steps=training.num_steps,
            prior_history=resume.prior_history,
            total_steps=resume.step_offset + num_updates * training.num_envs * training.num_steps,
            adaptive_spawn=adaptive_spawn,
            adaptive_maps=adaptive_maps,
            validation=validation_runner,
            run_dir=destination,
            metrics=latest_stats,
            final=True,
        )
        if map_suite_levels is None:
            _create_final_checkpoint_evaluation(
                model, level, checkpoint, destination, result=final_evaluation_result,
            )
    (destination / "metrics.json").write_text(
        json.dumps(latest_stats, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.finish()
    terminal_print("──────────────────────────────────────────────────────")
    terminal_print(f"Training complete ✓  checkpoint: {display_path(checkpoint) if checkpoint else 'saving disabled'}")
    return checkpoint, latest_stats


def _recordable_reward_components(rewards, terms):
    """Keep additive rewards; omit event counters and duplicate diagnostics."""
    diagnostics = {"newly_knows", "reward_by_reference_positive", "reward_by_reference_negative"}
    return {
        name: jnp.broadcast_to(value, rewards.shape)
        for name, value in terms.items() if name not in diagnostics
    }


def _collect_replay_impl(
    model: MAPPOModel,
    key: jax.Array,
    *,
    reset,
    step,
    observations,
    reward_cfg,
    cfg,
    horizon: int,
    target_position: tuple[float, float, float] | None = None,
    obstacle_min: jax.Array | None = None,
    obstacle_max: jax.Array | None = None,
):
    """Collect one deterministic replay entirely on device.

    The old implementation synchronised the accelerator on every frame to
    append a Python list.  Keeping the rollout in ``lax.scan`` makes replay
    collection use the same JAX execution model as training and batched eval.
    """
    if obstacle_min is None:
        state = reset(key) if target_position is None else reset(
            key, target_pos=jnp.asarray(target_position, dtype=jnp.float32)
        )
    else:
        vertices, distances = reset.build_roadmap(obstacle_min, obstacle_max)
        state = reset(
            key,
            target_pos=None if target_position is None else jnp.asarray(target_position, dtype=jnp.float32),
            obstacle_min=obstacle_min,
            obstacle_max=obstacle_max,
            stored_vertices=vertices,
            stored_distances=distances,
        )
    actor_hidden = model.initial_actor_hidden((), num_agents=cfg.num_agents)
    actor_signature = model.initial_actor_signature((), num_agents=cfg.num_agents)
    actor_value = model.initial_actor_value((), num_agents=cfg.num_agents)
    base_valid = jnp.bool_(False)
    base_signature = jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32)
    base_value = jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32)
    num_agents = cfg.num_agents
    comm_cadence = int(model.memory_comm_every_k_steps)

    def replay_step(carry, step_index):
        (
            current_state,
            current_hidden,
            current_signature,
            current_value,
            current_base_valid,
            current_base_signature,
            current_base_value,
            completed,
            episode_length,
        ) = carry
        delta = current_state.pos[:, None, :] - current_state.pos[None, :, :]
        comm_mask = (
            (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
            & current_state.active[:, None]
            & current_state.active[None, :]
            & ~jnp.eye(num_agents, dtype=jnp.bool_)
        )
        in_base_range = (
            jnp.linalg.norm(current_state.pos - current_state.base_pos[None, :], axis=-1)
            <= cfg.comm_radius_base
        ) & current_state.active
        if current_state.solid_min.shape[0]:
            comm_mask &= ~segments_blocked(
                current_state.pos[:, None, :], current_state.pos[None, :, :],
                current_state.solid_min, current_state.solid_max,
            )
            in_base_range &= ~segments_blocked(
                current_state.pos, current_state.base_pos,
                current_state.solid_min, current_state.solid_max,
            )
        share_now = communication_due(current_state.step, comm_cadence)
        comm_mask = comm_mask & share_now
        base_memory_mask = (
            current_base_valid & in_base_range
            & (jnp.ones_like(current_state.target_known) if cfg.base_keeps_informing
               else ~current_state.target_known)
            & share_now
        )
        (
            next_hidden,
            emitted_signature,
            emitted_value,
            means,
            _,
        ) = model.actor.__call_team__(
            observations(current_state),
            current_hidden,
            current_signature,
            current_value,
            reset=policy_resets(current_state),
            comm_mask=comm_mask,
            active=current_state.active,
            base_signature=current_base_signature,
            base_value=current_base_value,
            base_memory_mask=base_memory_mask,
            deterministic=True,
        )
        means = deterministic_actions(model.actor, next_hidden, means)
        candidate_state = step(current_state, physical_actions(means))
        rewards, terms = compute_rewards(current_state, candidate_state, reward_cfg, cfg)
        components = _recordable_reward_components(rewards, terms)

        reporters = current_state.target_known & in_base_range
        has_reporter = jnp.any(reporters)
        reporter_index = jnp.argmax(reporters.astype(jnp.int32))
        should_store = ~current_base_valid & has_reporter & share_now
        next_base_signature = jnp.where(
            should_store, emitted_signature[reporter_index], current_base_signature
        )
        next_base_value = jnp.where(
            should_store, emitted_value[reporter_index], current_base_value
        )
        next_base_valid = current_base_valid | should_store
        next_base_valid = jnp.where(candidate_state.done, False, next_base_valid)
        next_base_signature = jnp.where(candidate_state.done, 0.0, next_base_signature)
        next_base_value = jnp.where(candidate_state.done, 0.0, next_base_value)

        # After the terminal frame, retain the terminal state.  The full scan
        # remains static-shaped for JAX, while the host trims it to the real
        # episode length before writing the archive.
        next_state = jax.lax.cond(
            completed,
            lambda _: current_state,
            lambda _: candidate_state,
            operand=None,
        )
        live = ~completed
        next_carry = (
            next_state,
            next_hidden,
            emitted_signature,
            emitted_value,
            next_base_valid,
            next_base_signature,
            next_base_value,
            completed | candidate_state.done,
            episode_length + live.astype(jnp.int32),
        )
        recorded_rewards = jnp.where(live, rewards, jnp.zeros_like(rewards))
        recorded_components = jax.tree_util.tree_map(
            lambda value: jnp.where(live, value, jnp.zeros_like(value)), components,
        )
        return next_carry, (next_state, recorded_rewards, recorded_components)

    final_carry, (rollout_states, rollout_rewards, rollout_components) = jax.lax.scan(
        replay_step,
        (
            state,
            actor_hidden,
            actor_signature,
            actor_value,
            base_valid,
            base_signature,
            base_value,
            jnp.bool_(False),
            jnp.int32(0),
        ),
        jnp.arange(horizon, dtype=jnp.int32),
    )
    states = jax.tree_util.tree_map(
        lambda initial, rollout: jnp.concatenate((initial[None], rollout), axis=0),
        state,
        rollout_states,
    )
    rewards = jnp.concatenate(
        (jnp.zeros((1, num_agents), dtype=jnp.float32), rollout_rewards), axis=0
    )
    components = jax.tree_util.tree_map(
        lambda values: jnp.concatenate((jnp.zeros_like(values[:1]), values), axis=0),
        rollout_components,
    )
    return states, rewards, components, final_carry[-1]


@functools.partial(
    nnx.jit,
    static_argnames=(
        "reset",
        "step",
        "observations",
        "reward_cfg",
        "cfg",
        "horizon",
        "target_position",
    ),
)
def _collect_replay_jit(
    model: MAPPOModel,
    key: jax.Array,
    *,
    reset,
    step,
    observations,
    reward_cfg,
    cfg,
    horizon: int,
    target_position: tuple[float, float, float] | None = None,
    obstacle_min: jax.Array | None = None,
    obstacle_max: jax.Array | None = None,
):
    """Compile one scalar replay collector."""
    return _collect_replay_impl(
        model, key, reset=reset, step=step, observations=observations,
        reward_cfg=reward_cfg, cfg=cfg, horizon=horizon,
        target_position=target_position,
        obstacle_min=obstacle_min, obstacle_max=obstacle_max,
    )


def evaluate_model(
    model: MAPPOModel,
    level: Level,
    *,
    max_steps: int | None = None,
    seed: int | None = None,
    target_position: np.ndarray | tuple[float, float, float] | None = None,
    obstacle_min: np.ndarray | None = None,
    obstacle_max: np.ndarray | None = None,
) -> tuple[list, ReplayRewards]:
    """Run one deterministic trained-policy episode for replay/inspection."""
    level = resolve_evaluation_level(level)
    cfg = level.env
    reset, step, observations, _ = _evaluation_env_fns(level)
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps)
    timeline, rewards, components, episode_length = jax.device_get(
        _collect_replay_jit(
            model,
            jax.random.PRNGKey(seed if seed is not None else level.training.seed + 10_000),
            reset=reset,
            step=step,
            observations=observations,
            reward_cfg=level.reward,
            cfg=cfg,
            horizon=horizon,
            target_position=(
                None
                if target_position is None
                else tuple(float(value) for value in target_position)
            ),
            obstacle_min=None if obstacle_min is None else jnp.asarray(obstacle_min),
            obstacle_max=None if obstacle_max is None else jnp.asarray(obstacle_max),
        )
    )
    _validate_evaluation_pairs(level, timeline.spawn_pair_attempts[0])
    if cfg.adapt_size.enabled:
        validate_zones(timeline.adaptive.zone_valid)
    frames = int(episode_length) + 1
    timeline = jax.tree_util.tree_map(lambda item: item[:frames], timeline)
    states = [
        jax.tree_util.tree_map(lambda item, index=index: item[index], timeline)
        for index in range(frames)
    ]
    return states, ReplayRewards(
        np.asarray(rewards[:frames])[:, :, None],
        {name: np.asarray(values[:frames]) for name, values in components.items()},
    )


@functools.partial(
    nnx.jit,
    static_argnames=(
        "reset",
        "step",
        "observations",
        "env_metrics",
        "reward_cfg",
        "cfg",
        "num_envs",
        "horizon",
        "action_noise_max",
        "capture_actions",
        "capture_result",
        "capture_offset",
        "fixed_layout",
        "return_population",
    ),
)
def _run_parallel_evaluation_jit(
    model: MAPPOModel,
    key: jax.Array,
    action_noise_key: jax.Array,
    *,
    reset,
    step,
    observations,
    env_metrics,
    reward_cfg,
    cfg,
    num_envs: int,
    horizon: int,
    action_noise_max: float = 0.0,
    capture_actions: bool = False,
    capture_result: str = "success",
    capture_offset: int = 0,
    fixed_layout: bool = False,
    fixed_obstacle_min: jax.Array | None = None,
    fixed_obstacle_max: jax.Array | None = None,
    fixed_roadmap_vertices: jax.Array | None = None,
    fixed_roadmap_distances: jax.Array | None = None,
    case_map_ids: jax.Array | None = None,
    case_keys: jax.Array | None = None,
    case_targets: jax.Array | None = None,
    return_population: bool = False,
):
    """Run deterministic metrics and optionally retain one lane's actions."""
    if return_population and capture_actions:
        raise ValueError("Population summaries and action capture must be collected separately.")
    env_keys = jax.random.split(key, num_envs) if case_keys is None else case_keys
    env_action_noise_keys = jax.random.split(action_noise_key, num_envs)
    num_agents = cfg.num_agents
    comm_cadence = int(model.memory_comm_every_k_steps)

    def run_single_environment(env_key, env_action_noise_key, case_map_id=None, case_target=None):
        if case_map_id is not None:
            state = reset(env_key, map_id=case_map_id, target_pos=case_target)
        elif fixed_layout:
            state = reset(env_key, obstacle_min=fixed_obstacle_min, obstacle_max=fixed_obstacle_max,
                          stored_vertices=fixed_roadmap_vertices, stored_distances=fixed_roadmap_distances)
        else:
            state = reset(env_key)
        actor_hidden = model.initial_actor_hidden((), num_agents=cfg.num_agents)
        actor_signature = model.initial_actor_signature((), num_agents=cfg.num_agents)
        actor_value = model.initial_actor_value((), num_agents=cfg.num_agents)
        base_valid = jnp.bool_(False)
        base_signature = jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32)
        base_value = jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32)
        initial_carry = (
            state,
            actor_hidden,
            actor_signature,
            actor_value,
            base_valid,
            base_signature,
            base_value,
            jnp.bool_(False),  # completed
            jnp.float32(0.0),  # return
            jnp.int32(0),      # length
            jnp.bool_(False),  # success
            jnp.bool_(False),  # delivered target
            jnp.float32(0.0),  # terminal chain progress
            jnp.float32(0.0),  # terminal coverage
            jnp.bool_(False),  # target seen/known by any agent
            jnp.bool_(False),  # target delivered to base before termination
        )
        if return_population:
            initial_carry = (*initial_carry, jnp.sum(state.active))

        def evaluation_step(carry, step_index):
            (
                state,
                actor_hidden,
                actor_signature,
                actor_value,
                base_valid,
                base_signature,
                base_value,
                completed,
                returns,
                lengths,
                successes,
                target_found,
                chain_progress_pct,
                terminal_coverage,
                visually_found,
                base_delivered,
            ) = carry[:16]

            delta = state.pos[:, None, :] - state.pos[None, :, :]
            comm_mask = (
                (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
                & state.active[:, None]
                & state.active[None, :]
                & ~jnp.eye(num_agents, dtype=jnp.bool_)
            )
            in_base_range = (
                jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
                <= cfg.comm_radius_base
            ) & state.active
            if state.solid_min.shape[0]:
                comm_mask &= ~segments_blocked(
                    state.pos[:, None, :], state.pos[None, :, :],
                    state.solid_min, state.solid_max,
                )
                in_base_range &= ~segments_blocked(
                    state.pos, state.base_pos,
                    state.solid_min, state.solid_max,
                )
            receivers = jnp.ones_like(state.target_known) if cfg.base_keeps_informing else ~state.target_known
            base_memory_mask = base_valid & in_base_range & receivers
            share_now = communication_due(state.step, comm_cadence)
            comm_mask = comm_mask & share_now
            base_memory_mask = base_memory_mask & share_now

            observation = observations(state)
            (
                next_actor_hidden,
                emitted_signature,
                emitted_value,
                means,
                _,
            ) = model.actor.__call_team__(
                observation,
                actor_hidden,
                actor_signature,
                actor_value,
                reset=policy_resets(state),
                comm_mask=comm_mask,
                active=state.active,
                base_signature=base_signature,
                base_value=base_value,
                base_memory_mask=base_memory_mask,
                deterministic=True,
            )
            means = deterministic_actions(model.actor, next_actor_hidden, means)
            if action_noise_max > 0.0:
                # Probe the observed shape-dependent actor-mean discrepancy
                # without altering resets, target positions, or matrix shapes.
                step_noise_key = jax.random.fold_in(
                    env_action_noise_key, step_index
                )
                means = means.at[..., :3].add(jax.random.uniform(
                    step_noise_key,
                    means[..., :3].shape,
                    dtype=means.dtype,
                    minval=-action_noise_max,
                    maxval=action_noise_max,
                ))
            executed_actions = jnp.where(state.active[:, None], physical_actions(means), 0.0)
            next_state = step(state, executed_actions)
            rewards, _ = compute_rewards(state, next_state, reward_cfg, cfg)
            target_found_or_delivered = (
                next_state.base_target_known
                if reward_cfg.target_found_requires_delivery
                else jnp.any(next_state.target_known)
            )
            live = ~completed
            if cfg.success_condition == "coverage":
                step_chain_progress = jnp.float32(0.0)
            elif reward_cfg.chain_reward_system == "obstacle_geodesic":
                _, step_chain_progress, _, _ = obstacle_chain_diagnostics(next_state)
            else:
                _, step_chain_progress = chain_diagnostics(next_state)

            reporters = state.target_known & in_base_range
            has_reporter = jnp.any(reporters)
            reporter_index = jnp.argmax(reporters.astype(jnp.int32))
            should_store = ~base_valid & has_reporter & share_now
            next_base_signature = jnp.where(
                should_store, emitted_signature[reporter_index], base_signature
            )
            next_base_value = jnp.where(
                should_store, emitted_value[reporter_index], base_value
            )
            next_base_valid = base_valid | should_store
            next_base_valid = jnp.where(next_state.done, False, next_base_valid)
            next_base_signature = jnp.where(
                next_state.done, 0.0, next_base_signature
            )
            next_base_value = jnp.where(next_state.done, 0.0, next_base_value)

            next_carry = (
                next_state,
                next_actor_hidden,
                emitted_signature,
                emitted_value,
                next_base_valid,
                next_base_signature,
                next_base_value,
                completed | (live & next_state.done),
                jnp.where(live, returns + jnp.sum(rewards), returns),
                lengths + live.astype(jnp.int32),
                successes | (live & next_state.success),
                target_found | (live & target_found_or_delivered),
                jnp.where(live, step_chain_progress, chain_progress_pct),
                jnp.where(
                    live,
                    env_metrics(next_state)["coverage_fraction"],
                    terminal_coverage,
                ),
                visually_found | (live & jnp.any(next_state.target_known)),
                base_delivered | (live & next_state.base_target_known),
            )
            if return_population:
                next_carry = (*next_carry, jnp.where(live, jnp.sum(next_state.active), carry[16]))
            return next_carry, executed_actions if capture_actions else None

        final_carry, action_history = jax.lax.scan(
            evaluation_step,
            initial_carry,
            jnp.arange(horizon, dtype=jnp.int32),
        )
        metrics = (
            *final_carry[8:15],
            (final_chain_length(final_carry[0], cfg)
             if cfg.success_condition != "coverage" else jnp.float32(0.0)),
            state.target_pos, state.base_pos,
            state.obstacle_min, state.obstacle_max,
            state.spawn_pair_attempts,
            final_carry[15],
            state.adaptive.zone_valid if cfg.adapt_size.enabled else jnp.bool_(True),
        )
        if return_population:
            metrics = (*metrics, final_carry[16])
        return (*metrics, action_history) if capture_actions else metrics

    if case_map_ids is None:
        batched = jax.vmap(run_single_environment)(env_keys, env_action_noise_keys)
    elif case_targets is None:
        batched = jax.vmap(run_single_environment)(env_keys, env_action_noise_keys, case_map_ids)
    else:
        batched = jax.vmap(run_single_environment)(env_keys, env_action_noise_keys, case_map_ids, case_targets)
    if not capture_actions:
        return batched

    successes = batched[2]
    final_chain_lengths = batched[7]
    action_histories = batched[15]
    if capture_result == "success":
        scores = jnp.where(successes, final_chain_lengths, -jnp.inf)
        selected_lane = jnp.argsort(-scores, stable=True)[capture_offset]
    else:
        failure_lanes = jnp.nonzero(
            ~successes, size=num_envs, fill_value=-1
        )[0]
        selected_lane = failure_lanes[capture_offset]
    return batched[:15], selected_lane, action_histories[selected_lane]


def _validate_evaluation_pairs(level: Level, attempts: np.ndarray) -> None:
    if not (evaluation_minimum_geodesic_separation(level) or evaluation_dynamic_max_geo_distance(level)):
        return
    rejected = int(np.count_nonzero(np.asarray(attempts) == 0))
    if rejected:
        raise ValueError(
            f"Evaluation map {level.building_name!r} has {rejected} base/target pairs "
            "that failed the configured spawn feasibility rules. Choose a "
            "feasible fixed-base map or increase spawn_pair_max_attempts."
        )


def evaluate_suite(
    model: MAPPOModel,
    level: Level,
    *,
    episodes: int,
    max_steps: int | None = None,
    return_episode_info: bool = False,
    action_noise_max: float = 0.0,
    action_noise_seed: int | None = None,
    layout_mode: str | None = None,
    layout_seed_offset: int | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate a deterministic batch without collecting replay frames."""
    level = resolve_evaluation_level(level)
    cfg = level.env
    reset, step, observations, env_metrics = _evaluation_env_fns(level)
    layout_mode = layout_mode or ("fixed" if cfg.num_obstacles else "per_environment")
    fixed_layout = bool(cfg.num_obstacles)
    layout_seed_offset = 30_000 if layout_seed_offset is None else layout_seed_offset
    reference_state = None
    fixed_min = fixed_max = fixed_vertices = fixed_distances = None
    if fixed_layout:
        configured_bounds = getattr(
            level.evaluation, "eval_fixed_obstacle_bounds", None
        )
        if configured_bounds:
            bounds = jnp.asarray(configured_bounds, dtype=jnp.float32)
            if bounds.shape != (cfg.num_obstacles, 6):
                raise ValueError(
                    "eval_fixed_obstacle_bounds must contain six values per obstacle."
                )
            fixed_min, fixed_max = bounds[:, :3], bounds[:, 3:]
            fixed_vertices, fixed_distances = reset.build_roadmap(fixed_min, fixed_max)
        else:
            reference_state = reset(
                jax.random.PRNGKey(level.training.seed + layout_seed_offset)
            )
            fixed_min, fixed_max = reference_state.obstacle_min, reference_state.obstacle_max
            fixed_vertices = reference_state.roadmap_vertices
            fixed_distances = reference_state.roadmap_distances
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps)
    (
        returns,
        lengths,
        successes,
        target_found,
        chain_progress_pct,
        terminal_coverage,
        visually_found,
        final_chain_lengths,
        target_positions,
        base_positions,
        obstacle_min,
        obstacle_max,
        spawn_pair_attempts,
        base_delivered,
        zone_valid,
    ) = jax.device_get(
        _run_parallel_evaluation_jit(
            model,
            jax.random.PRNGKey(level.training.seed + 10_000),
            jax.random.PRNGKey(
                level.training.seed + 20_000
                if action_noise_seed is None
                else action_noise_seed
            ),
            reset=reset,
            step=step,
            observations=observations,
            env_metrics=env_metrics,
            reward_cfg=level.reward,
            cfg=cfg,
            num_envs=episodes,
            horizon=horizon,
            action_noise_max=action_noise_max,
            fixed_layout=fixed_layout,
            fixed_obstacle_min=fixed_min,
            fixed_obstacle_max=fixed_max,
            fixed_roadmap_vertices=fixed_vertices,
            fixed_roadmap_distances=fixed_distances,
        )
    )
    _validate_evaluation_pairs(level, spawn_pair_attempts)
    if cfg.adapt_size.enabled:
        validate_zones(zone_valid)
    success_episode_length = float(np.mean(lengths[successes])) if np.any(successes) else float(cfg.max_steps)
    metrics = {
        "eval_return": float(np.mean(returns)),
        "eval_success": float(np.mean(successes)),
        "eval_target_found_rate": float(np.mean(target_found)),
        "eval_visually_found_rate": float(np.mean(visually_found)),
        "eval_chain_progress_pct": float(np.mean(chain_progress_pct)),
        "eval_coverage": float(np.mean(terminal_coverage)),
        "eval_episode_length": float(np.mean(lengths)),
        "eval_success_episode_length": success_episode_length,
        "eval_success_episode_count": int(np.count_nonzero(successes)),
        "eval_success_length_reduction": 1.0 - success_episode_length / cfg.max_steps,
        "eval_success_chain_length": float(np.mean(final_chain_lengths[successes])) if np.any(successes) else 0.0,
    }
    if not return_episode_info:
        return metrics
    return metrics, {
        "target_positions": np.asarray(target_positions),
        "base_positions": np.asarray(base_positions),
        "successes": np.asarray(successes),
        "lengths": np.asarray(lengths),
        "delivered": np.asarray(base_delivered),
        "visually_found": np.asarray(visually_found),
        "final_chain_lengths": np.asarray(final_chain_lengths),
        "obstacle_min": np.asarray(obstacle_min),
        "obstacle_max": np.asarray(obstacle_max),
    }


def evaluate_suite_with_action_capture(
    model: MAPPOModel,
    level: Level,
    *,
    episodes: int,
    result: str,
    offset: int,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, object]]:
    """Evaluate and capture one selected lane's executed actions in one JIT."""
    level = resolve_evaluation_level(level)
    normalized = result.lower()
    if normalized in {"success", "successful"}:
        capture_result = "success"
    elif normalized in {"fail", "failure", "failed"}:
        capture_result = "fail"
    else:
        raise ValueError("result must be success or fail.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")

    cfg = level.env
    reset, step, observations, env_metrics = _evaluation_env_fns(level)
    fixed_layout = bool(cfg.num_obstacles)
    reference_state = (
        reset(jax.random.PRNGKey(level.training.seed + 30_000))
        if fixed_layout else None
    )
    fixed_min = fixed_max = fixed_vertices = fixed_distances = None
    if fixed_layout:
        configured_bounds = getattr(level.evaluation, "eval_fixed_obstacle_bounds", None)
        if configured_bounds:
            bounds = jnp.asarray(configured_bounds, dtype=jnp.float32)
            fixed_min, fixed_max = bounds[:, :3], bounds[:, 3:]
            fixed_vertices, fixed_distances = reset.build_roadmap(fixed_min, fixed_max)
        else:
            fixed_min, fixed_max = reference_state.obstacle_min, reference_state.obstacle_max
            fixed_vertices = reference_state.roadmap_vertices
            fixed_distances = reference_state.roadmap_distances
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps)
    batched, selected_lane, selected_actions = jax.device_get(
        _run_parallel_evaluation_jit(
            model,
            jax.random.PRNGKey(level.training.seed + 10_000),
            jax.random.PRNGKey(level.training.seed + 20_000),
            reset=reset,
            step=step,
            observations=observations,
            env_metrics=env_metrics,
            reward_cfg=level.reward,
            cfg=cfg,
            num_envs=episodes,
            horizon=horizon,
            capture_actions=True,
            capture_result=capture_result,
            capture_offset=offset,
            fixed_layout=fixed_layout,
            fixed_obstacle_min=fixed_min,
            fixed_obstacle_max=fixed_max,
            fixed_roadmap_vertices=fixed_vertices,
            fixed_roadmap_distances=fixed_distances,
        )
    )
    (
        returns,
        lengths,
        successes,
        target_found,
        chain_progress_pct,
        terminal_coverage,
        visually_found,
        final_chain_lengths,
        target_positions,
        base_positions,
        obstacle_min,
        obstacle_max,
        spawn_pair_attempts,
        base_delivered,
        zone_valid,
    ) = batched
    _validate_evaluation_pairs(level, spawn_pair_attempts)
    if cfg.adapt_size.enabled:
        validate_zones(zone_valid)
    matching_count = int(np.count_nonzero(successes if capture_result == "success" else ~successes))
    if offset >= matching_count:
        raise ValueError(
            f"Requested {capture_result.upper()}_{offset}, but the captured "
            f"evaluation contains only {matching_count} matching episodes."
        )
    lane = int(selected_lane)
    success_episode_length = float(np.mean(lengths[successes])) if np.any(successes) else float(cfg.max_steps)
    metrics = {
        "eval_return": float(np.mean(returns)),
        "eval_success": float(np.mean(successes)),
        "eval_target_found_rate": float(np.mean(target_found)),
        "eval_visually_found_rate": float(np.mean(visually_found)),
        "eval_chain_progress_pct": float(np.mean(chain_progress_pct)),
        "eval_coverage": float(np.mean(terminal_coverage)),
        "eval_episode_length": float(np.mean(lengths)),
        "eval_success_episode_length": success_episode_length,
        "eval_success_episode_count": int(np.count_nonzero(successes)),
        "eval_success_length_reduction": 1.0 - success_episode_length / cfg.max_steps,
        "eval_success_chain_length": float(np.mean(final_chain_lengths[successes])) if np.any(successes) else 0.0,
    }
    episode_info = {
        "target_positions": np.asarray(target_positions),
        "base_positions": np.asarray(base_positions),
        "successes": np.asarray(successes),
        "lengths": np.asarray(lengths),
        "delivered": np.asarray(base_delivered),
        "visually_found": np.asarray(visually_found),
        "final_chain_lengths": np.asarray(final_chain_lengths),
        "obstacle_min": np.asarray(obstacle_min),
        "obstacle_max": np.asarray(obstacle_max),
    }
    capture = {
        "lane": lane,
        "actions": np.asarray(selected_actions),
        "length": int(np.asarray(lengths)[lane]),
    }
    return metrics, episode_info, capture


@functools.partial(
    nnx.jit,
    static_argnames=(
        "reset",
        "step",
        "reward_cfg",
        "cfg",
        "horizon",
        "num_envs",
        "capture_lane",
    ),
)
def _replay_recorded_actions_jit(
    root_key: jax.Array,
    actions: jax.Array,
    *,
    reset,
    step,
    reward_cfg,
    cfg,
    horizon: int,
    num_envs: int,
    capture_lane: int,
):
    """Rebuild one lane from actions emitted by the authoritative evaluation."""
    lane_key = jax.random.split(root_key, num_envs)[capture_lane]
    initial_state = reset(lane_key)

    def replay_step(carry, action):
        state, completed, episode_length = carry
        candidate_state = step(state, action)
        rewards, terms = compute_rewards(state, candidate_state, reward_cfg, cfg)
        components = _recordable_reward_components(rewards, terms)
        next_state = jax.lax.cond(
            completed,
            lambda _: state,
            lambda _: candidate_state,
            operand=None,
        )
        live = ~completed
        return (
            next_state,
            completed | candidate_state.done,
            episode_length + live.astype(jnp.int32),
        ), (
            next_state, jnp.where(live, rewards, jnp.zeros_like(rewards)),
            jax.tree_util.tree_map(lambda value: jnp.where(live, value, jnp.zeros_like(value)), components),
        )

    final_carry, (rollout_states, rollout_rewards, rollout_components) = jax.lax.scan(
        replay_step,
        (initial_state, jnp.bool_(False), jnp.int32(0)),
        actions[:horizon],
    )
    states = jax.tree_util.tree_map(
        lambda initial, rollout: jnp.concatenate((initial[None], rollout), axis=0),
        initial_state,
        rollout_states,
    )
    rewards = jnp.concatenate(
        (jnp.zeros((1, cfg.num_agents), dtype=jnp.float32), rollout_rewards), axis=0
    )
    components = jax.tree_util.tree_map(
        lambda values: jnp.concatenate((jnp.zeros_like(values[:1]), values), axis=0),
        rollout_components,
    )
    return states, rewards, components, final_carry[-1]


def replay_recorded_actions(
    level: Level,
    actions: np.ndarray,
    *,
    capture_lane: int,
    batch_size: int,
    max_steps: int | None = None,
) -> tuple[list, ReplayRewards]:
    """Materialise one evaluation lane without invoking the policy again."""
    level = resolve_evaluation_level(level)
    cfg = level.env
    reset, step, _, _ = _evaluation_env_fns(level)
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps, len(actions))
    timeline, rewards, components, episode_length = jax.device_get(
        _replay_recorded_actions_jit(
            jax.random.PRNGKey(level.training.seed + 10_000),
            jnp.asarray(actions, dtype=jnp.float32),
            reset=reset,
            step=step,
            reward_cfg=level.reward,
            cfg=cfg,
            horizon=horizon,
            num_envs=batch_size,
            capture_lane=capture_lane,
        )
    )
    _validate_evaluation_pairs(level, timeline.spawn_pair_attempts[0])
    if cfg.adapt_size.enabled:
        validate_zones(timeline.adaptive.zone_valid)
    frames = int(episode_length) + 1
    timeline = jax.tree_util.tree_map(lambda item: item[:frames], timeline)
    states = [
        jax.tree_util.tree_map(lambda item, index=index: item[index], timeline)
        for index in range(frames)
    ]
    return states, ReplayRewards(
        np.asarray(rewards[:frames])[:, :, None],
        {name: np.asarray(values[:frames]) for name, values in components.items()},
    )


def main() -> None:
    from swarmecho.training.launcher import main as launch
    launch()


if __name__ == "__main__":
    main()
