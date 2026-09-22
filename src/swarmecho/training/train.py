"""Config-driven recurrent MAPPO training runner for the baseline."""

from __future__ import annotations

import os
import sys

# XLA/absl verbosity must be configured before importing JAX. Doing this in the
# training function is too late because plugin discovery happens at import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import functools
import json
import time
from collections import OrderedDict, deque
from dataclasses import asdict, fields
from pathlib import Path
from swarmecho.core.terminal import terminal_print, display_path, evaluation_row, init_logging

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx

from swarmecho.core.config import (
    Level,
    load_level_cli,
    resolve_evaluation_level,
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
from swarmecho.env.critic import (
    assemble_privileged_observations,
    make_privileged_features,
    privileged_global_dim,
    privileged_input_dim,
)
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.artifacts import (
    artifact_suffix,
    save_eval_info_csv,
    save_eval_layout,
    train_replay_root,
    write_manifest,
)
from swarmecho.training.checkpoints import restore_model_checkpoint, save_model_checkpoint
from swarmecho.training.mappo_buffer import MAPPORolloutBuffer, MAPPOTransition
from swarmecho.training.ppo import PPOTrainer, RunningValueNormalizer
from swarmecho.training.run_lifecycle import (
    EvaluationHold,
    broadcast_early_stop_metrics,
    init_wandb,
    resolve_resume_state,
    resolve_run_layout,
    schedule_due,
)
from swarmecho.visualize.replay import building_snapshot, write_replay


def communication_due(episode_step, every_k_steps):
    """Shared episode clock: independent of rollout boundaries and auto-reset lanes."""
    return (episode_step % every_k_steps) == 0


def build_model(level: Level, hidden_dim: int | None = None) -> MAPPOModel:
    """Construct the canonical recurrent MAPPO/TarMAC model."""
    cfg = level.env
    if not level.network.actor_memory or not level.network.critic_memory:
        raise ValueError("The runtime requires recurrent actor and critic memory.")
    if level.network.critic_type not in {"observation", "privileged"}:
        raise ValueError("network.critic_type must be 'observation' or 'privileged'.")
    privileged = level.network.critic_type == "privileged"
    model = MAPPOModel(
        obs_dim=observation_dim(cfg),
        act_dim=3,
        num_agents=cfg.num_agents,
        hidden_dim=hidden_dim or level.network.hidden_dim,
        num_layers=level.network.num_layers,
        actor_num_layers=level.network.actor_num_layers,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
        critic_type=level.network.critic_type,
        critic_input_dim=privileged_input_dim(cfg) if privileged else None,
        memory_comm_enabled=level.network.memory_comm_enabled,
        memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
        tarmac_sig_dim=level.network.tarmac_sig_dim,
        tarmac_val_dim=level.network.tarmac_val_dim,
        tarmac_include_self=level.network.tarmac_include_self,
        rngs=nnx.Rngs(level.training.seed),
    )
    model.mask_inactive = True
    model.actor.mask_inactive = True
    if privileged:
        model.privileged_inputs = True
        model.privileged_include_base_vector = not cfg.observe_base_vector
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
        "communication_clock": "episode",
        "inactive_agents": "excluded",
        **({"critic_type": "privileged", "privileged_layout": "compact_v1",
            "critic_input_dim": privileged_input_dim(cfg)} if privileged else {}),
        **{field.name: getattr(cfg, field.name) for field in fields(cfg)
           if field.name.startswith("observe_") or field.name == "radar_bins"},
    }.items()))
    return model


_EVALUATION_ENV_CACHE = OrderedDict()


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
    key = (
        tuple(geometry_key),
        level.env,
        plan_geodesic,
        level.reward.allow_redundancy_reward,
        level.reward.target_found_requires_delivery,
    )
    if key not in _EVALUATION_ENV_CACHE:
        with init_logging(level.logging.terminal_log_init):
            _EVALUATION_ENV_CACHE[key] = make_env_fns(
                level.building, level.env, plan_geodesic=plan_geodesic,
                allow_redundancy_reward=level.reward.allow_redundancy_reward,
                target_found_requires_delivery=level.reward.target_found_requires_delivery,
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
    base_memory_masks = base_valid[:, None] & in_base_range & ~states.target_known
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
):
    reporters = states.target_known & in_base_range
    has_reporter = jnp.any(reporters, axis=-1)
    reporter_index = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
    env_index = jnp.arange(states.pos.shape[0])
    should_store = ~base_valid & has_reporter
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
            model, level, episodes=evaluation.eval_parallel_envs,
            return_episode_info=evaluation.training_heatmap_creation,
        )
        if evaluation.training_heatmap_creation:
            metrics, episode_info = result
            successes = np.asarray(episode_info["successes"], dtype=bool)
            delivered = np.asarray(episode_info["delivered"], dtype=bool) | successes
            visually_found = np.asarray(episode_info["visually_found"], dtype=bool) | delivered
            episode_info["stage_rates"] = {
                "chain_success": successes.astype(float),
                "found_and_delivered": delivered.astype(float),
                "visually_found": visually_found.astype(float),
            }
        else:
            metrics = result
        metrics = {**metrics, "eval_robustness_runs": 1}
        if evaluation.training_heatmap_creation:
            evaluation_result = metrics, episode_info
    stop = evaluation.early_exit and hold.observe(metrics["eval_success"])
    return metrics, evaluation_result, stop


def _report_evaluation(metrics, level, *, update, total, steps, phase, wandb_run, history_path, hold):
    """Publish one measurement with explicit provenance to all logging sinks."""
    runs = int(metrics.get("eval_robustness_runs", 1))
    prefix = "FINAL" if phase == "final_eval" else "EVAL"
    tail = (
        f"{f'est={level.evaluation.early_exit_threshold:.0%}':<10}  "
        f"hold={hold.consecutive}/{max(1, level.evaluation.early_exit_hold_evals)}"
        if level.evaluation.early_exit else "est=off"
    )
    if phase == "final_eval" and level.evaluation.early_exit:
        tail += " (unchanged)"
    evaluation_row(
        metrics, label=f"[{prefix} {'UNAN' if runs > 1 else '1/1'}]",
        update=update, total=total, episodes=level.evaluation.eval_parallel_envs, tail=tail,
    )
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "update": update, "global_step": steps, "phase": phase,
            "mode": "robust" if runs > 1 else "single_pass", **metrics,
        }) + "\n")
    values = {
        "ep_return": metrics["eval_return"],
        "ep_length": metrics["eval_episode_length"],
        "chain_progress_pct": metrics["eval_chain_progress_pct"],
        "ep_length_reduction": (1.0 - metrics["eval_episode_length"] / level.env.max_steps) * 100.0,
        "success_rate": metrics["eval_success"],
        "target_found_rate": metrics["eval_target_found_rate"],
        "visually_found_rate": metrics["eval_visually_found_rate"],
        "map_coverage_pct": metrics["eval_coverage"] * 100.0,
        "robustness_runs": runs,
    }
    logs = {f"{phase}/{key}": value for key, value in values.items()}
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
    evaluation_level = resolve_evaluation_level(level)
    logging = level.logging
    num_updates = level.num_updates
    layout = resolve_run_layout(logging, evaluation, media_dir="replays")
    run_name = layout.run_name
    destination = Path(output_dir).absolute() if output_dir is not None else layout.run_dir
    destination.mkdir(parents=True, exist_ok=True)
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
    if level.random_buildings.enabled:
        from swarmecho.env.random_buildings import generate_maps
        from swarmecho.env.building_bank import prepare_bank
        init_log(f"Generating {training.num_envs} persistent random buildings...")
        buildings, generation_manifest = generate_maps(
            level.random_buildings, training.seed, training.num_envs,
            directory=destination / "generated_maps" if level.random_buildings.save_training_maps else None,
            log=init_log, drone_clearance=cfg.drone_radius + cfg.obstacle_planning_clearance_m)
        write_manifest(destination / "random_buildings.json", generation_manifest)
        building_bank = prepare_bank(buildings, cfg,
            plan=level.reward.chain_reward_system == "obstacle_geodesic" or level.reward.enable_chain_efficiency_reward,
            corner_bonus_m=cfg.roadmap_corner_bonus_m if training.minimum_geodesic_separation else 0., log=init_log)
        del buildings
    random_eval_levels = None
    if evaluation.random_eval:
        from swarmecho.training.random_evaluation import prepare_evaluation_maps
        random_eval_levels = prepare_evaluation_maps(level, destination / "random_eval_maps", log=init_log)
    init_log("Initializing JAX devices...")
    devices = jax.devices()
    init_log(f"JAX devices ready: {devices}")
    voxel_size, coverage_shape = coverage_grid_geometry(level.building, cfg)
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
            minimum_geodesic_separation_multiplier=training.minimum_geodesic_separation_multiplier,
            spawn_pair_max_attempts=training.spawn_pair_max_attempts,
            building_bank=building_bank,
        )
    obs_dim = observation_dim(cfg)
    sig_dim = network.tarmac_sig_dim
    val_dim = network.tarmac_val_dim
    init_log("Environment functions constructed. Initializing model...")
    model = build_model(level)
    privileged = network.critic_type == "privileged"
    collect_critic_features = make_privileged_features(level.building, cfg) if privileged else None
    init_log("Model constructed. Preparing checkpoint, optimizer and rollout buffer...")
    resume_from = checkpoint_path or training.checkpoint_path
    resume = resolve_resume_state(training, run_name, training.num_envs, training.num_steps)
    if resume_from:
        restored = restore_model_checkpoint(model, resume_from)
        terminal_print(f"  Restored weights  ✓  {display_path(restored)}")
    trainer = PPOTrainer(model, training)
    trainer.updates = resume.start_update
    buffer = MAPPORolloutBuffer(
        num_steps=training.num_steps,
        num_envs=training.num_envs,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        act_dim=3,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        recurrent=True,
        hidden_dim=network.hidden_dim,
        tarmac_sig_dim=sig_dim,
        tarmac_val_dim=val_dim,
        actor_memory=level.network.actor_memory,
        critic_memory=level.network.critic_memory,
        mask_inactive=True,
        critic_agent_dim=9 if privileged else 0,
        critic_global_dim=privileged_global_dim(cfg) if privileged else 0,
    )
    init_log("Optimizer and buffer prepared. Initial batched environment reset...")
    keys = jax.random.split(jax.random.PRNGKey(training.seed + 1), training.num_envs)
    if training.randomize_base or training.minimum_geodesic_separation or building_bank is not None:
        reset_batch_size = min(32, training.num_envs)
        reset_batch_count = (training.num_envs + reset_batch_size - 1) // reset_batch_size
        reset_padded_count = reset_batch_count * reset_batch_size
        reset_keys = jnp.concatenate((keys, jnp.repeat(keys[-1:], reset_padded_count - training.num_envs, axis=0)))
        reset_keys = reset_keys.reshape((reset_batch_count, reset_batch_size) + keys.shape[1:])
        if building_bank is not None:
            ids = jnp.minimum(jnp.arange(reset_padded_count), training.num_envs - 1).reshape(reset_batch_count, reset_batch_size)
            reset_batches = jax.lax.map(lambda pair: jax.vmap(lambda key, index: reset(key, map_id=index))(*pair), (reset_keys, ids))
        else:
            reset_batches = jax.lax.map(jax.vmap(reset), reset_keys)
        # Explicit dimensions avoid lax.map(batch_size=...)'s inferred -1
        # reshape, which fails for legitimate zero-sized geometry leaves.
        states = jax.tree_util.tree_map(
            lambda value: value.reshape((reset_padded_count,) + value.shape[2:])[:training.num_envs],
            reset_batches,
        )
    else:
        states = jax.vmap(reset)(keys)
    # Synchronize once at startup so this milestone reports completion, not dispatch.
    jax.block_until_ready(states)
    def check_spawn_pairs(attempts):
        if np.any(np.asarray(attempts) == 0):
            raise ValueError(
                f"No valid base/target pair in {training.spawn_pair_max_attempts} attempts per environment. "
                f"Required roadmap separation: {training.minimum_geodesic_separation_multiplier * cfg.comm_radius if training.minimum_geodesic_separation else 0:g} m. "
                "Randomized bases keep the target fixed; it may have no qualifying base location. "
                "Reduce the separation multiplier or check spawn geometry; increasing the attempt limit only helps feasible targets."
            )

    if training.randomize_base or training.minimum_geodesic_separation:
        check_spawn_pairs(states.spawn_pair_attempts)
    init_log("Initial reset complete on device. Preparing recurrent state and logging...")
    actor_hidden = model.initial_actor_hidden((training.num_envs,))
    actor_signature = model.initial_actor_signature((training.num_envs,))
    actor_value = model.initial_actor_value((training.num_envs,))
    critic_hidden = model.initial_critic_hidden((training.num_envs,))
    base_valid = jnp.zeros(training.num_envs, dtype=jnp.bool_)
    base_signature = jnp.zeros((training.num_envs, sig_dim), dtype=jnp.float32)
    base_value = jnp.zeros((training.num_envs, val_dim), dtype=jnp.float32)
    resets = jnp.zeros((training.num_envs, cfg.num_agents), dtype=jnp.bool_)
    latest_stats: dict[str, float] = {}
    eval_hold = EvaluationHold(evaluation.early_exit_threshold, evaluation.early_exit_hold_evals)
    final_robust_result = None
    episode_returns = np.zeros(training.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(training.num_envs, dtype=np.int32)
    episode_success = np.zeros(training.num_envs, dtype=np.float32)
    episode_found = np.zeros(training.num_envs, dtype=np.float32)
    episode_idle = np.zeros(training.num_envs, dtype=np.float32)
    episode_gap = np.zeros(training.num_envs, dtype=np.float32)
    episode_progress_pct = np.zeros(training.num_envs, dtype=np.float32)
    episode_coverage = np.zeros(training.num_envs, dtype=np.float32)
    episode_rewards = {
        name: np.zeros(training.num_envs, dtype=np.float64)
        for name in ("coverage", "collision", "finder", "chain_gap", "target_found", "success", "no_movement_termination")
    }
        # Collect one completed episode per parallel environment,
    # rather than a fixed-size history of partial/live rollouts.
    window_size = training.num_envs
    window_ret: deque[float] = deque(maxlen=window_size)
    window_len: deque[int] = deque(maxlen=window_size)
    window_succ: deque[float] = deque(maxlen=window_size)
    window_fnd: deque[float] = deque(maxlen=window_size)
    window_idle: deque[float] = deque(maxlen=window_size)
    window_gap: deque[float] = deque(maxlen=window_size)
    window_prog_pct: deque[float] = deque(maxlen=window_size)
    window_cov: deque[float] = deque(maxlen=window_size)
    window_pair_attempts: deque[int] = deque(maxlen=window_size)
    window_success_target_distance: deque[float] = deque(maxlen=window_size)
    window_rewards = {name: deque(maxlen=window_size) for name in episode_rewards}
    path_stat_names = (["number_of_valid_paths"] if level.reward.allow_redundancy_reward or level.reward.enable_chain_efficiency_reward else [])
    if level.reward.enable_chain_efficiency_reward:
        path_stat_names.append("chain_efficiency")
    window_path_stats = {name: deque(maxlen=window_size) for name in path_stat_names}
    completed_eps_count = 0
    history_path = destination / "training_history.jsonl"
    config_snapshot = {
        "name": level.name,
        "building": level.building_name,
        "env": asdict(level.env),
        "reward": asdict(level.reward),
        "training": asdict(training),
        "network": asdict(network),
        "evaluation": asdict(evaluation),
        "random_buildings": asdict(level.random_buildings),
        "logging": asdict(logging),
    }
    (destination / "config.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8"
    )
    init_log("State and configuration prepared. Initializing W&B...")
    wandb_run = init_wandb(logging, training, run_name, destination, config_snapshot)
    init_log("W&B initialization returned. Preparing first rollout...")

    _, params = nnx.split(model)
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
    total_steps = num_updates * training.num_envs * training.num_steps
    terminal_print("\n══════════════════════════════════════════════════════")
    terminal_print("  SwarmEcho — Recurrent MAPPO + TarMAC")
    terminal_print("══════════════════════════════════════════════════════")
    terminal_print(f"  level / building : {level.name} / {level.building_name}")
    terminal_print(f"  devices          : {jax.devices()}")
    terminal_print(f"  agents           : {cfg.num_agents}")
    terminal_print(f"  observation      : {obs_dim}  (radar bins: {cfg.radar_bins})")
    terminal_print("  action           : XYZ continuous force")
    terminal_print(f"  model parameters : {parameter_count:,}")
    terminal_print(f"  environments     : {training.num_envs:,}")
    terminal_print(f"  rollout / updates: {training.num_steps} / {num_updates}")
    terminal_print(f"  eval map         : {evaluation_level.building_name}")
    if evaluation.random_eval:
        terminal_print(f"  eval mode        : {evaluation.random_eval_envs} persistent maps, sequential, no action noise")
    else:
        terminal_print(f"  eval mode        : {'robust' if evaluation.training_robustness else 'single-pass'}; final robust {evaluation.eval_robustness_runs} runs; noise +/-{evaluation.eval_action_noise_max:g}")
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
            critic_obs = obs
            if privileged:
                critic_agent, critic_global = jax.vmap(collect_critic_features)(states)
                critic_obs = assemble_privileged_observations(
                    obs, critic_agent, critic_global, states.active,
                    include_base_vector=not cfg.observe_base_vector,
                )
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

            def policy_one(obs_e, keys_e, ah, sig, val, ch, cm, active, bs, bv, bm, rst, critic_e=None):
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
                    critic_obs=critic_e if privileged else None,
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
                resets | ~states.active,
                **({"critic_e": critic_obs} if privileged else {}),
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
                executed_actions = actions + action_noise
            else:
                executed_actions = actions
            executed_actions = jnp.where(states.active[..., None], executed_actions, 0.0)
            next_states, rewards, dones, info = env_step.batched(
                states, jnp.tanh(executed_actions)
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
            )
            next_resets = jnp.broadcast_to(dones[:, None], resets.shape)
            transition = {
                "obs": obs,
                "actions": actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": rewards,
                "dones": dones,
                "rnn_resets": resets,
                "comm_masks": comm_masks,
                "active_masks": states.active,
                "base_signatures": base_sig_in,
                "base_values": base_val_in,
                "base_memory_masks": base_masks,
                "coverage": info["coverage"],
                "collision": info["collision"],
                "finder": info["finder"],
                "chain_gap": info["chain_gap"],
                "target_found": info["target_found"],
                "reward_success": info["reward_success"],
                "no_movement_termination": info["no_movement_termination"],
                "success": info["success"],
                "global_target_found": info["global_target_found"],
                "idle_terminated": info["idle_terminated"],
                "chain_gap_dist": info["chain_gap_dist"],
                "chain_progress_pct": info["chain_progress_pct"],
                "global_coverage": info["global_coverage"],
                "terminal_target_pos": info["terminal_target_pos"],
                "terminal_base_pos": states.base_pos,
                "spawn_pair_attempts": states.spawn_pair_attempts,
                **{name: info[name] for name in path_stat_names},
            }
            if privileged:
                transition["critic_agent_features"] = critic_agent
                transition["critic_global_features"] = critic_global
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
        final_states, _, _, _, _, final_critic_hidden, _, _, _, _ = carry
        final_obs = jax.vmap(observations)(final_states)
        final_critic_obs = final_obs
        if privileged:
            final_agent, final_global = jax.vmap(collect_critic_features)(final_states)
            final_critic_obs = assemble_privileged_observations(
                final_obs, final_agent, final_global, final_states.active,
                include_base_vector=not cfg.observe_base_vector,
            )

        def value_one(obs_e, hidden_e, active_e, critic_e=None):
            return model.get_value_recurrent(
                obs_e, hidden_e, ~active_e, active=active_e,
                critic_obs=critic_e if privileged else None,
            )[1]

        final_values = jax.vmap(value_one)(
            final_obs, final_critic_hidden, final_states.active,
            **({"critic_e": final_critic_obs} if privileged else {}),
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
        if training.randomize_base or training.minimum_geodesic_separation:
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
            episode_idle = np.maximum(
                episode_idle, rollout_host["idle_terminated"][step_index]
            )
            episode_gap = rollout_host["chain_gap_dist"][step_index]
            episode_progress_pct = rollout_host["chain_progress_pct"][step_index]
            episode_coverage = rollout_host["global_coverage"][step_index]
            for reward_name, accumulator in episode_rewards.items():
                info_name = "reward_success" if reward_name == "success" else reward_name
                accumulator += rollout_host[info_name][step_index].sum(axis=-1)
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
                    base_signatures=rollout_host["base_signatures"][step_index],
                    base_values=rollout_host["base_values"][step_index],
                    base_memory_masks=rollout_host["base_memory_masks"][step_index],
                    critic_obs=rollout_host["obs"][step_index],
                    critic_agent_features=(rollout_host["critic_agent_features"][step_index]
                                           if privileged else None),
                    critic_global_features=(rollout_host["critic_global_features"][step_index]
                                            if privileged else None),
                )
            )
            completed_indices = np.flatnonzero(dones_host)
            for index in completed_indices:
                window_ret.append(float(episode_returns[index]))
                window_len.append(int(episode_lengths[index]))
                window_succ.append(float(episode_success[index]))
                window_fnd.append(float(episode_found[index]))
                window_idle.append(float(episode_idle[index]))
                window_gap.append(float(episode_gap[index]))
                window_prog_pct.append(float(episode_progress_pct[index]))
                window_cov.append(float(episode_coverage[index]))
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
            episode_idle[completed_indices] = 0.0
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
        completed_session_steps = (update - resume.start_update) * training.num_envs * training.num_steps
        sps = completed_session_steps / max(elapsed, 1e-6)
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
                "chain_gap": float(np.mean(window_gap)) if window_gap else 0.0,
                "chain_progress_pct": float(np.mean(window_prog_pct)) if window_prog_pct else 0.0,
                "mean_terminal_coverage": float(np.mean(window_cov)) if window_cov else 0.0,
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
            if training.minimum_geodesic_separation:
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
                    "train/chain_progress_pct": latest_stats["chain_progress_pct"],
                    "train/ep_length_reduction": (1.0 - latest_stats["mean_episode_length"] / cfg.max_steps) * 100.0,
                    "train/map_coverage_pct": latest_stats["mean_terminal_coverage"] * 100.0,
                    "train/episodes_completed": completed_eps_count,
                    "train/success_target_distance_mean_norm": latest_stats["success_target_distance_mean_norm"],
                    "train/success_target_distance_max_norm": latest_stats["success_target_distance_max_norm"],
                    **{f"rewards/{name}": latest_stats[f"reward_{name}"] for name in window_rewards},
                })
            for name, window in window_path_stats.items():
                if len(window) == window.maxlen:
                    wandb_logs[f"train/{name}"] = float(np.mean(window))
            # Keep this step open for evaluation metrics from the same update.
            wandb.log(wandb_logs, step=steps_done, commit=False)
        checkpoint_due = evaluation.save_model and schedule_due(
            update, evaluation.checkpoint_freq, evaluation.checkpoint_offset
        )
        if checkpoint_due and update < num_updates:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_model_checkpoint(
                model,
                checkpoint_dir / f"ckpt_{update:06d}",
                run_name=run_name,
                update=update,
                num_envs=training.num_envs,
                num_steps=training.num_steps,
                prior_history=resume.prior_history,
            )
        # The final update bypasses the gate. Its training metric uses the same
        # mode as earlier evaluations; final robustness is a separate phase.
        is_final_update = update == num_updates
        threshold_met = (
            latest_stats["rolling_success_rate"] >= evaluation.eval_min_train_success
        )
        evaluation_enabled = threshold_met or is_final_update
        if schedule_due(update, evaluation.eval_freq, evaluation.eval_offset) and not evaluation_enabled:
            terminal_print(
                f"[EVAL update={update}] skipped: training success "
                f"{latest_stats['rolling_success_rate']:.1%} < "
                f"{evaluation.eval_min_train_success:.1%} required.",
                flush=True,
            )
        eval_due = is_final_update or (
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
                    total=num_updates, episodes=evaluation.eval_parallel_envs,
                )
            # Replay-only updates do not create metrics or advance the hold.
            if random_eval_levels is not None:
                from swarmecho.training.random_evaluation import evaluate_random_maps
                eval_metrics = evaluate_random_maps(model, random_eval_levels, destination,
                    update=update, steps=steps_done, eval_due=eval_due, replay_due=replay_due, on_run=report_run)
                evaluation_result = None
                stop_training = bool(eval_metrics is not None and evaluation.early_exit and eval_hold.observe(eval_metrics["eval_success"]))
            else:
                eval_metrics, evaluation_result, stop_training = _evaluate_training_metrics(
                    model, evaluation_level, eval_hold,
                    eval_due=eval_due, on_run=report_run,
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
            if eval_metrics is not None and (is_final_update or stop_training) and random_eval_levels is None:
                if evaluation_result is None or int(evaluation_result[0]["eval_robustness_runs"]) == 1:
                    from swarmecho.training.evaluate import collect_robust_evaluation
                    evaluation_result = collect_robust_evaluation(
                        model, evaluation_level,
                        on_run=functools.partial(report_run, prefix="FINAL"),
                    )
                else:
                    terminal_print("[FINAL] reusing this update's robust runs.")
                final_robust_result = evaluation_result
                final_metrics, _ = final_robust_result
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
                        "episodes": evaluation.eval_parallel_envs,
                        "data_path": str(info_path),
                        **artifact_metrics,
                    },
                )
            if replay_due and not stop_training and random_eval_levels is None:
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
    if evaluation.save_model:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = save_model_checkpoint(
            model,
            checkpoint_dir / f"ckpt_{num_updates:06d}",
            run_name=run_name,
            update=num_updates,
            num_envs=training.num_envs,
            num_steps=training.num_steps,
            prior_history=resume.prior_history,
        )
        if random_eval_levels is None:
            _create_final_checkpoint_evaluation(
                model, level, checkpoint, destination, result=final_robust_result,
            )
    (destination / "metrics.json").write_text(
        json.dumps(latest_stats, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.finish()
    terminal_print("──────────────────────────────────────────────────────")
    terminal_print(f"Training complete ✓  checkpoint: {display_path(checkpoint) if checkpoint else 'saving disabled'}")
    return checkpoint, latest_stats


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
    actor_hidden = model.initial_actor_hidden(())
    actor_signature = model.initial_actor_signature(())
    actor_value = model.initial_actor_value(())
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
            current_base_valid & in_base_range & ~current_state.target_known & share_now
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
            reset=jnp.zeros(num_agents, dtype=jnp.bool_),
            comm_mask=comm_mask,
            active=current_state.active,
            base_signature=current_base_signature,
            base_value=current_base_value,
            base_memory_mask=base_memory_mask,
            deterministic=True,
        )
        candidate_state = step(current_state, jnp.tanh(means))
        rewards, _ = compute_rewards(current_state, candidate_state, reward_cfg, cfg)

        reporters = current_state.target_known & in_base_range
        has_reporter = jnp.any(reporters)
        reporter_index = jnp.argmax(reporters.astype(jnp.int32))
        should_store = ~current_base_valid & has_reporter
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
        return next_carry, (next_state, recorded_rewards)

    final_carry, (rollout_states, rollout_rewards) = jax.lax.scan(
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
    return states, rewards, final_carry[-1]


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
) -> tuple[list, np.ndarray]:
    """Run one deterministic trained-policy episode for replay/inspection."""
    level = resolve_evaluation_level(level)
    cfg = level.env
    reset, step, observations, _ = _evaluation_env_fns(level)
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps)
    timeline, rewards, episode_length = jax.device_get(
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
    frames = int(episode_length) + 1
    timeline = jax.tree_util.tree_map(lambda item: item[:frames], timeline)
    states = [
        jax.tree_util.tree_map(lambda item, index=index: item[index], timeline)
        for index in range(frames)
    ]
    return states, np.asarray(rewards[:frames])[:, :, None]


@functools.partial(
    nnx.jit,
    static_argnames=(
        "reset",
        "step",
        "observations",
        "reward_cfg",
        "cfg",
        "num_envs",
        "horizon",
        "action_noise_max",
        "capture_actions",
        "capture_result",
        "capture_offset",
        "fixed_layout",
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
):
    """Run deterministic metrics and optionally retain one lane's actions."""
    env_keys = jax.random.split(key, num_envs)
    env_action_noise_keys = jax.random.split(action_noise_key, num_envs)
    num_agents = cfg.num_agents
    comm_cadence = int(model.memory_comm_every_k_steps)

    def run_single_environment(env_key, env_action_noise_key):
        state = reset(
            env_key,
            obstacle_min=fixed_obstacle_min,
            obstacle_max=fixed_obstacle_max,
            stored_vertices=fixed_roadmap_vertices,
            stored_distances=fixed_roadmap_distances,
        ) if fixed_layout else reset(env_key)
        actor_hidden = model.initial_actor_hidden(())
        actor_signature = model.initial_actor_signature(())
        actor_value = model.initial_actor_value(())
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
        )

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
            ) = carry

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
            base_memory_mask = base_valid & in_base_range & ~state.target_known
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
                reset=jnp.zeros(num_agents, dtype=jnp.bool_),
                comm_mask=comm_mask,
                active=state.active,
                base_signature=base_signature,
                base_value=base_value,
                base_memory_mask=base_memory_mask,
                deterministic=True,
            )
            if action_noise_max > 0.0:
                # Probe the observed shape-dependent actor-mean discrepancy
                # without altering resets, target positions, or matrix shapes.
                step_noise_key = jax.random.fold_in(
                    env_action_noise_key, step_index
                )
                means = means + jax.random.uniform(
                    step_noise_key,
                    means.shape,
                    dtype=means.dtype,
                    minval=-action_noise_max,
                    maxval=action_noise_max,
                )
            executed_actions = jnp.where(state.active[:, None], jnp.tanh(means), 0.0)
            next_state = step(state, executed_actions)
            rewards, _ = compute_rewards(state, next_state, reward_cfg, cfg)
            target_found_or_delivered = (
                next_state.base_target_known
                if reward_cfg.target_found_requires_delivery
                else jnp.any(next_state.target_known)
            )
            live = ~completed
            if reward_cfg.chain_reward_system == "obstacle_geodesic":
                _, step_chain_progress, _, _ = obstacle_chain_diagnostics(next_state)
            else:
                _, step_chain_progress = chain_diagnostics(next_state)

            reporters = state.target_known & in_base_range
            has_reporter = jnp.any(reporters)
            reporter_index = jnp.argmax(reporters.astype(jnp.int32))
            should_store = ~base_valid & has_reporter
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
                    jnp.mean(next_state.coverage),
                    terminal_coverage,
                ),
                visually_found | (live & jnp.any(next_state.target_known)),
            )
            return next_carry, executed_actions if capture_actions else None

        final_carry, action_history = jax.lax.scan(
            evaluation_step,
            initial_carry,
            jnp.arange(horizon, dtype=jnp.int32),
        )
        metrics = (
            *final_carry[8:], final_chain_length(final_carry[0], cfg),
            state.target_pos, state.base_pos,
            state.obstacle_min, state.obstacle_max,
        )
        return (*metrics, action_history) if capture_actions else metrics

    batched = jax.vmap(run_single_environment)(
        env_keys, env_action_noise_keys
    )
    if not capture_actions:
        return batched

    successes = batched[2]
    final_chain_lengths = batched[7]
    action_histories = batched[12]
    if capture_result == "success":
        scores = jnp.where(successes, final_chain_lengths, -jnp.inf)
        selected_lane = jnp.argsort(-scores, stable=True)[capture_offset]
    else:
        failure_lanes = jnp.nonzero(
            ~successes, size=num_envs, fill_value=-1
        )[0]
        selected_lane = failure_lanes[capture_offset]
    return batched[:12], selected_lane, action_histories[selected_lane]


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
    reset, step, observations, _ = _evaluation_env_fns(level)
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
    metrics = {
        "eval_return": float(np.mean(returns)),
        "eval_success": float(np.mean(successes)),
        "eval_target_found_rate": float(np.mean(target_found)),
        "eval_visually_found_rate": float(np.mean(visually_found)),
        "eval_chain_progress_pct": float(np.mean(chain_progress_pct)),
        "eval_coverage": float(np.mean(terminal_coverage)),
        "eval_episode_length": float(np.mean(lengths)),
        "eval_success_chain_length": float(np.mean(final_chain_lengths[successes])) if np.any(successes) else 0.0,
    }
    if not return_episode_info:
        return metrics
    return metrics, {
        "target_positions": np.asarray(target_positions),
        "base_positions": np.asarray(base_positions),
        "successes": np.asarray(successes),
        "delivered": np.asarray(target_found),
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
    reset, step, observations, _ = _evaluation_env_fns(level)
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
    ) = batched
    matching_count = int(np.count_nonzero(successes if capture_result == "success" else ~successes))
    if offset >= matching_count:
        raise ValueError(
            f"Requested {capture_result.upper()}_{offset}, but the captured "
            f"evaluation contains only {matching_count} matching episodes."
        )
    lane = int(selected_lane)
    metrics = {
        "eval_return": float(np.mean(returns)),
        "eval_success": float(np.mean(successes)),
        "eval_target_found_rate": float(np.mean(target_found)),
        "eval_visually_found_rate": float(np.mean(visually_found)),
        "eval_chain_progress_pct": float(np.mean(chain_progress_pct)),
        "eval_coverage": float(np.mean(terminal_coverage)),
        "eval_episode_length": float(np.mean(lengths)),
        "eval_success_chain_length": float(np.mean(final_chain_lengths[successes])) if np.any(successes) else 0.0,
    }
    episode_info = {
        "target_positions": np.asarray(target_positions),
        "base_positions": np.asarray(base_positions),
        "successes": np.asarray(successes),
        "delivered": np.asarray(target_found),
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
        rewards, _ = compute_rewards(state, candidate_state, reward_cfg, cfg)
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
        ), (next_state, jnp.where(live, rewards, jnp.zeros_like(rewards)))

    final_carry, (rollout_states, rollout_rewards) = jax.lax.scan(
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
    return states, rewards, final_carry[-1]


def replay_recorded_actions(
    level: Level,
    actions: np.ndarray,
    *,
    capture_lane: int,
    batch_size: int,
    max_steps: int | None = None,
) -> tuple[list, np.ndarray]:
    """Materialise one evaluation lane without invoking the policy again."""
    level = resolve_evaluation_level(level)
    cfg = level.env
    reset, step, _, _ = _evaluation_env_fns(level)
    horizon = min(max_steps or cfg.max_steps, cfg.max_steps, len(actions))
    timeline, rewards, episode_length = jax.device_get(
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
    frames = int(episode_length) + 1
    timeline = jax.tree_util.tree_map(lambda item: item[:frames], timeline)
    states = [
        jax.tree_util.tree_map(lambda item, index=index: item[index], timeline)
        for index in range(frames)
    ]
    return states, np.asarray(rewards[:frames])[:, :, None]


def main() -> None:
    level = load_level_cli(sys.argv[1:])
    if level.logging.terminal_log_init:
        terminal_print("[INIT] Level and building loaded.")
    train(level)


if __name__ == "__main__":
    main()
