"""Checkpoint evaluation and target-selected replay entry point for 3D."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

# These must be configured before importing Flax/JAX.  ``checkpoints`` imports
# Flax below, so relying on train3d.py to set them is too late for this entry
# point and XLA backend diagnostics leak to stderr during replay compilation.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import yaml

from swarmecho.core.config import load_level_3d_cli
from swarmecho.core.paths import normalize_wsl_path
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    evaluation_stage,
    eval_checkpoint_artifact_root,
    eval_checkpoint_replay_root,
    load_eval_info_csv,
    load_eval_layout,
    parse_checkpoint_update,
    save_eval_info_csv,
    save_eval_layout,
    write_manifest,
)
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train3d import (
    build_model_3d,
    evaluate_model_3d,
    evaluate_suite_3d_with_action_capture,
    evaluate_suite_3d,
    replay_recorded_actions_3d,
)
from swarmecho.visualize.replay3d import write_replay


_EVAL_INFO_UPDATE = re.compile(r"eval_info_u(\d+)_")
_EVAL_MODES = {"parallel", "selective_auto_pick", "selective_manual_pick"}
_REPLAY_EXECUTIONS = {"single", "parallel_capture"}


def _as_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _target_artifact_suffix(target: np.ndarray) -> str:
    """Encode a manual target into a filesystem-safe, collision-resistant name."""
    encoded: list[str] = []
    for axis, value in zip("xyz", target, strict=True):
        coordinate = f"{float(value):.6f}".replace("-", "m").replace(".", "p")
        encoded.append(f"{axis}{coordinate}")
    return "TARGET-REPLAY_" + "_".join(encoded)


def _checkpoint_level_name(checkpoint: Path) -> str | None:
    """Read the level name saved alongside a training run's checkpoint."""
    config_path = checkpoint.parents[1] / "config.yaml"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return None
    level_name = config.get("name")
    if isinstance(level_name, str) and level_name:
        return level_name
    return None


def _heatmap_manifest(
    info_path: Path,
    level,
    *,
    checkpoint: Path,
    robustness_runs: int | None = None,
    action_noise_max: float | None = None,
    obstacle_min: np.ndarray | None = None,
    obstacle_max: np.ndarray | None = None,
) -> None:
    """Write the small inspector metadata sidecar for one 3D eval CSV."""
    manifest = {
        "format": "swarmecho-3d-eval-heatmap/v1",
        "data_file": info_path.name,
        "map_name": level.building_name,
        "world_size_m": level.building.world_size_m.tolist(),
        "checkpoint": str(checkpoint),
        "artifact_scope": "eval",
    }
    if robustness_runs is not None:
        manifest["robustness_runs"] = int(robustness_runs)
    if action_noise_max is not None:
        manifest["action_noise_max"] = float(action_noise_max)
    if obstacle_min is not None and np.asarray(obstacle_min).shape[-2] > 0:
        layout_path = save_eval_layout(
            info_path.with_suffix(".layout.json"), obstacle_min, obstacle_max
        )
        manifest["obstacle_layout_mode"] = "fixed"
        manifest["layout_file"] = layout_path.name
    write_manifest(info_path.with_suffix(".heatmap.json"), manifest)


def run_parallel_evaluation_3d(
    model, level, checkpoint: Path, run_dir: Path,
) -> Path:
    """Create confidence rates from repeated, identically reset 3D evals."""
    episodes = level.evaluation.eval_parallel_envs
    robustness_runs = level.evaluation.eval_robustness_runs
    action_noise_max = level.evaluation.eval_action_noise_max
    layout_mode = "fixed" if getattr(getattr(level, "env", None), "num_obstacles", 0) else None
    print(
        f"[EVAL] evaluating {episodes} targets x {robustness_runs} seeded "
        f"action-noise runs (max |noise|={action_noise_max:g})...",
        flush=True,
    )
    started = time.perf_counter()
    metric_runs: list[dict[str, float]] = []
    success_runs: list[np.ndarray] = []
    delivered_runs: list[np.ndarray] = []
    visually_found_runs: list[np.ndarray] = []
    reference_targets: np.ndarray | None = None
    reference_bases: np.ndarray | None = None
    reference_chain_lengths: np.ndarray | None = None
    reference_obstacle_min: np.ndarray | None = None
    reference_obstacle_max: np.ndarray | None = None
    for run_index in range(robustness_runs):
        print(
            f"[EVAL] robustness run {run_index + 1}/{robustness_runs}...",
            flush=True,
        )
        metrics, episode_info = evaluate_suite_3d(
            model,
            level,
            episodes=episodes,
            return_episode_info=True,
            action_noise_max=action_noise_max,
            action_noise_seed=level.training.seed + 20_000 + run_index,
            layout_mode=layout_mode,
        )
        targets = np.asarray(episode_info["target_positions"])
        bases = np.asarray(episode_info["base_positions"])
        chain_lengths = np.asarray(episode_info["final_chain_lengths"])
        obstacle_min = np.asarray(episode_info["obstacle_min"])
        obstacle_max = np.asarray(episode_info["obstacle_max"])
        if reference_targets is None:
            reference_targets = targets
            reference_bases = bases
            reference_chain_lengths = chain_lengths
            reference_obstacle_min = obstacle_min
            reference_obstacle_max = obstacle_max
        elif not (
            np.array_equal(targets, reference_targets)
            and np.array_equal(bases, reference_bases)
            and np.array_equal(obstacle_min, reference_obstacle_min)
            and np.array_equal(obstacle_max, reference_obstacle_max)
        ):
            raise RuntimeError(
                "Robust evaluation reset provenance changed between runs; "
                "target and base positions must be identical."
            )

        successes = np.asarray(episode_info["successes"], dtype=bool)
        delivered = (
            np.asarray(episode_info["delivered"], dtype=bool) | successes
        )
        visually_found = (
            np.asarray(episode_info["visually_found"], dtype=bool) | delivered
        )
        metric_runs.append(metrics)
        success_runs.append(successes)
        delivered_runs.append(delivered)
        visually_found_runs.append(visually_found)

    assert reference_targets is not None and reference_bases is not None
    assert reference_chain_lengths is not None
    chain_success_rate = np.mean(np.stack(success_runs), axis=0)
    found_and_delivered_rate = np.mean(np.stack(delivered_runs), axis=0)
    visually_found_rate = np.mean(np.stack(visually_found_runs), axis=0)
    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    artifact_root = eval_checkpoint_artifact_root(run_dir, checkpoint, level)
    info_path = save_eval_info_csv(
        artifact_root / "data" / f"eval_info_{artifact_tag}.csv",
        target_positions=reference_targets,
        base_positions=reference_bases,
        stage_rates={
            "chain_success": chain_success_rate,
            "found_and_delivered": found_and_delivered_rate,
            "visually_found": visually_found_rate,
        },
        final_chain_lengths=reference_chain_lengths,
    )
    _heatmap_manifest(
        info_path,
        level,
        checkpoint=checkpoint,
        robustness_runs=robustness_runs,
        action_noise_max=action_noise_max,
        obstacle_min=reference_obstacle_min,
        obstacle_max=reference_obstacle_max,
    )
    mean_success = float(
        np.mean([metrics["eval_success"] for metrics in metric_runs])
    )
    unanimous_success = float(np.mean(chain_success_rate == 1.0))
    print(
        f"[EVAL] complete in {time.perf_counter() - started:.1f}s; "
        f"mean success={mean_success:.1%}; unanimous success="
        f"{unanimous_success:.1%}; csv={info_path}",
        flush=True,
    )
    return info_path


def run_action_capturing_evaluation_3d(
    model,
    level,
    checkpoint: Path,
    run_dir: Path,
    *,
    result: str,
    offset: int,
    max_steps: int | None = None,
) -> tuple[Path, np.ndarray, str, int, list, np.ndarray]:
    """Evaluate, select, and replay actions originating in the same actor run."""
    episodes = level.evaluation.eval_parallel_envs
    print(
        f"[EVAL] evaluating {episodes} deterministic 3D episodes while capturing "
        "executed actions...",
        flush=True,
    )
    started = time.perf_counter()
    metrics, episode_info, capture = evaluate_suite_3d_with_action_capture(
        model,
        level,
        episodes=episodes,
        result=result,
        offset=offset,
        max_steps=max_steps,
    )
    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    artifact_root = eval_checkpoint_artifact_root(run_dir, checkpoint, level)
    info_path = save_eval_info_csv(
        artifact_root / "data" / f"eval_capture_info_{artifact_tag}.csv",
        target_positions=episode_info["target_positions"],
        base_positions=episode_info["base_positions"],
        successes=episode_info["successes"],
        delivered=episode_info["delivered"],
        visually_found=episode_info["visually_found"],
        final_chain_lengths=episode_info["final_chain_lengths"],
    )
    _heatmap_manifest(
        info_path,
        level,
        checkpoint=checkpoint,
        obstacle_min=episode_info["obstacle_min"],
        obstacle_max=episode_info["obstacle_max"],
    )
    lane = int(capture["lane"])
    selected_target, replay_tag, csv_lane = select_eval_target_with_lane(
        info_path, result=result, offset=offset
    )
    if csv_lane != lane:
        raise RuntimeError(
            f"On-device action selection chose lane {lane}, but the saved CSV "
            f"selected lane {csv_lane}."
        )
    print(
        f"[EVAL] complete in {time.perf_counter() - started:.1f}s; "
        f"success={metrics['eval_success']:.1%}; csv={info_path}",
        flush=True,
    )
    print(
        f"[REPLAY] selected {replay_tag} from the same evaluation: "
        f"({selected_target[0]:.2f}, {selected_target[1]:.2f}, "
        f"{selected_target[2]:.2f}); evaluation lane={lane}",
        flush=True,
    )
    states, rewards = replay_recorded_actions_3d(
        level,
        capture["actions"],
        capture_lane=lane,
        batch_size=episodes,
        max_steps=max_steps,
    )
    reconstructed_success = bool(np.asarray(states[-1].success))
    expected_success = result.lower() in {"success", "successful"}
    if reconstructed_success != expected_success:
        raise RuntimeError(
            "The actor-free replay reconstructed a different success result "
            "from the action-capturing evaluation; no misleading archive was written."
        )
    if not np.array_equal(np.asarray(states[0].target_pos), selected_target):
        raise RuntimeError(
            "The action replay's initial target differs from its captured CSV lane."
        )
    selected_records = load_eval_info_csv(info_path)
    recorded_min, _ = eval_layout_for_csv(info_path)
    if recorded_min is None:
        recorded_min = selected_records["obstacle_min"][lane]
    if not np.array_equal(
        np.asarray(states[0].obstacle_min), recorded_min
    ):
        raise RuntimeError(
            "The action replay's obstacle layout differs from its captured CSV lane."
        )
    return info_path, selected_target, replay_tag, lane, states, rewards


def find_nearest_eval_info_csv(run_dir: Path, checkpoint: Path, level) -> Path | None:
    """Find the closest training/checkpoint eval CSV by PPO update number."""
    expected = f"eval_info_{checkpoint_artifact_suffix(checkpoint, level)}.csv"
    candidates: list[Path] = []
    for path in (run_dir / "artifacts").glob("**/data/eval_info_*.csv"):
        try:
            if load_eval_info_csv(path)["positions"].shape[-1] == 3:
                candidates.append(path)
        except (OSError, ValueError):
            continue
    if not candidates:
        return None
    exact = [path for path in candidates if path.name == expected]
    if exact:
        return sorted(exact, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    checkpoint_update = parse_checkpoint_update(checkpoint) or 0

    def rank(path: Path) -> tuple[int, float]:
        match = _EVAL_INFO_UPDATE.search(path.name)
        update = int(match.group(1)) if match else 0
        return abs(update - checkpoint_update), -path.stat().st_mtime

    return min(candidates, key=rank)


def select_eval_target_with_lane(
    info_path: Path, *, result: str, offset: int
) -> tuple[np.ndarray, str, int]:
    """Select a replay target together with its immutable evaluation lane."""
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    records = load_eval_info_csv(info_path)
    positions = records["positions"]
    if positions.shape[-1] != 3:
        raise ValueError(f"Expected a 3D evaluation CSV, got {info_path}.")
    normalized = result.lower()
    if normalized in {"success", "successful"}:
        candidate_lanes, label = np.flatnonzero(records["stages"] == "chain_success"), "SUCCESS"
    elif normalized in {"fail", "failure", "failed"}:
        candidate_lanes, label = np.flatnonzero(records["stages"] != "chain_success"), "FAIL"
    else:
        raise ValueError("result must be success or fail.")
    if label == "SUCCESS":
        candidate_lanes = diverse_success_lanes(
            records, candidate_lanes, limit=offset + 1
        )
    if offset >= len(candidate_lanes):
        raise ValueError(
            f"Requested {label}_{offset}, but {info_path.name} contains only "
            f"{len(candidate_lanes)} matching targets."
        )
    lane = int(candidate_lanes[offset])
    return positions[lane], f"{label}_{offset}", lane


def select_eval_replay_lanes(
    records: dict[str, np.ndarray], *, result: str, count: int, start_offset: int = 0
) -> tuple[np.ndarray, str]:
    """Select replay configurations using only persisted CSV metrics."""
    normalized = result.lower()
    if normalized in {"success", "successful"}:
        lanes = np.flatnonzero(records["stages"] == "chain_success")
        label = "SUCCESS"
        lanes = diverse_success_lanes(
            records, lanes, limit=min(len(lanes), start_offset + count)
        )
    elif normalized in {"fail", "failure", "failed"}:
        lanes = np.flatnonzero(records["stages"] != "chain_success")
        label = "FAIL"
    else:
        raise ValueError("result must be success or fail.")
    selected = lanes[start_offset:start_offset + count]
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} {label} replay(s) starting at {start_offset}, "
            f"but only {len(lanes)} matching CSV rows exist."
        )
    return selected, label


def eval_layout_for_csv(info_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load the shared layout referenced by a heatmap sidecar, if present."""
    sidecar = info_path.with_suffix(".heatmap.json")
    if not sidecar.exists():
        return None, None
    import json
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    layout_file = metadata.get("layout_file")
    return load_eval_layout(info_path.parent / layout_file) if layout_file else (None, None)


def diverse_success_lanes(
    records: dict[str, np.ndarray], lanes: np.ndarray, *, limit: int | None = None
) -> np.ndarray:
    """Select at most ``limit`` long, spatially distinct representatives."""
    lanes = np.asarray(lanes, dtype=np.int64)
    selection_count = min(len(lanes), len(lanes) if limit is None else limit)
    if selection_count == 0:
        return lanes
    positions = records["positions"][lanes]
    lengths = records["final_chain_length"][lanes]
    length_span = float(np.ptp(lengths))
    length_score = ((lengths - np.min(lengths)) / length_span) if length_span else np.ones(len(lanes))
    world_span = float(np.linalg.norm(np.ptp(positions, axis=0))) or 1.0
    selected = [int(np.argmax(length_score))]
    if selection_count == 1:
        return lanes[np.asarray(selected)]
    available = np.ones(len(lanes), dtype=bool)
    available[selected[0]] = False
    nearest_separation = np.linalg.norm(
        positions - positions[selected[0]], axis=1
    ) / world_span
    while len(selected) < selection_count:
        score = 0.35 * length_score + 0.65 * nearest_separation
        choice = int(np.argmax(np.where(available, score, -np.inf)))
        selected.append(choice)
        available[choice] = False
        nearest_separation = np.minimum(
            nearest_separation,
            np.linalg.norm(positions - positions[choice], axis=1) / world_span,
        )
    return lanes[np.asarray(selected)]


def render_csv_replays(
    model, level, checkpoint: Path, run_dir: Path, *, result: str,
    replay_count: int, start_offset: int = 0, max_steps: int | None = None,
) -> None:
    """Render selected CSV lanes without retaining the 4k evaluation batch."""
    source_csv = find_nearest_eval_info_csv(run_dir, checkpoint, level)
    if source_csv is None:
        raise FileNotFoundError(
            "No evaluation CSV was found. Run mode=parallel before replay-only mode."
        )
    records = load_eval_info_csv(source_csv)
    shared_min, shared_max = eval_layout_for_csv(source_csv)
    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    horizon = min(max_steps or level.env.max_steps, level.env.max_steps)
    print(
        f"[REPLAY] using {source_csv} for {replay_count} replay(s); "
        "selecting spatially diverse representatives...", flush=True,
    )
    selected_lanes, label = select_eval_replay_lanes(
        records, result=result, count=replay_count, start_offset=start_offset
    )
    selection_summary = ", ".join(
        f"lane {lane}: {records['final_chain_length'][lane]:.2f}m @ "
        f"({records['positions'][lane, 0]:.2f}, {records['positions'][lane, 1]:.2f}, "
        f"{records['positions'][lane, 2]:.2f})"
        for lane in selected_lanes
    )
    print(
        f"[REPLAY] selected from CSV chain lengths and target positions: "
        f"{selection_summary}", flush=True,
    )
    for replay_number, lane_value in enumerate(selected_lanes):
        lane = int(lane_value)
        selection_offset = start_offset + replay_number
        target = records["positions"][lane]
        replay_tag = f"{label}_{selection_offset}"
        obstacle_min, obstacle_max = shared_min, shared_max
        if obstacle_min is None and records["obstacle_min"].shape[1]:
            obstacle_min = records["obstacle_min"][lane]
            obstacle_max = records["obstacle_max"][lane]
        print(
            f"[REPLAY {replay_number + 1}/{replay_count}] {replay_tag}, lane={lane}, "
            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}); "
            f"collecting up to {horizon} steps...", flush=True,
        )
        started = time.perf_counter()
        states, rewards = evaluate_model_3d(
            model, level, max_steps=max_steps, target_position=target,
            obstacle_min=obstacle_min, obstacle_max=obstacle_max,
        )
        print(
            f"[REPLAY {replay_number + 1}/{replay_count}] collected "
            f"{len(states) - 1} steps in {time.perf_counter() - started:.1f}s; "
            "compressing archive...", flush=True,
        )
        output = eval_checkpoint_replay_root(run_dir, checkpoint, level) / (
            f"eval_{artifact_tag}_{replay_tag}"
        )
        _, manifest = write_replay(
            output, states, map_name=level.building_name, dt=level.env.dt,
            reward_terms=rewards,
            metadata={
                "world_size_m": level.building.world_size_m.tolist(),
                "cell_size_m": level.building.cell_size_m,
                "coverage_voxel_size_m": level.building.cell_size_m
                if level.env.coverage_voxel_size is None
                else level.env.coverage_voxel_size,
                "comm_radius_m": level.env.comm_radius,
                "comm_radius_base_m": level.env.comm_radius_base,
                "visual_radius_m": level.env.visual_radius,
                "checkpoint": str(checkpoint),
                "target_source_csv": str(source_csv),
                "target_selection": replay_tag,
                "replay_execution": "single",
                "replay_source_lane": lane,
                "selected_target_position": target.tolist(),
                "artifact_scope": "eval",
            },
        )
        print(
            f"[REPLAY {replay_number + 1}/{replay_count}] ready: {manifest}",
            flush=True,
        )


def main() -> None:
    checkpoint: Path | None = None
    output: Path | None = None
    max_steps: int | None = None
    mode = "parallel"
    mode_explicit = False
    result = "success"
    offset = 0
    replay_after = False
    replay_count = 1
    target_mode = "csv"
    replay_execution = "single"
    explicit_target: np.ndarray | None = None
    selection_argument_seen = False
    target_argument_seen = False
    config_arguments: list[str] = []
    for argument in sys.argv[1:]:
        if "=" not in argument:
            raise ValueError(
                "Use checkpoint=<path>, mode=parallel|selective_auto_pick|"
                "selective_manual_pick, result=success|fail, offset=<n>, "
                "replay_after=true|false, replays=<positive count>, "
                "target_position=x,y,z, replay_execution=single|"
                "parallel_capture, and "
                "key=value overrides."
            )
        key, value = argument.split("=", 1)
        if key == "checkpoint":
            checkpoint = normalize_wsl_path(value).absolute()
        elif key == "output":
            output = Path(value)
        elif key == "max_steps":
            max_steps = int(value)
        elif key == "mode":
            mode = value.lower()
            mode_explicit = True
        elif key in {"result", "target_result"}:
            result = value
            selection_argument_seen = True
        elif key == "offset":
            offset = int(value)
            selection_argument_seen = True
        elif key == "replay_after":
            replay_after = _as_bool(value)
        elif key == "replays":
            replay_count = int(value)
        elif key == "target":
            target_mode = value.lower()
            target_argument_seen = True
        elif key == "target_position":
            coordinates = tuple(float(part) for part in value.split(","))
            if len(coordinates) != 3:
                raise ValueError("target_position must be x,y,z.")
            explicit_target = np.asarray(coordinates, dtype=np.float32)
        elif key == "replay_execution":
            replay_execution = value.lower()
        else:
            config_arguments.append(argument)
    if checkpoint is None:
        raise ValueError("Specify checkpoint=<path>.")
    if mode not in _EVAL_MODES:
        raise ValueError(
            "mode must be parallel, selective_auto_pick, or "
            "selective_manual_pick."
        )
    if replay_execution not in _REPLAY_EXECUTIONS:
        raise ValueError("replay_execution must be single or parallel_capture.")
    if replay_count < 1:
        raise ValueError("replays must be positive.")

    if not mode_explicit:
        # Existing inspector/manual commands omitted mode. Preserve their
        # effective behavior while making an argument-free invocation parallel.
        if explicit_target is not None:
            mode = "selective_manual_pick"
        elif selection_argument_seen or target_argument_seen:
            mode = "selective_auto_pick"

    if target_mode not in {"csv", "random"}:
        raise ValueError("target must be csv or random.")

    run_dir = checkpoint.parents[1]
    explicit_level = any(
        argument.split("=", 1)[0] == "level" for argument in config_arguments
    )
    checkpoint_level = None if explicit_level else _checkpoint_level_name(checkpoint)
    if checkpoint_level is not None:
        config_arguments.insert(0, f"level={checkpoint_level}")
        print(f"[EVAL] using checkpoint-trained level: {checkpoint_level}", flush=True)
    level = load_level_3d_cli(config_arguments)
    model = build_model_3d(level)
    restore_model_checkpoint(model, checkpoint)
    if mode == "parallel":
        run_parallel_evaluation_3d(model, level, checkpoint, run_dir)
        # Parallel mode is independent of selective arguments unless
        # replay_after=true requests the combined workflow.
        if replay_after:
            print(
                "[EVAL] evaluation artifacts complete; releasing batched results "
                "and starting replay rendering in this process...", flush=True,
            )
            render_csv_replays(
                model, level, checkpoint, run_dir, result=result,
                replay_count=replay_count, start_offset=offset, max_steps=max_steps,
            )
            return
        else:
            return

    if mode == "selective_auto_pick" and replay_execution == "single":
        render_csv_replays(
            model, level, checkpoint, run_dir, result=result,
            replay_count=replay_count, start_offset=offset, max_steps=max_steps,
        )
        return

    selected_target: np.ndarray | None = None
    selected_obstacle_min: np.ndarray | None = None
    selected_obstacle_max: np.ndarray | None = None
    selected_lane: int | None = None
    replay_tag = "RANDOM"
    source_csv: Path | None = None
    captured_states: list | None = None
    captured_rewards: np.ndarray | None = None
    if mode == "selective_manual_pick" and explicit_target is None:
        raise ValueError(
            "mode=selective_manual_pick requires target_position=x,y,z."
        )

    if mode == "selective_auto_pick" and replay_execution == "parallel_capture":
        (
            source_csv,
            selected_target,
            replay_tag,
            selected_lane,
            captured_states,
            captured_rewards,
        ) = run_action_capturing_evaluation_3d(
            model,
            level,
            checkpoint,
            run_dir,
            result=result,
            offset=offset,
            max_steps=max_steps,
        )
    elif mode == "selective_manual_pick":
        selected_target = explicit_target
        replay_tag = "TARGET"
        print(
            f"[REPLAY] selected explicit target: ({selected_target[0]:.2f}, "
            f"{selected_target[1]:.2f}, {selected_target[2]:.2f})",
            flush=True,
        )
    elif (
        mode == "selective_auto_pick"
    ):
        source_csv = find_nearest_eval_info_csv(run_dir, checkpoint, level)
        if source_csv is None:
            raise FileNotFoundError(
                "No evaluation CSV was found for this run. Run again with "
                "mode=parallel to create checkpoint-scoped target data."
            )
        selected_target, replay_tag, selected_lane = select_eval_target_with_lane(
            source_csv, result=result, offset=offset
        )
        selected_records = load_eval_info_csv(source_csv)
        selected_obstacle_min, selected_obstacle_max = eval_layout_for_csv(source_csv)
        if selected_obstacle_min is None and selected_records["obstacle_min"].shape[1]:
            selected_obstacle_min = selected_records["obstacle_min"][selected_lane]
            selected_obstacle_max = selected_records["obstacle_max"][selected_lane]
        print(
            f"[REPLAY] selected {replay_tag} target from {source_csv.name}: "
            f"({selected_target[0]:.2f}, {selected_target[1]:.2f}, {selected_target[2]:.2f}); "
            f"evaluation lane={selected_lane}",
            flush=True,
        )
    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    output = output or (
        eval_checkpoint_replay_root(run_dir, checkpoint, level) / f"eval_{artifact_tag}"
    )
    artifact_replay_tag = replay_tag
    if replay_tag == "TARGET" and selected_target is not None:
        artifact_replay_tag = _target_artifact_suffix(selected_target)
    if replay_execution == "parallel_capture" and selected_lane is None:
        raise ValueError(
            "replay_execution=parallel_capture requires mode=selective_auto_pick "
            "with a CSV-selected target."
        )
    if replay_execution == "parallel_capture":
        artifact_replay_tag = f"{artifact_replay_tag}_PARALLEL-CAPTURE"
    output = output.with_name(f"{output.name}_{artifact_replay_tag}")
    horizon = min(max_steps or level.env.max_steps, level.env.max_steps)
    if replay_execution == "parallel_capture":
        replay_batch_size = level.evaluation.eval_parallel_envs
        print(
            f"[REPLAY] materialising lane {selected_lane} from actions captured "
            "by its authoritative evaluation...",
            flush=True,
        )
    else:
        replay_batch_size = None
        print(
            f"[REPLAY] collecting one deterministic episode (up to {horizon} steps; "
            "first call may compile JAX)...",
            flush=True,
        )
    started = time.perf_counter()
    if replay_execution == "parallel_capture":
        if captured_states is None or captured_rewards is None:
            raise RuntimeError("The action-capturing evaluation produced no replay data.")
        states, rewards = captured_states, captured_rewards
    else:
        states, rewards = evaluate_model_3d(
            model,
            level,
            max_steps=max_steps,
            target_position=selected_target,
            obstacle_min=selected_obstacle_min,
            obstacle_max=selected_obstacle_max,
        )
    print(
        f"[REPLAY] collected {len(states) - 1} steps in {time.perf_counter() - started:.1f}s; "
        "preparing archive...",
        flush=True,
    )
    final_state = states[-1]
    final_stage = evaluation_stage(
        success=bool(np.asarray(final_state.success)),
        delivered=bool(np.asarray(final_state.base_target_known)),
        visually_found=bool(np.any(np.asarray(final_state.target_known))),
    )
    print(f"[REPLAY] final stage: {final_stage}", flush=True)
    _, manifest = write_replay(
        output,
        states,
        map_name=level.building_name,
        dt=level.env.dt,
        reward_terms=rewards,
        metadata={
            "world_size_m": level.building.world_size_m.tolist(),
            "cell_size_m": level.building.cell_size_m,
            "coverage_voxel_size_m": (
                level.building.cell_size_m
                if level.env.coverage_voxel_size is None
                else level.env.coverage_voxel_size
            ),
            "comm_radius_m": level.env.comm_radius,
            "comm_radius_base_m": level.env.comm_radius_base,
            "visual_radius_m": level.env.visual_radius,
            "checkpoint": str(checkpoint),
            "target_source_csv": None if source_csv is None else str(source_csv),
            "target_selection": replay_tag,
            "replay_execution": replay_execution,
            "replay_batch_size": replay_batch_size,
            "replay_source_lane": selected_lane,
            "selected_target_position": (
                None
                if selected_target is None
                else [float(value) for value in selected_target]
            ),
            "artifact_scope": "eval",
        },
    )
    print(f"3D evaluation replay: {manifest}")


if __name__ == "__main__":
    main()
