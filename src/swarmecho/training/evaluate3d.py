"""Checkpoint evaluation and target-selected replay entry point for 3D."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from swarmecho.core.config import load_level_3d_cli
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_artifact_root,
    eval_checkpoint_replay_root,
    load_eval_info_csv,
    parse_checkpoint_update,
    save_eval_info_csv,
    write_manifest,
)
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train3d import (
    build_model_3d,
    evaluate_model_3d,
    evaluate_suite_3d,
)
from swarmecho.visualize.replay3d import write_replay


_EVAL_INFO_UPDATE = re.compile(r"eval_info_u(\d+)_")
_EVAL_MODES = {"parallel", "selective_auto_pick", "selective_manual_pick"}


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


def _heatmap_manifest(info_path: Path, level, *, checkpoint: Path) -> None:
    """Write the small inspector metadata sidecar for one 3D eval CSV."""
    write_manifest(
        info_path.with_suffix(".heatmap.json"),
        {
            "format": "swarmecho-3d-eval-heatmap/v1",
            "data_file": info_path.name,
            "map_name": level.building_name,
            "world_size_m": level.building.world_size_m.tolist(),
            "checkpoint": str(checkpoint),
            "artifact_scope": "eval",
        },
    )


def run_parallel_evaluation_3d(model, level, checkpoint: Path, run_dir: Path) -> Path:
    """Create checkpoint-scoped per-target CSV data for one full 3D eval."""
    episodes = level.evaluation.eval_parallel_envs
    print(f"[EVAL] evaluating {episodes} deterministic 3D episodes...", flush=True)
    started = time.perf_counter()
    metrics, episode_info = evaluate_suite_3d(
        model,
        level,
        episodes=episodes,
        return_episode_info=True,
    )
    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    artifact_root = eval_checkpoint_artifact_root(run_dir, checkpoint, level)
    info_path = save_eval_info_csv(
        artifact_root / "data" / f"eval_info_{artifact_tag}.csv",
        target_positions=episode_info["target_positions"],
        base_positions=episode_info["base_positions"],
        successes=episode_info["successes"],
        delivered=episode_info["delivered"],
        visually_found=episode_info["visually_found"],
    )
    _heatmap_manifest(info_path, level, checkpoint=checkpoint)
    print(
        f"[EVAL] complete in {time.perf_counter() - started:.1f}s; "
        f"success={metrics['eval_success']:.1%}; csv={info_path}",
        flush=True,
    )
    return info_path


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


def select_eval_target(
    info_path: Path, *, result: str, offset: int
) -> tuple[np.ndarray, str]:
    """Select a reproducible target, prioritising distant successful targets."""
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    records = load_eval_info_csv(info_path)
    positions = records["positions"]
    if positions.shape[-1] != 3:
        raise ValueError(f"Expected a 3D evaluation CSV, got {info_path}.")
    normalized = result.lower()
    if normalized in {"success", "successful"}:
        mask, label = records["stages"] == "chain_success", "SUCCESS"
    elif normalized in {"fail", "failure", "failed"}:
        mask, label = records["stages"] != "chain_success", "FAIL"
    else:
        raise ValueError("result must be success or fail.")
    selected = positions[mask]
    if label == "SUCCESS":
        # Keep ties in their original CSV order while making SUCCESS_0 the
        # successful target furthest from the base.
        distances = records["distance_to_base"][mask]
        selected = selected[np.argsort(-distances, kind="stable")]
    if offset >= len(selected):
        raise ValueError(
            f"Requested {label}_{offset}, but {info_path.name} contains only "
            f"{len(selected)} matching targets."
        )
    return selected[offset], f"{label}_{offset}"


def main() -> None:
    checkpoint: Path | None = None
    output: Path | None = None
    max_steps: int | None = None
    mode = "parallel"
    mode_explicit = False
    legacy_parallel_eval: bool | None = None
    result = "success"
    offset = 0
    target_mode = "csv"
    explicit_target: np.ndarray | None = None
    selection_argument_seen = False
    target_argument_seen = False
    config_arguments: list[str] = []
    for argument in sys.argv[1:]:
        if "=" not in argument:
            raise ValueError(
                "Use checkpoint=<path>, mode=parallel|selective_auto_pick|"
                "selective_manual_pick, result=success|fail, offset=<n>, "
                "target_position=x,y,z, and key=value overrides."
            )
        key, value = argument.split("=", 1)
        if key == "checkpoint":
            checkpoint = Path(value.replace("\\", "/")).absolute()
        elif key == "output":
            output = Path(value)
        elif key == "max_steps":
            max_steps = int(value)
        elif key == "mode":
            mode = value.lower()
            mode_explicit = True
        elif key == "parallel_eval":
            # Backwards-compatible spelling. New callers should use mode=.
            legacy_parallel_eval = _as_bool(value)
        elif key in {"result", "target_result"}:
            result = value
            selection_argument_seen = True
        elif key == "offset":
            offset = int(value)
            selection_argument_seen = True
        elif key == "target":
            target_mode = value.lower()
            target_argument_seen = True
        elif key == "target_position":
            coordinates = tuple(float(part) for part in value.split(","))
            if len(coordinates) != 3:
                raise ValueError("target_position must be x,y,z.")
            explicit_target = np.asarray(coordinates, dtype=np.float32)
        else:
            config_arguments.append(argument)
    if checkpoint is None:
        raise ValueError("Specify checkpoint=<path>.")
    if mode not in _EVAL_MODES:
        raise ValueError(
            "mode must be parallel, selective_auto_pick, or "
            "selective_manual_pick."
        )

    # Keep old command lines working. In particular, parallel_eval=true used
    # to run the suite and then create one selected replay, while
    # parallel_eval=false used to create only the selected replay.
    legacy_selection = legacy_parallel_eval is not None
    if legacy_parallel_eval is not None:
        legacy_mode = "parallel" if legacy_parallel_eval else "selective_auto_pick"
        if mode_explicit and mode != legacy_mode:
            raise ValueError("mode and legacy parallel_eval specify different modes.")
        if not mode_explicit:
            mode = legacy_mode
    elif not mode_explicit:
        # Existing inspector/manual commands omitted mode. Preserve their
        # effective behavior while making an argument-free invocation parallel.
        if explicit_target is not None:
            mode = "selective_manual_pick"
        elif selection_argument_seen or target_argument_seen:
            mode = "selective_auto_pick"
            legacy_selection = target_argument_seen

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
        # The legacy flag retained the historical follow-up replay. New
        # parallel mode is intentionally independent of selective arguments.
        if not legacy_selection:
            return

    selected_target: np.ndarray | None = None
    replay_tag = "RANDOM"
    source_csv: Path | None = None
    if mode == "selective_manual_pick" and explicit_target is None:
        raise ValueError(
            "mode=selective_manual_pick requires target_position=x,y,z."
        )

    if mode == "selective_manual_pick":
        selected_target = explicit_target
        replay_tag = "TARGET"
        print(
            f"[REPLAY] selected explicit target: ({selected_target[0]:.2f}, "
            f"{selected_target[1]:.2f}, {selected_target[2]:.2f})",
            flush=True,
        )
    elif (
        mode == "selective_auto_pick"
        and not (
            legacy_selection
            and (target_mode == "random" or explicit_target is not None)
        )
    ):
        source_csv = find_nearest_eval_info_csv(run_dir, checkpoint, level)
        if source_csv is None:
            raise FileNotFoundError(
                "No evaluation CSV was found for this run. Run again with "
                "mode=parallel to create checkpoint-scoped target data."
            )
        selected_target, replay_tag = select_eval_target(
            source_csv, result=result, offset=offset
        )
        print(
            f"[REPLAY] selected {replay_tag} target from {source_csv.name}: "
            f"({selected_target[0]:.2f}, {selected_target[1]:.2f}, {selected_target[2]:.2f})",
            flush=True,
        )
    elif legacy_selection and explicit_target is not None:
        # Exact compatibility path for old target_position= commands.
        selected_target = explicit_target
        replay_tag = "TARGET"
        print(
            f"[REPLAY] selected explicit target: ({selected_target[0]:.2f}, "
            f"{selected_target[1]:.2f}, {selected_target[2]:.2f})",
            flush=True,
        )
    elif legacy_selection and target_mode == "csv":
        source_csv = find_nearest_eval_info_csv(run_dir, checkpoint, level)
        if source_csv is None:
            raise FileNotFoundError(
                "No evaluation CSV was found for this run. Run again with "
                "parallel_eval=true to create checkpoint-scoped target data."
            )
        selected_target, replay_tag = select_eval_target(
            source_csv, result=result, offset=offset
        )
        print(
            f"[REPLAY] selected {replay_tag} target from {source_csv.name}: "
            f"({selected_target[0]:.2f}, {selected_target[1]:.2f}, {selected_target[2]:.2f})",
            flush=True,
        )

    artifact_tag = checkpoint_artifact_suffix(checkpoint, level)
    output = output or (
        eval_checkpoint_replay_root(run_dir, checkpoint, level) / f"eval_{artifact_tag}"
    )
    artifact_replay_tag = replay_tag
    if replay_tag == "TARGET" and selected_target is not None:
        artifact_replay_tag = _target_artifact_suffix(selected_target)
    output = output.with_name(f"{output.name}_{artifact_replay_tag}")
    horizon = min(max_steps or level.env.max_steps, level.env.max_steps)
    print(
        f"[REPLAY] collecting one deterministic episode (up to {horizon} steps; "
        "first call may compile JAX)...",
        flush=True,
    )
    started = time.perf_counter()
    states, rewards = evaluate_model_3d(
        model,
        level,
        max_steps=max_steps,
        target_position=selected_target,
    )
    print(
        f"[REPLAY] collected {len(states) - 1} steps in {time.perf_counter() - started:.1f}s; "
        "preparing archive...",
        flush=True,
    )
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
