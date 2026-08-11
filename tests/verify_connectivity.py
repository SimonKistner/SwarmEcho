"""Record or compare one deterministic connectivity-verification episode.

This is an investigation harness, not a maintained behavioral test. It does
not change communication, observation, reward, or renderer semantics.

Recommended before/after workflow:

    # Before changing connectivity ownership:
    uv run swarmecho-verify-connectivity baseline=record

    # After the change, using the same checkpoint and seed:
    uv run swarmecho-verify-connectivity baseline=compare strict=true

The default checkpoint was trained on the retained tiny-maze geometry with the
current 37-dimensional observation. The old run configuration is intentionally
not loaded wholesale: it contains removed configuration keys, the pre-cleanup
map name, and TarMAC widths that disagree with the checkpoint itself. Instead,
the maintained M01 level supplies environment behavior while weight-shaping
network settings are recovered from Orbax metadata.

Coverage gaps
--------------
The harness compares physics state, actor observation flags, reward completion,
a NumPy/CPU reference graph, and a model of the current renderer graph for every
saved post-step state. It also renders the normal evaluation video with the
physics adjacency matrix visible.

It also records a synchronized, parallel, no-render performance sample of
reset, observations, environment stepping, and rewards. Model inference and
PPO are excluded so connectivity costs remain visible.

It does not prove numerical equivalence across devices, learning behavior,
TarMAC message contents, or correctness of the chosen graph semantics. A
matching behavioral baseline proves only that this deterministic trajectory
and its recorded connectivity decisions did not change. Performance results
are comparative measurements, not pass/fail assertions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from swarmecho.core.config import (
    compute_action_dim,
    compute_obs_dim,
    load_config,
    validate_config,
)
from swarmecho.env.raycast import dda_raycast_np
from swarmecho.training.evaluation import collect_video_episode
from swarmecho.training.runtime import build_evaluation_runtime
from swarmecho.training.video_worker import render_eval_video


REPO_ROOT = Path.cwd()
DEFAULT_CHECKPOINT = Path(
    "outputs/tiny_maze_nCP_simpleCurr_v6_seed_1/"
    "checkpoints/ckpt_000250"
)
DEFAULT_LEVEL = "M01_small_maze"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Options:
    checkpoint: Path
    level: str
    seed: int
    steps: int | None
    output_dir: Path | None
    baseline: str
    video: bool
    strict: bool
    overwrite: bool
    benchmark: bool
    benchmark_envs: int
    benchmark_steps: int
    benchmark_warmup: int
    benchmark_repeats: int
    config_overrides: tuple[str, ...]


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got '{value}'.")


def _parse_args() -> Options:
    values: dict[str, str] = {}
    config_overrides: list[str] = []
    known = {
        "checkpoint",
        "level",
        "seed",
        "steps",
        "output_dir",
        "baseline",
        "video",
        "strict",
        "overwrite",
        "benchmark",
        "benchmark_envs",
        "benchmark_steps",
        "benchmark_warmup",
        "benchmark_repeats",
    }

    for arg in sys.argv[1:]:
        if arg == "--no-video":
            values["video"] = "false"
            continue
        if "=" not in arg:
            raise ValueError(
                f"Unknown argument '{arg}'. Use key=value arguments."
            )
        key, value = arg.split("=", 1)
        if key in known:
            values[key] = value
        else:
            config_overrides.append(arg)

    checkpoint = Path(values.get("checkpoint", str(DEFAULT_CHECKPOINT)))
    output = values.get("output_dir")
    steps = values.get("steps")
    return Options(
        checkpoint=checkpoint,
        level=values.get("level", DEFAULT_LEVEL),
        seed=int(values.get("seed", "0")),
        steps=int(steps) if steps is not None else None,
        output_dir=Path(output) if output else None,
        baseline=values.get("baseline", "none"),
        video=_as_bool(values.get("video", "true")),
        strict=_as_bool(values.get("strict", "false")),
        overwrite=_as_bool(values.get("overwrite", "false")),
        benchmark=_as_bool(values.get("benchmark", "true")),
        benchmark_envs=int(values.get("benchmark_envs", "4000")),
        benchmark_steps=int(values.get("benchmark_steps", "100")),
        benchmark_warmup=int(values.get("benchmark_warmup", "2")),
        benchmark_repeats=int(values.get("benchmark_repeats", "5")),
        config_overrides=tuple(config_overrides),
    )


def _resolve_checkpoint(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved}\n"
            "Pass checkpoint=<path>. The default expects the retained local "
            "run tiny_maze_nCP_simpleCurr_v6_seed_1/ckpt_000250."
        )
    return resolved


def _default_output_dir(checkpoint: Path, seed: int) -> Path:
    if checkpoint.parent.name == "checkpoints":
        run_dir = checkpoint.parents[1]
    else:
        run_dir = REPO_ROOT / "outputs"
    return (
        run_dir
        / "artifacts"
        / "verification"
        / "connectivity"
        / f"{checkpoint.name}_seed{seed}"
    )


def _checkpoint_parameter_shapes(
    checkpoint: Path,
) -> dict[tuple[str, ...], tuple[int, ...]]:
    metadata_path = checkpoint / "_METADATA"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tree = metadata.get("tree_metadata", {})
    shapes: dict[tuple[str, ...], tuple[int, ...]] = {}
    for key, item in tree.items():
        try:
            path = ast.literal_eval(key)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(path, tuple) or not all(
            isinstance(part, str) for part in path
        ):
            continue
        shape = item.get("value_metadata", {}).get("write_shape")
        if shape:
            shapes[path] = tuple(int(size) for size in shape)
    return shapes


def _layer_count(
    shapes: dict[tuple[str, ...], tuple[int, ...]],
    prefix: tuple[str, ...],
) -> int | None:
    indices = {
        int(path[len(prefix)])
        for path in shapes
        if (
            len(path) == len(prefix) + 3
            and path[:len(prefix)] == prefix
            and path[len(prefix)].isdigit()
            and path[-2:] == ("kernel", "value")
        )
    }
    return max(indices) + 1 if indices else None


def _checkpoint_model_spec(checkpoint: Path) -> dict[str, int | bool]:
    """Infer weight-shaping model settings from Orbax metadata."""
    shapes = _checkpoint_parameter_shapes(checkpoint)
    if not shapes:
        return {}

    actor_input = shapes.get(
        ("actor", "encoder", "layers", "0", "kernel", "value")
    ) or shapes.get(
        ("actor", "trunk", "layers", "0", "kernel", "value")
    )
    actor_output = shapes.get(
        ("actor", "mu_head", "kernel", "value")
    )
    tarmac_signature = shapes.get(
        ("actor", "tarmac_signature", "kernel", "value")
    )
    tarmac_value = shapes.get(
        ("actor", "tarmac_value", "kernel", "value")
    )

    spec: dict[str, int | bool] = {
        "actor_memory": any(path[:2] == ("actor", "gru") for path in shapes),
        "critic_memory": any(path[:2] == ("critic", "gru") for path in shapes),
        "memory_comm_enabled": tarmac_signature is not None,
    }
    if actor_input:
        spec["obs_dim"] = actor_input[0]
        spec["hidden_dim"] = actor_input[1]
    if actor_output:
        spec["act_dim"] = actor_output[-1]
    if tarmac_signature:
        spec["tarmac_sig_dim"] = tarmac_signature[-1]
    if tarmac_value:
        spec["tarmac_val_dim"] = tarmac_value[-1]

    actor_layers = (
        _layer_count(shapes, ("actor", "policy_trunk", "layers"))
        or _layer_count(shapes, ("actor", "trunk", "layers"))
    )
    critic_layers = _layer_count(
        shapes, ("critic", "value_head", "layers")
    )
    if actor_layers is not None:
        spec["actor_num_layers"] = actor_layers
    if critic_layers is not None:
        spec["num_layers"] = critic_layers
    return spec


def _apply_checkpoint_model_spec(
    cfg: Any,
    checkpoint: Path,
) -> tuple[dict[str, int | bool], list[str]]:
    """Recreate the saved parameter shapes without loading stale run config."""
    spec = _checkpoint_model_spec(checkpoint)
    configured_obs_dim = compute_obs_dim(cfg)
    configured_act_dim = compute_action_dim(cfg)

    checkpoint_obs_dim = spec.get("obs_dim")
    if (
        checkpoint_obs_dim is not None
        and int(checkpoint_obs_dim) != configured_obs_dim
    ):
        raise ValueError(
            "Checkpoint/config observation mismatch: checkpoint actor expects "
            f"{checkpoint_obs_dim}, verification config produces "
            f"{configured_obs_dim}. Choose a matching level or pass observation "
            "switch overrides explicitly."
        )
    checkpoint_act_dim = spec.get("act_dim")
    if (
        checkpoint_act_dim is not None
        and int(checkpoint_act_dim) != configured_act_dim
    ):
        raise ValueError(
            "Checkpoint/config action mismatch: checkpoint actor expects "
            f"{checkpoint_act_dim}, verification config produces "
            f"{configured_act_dim}."
        )

    changes: list[str] = []
    updates: dict[str, int | bool] = {}
    for key in (
        "hidden_dim",
        "num_layers",
        "actor_num_layers",
        "actor_memory",
        "critic_memory",
        "memory_comm_enabled",
        "tarmac_sig_dim",
        "tarmac_val_dim",
    ):
        if key not in spec:
            continue
        previous = cfg.network.get(key)
        saved = spec[key]
        if previous != saved:
            changes.append(f"network.{key}: {previous} -> {saved}")
            updates[key] = saved
    # load_config() intentionally returns a frozen structured config. This
    # verifier changes only its private runtime copy to match checkpoint
    # parameter shapes, then immediately restores the normal invariant.
    OmegaConf.set_readonly(cfg, False)
    try:
        for key, saved in updates.items():
            cfg.network[key] = saved
    finally:
        OmegaConf.set_readonly(cfg, True)
    return spec, changes


def _transitive_closure(adjacency: np.ndarray) -> np.ndarray:
    reach = np.asarray(adjacency, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for intermediate in range(reach.shape[0]):
        reach |= (
            reach[:, intermediate, None]
            & reach[None, intermediate, :]
        )
    return reach


def _cpu_reference_graph(
    state: Any,
    cfg: Any,
    occupancy_grid: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    """Independent CPU expression of the intended physics graph."""
    pos = np.asarray(state.physics.pos, dtype=np.float32)
    base_pos = np.asarray(state.physics.base_pos, dtype=np.float32)
    target_pos = np.asarray(state.physics.target_pos, dtype=np.float32)
    active = np.asarray(state.physics.active, dtype=bool)
    num_agents = pos.shape[0]
    adjacency = np.zeros((num_agents + 1, num_agents + 1), dtype=bool)

    comm_radius = float(cfg.env.comm_radius)
    base_radius = float(
        cfg.env.get("comm_radius_base", cfg.env.comm_radius)
    )
    visual_radius = float(cfg.env.visual_radius)

    for left in range(num_agents):
        for right in range(left + 1, num_agents):
            in_range = (
                active[left]
                and active[right]
                and np.linalg.norm(pos[left] - pos[right]) <= comm_radius
            )
            if in_range and dda_raycast_np(
                pos[left], pos[right], occupancy_grid
            ):
                adjacency[left, right] = True
                adjacency[right, left] = True

    base_index = num_agents
    for agent in range(num_agents):
        in_range = (
            active[agent]
            and np.linalg.norm(pos[agent] - base_pos) <= base_radius
        )
        if in_range and dda_raycast_np(
            pos[agent], base_pos, occupancy_grid
        ):
            adjacency[agent, base_index] = True
            adjacency[base_index, agent] = True

    directly_sees = np.zeros(num_agents, dtype=bool)
    for agent in range(num_agents):
        in_range = (
            active[agent]
            and np.linalg.norm(pos[agent] - target_pos) <= visual_radius
        )
        directly_sees[agent] = in_range and dda_raycast_np(
            pos[agent], target_pos, occupancy_grid
        )

    reach = _transitive_closure(adjacency)
    connected_base = reach[:num_agents, base_index]
    connected_target = np.any(
        reach[:num_agents, :num_agents]
        & directly_sees[None, :],
        axis=1,
    )
    return {
        "adjacency": adjacency,
        "directly_sees": directly_sees,
        "connected_base": connected_base,
        "connected_target": connected_target,
        "fully_connected": bool(
            np.any(connected_base & connected_target)
        ),
    }


def _renderer_graph_model(
    state: Any,
    _cfg: Any,
    _occupancy_grid: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    """Mirror the renderer translation of physics-owned compact graph data."""
    pos = np.asarray(state.physics.pos, dtype=np.float32)
    base_pos = np.asarray(state.physics.base_pos, dtype=np.float32)
    target_pos = np.asarray(state.physics.target_pos, dtype=np.float32)
    entities = np.concatenate(
        [base_pos[None, :], target_pos[None, :], pos],
        axis=0,
    )
    base_index = 0
    target_index = 1
    drone_start = 2
    adjacency = np.zeros(
        (entities.shape[0], entities.shape[0]),
        dtype=bool,
    )
    physics_adjacency = np.asarray(
        state.communication.adj_matrix,
        dtype=bool,
    )
    num_agents = pos.shape[0]
    adjacency[drone_start:, drone_start:] = (
        physics_adjacency[:num_agents, :num_agents]
    )
    adjacency[base_index, drone_start:] = (
        physics_adjacency[num_agents, :num_agents]
    )
    adjacency[drone_start:, base_index] = (
        physics_adjacency[:num_agents, num_agents]
    )
    target_edges = np.asarray(
        state.communication.directly_sees_target,
        dtype=bool,
    )
    adjacency[target_index, drone_start:] = target_edges
    adjacency[drone_start:, target_index] = target_edges

    reach = _transitive_closure(adjacency)
    connected_base = reach[drone_start:, base_index]
    connected_target = reach[drone_start:, target_index]
    return {
        "adjacency": adjacency,
        "connected_base": connected_base,
        "connected_target": connected_target,
        "fully_connected": bool(reach[base_index, target_index]),
    }


def _distance_edges_blocked_by_walls(
    state: Any,
    cfg: Any,
    occupancy_grid: np.ndarray,
) -> int:
    """Count edges the removed range-only reward graph would have admitted."""
    pos = np.asarray(state.physics.pos, dtype=np.float32)
    base_pos = np.asarray(state.physics.base_pos, dtype=np.float32)
    target_pos = np.asarray(state.physics.target_pos, dtype=np.float32)
    active = np.asarray(state.physics.active, dtype=bool)
    comm_radius = float(cfg.env.comm_radius)
    base_radius = float(
        cfg.env.get("comm_radius_base", cfg.env.comm_radius)
    )
    visual_radius = float(cfg.env.visual_radius)
    blocked = 0

    for left in range(pos.shape[0]):
        for right in range(left + 1, pos.shape[0]):
            candidate = (
                active[left]
                and active[right]
                and np.linalg.norm(pos[left] - pos[right]) <= comm_radius
            )
            if candidate and not dda_raycast_np(
                pos[left], pos[right], occupancy_grid
            ):
                blocked += 1

    for agent in range(pos.shape[0]):
        if not active[agent]:
            continue
        if (
            np.linalg.norm(pos[agent] - base_pos) <= base_radius
            and not dda_raycast_np(pos[agent], base_pos, occupancy_grid)
        ):
            blocked += 1
        if (
            np.linalg.norm(pos[agent] - target_pos) <= visual_radius
            and not dda_raycast_np(pos[agent], target_pos, occupancy_grid)
        ):
            blocked += 1
    return blocked


def _bool_list(value: Any) -> list[bool]:
    return np.asarray(value, dtype=bool).tolist()


def _int_matrix(value: Any) -> list[list[int]]:
    return np.asarray(value, dtype=np.int8).tolist()


def _rounded(value: Any) -> list:
    return np.round(np.asarray(value, dtype=np.float64), 6).tolist()


def _observation_connectivity(obs: np.ndarray, cfg: Any) -> tuple:
    offset = 2
    if bool(cfg.env.get("observe_base_vector", True)):
        offset += 2
    connected_base = obs[:, offset] > 0.5
    connected_target = obs[:, offset + 1] > 0.5
    target_known = obs[:, offset + 2] > 0.5
    return connected_base, connected_target, target_known


def _analyse_trajectory(
    states: list[Any],
    cfg: Any,
    compute_obs,
    compute_reward,
    occupancy_grid: Any,
) -> tuple[list[dict], dict]:
    grid = np.asarray(occupancy_grid, dtype=bool)
    obs_jit = jax.jit(compute_obs)
    reward_jit = jax.jit(compute_reward)
    frames: list[dict] = []

    counts = {
        "post_step_frames": 0,
        "physics_vs_cpu_reference_mismatch_frames": 0,
        "physics_vs_observation_mismatch_frames": 0,
        "reward_vs_physics_mismatch_frames": 0,
        "reward_contributor_outside_physics_components_frames": 0,
        "renderer_vs_physics_mismatch_frames": 0,
        "reward_range_graph_has_wall_blocked_edges_frames": 0,
        "reset_physics_uninitialized_mismatch": False,
    }

    for frame_index, state in enumerate(states):
        step = int(np.asarray(state.physics.step))
        active = np.asarray(state.physics.active, dtype=bool)
        physics_base = np.asarray(
            state.communication.is_conn_base,
            dtype=bool,
        )
        physics_target = np.asarray(
            state.communication.is_conn_target,
            dtype=bool,
        )
        physics_direct_target = np.asarray(
            state.communication.directly_sees_target,
            dtype=bool,
        )
        physics_full = bool(np.any(physics_base & physics_target))
        physics_adjacency = np.asarray(
            state.communication.adj_matrix,
            dtype=bool,
        )

        obs = np.asarray(jax.device_get(obs_jit(state)))
        obs_base, obs_target, obs_known = _observation_connectivity(
            obs, cfg
        )
        reference = _cpu_reference_graph(state, cfg, grid)
        renderer = _renderer_graph_model(state, cfg, grid)
        blocked_edges = _distance_edges_blocked_by_walls(
            state, cfg, grid
        )

        adjacency_matches = (
            physics_adjacency.shape
            == np.asarray(reference["adjacency"]).shape
            and np.array_equal(
                physics_adjacency,
                reference["adjacency"],
            )
        )
        physics_cpu_matches = (
            adjacency_matches
            and np.array_equal(
                physics_base, reference["connected_base"]
            )
            and np.array_equal(
                physics_target, reference["connected_target"]
            )
            and np.array_equal(
                physics_direct_target, reference["directly_sees"]
            )
        )
        physics_obs_matches = (
            np.array_equal(physics_base, obs_base)
            and np.array_equal(physics_target, obs_target)
        )
        renderer_matches = (
            np.array_equal(
                physics_base, renderer["connected_base"]
            )
            and np.array_equal(
                physics_target, renderer["connected_target"]
            )
            and physics_full == renderer["fully_connected"]
        )

        reward_full: bool | None = None
        reward_contributing: list[bool] | None = None
        reward_contributor_outside = np.zeros_like(
            physics_base, dtype=bool
        )
        reward_matches = True
        if frame_index > 0:
            _, reward_info = reward_jit(
                states[frame_index - 1],
                state,
                np.bool_(False),
            )
            reward_full = bool(
                np.asarray(
                    jax.device_get(reward_info["fully_connected"])
                )
                > 0.5
            )
            reward_contributing_array = np.asarray(
                jax.device_get(reward_info["is_contributing"]),
                dtype=bool,
            )
            reward_contributing = _bool_list(
                reward_contributing_array
            )
            reward_contributor_outside = (
                reward_contributing_array
                & ~(physics_base | physics_target)
            )
            reward_matches = reward_full == physics_full

        is_post_step = step > 0
        if is_post_step:
            counts["post_step_frames"] += 1
            if not physics_cpu_matches:
                counts[
                    "physics_vs_cpu_reference_mismatch_frames"
                ] += 1
            if not physics_obs_matches:
                counts[
                    "physics_vs_observation_mismatch_frames"
                ] += 1
            if not reward_matches:
                counts[
                    "reward_vs_physics_mismatch_frames"
                ] += 1
            if np.any(reward_contributor_outside):
                counts[
                    "reward_contributor_outside_physics_components_frames"
                ] += 1
            if not renderer_matches:
                counts[
                    "renderer_vs_physics_mismatch_frames"
                ] += 1
            if blocked_edges:
                counts[
                    "reward_range_graph_has_wall_blocked_edges_frames"
                ] += 1
        elif (
            not physics_cpu_matches
            or not physics_obs_matches
            or not renderer_matches
        ):
            counts["reset_physics_uninitialized_mismatch"] = True

        frames.append(
            {
                "frame_index": frame_index,
                "step": step,
                "pos": _rounded(state.physics.pos),
                "base_pos": _rounded(state.physics.base_pos),
                "target_pos": _rounded(state.physics.target_pos),
                "active": _bool_list(active),
                "physics": {
                    "adjacency": _int_matrix(physics_adjacency),
                    "connected_base": _bool_list(physics_base),
                    "connected_target": _bool_list(physics_target),
                    "directly_sees_target": _bool_list(
                        physics_direct_target
                    ),
                    "fully_connected": physics_full,
                },
                "observation": {
                    "connected_base": _bool_list(obs_base),
                    "connected_target": _bool_list(obs_target),
                    "target_known": _bool_list(obs_known),
                },
                "cpu_reference": {
                    "adjacency": _int_matrix(reference["adjacency"]),
                    "directly_sees": _bool_list(
                        reference["directly_sees"]
                    ),
                    "connected_base": _bool_list(
                        reference["connected_base"]
                    ),
                    "connected_target": _bool_list(
                        reference["connected_target"]
                    ),
                    "fully_connected": bool(
                        reference["fully_connected"]
                    ),
                },
                "reward": {
                    "fully_connected": reward_full,
                    "is_contributing": reward_contributing,
                    "contributing_outside_physics_components": _bool_list(
                        reward_contributor_outside
                    ),
                    "range_only_edges_blocked_by_walls": blocked_edges,
                },
                "renderer_model": {
                    "adjacency": _int_matrix(renderer["adjacency"]),
                    "connected_base": _bool_list(
                        renderer["connected_base"]
                    ),
                    "connected_target": _bool_list(
                        renderer["connected_target"]
                    ),
                    "fully_connected": bool(
                        renderer["fully_connected"]
                    ),
                },
                "matches": {
                    "physics_cpu_reference": physics_cpu_matches,
                    "physics_observation": physics_obs_matches,
                    "reward_physics": reward_matches,
                    "renderer_physics": renderer_matches,
                },
            }
        )

    counts["core_post_step_consistent"] = all(
        counts[name] == 0
        for name in (
            "physics_vs_cpu_reference_mismatch_frames",
            "physics_vs_observation_mismatch_frames",
            "reward_vs_physics_mismatch_frames",
        )
    )
    counts["all_consumers_consistent"] = (
        counts["core_post_step_consistent"]
        and not counts["reset_physics_uninitialized_mismatch"]
        and counts["renderer_vs_physics_mismatch_frames"] == 0
        and counts[
            "reward_contributor_outside_physics_components_frames"
        ] == 0
    )
    return frames, counts


def _comparison_payload(
    checkpoint: Path,
    options: Options,
    cfg: Any,
    frames: list[dict],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_name": checkpoint.name,
        "level": options.level,
        "seed": options.seed,
        "configured_max_steps": int(cfg.env.max_steps),
        "observation_dim": compute_obs_dim(cfg),
        "map_names": [str(name) for name in cfg.env.map_names],
        "frames": frames,
    }


def _behavior_projection(comparison: dict[str, Any]) -> dict[str, Any]:
    """Select state facts that must remain unchanged by graph consolidation."""
    return {
        "checkpoint_name": comparison["checkpoint_name"],
        "level": comparison["level"],
        "seed": comparison["seed"],
        "configured_max_steps": comparison["configured_max_steps"],
        "observation_dim": comparison["observation_dim"],
        "map_names": comparison["map_names"],
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "step": frame["step"],
                "pos": frame["pos"],
                "base_pos": frame["base_pos"],
                "target_pos": frame["target_pos"],
                "active": frame["active"],
                "target_known": frame["observation"]["target_known"],
            }
            for frame in comparison["frames"]
        ],
    }


def _digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_difference(
    before: Any,
    after: Any,
    path: str = "comparison",
) -> str | None:
    if type(before) is not type(after):
        return f"{path}: type {type(before).__name__} != {type(after).__name__}"
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return f"{path}: keys {sorted(before)} != {sorted(after)}"
        for key in before:
            difference = _first_difference(
                before[key], after[key], f"{path}.{key}"
            )
            if difference:
                return difference
        return None
    if isinstance(before, list):
        if len(before) != len(after):
            return f"{path}: length {len(before)} != {len(after)}"
        for index, (left, right) in enumerate(zip(before, after)):
            difference = _first_difference(
                left, right, f"{path}[{index}]"
            )
            if difference:
                return difference
        return None
    if before != after:
        return f"{path}: {before!r} != {after!r}"
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _device_memory_stats() -> dict[str, int]:
    """Return portable integer memory counters when the JAX backend has them."""
    try:
        raw = jax.devices()[0].memory_stats() or {}
    except (AttributeError, RuntimeError):
        return {}
    return {
        key: int(value)
        for key, value in raw.items()
        if isinstance(value, (int, np.integer))
        and (
            "byte" in key.lower()
            or "memory" in key.lower()
        )
    }


def _run_performance_benchmark(
    runtime: Any,
    cfg: Any,
    options: Options,
) -> dict[str, Any]:
    """Time the no-render environment/observation/reward connectivity path.

    Reset is included because a consolidated graph must also initialise reset
    state correctly. Model inference, PPO, host conversion, JSON, and rendering
    are intentionally excluded so they cannot hide connectivity costs.
    """
    num_envs = options.benchmark_envs
    num_steps = options.benchmark_steps
    num_agents = int(cfg.env.num_agents)
    action_dim = compute_action_dim(cfg)
    max_force = float(cfg.env.max_force)

    reset_v = jax.vmap(runtime.environment.reset)
    env_step_v = jax.vmap(runtime.environment.env_step)
    obs_v = jax.vmap(runtime.environment.compute_obs)
    reward_v = jax.vmap(
        runtime.environment.compute_reward,
        in_axes=(0, 0, None),
    )

    action_rng = np.random.default_rng(options.seed + 17_031)
    actions_host = action_rng.uniform(
        low=-0.25 * max_force,
        high=0.25 * max_force,
        size=(num_steps, num_envs, num_agents, action_dim),
    ).astype(np.float32)
    actions = jax.device_put(actions_host)
    jax.block_until_ready(actions)
    del actions_host

    def benchmark_once(key: jax.Array, action_sequence: jax.Array):
        states = reset_v(jax.random.split(key, num_envs))

        def one_step(current_states, step_actions):
            observations = obs_v(current_states)
            next_states = env_step_v(current_states, step_actions)
            rewards, info = reward_v(
                current_states,
                next_states,
                jnp.bool_(False),
            )
            checksum = (
                jnp.sum(observations, dtype=jnp.float32)
                + jnp.sum(rewards, dtype=jnp.float32)
                + jnp.sum(
                    info["fully_connected"].astype(jnp.float32),
                    dtype=jnp.float32,
                )
            )
            return next_states, checksum

        final_states, checksums = jax.lax.scan(
            one_step,
            states,
            action_sequence,
        )
        return (
            jnp.sum(checksums, dtype=jnp.float32)
            + jnp.sum(final_states.physics.step, dtype=jnp.float32)
        )

    benchmark_jit = jax.jit(benchmark_once)
    run_keys = jax.random.split(
        jax.random.PRNGKey(options.seed + 81_911),
        options.benchmark_warmup + options.benchmark_repeats,
    )

    memory_before = _device_memory_stats()
    compile_started = time.perf_counter()
    first_result = benchmark_jit(run_keys[0], actions)
    jax.block_until_ready(first_result)
    compile_and_first_run_seconds = time.perf_counter() - compile_started

    for warmup_index in range(1, options.benchmark_warmup):
        result = benchmark_jit(run_keys[warmup_index], actions)
        jax.block_until_ready(result)

    elapsed_seconds: list[float] = []
    checksum = float(np.asarray(jax.device_get(first_result)))
    for repeat_index in range(options.benchmark_repeats):
        key_index = options.benchmark_warmup + repeat_index
        started = time.perf_counter()
        result = benchmark_jit(run_keys[key_index], actions)
        jax.block_until_ready(result)
        elapsed_seconds.append(time.perf_counter() - started)
        checksum = float(np.asarray(jax.device_get(result)))

    median_seconds = statistics.median(elapsed_seconds)
    total_env_steps = num_envs * num_steps
    memory_after = _device_memory_stats()
    return {
        "scope": (
            "JIT reset + batched observations + environment step + rewards; "
            "model, PPO, host trajectory conversion, JSON, and video excluded"
        ),
        "num_envs": num_envs,
        "num_agents": num_agents,
        "num_steps": num_steps,
        "warmup_runs": options.benchmark_warmup,
        "timed_repeats": options.benchmark_repeats,
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "elapsed_seconds": elapsed_seconds,
        "median_seconds": median_seconds,
        "median_env_steps_per_second": total_env_steps / median_seconds,
        "checksum": checksum,
        "device": str(jax.devices()[0]),
        "jax_version": str(jax.__version__),
        "observation_dim": compute_obs_dim(cfg),
        "map_names": [str(name) for name in cfg.env.map_names],
        "memory_comm_enabled": bool(
            cfg.network.get("memory_comm_enabled", False)
        ),
        "chain_reward_system": str(
            cfg.reward.get("chain_reward_system", "euclidean")
        ),
        "memory_before": memory_before,
        "memory_after": memory_after,
    }


def _performance_comparison(
    baseline: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None or current is None:
        return None
    compatible_fields = (
        "num_envs",
        "num_agents",
        "num_steps",
        "device",
        "jax_version",
        "observation_dim",
        "map_names",
        "memory_comm_enabled",
        "chain_reward_system",
    )
    differences = {
        field: [baseline.get(field), current.get(field)]
        for field in compatible_fields
        if baseline.get(field) != current.get(field)
    }
    if differences:
        return {
            "comparable": False,
            "configuration_differences": differences,
        }

    baseline_rate = float(baseline["median_env_steps_per_second"])
    current_rate = float(current["median_env_steps_per_second"])
    comparison = {
        "comparable": True,
        "baseline_median_env_steps_per_second": baseline_rate,
        "current_median_env_steps_per_second": current_rate,
        "throughput_change_percent": (
            (current_rate / baseline_rate - 1.0) * 100.0
        ),
        "baseline_median_seconds": float(baseline["median_seconds"]),
        "current_median_seconds": float(current["median_seconds"]),
    }
    baseline_peak = baseline.get("memory_after", {}).get(
        "peak_bytes_in_use"
    )
    current_peak = current.get("memory_after", {}).get(
        "peak_bytes_in_use"
    )
    if baseline_peak is not None and current_peak is not None:
        comparison["baseline_peak_bytes_in_use"] = int(baseline_peak)
        comparison["current_peak_bytes_in_use"] = int(current_peak)
        comparison["peak_memory_change_mib"] = (
            (int(current_peak) - int(baseline_peak)) / (1024.0 ** 2)
        )
    return comparison


def main() -> None:
    options = _parse_args()
    if options.benchmark_envs < 1:
        raise ValueError("benchmark_envs must be >= 1.")
    if options.benchmark_steps < 1:
        raise ValueError("benchmark_steps must be >= 1.")
    if options.benchmark_warmup < 1:
        raise ValueError("benchmark_warmup must be >= 1.")
    if options.benchmark_repeats < 1:
        raise ValueError("benchmark_repeats must be >= 1.")

    checkpoint = _resolve_checkpoint(options.checkpoint)
    output_dir = (
        options.output_dir.resolve()
        if options.output_dir is not None
        else _default_output_dir(checkpoint, options.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"level={options.level}",
        "visualize.render_conn_matrix=true",
        "evaluation.eval_video=true",
        *options.config_overrides,
    ]
    if options.steps is not None:
        overrides.append(f"env.max_steps={options.steps}")
    cfg = load_config(cli_overrides=True, overrides=overrides)
    validate_config(cfg)
    checkpoint_spec, architecture_changes = _apply_checkpoint_model_spec(
        cfg, checkpoint
    )
    validate_config(cfg)
    configured_obs_dim = compute_obs_dim(cfg)

    print("\nConnectivity verification")
    print(f"  checkpoint : {checkpoint}")
    print(f"  level      : {options.level}")
    print(f"  seed       : {options.seed}")
    print(f"  obs dim    : {configured_obs_dim}")
    print(f"  output     : {output_dir}")
    if checkpoint_spec:
        print(
            "  saved net  : "
            f"H={checkpoint_spec.get('hidden_dim', '?')}, "
            f"actor={checkpoint_spec.get('actor_num_layers', '?')} layers, "
            f"critic={checkpoint_spec.get('num_layers', '?')} layers, "
            "TarMAC="
            f"{checkpoint_spec.get('tarmac_sig_dim', '-')}/"
            f"{checkpoint_spec.get('tarmac_val_dim', '-')}"
        )
    if architecture_changes:
        print("  restored architecture overrides:")
        for change in architecture_changes:
            print(f"    {change}")

    runtime = build_evaluation_runtime(
        cfg, checkpoint, rng_seed=options.seed
    )
    performance: dict[str, Any] | None = None
    if options.benchmark:
        print(
            "\nPerformance benchmark "
            f"({options.benchmark_envs} envs x "
            f"{options.benchmark_steps} steps, "
            f"{options.benchmark_repeats} timed repeats)"
        )
        performance = _run_performance_benchmark(
            runtime, cfg, options
        )
        print(
            "  median throughput : "
            f"{performance['median_env_steps_per_second']:,.0f} env steps/s"
        )
        print(
            "  median duration   : "
            f"{performance['median_seconds']:.4f} s"
        )
        print(
            "  compile + first   : "
            f"{performance['compile_and_first_run_seconds']:.4f} s"
        )

    states_all, rewards_all, metrics_all = collect_video_episode(
        runtime.model,
        jax.jit(runtime.environment.reset),
        jax.jit(runtime.environment.env_step),
        jax.jit(runtime.environment.compute_obs),
        jax.jit(runtime.environment.compute_reward),
        cfg,
        jax.random.PRNGKey(options.seed),
    )
    states = states_all[0]
    rewards = rewards_all[0]
    metrics = metrics_all[0]

    frames, summary = _analyse_trajectory(
        states,
        cfg,
        runtime.environment.compute_obs,
        runtime.environment.compute_reward,
        runtime.environment.occupancy_grid,
    )
    comparison = _comparison_payload(
        checkpoint, options, cfg, frames
    )
    report = {
        "comparison": comparison,
        "comparison_sha256": _digest(comparison),
        "checkpoint": str(checkpoint),
        "checkpoint_model_spec": checkpoint_spec,
        "performance": performance,
        "summary": summary,
        "notes": [
            "Step zero is reported separately to verify that reset now "
            "initializes the same physics-owned connectivity consumed by "
            "observations and TarMAC.",
            "Reward completion consumes physics connectivity, so equality there "
            "checks wiring rather than an independent graph implementation.",
            "The legacy range-only reward graph's wall-blocked candidates are "
            "still counted to show the risk removed by physics-owned routing.",
            "The renderer model translates communication adjacency plus "
            "direct target visibility into renderer node order.",
        ],
    }
    current_path = output_dir / "current.json"
    _write_json(current_path, report)

    video_path: str | None = None
    if options.video:
        video_path = render_eval_video(
            ep_states=states,
            ep_rewards=rewards,
            ep_metrics=metrics,
            cfg=cfg,
            out_dir=output_dir / "vids",
            filename_stem="connectivity_verification",
        )

    baseline_path = output_dir / "baseline.json"
    baseline_result = "not requested"
    baseline_mismatch = False
    behavior_mismatch = False
    baseline_report: dict[str, Any] | None = None
    if options.baseline == "record":
        if baseline_path.exists() and not options.overwrite:
            raise FileExistsError(
                f"Baseline already exists: {baseline_path}\n"
                "Use baseline=compare, or overwrite=true only when deliberately "
                "replacing the before-change reference."
            )
        _write_json(baseline_path, report)
        baseline_result = f"recorded: {baseline_path}"
    elif options.baseline == "compare":
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline not found: {baseline_path}\n"
                "Run once with baseline=record before changing connectivity."
            )
        baseline_report = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        difference = _first_difference(
            baseline_report["comparison"], comparison
        )
        behavior_difference = _first_difference(
            _behavior_projection(baseline_report["comparison"]),
            _behavior_projection(comparison),
        )
        if behavior_difference:
            behavior_mismatch = True
        if difference:
            baseline_mismatch = behavior_mismatch
            baseline_result = (
                "BEHAVIOR DIFFERENT — first difference: "
                f"{behavior_difference}"
                if behavior_mismatch
                else "EXPECTED CONNECTIVITY DIFFERENCES — first full-report "
                f"difference: {difference}"
            )
        else:
            baseline_result = "FULL MATCH"
    elif options.baseline not in {"none", ""}:
        explicit_path = Path(options.baseline)
        if not explicit_path.is_absolute():
            explicit_path = (REPO_ROOT / explicit_path).resolve()
        baseline_report = json.loads(
            explicit_path.read_text(encoding="utf-8")
        )
        difference = _first_difference(
            baseline_report["comparison"], comparison
        )
        behavior_difference = _first_difference(
            _behavior_projection(baseline_report["comparison"]),
            _behavior_projection(comparison),
        )
        if behavior_difference:
            behavior_mismatch = True
        if difference:
            baseline_mismatch = behavior_mismatch
            baseline_result = (
                "BEHAVIOR DIFFERENT — first difference: "
                f"{behavior_difference}"
                if behavior_mismatch
                else "EXPECTED CONNECTIVITY DIFFERENCES — first full-report "
                f"difference: {difference}"
            )
        else:
            baseline_result = f"FULL MATCH: {explicit_path}"

    performance_result = _performance_comparison(
        baseline_report.get("performance") if baseline_report else None,
        performance,
    )

    print("\nSubsystem comparison")
    print(
        "  physics vs CPU reference : "
        f"{summary['physics_vs_cpu_reference_mismatch_frames']} mismatch frames"
    )
    print(
        "  physics vs observations  : "
        f"{summary['physics_vs_observation_mismatch_frames']} mismatch frames"
    )
    print(
        "  reward vs physics result  : "
        f"{summary['reward_vs_physics_mismatch_frames']} mismatch frames"
    )
    print(
        "  reward contributor drift  : "
        f"{summary['reward_contributor_outside_physics_components_frames']} "
        "frames"
    )
    print(
        "  renderer vs physics       : "
        f"{summary['renderer_vs_physics_mismatch_frames']} mismatch frames"
    )
    print(
        "  legacy wall-risk frames   : "
        f"{summary['reward_range_graph_has_wall_blocked_edges_frames']}"
    )
    print(
        "  reset consumers differ    : "
        f"{summary['reset_physics_uninitialized_mismatch']}"
    )
    if baseline_report is not None:
        print(
            "  deterministic behavior    : "
            f"{'DIFFERENT' if behavior_mismatch else 'MATCH'}"
        )
    if performance_result is not None:
        if performance_result["comparable"]:
            print(
                "  benchmark throughput      : "
                f"{performance_result['throughput_change_percent']:+.2f}% "
                "vs baseline"
            )
            if "peak_memory_change_mib" in performance_result:
                print(
                    "  benchmark peak memory     : "
                    f"{performance_result['peak_memory_change_mib']:+.1f} MiB "
                    "vs baseline"
                )
        else:
            print(
                "  benchmark throughput      : NOT COMPARABLE "
                f"{performance_result['configuration_differences']}"
            )
    elif options.benchmark:
        benchmark_message = (
            "recorded in baseline"
            if options.baseline == "record"
            else "current result only; comparison needs a baseline"
        )
        print(f"  benchmark throughput      : {benchmark_message}")
    print(f"  baseline                  : {baseline_result}")
    print(f"  report                    : {current_path}")
    if video_path:
        print(f"  video                     : {video_path}")

    if baseline_mismatch:
        raise SystemExit(2)
    if options.strict and not summary["all_consumers_consistent"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
