"""Versioned, renderer-independent 3D replay artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from swarmecho.env.baseline3d import Baseline3DState


REPLAY_FORMAT = "swarmecho-replay/v1"


def write_replay(
    output: str | Path,
    states: Iterable[Baseline3DState],
    *,
    map_name: str,
    dt: float,
    reward_terms: np.ndarray | None = None,
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Atomically write one replay NPZ and its small JSON manifest.

    The caller launches no server and owns no renderer process. A separately
    started inspector can poll for the JSON manifest, which is renamed into
    place only after the corresponding data file is complete.
    """
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_list = list(states)
    if not state_list:
        raise ValueError("A replay requires at least one state.")

    def stack(name: str):
        return np.stack([np.asarray(getattr(state, name)) for state in state_list])

    arrays = {
        "position": stack("pos"),
        "velocity": stack("vel"),
        "active": stack("active"),
        "target_position": stack("target_pos"),
        "base_position": stack("base_pos"),
        "directly_sees_target": stack("directly_sees_target"),
        "connected_to_base": stack("is_conn_base"),
        "connected_to_target": stack("is_conn_target"),
        "target_known": stack("target_known"),
        "success": stack("success"),
        "fully_connected": stack("fully_connected"),
        "chain_held_steps": stack("chain_held_steps"),
        "done": stack("done"),
        "collided": stack("collided"),
        "coverage_credit": stack("coverage_credit"),
        "coverage": stack("coverage"),
        "step": stack("step"),
    }
    if reward_terms is not None:
        rewards = np.asarray(reward_terms)
        if rewards.shape[0] != len(state_list):
            raise ValueError("reward_terms first dimension must match replay frames.")
        arrays["reward_terms"] = rewards

    data_path = destination.with_suffix(".npz")
    manifest_path = destination.with_suffix(".json")
    data_temporary = data_path.with_suffix(".npz.tmp")
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    with data_temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    data_temporary.replace(data_path)

    manifest = {
        "format": REPLAY_FORMAT,
        "data_file": data_path.name,
        "map_name": map_name,
        "dt": float(dt),
        "frames": len(state_list),
        "agents": int(arrays["position"].shape[1]),
        "coverage_shape": list(arrays["coverage"].shape[1:]),
        "fields": sorted(arrays),
    }
    if metadata:
        protected = {"format", "data_file", "frames", "agents", "fields"}
        if protected & set(metadata):
            raise ValueError(f"Replay metadata cannot replace protected keys: {sorted(protected & set(metadata))}")
        manifest.update(metadata)
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_temporary.replace(manifest_path)
    return data_path, manifest_path


def load_replay(manifest: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read and validate the format marker of one completed replay."""
    manifest_path = Path(manifest)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("format") != REPLAY_FORMAT:
        raise ValueError(f"Unsupported replay format: {metadata.get('format')!r}")
    data_path = manifest_path.parent / metadata["data_file"]
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if arrays["position"].shape[0] != metadata["frames"]:
        raise ValueError("Replay frame count does not match its manifest.")
    return metadata, arrays
