"""Checkpoint persistence and resume helpers shared by training and evaluation."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from swarmecho.core.terminal import terminal_print, display_path
from typing import Any

from flax import nnx


def validate_checkpoint_contract(model: Any, path: Path) -> None:
    """Reject incompatible value units/input meanings before restoring tensors."""
    if not hasattr(model, "checkpoint_contract"):
        return
    expected = dict(model.checkpoint_contract)
    source = path / "training_contract.json"
    if not source.exists():
        if expected.get("observe_visually_gated_target_displacement", False):
            raise ValueError("Visually gated target displacement requires matching checkpoint observation metadata.")
        if expected.get("adaptive_size", False):
            raise ValueError("Adaptive voting checkpoints require matching action/observation metadata.")
        if expected["value_normalization"] != "none":
            raise ValueError(
                "Checkpoint has no value-normalization metadata. It cannot be loaded as a "
                "running-normalized critic. Use the original raw-value settings or an explicitly "
                "converted checkpoint; changing the flag alone changes critic units."
            )
        if expected.get("radar_layout", "legacy_interleaved") != "legacy_interleaved":
            raise ValueError("Checkpoint has no radar metadata; its input layout cannot be verified. "
                             "Load it with the original legacy radar settings.")
        if expected.get("wall_sensing_layout", "fibonacci") != "fibonacci":
            raise ValueError("Checkpoint has no wall sensing metadata; load it with the original Fibonacci layout.")
        warnings.warn("Checkpoint has no training contract; input semantics cannot be verified. "
                      "Communication timing and inactive-agent handling now use corrected behavior.",
                      stacklevel=2)
        return
    actual = json.loads(source.read_text(encoding="utf-8"))
    if actual.get("critic_type", "observation") != "observation":
        raise ValueError("This checkpoint uses a critic architecture that is no longer supported.")
    if actual.get("format") != expected["format"]:
        raise ValueError("Unsupported checkpoint training-contract version.")
    if "wall_sensing_ray_max_distance" not in actual:
        # Old contracts omitted the wall range, which was visual_radius.
        # Recover that value from the frozen run config when available.
        from swarmecho.core.config import EnvConfig
        legacy_range = EnvConfig.visual_radius
        config_source = path.parent.parent / "config.yaml"
        if config_source.is_file():
            import yaml
            saved_env = (yaml.safe_load(config_source.read_text(encoding="utf-8")) or {}).get("env", {})
            legacy_range = saved_env.get("visual_radius", legacy_range)
        actual["wall_sensing_ray_max_distance"] = legacy_range
    for contract in (actual, expected):
        contract.setdefault("progress_based_cluster_tiebreaker_soft", False)
        contract.setdefault("adaptive_size", False)
        contract.setdefault("vote_mode", None)
        contract.setdefault("observe_seeing_target", False)
        contract.setdefault("observe_visually_gated_target_displacement", False)
        # Normalize both generations of renamed counts without changing rays.
        old_wall_rays = contract.pop("wall_radar_bins", None)
        if old_wall_rays is not None:
            contract.setdefault("wall_sensing_rays", old_wall_rays)
        old_bins = contract.pop("radar_bins", None)
        if old_bins is not None:
            contract.setdefault("wall_sensing_rays", old_bins)
            contract.setdefault("drone_radar_bins", old_bins)
        contract.setdefault("wall_sensing_layout", "fibonacci")
        contract.setdefault("radar_mode", "legacy")
        contract.setdefault("distance_bands", 1)
        contract.setdefault("radar_count_cap", 3)
        contract.setdefault(
            "radar_layout",
            # Missing layout metadata predates counts-only radar. Keep the
            # old inference so it cannot match the new band_counts_v2 layout.
            "legacy_interleaved" if contract["radar_mode"] == "legacy"
            and contract.get("wall_sensing_rays") == contract.get("drone_radar_bins")
            else "radar_blocks_v1",
        )
        if contract["radar_mode"] == "legacy":
            # These settings do not affect legacy observations.
            contract["distance_bands"] = 1
            contract["radar_count_cap"] = None
        else:
            contract["distance_bands"] = max(1, contract["distance_bands"])
    critical = {
        "adaptive_size", "vote_mode",
        "obs_dim", "hidden_dim", "num_layers", "actor_num_layers", "tarmac_sig_dim",
        "tarmac_val_dim", "memory_comm_enabled", "value_normalization",
        "wall_sensing_rays", "wall_sensing_layout", "wall_sensing_ray_max_distance",
        "drone_radar_bins", "radar_mode", "radar_layout",
        "distance_bands", "radar_count_cap",
        "visually_gated_target_displacement_scale_m",
        *[name for name in expected if name.startswith("observe_")],
    }
    critical.intersection_update(expected)
    mismatches = [f"{name}: checkpoint={actual.get(name)!r}, level={expected[name]!r}"
                  for name in sorted(critical) if actual.get(name) != expected[name]]
    if mismatches:
        raise ValueError("Incompatible checkpoint settings: " + "; ".join(mismatches))
    changed = [name for name in expected if name not in critical and actual.get(name) != expected[name]]
    if changed:
        warnings.warn("Checkpoint loaded with changed training behavior: " + ", ".join(changed),
                      stacklevel=2)


def restore_model_checkpoint(model: Any, checkpoint_path: str | Path) -> Path:
    """Restore model weights from one Orbax checkpoint directory."""
    import orbax.checkpoint as ocp

    path = Path(str(checkpoint_path).replace("\\", "/")).absolute()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    validate_checkpoint_contract(model, path)

    _, empty_state = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored_state = checkpointer.restore(
        str(path),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)
    return path


def save_checkpoint_history(
    checkpoint_path: str | Path,
    *,
    run_name: str,
    update: int,
    num_envs: int,
    num_steps: int,
    prior_history: list,
    total_steps: int | None = None,
) -> None:
    """Write cumulative training-step history next to an Orbax checkpoint."""
    path = Path(checkpoint_path)
    try:
        current_session_steps = (int(update) * int(num_envs) * int(num_steps)
                                 if total_steps is None else
                                 int(total_steps) - sum(int(item["steps"]) for item in prior_history))
        updated_history = list(prior_history)
        updated_history.append({"run_name": run_name, "steps": current_session_steps})
        total_steps = sum(item["steps"] for item in updated_history)
        (path / "step_history.json").write_text(
            json.dumps(
                {
                    "total_steps": total_steps,
                    "history": updated_history,
                    "update": int(update),
                    "num_envs": int(num_envs),
                    "num_steps": int(num_steps),
                },
                indent=2,
            )
        )
    except MemoryError:
        raise
    except Exception as exc:
        terminal_print(f"[checkpoint-history] Failed to save step_history.json to {display_path(path)}: {exc}")


def save_model_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    *,
    run_name: str,
    update: int,
    num_envs: int,
    num_steps: int,
    prior_history: list,
    total_steps: int | None = None,
) -> Path:
    """Replace and save one complete Orbax model checkpoint."""
    import orbax.checkpoint as ocp

    path = Path(checkpoint_path).absolute()
    if path.exists():
        shutil.rmtree(path)

    _, state_dict = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(str(path), args=ocp.args.StandardSave(state_dict))
    if hasattr(model, "checkpoint_contract"):
        (path / "training_contract.json").write_text(
            json.dumps(dict(model.checkpoint_contract), indent=2) + "\n", encoding="utf-8"
        )
    save_checkpoint_history(
        path,
        run_name=run_name,
        update=update,
        num_envs=num_envs,
        num_steps=num_steps,
        prior_history=prior_history,
        total_steps=total_steps,
    )
    return path


def save_training_checkpoint(model, path, *, adaptive_spawn=None, adaptive_maps=None, final=False,
                             metrics=None, run_dir=None, **kwargs):
    """Publish model, history and adaptive state together before marking ready."""
    path = Path(path).absolute()
    pending = path.with_name("." + path.name + "_pending")
    save_model_checkpoint(model, pending, **kwargs)
    if adaptive_spawn is not None:
        adaptive_spawn.save(pending)
    if adaptive_maps is not None:
        adaptive_maps.save(pending)
        if run_dir is not None:
            selection = Path(run_dir) / "adaptive_maps" / "selection.json"
            if selection.is_file():
                shutil.copy2(selection, pending / "adaptive_map_selection.json")
    if not (pending / "step_history.json").is_file():
        raise RuntimeError("Checkpoint step history was not saved; recovery checkpoint cannot be published.")
    (pending / "recovery_ready.json").write_text(json.dumps({
        "recovery_id": os.environ.get("SWARMECHO_RECOVERY_ID"),
        "run_name": kwargs["run_name"], "run_dir": str(run_dir) if run_dir else None,
        "update": kwargs["update"], "num_envs": kwargs["num_envs"],
        "final": bool(final), "metrics": metrics or {},
    }, indent=2) + "\n", encoding="utf-8")
    # No GPU work follows the readiness marker. Keep the previous published
    # checkpoint intact until all device copies and controller saves succeed.
    if path.exists():
        shutil.rmtree(path)
    pending.rename(path)
    return path

