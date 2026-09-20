"""Checkpoint persistence and resume helpers shared by training and evaluation."""

from __future__ import annotations

import json
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
        if expected.get("critic_type", "observation") == "privileged":
            raise ValueError("Privileged 3D critics require checkpoint input-layout metadata; "
                             "an observation critic checkpoint cannot be restored into this architecture.")
        if expected["value_normalization"] != "none":
            raise ValueError(
                "Checkpoint has no value-normalization metadata. It cannot be loaded as a "
                "running-normalized critic. Use the original raw-value settings or an explicitly "
                "converted checkpoint; changing the flag alone changes critic units."
            )
        warnings.warn("Checkpoint has no training contract; input semantics cannot be verified. "
                      "Communication timing and inactive-agent handling now use corrected 3D behavior.",
                      stacklevel=2)
        return
    actual = json.loads(source.read_text(encoding="utf-8"))
    if actual.get("critic_type", "observation") != expected.get("critic_type", "observation"):
        raise ValueError("Incompatible checkpoint critic_type: observation and privileged "
                         "3D critics have different input architectures.")
    if actual.get("format") != expected["format"]:
        raise ValueError("Unsupported checkpoint training-contract version.")
    critical = {
        "obs_dim", "hidden_dim", "num_layers", "actor_num_layers", "tarmac_sig_dim",
        "tarmac_val_dim", "memory_comm_enabled", "value_normalization", "radar_bins",
        "critic_type", "privileged_layout", "critic_input_dim",
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
) -> None:
    """Write cumulative training-step history next to an Orbax checkpoint."""
    path = Path(checkpoint_path)
    try:
        current_session_steps = int(update) * int(num_envs) * int(num_steps)
        updated_history = list(prior_history)
        updated_history.append({"run_name": run_name, "steps": current_session_steps})
        total_steps = sum(item["steps"] for item in updated_history)
        (path / "step_history.json").write_text(
            json.dumps(
                {
                    "total_steps": total_steps,
                    "history": updated_history,
                },
                indent=2,
            )
        )
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
    )
    return path

