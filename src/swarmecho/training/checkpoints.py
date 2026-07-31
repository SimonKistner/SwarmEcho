"""Checkpoint persistence and resume helpers shared by training and evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from flax import nnx


def restore_model_checkpoint(model: Any, checkpoint_path: str | Path) -> Path:
    """Restore model weights from one Orbax checkpoint directory."""
    import orbax.checkpoint as ocp

    path = Path(str(checkpoint_path).replace("\\", "/")).absolute()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

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
        print(f"  [checkpoint-history] Failed to save step_history.json to {path}: {exc}")


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
    save_checkpoint_history(
        path,
        run_name=run_name,
        update=update,
        num_envs=num_envs,
        num_steps=num_steps,
        prior_history=prior_history,
    )
    return path

