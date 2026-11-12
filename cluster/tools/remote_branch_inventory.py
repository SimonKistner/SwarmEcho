"""Read-only live metadata for published checkpoints; no map/model downloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect(request):
    root = Path(request["persist_root"]).resolve(strict=True)
    result = []
    for name in request["run_names"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid source run name")
        folder = root / "outputs" / name
        if folder.is_symlink() or root not in folder.resolve(strict=True).parents:
            raise ValueError("Source run is outside the project root")
        candidates = []
        for checkpoint in (folder / "checkpoints").glob("ckpt_*"):
            marker = checkpoint / "recovery_ready.json"
            history = checkpoint / "step_history.json"
            if checkpoint.is_symlink() or not marker.is_file() or not history.is_file():
                continue
            try:
                ready, steps = read(marker), read(history)
                update = int(ready["update"])
                if checkpoint.name != "ckpt_{:06d}".format(update) or ready["run_name"] != name:
                    continue
                if int(steps["update"]) != update:
                    continue
                candidates.append((update, checkpoint, ready, steps))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if not candidates:
            raise ValueError("No complete numbered checkpoint for " + name)
        update, checkpoint, ready, history = max(candidates, key=lambda row: row[0])
        selection = read(checkpoint / "adaptive_map_selection.json")
        curriculum_path = checkpoint / "target_curriculum_state.json"
        curriculum = read(curriculum_path) if curriculum_path.is_file() else None
        contract = read(checkpoint / "training_contract.json")
        prior_sessions = []
        cumulative_steps = 0
        for item in (history["history"][:-1] if request.get("include_prior_sessions", True) else []):
            ancestor_name = item["run_name"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", ancestor_name):
                raise ValueError("Invalid ancestor run name")
            cumulative_steps += int(item["steps"])
            ancestor = root / "outputs" / ancestor_name
            timing = None
            for marker in (ancestor / "checkpoints").glob("ckpt_*/recovery_ready.json"):
                saved = read(marker)
                saved_history = read(marker.parent / "step_history.json")
                if saved_history["total_steps"] == cumulative_steps and saved.get("final"):
                    timing = saved.get("metrics", {}).get("training_seconds")
                    if timing is not None:
                        break
            if timing is None:
                raise ValueError("Missing final training-clock metadata for ancestor " + ancestor_name)
            prior_sessions.append({"run_name": ancestor_name, "training_seconds": timing,
                                   "total_steps": cumulative_steps})
        wandb_runs = sorted((p for p in (folder / "wandb").glob("run-*") if p.is_dir()),
                            key=lambda p: p.stat().st_mtime)
        if not wandb_runs:
            raise ValueError("No W&B run identity for " + name)
        result.append({
            "run_name": name, "checkpoint": checkpoint.as_posix(), "update": update,
            "total_steps": history["total_steps"], "history": history,
            "curriculum": curriculum, "pool": selection["pool"], "contract": contract,
            "optimizer_state_present": (checkpoint / "optimizer_state").is_dir(),
            "ready_metrics": ready.get("metrics", {}), "final": bool(ready.get("final")),
            "prior_sessions": prior_sessions,
            "wandb_id": wandb_runs[-1].name.rsplit("-", 1)[-1],
            "checkpoint_published_at": (checkpoint / "recovery_ready.json").stat().st_mtime,
            "ready_sha256": hashlib.sha256((checkpoint / "recovery_ready.json").read_bytes()).hexdigest(),
            "config_yaml": (folder / "config.yaml").read_text(encoding="utf-8"),
        })
    return {"ok": True, "parents": result}


def main():
    for line in sys.stdin.buffer:
        try:
            response = inspect(json.loads(line))
        except Exception as error:
            response = {"ok": False, "error": str(error)}
        payload = json.dumps(response).encode("utf-8")
        sys.stdout.buffer.write(struct.pack("!Q", len(payload)) + payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
