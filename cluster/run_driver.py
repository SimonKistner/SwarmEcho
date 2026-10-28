"""Run the uploaded training configuration inside a Determined container."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess

import determined as det


def main():
    run = json.loads((Path(__file__).parent / "run.json").read_text(encoding="utf-8"))
    root = Path(os.environ.get("SWARMECHO_PERSIST_ROOT", ""))
    if not root.is_absolute() or not root.is_dir() or not os.access(root, os.W_OK):
        raise RuntimeError("Persistent root is missing or not writable: " + str(root))
    train_python = Path(os.environ.get("SWARMECHO_TRAIN_PYTHON", ""))
    if not train_python.is_file():
        raise RuntimeError("Training Python is unavailable: " + str(train_python))
    source = Path.cwd() / "src"
    if not (source / "swarmecho/training/launcher.py").is_file():
        raise RuntimeError("SwarmEcho source is missing from the Determined context.")
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(source) + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env["SWARMECHO_POOL_ROOT"] = str(root / "map_pools")
    if run["wandb_mode"] == "online" and not child_env.get("WANDB_API_KEY"):
        key_file = root / ".wandb_api_key"
        if not key_file.is_file():
            raise RuntimeError("W&B key missing: " + str(key_file))
        if stat.S_IMODE(key_file.stat().st_mode) & 0o077:
            raise RuntimeError("W&B key file must be readable only by its owner: " + str(key_file))
        child_env["WANDB_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
        if not child_env["WANDB_API_KEY"]:
            raise RuntimeError("W&B key file is empty.")
    command = [str(train_python), "-m", "swarmecho.training.launcher", "level=" + run["level"],
               "logging.log_dir=" + str(root / "outputs"),
               "logging.run_name=" + run["run_name"], "logging.wandb_mode=" + run["wandb_mode"],
               *run["overrides"]]
    print("Starting {}: {} under {}".format(run["run_name"], run["level"], root), flush=True)
    print("Training arguments: " + json.dumps(command[3:]), flush=True)
    with det.core.init() as core:
        for operation in core.searcher.operations():
            subprocess.run(command, env=child_env, check=True)
            core.train.report_validation_metrics(steps_completed=operation.length, metrics={"job_completed": 1.0})
            operation.report_completed(1.0)


if __name__ == "__main__":
    main()
