"""Run the uploaded training configuration inside a Determined container."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

import determined as det


def latest_checkpoint(directory, run_name):
    candidates = []
    for checkpoint in (directory / "checkpoints").glob("ckpt_*"):
        try:
            if checkpoint.is_symlink():
                continue
            ready = json.loads((checkpoint / "recovery_ready.json").read_text(encoding="utf-8"))
            history = json.loads((checkpoint / "step_history.json").read_text(encoding="utf-8"))
            update = int(ready["update"])
            if (ready["run_name"] == run_name and int(history["update"]) == update
                    and checkpoint.name == "ckpt_{:06d}".format(update)
                    and (checkpoint / "optimizer_state").is_dir()):
                candidates.append((update, checkpoint, ready))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return max(candidates, key=lambda row: row[0]) if candidates else None


def resume_allocation(run, root, child_env, train_python):
    """Restart this output's own checkpoint with its remaining job allowance."""
    directory = root / "outputs" / run["run_name"]
    found = latest_checkpoint(directory, run["run_name"])
    if found is None:
        # Never silently rebranch into an existing validation history.
        history = directory / "validation/history.jsonl"
        if history.is_file() and history.stat().st_size:
            raise RuntimeError("Existing validation history has no complete resumable checkpoint: " + str(directory))
        child_env["SWARMECHO_CLUSTER_PRIOR_TRAINING_SECONDS"] = "0"
        return run, False
    _, checkpoint, ready = found
    metrics = ready.get("metrics", {})
    spent = float(metrics.get("cluster_training_seconds", metrics.get("training_seconds", -1)))
    if not math.isfinite(spent) or spent < 0:
        raise RuntimeError("Checkpoint lacks a valid portable training clock: " + str(checkpoint))
    if ready.get("final"):
        print("[CLUSTER RESUME] Final training checkpoint already published: " + str(checkpoint), flush=True)
        return run, True
    overrides = dict(value.split("=", 1) for value in run["overrides"])
    allowance = overrides.get("training.max_training_seconds")
    if allowance is None:
        # Resolve the original frozen level, since a previous restart may have
        # saved a config containing only that allocation's remaining allowance.
        command = [str(train_python), "-c", (
            "import json, sys\n"
            "from swarmecho.core.config import load_level\n"
            "level = load_level(sys.argv[1], sys.argv[2:])\n"
            "print(json.dumps(level.training.max_training_seconds))\n"
        ), run["level"], *run["overrides"]]
        allowance = json.loads(subprocess.check_output(command, env=child_env, text=True))
    if allowance is None or allowance == "null":
        raise RuntimeError("Allocation recovery requires training.max_training_seconds to be set.")
    remaining = float(allowance) - spent
    if remaining <= 0:
        print("[CLUSTER RESUME] Training allowance exhausted at complete checkpoint.", flush=True)
        return run, True
    overrides.update({"training.checkpoint_path": str(checkpoint), "training.ckpt_loading_mode": "resume",
                      "training.max_training_seconds": str(remaining)})
    child_env["SWARMECHO_CLUSTER_PRIOR_TRAINING_SECONDS"] = str(spent)
    print("[CLUSTER RESUME] {} ; {:.1f}s consumed, {:.1f}s remaining.".format(
        checkpoint, spent, remaining), flush=True)
    return dict(run, overrides=[key + "=" + value for key, value in overrides.items()]), False


def run_preemptible(core, command, child_env, root, run):
    request = root / "outputs" / run["run_name"] / ".cluster/preempt_request.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    if request.exists():
        request.unlink()
    child_env["SWARMECHO_PREEMPT_REQUEST"] = str(request)
    process = subprocess.Popen(command, env=child_env)
    pending = False
    try:
        while True:
            try:
                code = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if not pending and core.preempt.should_preempt(auto_ack=False):
                    request.write_text("{}\n", encoding="utf-8")
                    pending = True
                    print("[CLUSTER PREEMPT] Checkpoint requested at the next complete update.", flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)
        if pending:
            if latest_checkpoint(root / "outputs" / run["run_name"], run["run_name"]) is None:
                raise RuntimeError("Preempted worker exited without publishing a resumable checkpoint.")
            core.preempt.acknowledge_preemption_signal()
            print("[CLUSTER PREEMPT] Checkpoint saved; preemption acknowledged.", flush=True)
            return True
        return False
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def main():
    run = json.loads((Path(__file__).resolve().parents[1] / "run.json").read_text(encoding="utf-8"))
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
        key_error = None
        if not key_file.is_file():
            key_error = "W&B key missing: " + str(key_file)
        elif stat.S_IMODE(key_file.stat().st_mode) & 0o077:
            key_error = "W&B key file must be readable only by its owner: " + str(key_file)
        else:
            child_env["WANDB_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
            if not child_env["WANDB_API_KEY"]:
                key_error = "W&B key file is empty."
        if key_error:
            if "tuning" not in run:
                raise RuntimeError(key_error)
            print("[TUNING WARNING] " + key_error + "; continuing with local JSON curves only.", flush=True)
            run = dict(run, wandb_mode="disabled")
    curriculum = "levels" in run
    completed = False
    if not curriculum and "tuning" not in run:
        run, completed = resume_allocation(run, root, child_env, train_python)
    module = "swarmecho.training.curriculum" if curriculum else "swarmecho.training.launcher"
    selection = "levels=" + ",".join(run["levels"]) if curriculum else "level=" + run["level"]
    command = [str(train_python), "-m", module, selection,
               "logging.log_dir=" + str(root / "outputs"),
               "logging.run_name=" + run["run_name"], "logging.wandb_mode=" + run["wandb_mode"],
               "logging.wandb_group=" + run.get("wandb_group", run["run_name"]), "logging.use_timestamp_postfix=false",
               *run["overrides"]]
    print("Starting {}: {} under {}".format(run["run_name"], selection, root), flush=True)
    print("Training arguments: " + json.dumps(command[3:]), flush=True)
    with det.core.init() as core:
        for operation in core.searcher.operations():
            if "tuning" in run:
                sys.path.insert(0, str(source))
                from swarmecho.training.cluster_tuning import run as run_tuning
                progress_step = 0
                def report(metrics):
                    nonlocal progress_step
                    progress_step += 1
                    core.train.report_training_metrics(steps_completed=progress_step, metrics=metrics)
                run_tuning(run, root, train_python, child_env, report)
            else:
                if curriculum:
                    subprocess.run(command, env=child_env, check=True)
                elif not completed and run_preemptible(core, command, child_env, root, run):
                    return
            core.train.report_validation_metrics(steps_completed=operation.length, metrics={"job_completed": 1.0})
            operation.report_completed(1.0)


if __name__ == "__main__":
    main()
