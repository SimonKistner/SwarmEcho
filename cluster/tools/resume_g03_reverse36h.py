"""Extend completed recovery 55490 by twelve hours, matching the other originals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from capacity import Det, collect
from fetch import Transfer
from patch_preemption import install
from plan import merge_overrides
from prepare_context import ROOT
from resume_g03_nov04 import level_steps
from submit import (CLUSTER, ensure_unused_run_name, prepare_submission, read_json,
                    submission_run_name, submit, validate, write_json)


SOURCE_ID = 55490
SOURCE_NAME = "CLU_G03_reverse_aux_l0p3_seed456790_resume24h_recovery1"
NAME = "G03_reverse_aux_l0p3_seed456790"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    settings = read_json(CLUSTER / "settings.json")
    tracked = read_json(ROOT / "outputs" / SOURCE_NAME / "cluster_run.json")
    receipt = Path(tracked["submission_dir"]).resolve()
    if (CLUSTER / ".submissions").resolve() not in receipt.parents:
        raise ValueError("Unexpected source receipt location.")
    original = read_json(receipt / "summary.json")
    if (tracked["experiment_id"] != SOURCE_ID or original["run"]["run_name"] != SOURCE_NAME
            or read_json(receipt / "result.json")["experiment_id"] != SOURCE_ID
            or original["settings"] != settings):
        raise ValueError("Source identity/settings changed.")
    run = {key: value for key, value in original["run"].items() if key != "run_name"}
    run.update(run_postfix="resume36h_v1",
               description="Another 12h continuation of completed recovery 55490, matching the 36h original-run target")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ensure_unused_run_name(submission_run_name(NAME, run, stamp))
    client = Det(settings)
    client.environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    source = client.json("experiment", "describe", str(SOURCE_ID))[0]["experiment"]
    if (source["name"] != SOURCE_NAME or source["state"] != "STATE_COMPLETED"
            or source["projectId"] != settings["project_id"] or source["username"] != "ar44ruca"):
        raise ValueError("Expected completed source experiment 55490.")
    with Transfer(settings, service_file=CLUSTER / "tools/remote_branch_inventory.py") as connection:
        inventory = connection.request({"persist_root": settings["persist_root"],
            "run_names": [SOURCE_NAME], "include_prior_sessions": False})
    parent = inventory["parents"][0]
    expected_id = original["continuation"]["wandb_id"]
    if (parent["run_name"] != SOURCE_NAME or not parent["final"]
            or not parent["optimizer_state_present"] or parent["wandb_id"] != expected_id
            or parent["curriculum"]["config"] != {"mode": "reverse", "success_threshold": 0.75}):
        raise ValueError("Final checkpoint, optimizer, curriculum or W&B identity is inconsistent.")
    remaining_steps = (level_steps(original, receipt / "model")
                       - int(parent["history"]["history"][-1]["steps"]))
    if remaining_steps <= 0:
        raise ValueError("Original lifetime step allowance exhausted.")
    run["overrides"] = merge_overrides(run["overrides"], [
        "training.checkpoint_path=" + parent["checkpoint"],
        "training.ckpt_loading_mode=resume", "training.max_training_seconds=43200",
        "training.max_wall_seconds=null", "training.total_timesteps=" + str(remaining_steps),
        "training.map_pool_path=" + parent["pool"],
    ])
    validate(settings, run)
    output, config, context = prepare_submission(settings, original["dependency_image"], NAME,
        run, stamp, False, source_context=(receipt / "model", original["source_files"], original["source_bytes"]))
    install(context[0])
    experiment = read_json(config)
    experiment["environment"]["environment_variables"].extend([
        "WANDB_RUN_ID=" + expected_id, "WANDB_RESUME=must"])
    write_json(config, experiment)
    summary = read_json(output / "summary.json")
    summary.update(continuation={"source_experiment_id": SOURCE_ID, "source_run": SOURCE_NAME,
        "source_receipt": str(receipt), "checkpoint": parent["checkpoint"],
        "completed_steps": parent["total_steps"], "wandb_id": expected_id,
        "remaining_training_seconds": 43200, "checkpoint_ready_sha256": parent["ready_sha256"]},
        lifecycle_patch="preemption-checkpoint-resume-v1")
    write_json(output / "summary.json", summary)
    write_json(output / "source_inventory.json", inventory)
    occupancy = collect(settings, client)
    print("LIVE CAPACITY: {} free slots; one requested.".format(occupancy["free_slots"]), flush=True)
    if occupancy["free_slots"] < 1:
        raise RuntimeError("No free slot; continuation prepared but not submitted.")
    print("FINAL CHECKPOINT: " + parent["checkpoint"], flush=True)
    print("W&B GROUP: " + run["wandb_group"] + "; existing run " + expected_id, flush=True)
    submit(settings, config, context[0], output)


if __name__ == "__main__":
    main()
