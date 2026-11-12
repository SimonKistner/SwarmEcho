"""Add twelve training hours to all nine originals and eight efficiency-25/50 runs.

Keep the exact existing W&B identities and live, manually corrected group names.
Inventory, preparation and submission are separate, inspectable operations.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import urllib.request
import uuid

from capacity import Det, collect
from fetch import Transfer
from patch_preemption import install
from plan import merge_overrides
from prepare_context import ROOT
from resume_g03_nov04 import checkpoint_clock, level_steps
from setup_wandb import local_key
from submit import (CLUSTER, ensure_unused_run_name, prepare_submission, read_json,
                    submission_run_name, submit, validate, write_json)


SOURCE_IDS = {55511, 55512, 55513, 55514, 55515, 55516, 55517, 55518, 55524,
              55489, 55492, 55496, 55498, 55503, 55519, 55521, 55522}
SELECTION = CLUSTER / ".submissions/g03_next12h_selection.json"
INVENTORY = CLUSTER / ".submissions/g03_next12h_inventory.json"
TERMINAL = {"STATE_COMPLETED", "STATE_ERROR", "STATE_CANCELED"}


def selection():
    rows = read_json(SELECTION)
    if (len(rows) != 17 or {row["source_id"] for row in rows} != SOURCE_IDS
            or sum(row["kind"] == "original" for row in rows) != 9
            or sum(row["kind"] == "efficiency" for row in rows) != 8
            or any("eff100" in row["name"] for row in rows)):
        raise ValueError("Expected all nine originals plus eight efficiency-25/50 runs, with no efficiency-100 runs.")
    if any(row["live_group"] != row["group"] for row in rows):
        raise ValueError("Live groups differ from the user-corrected group names; review selection.")
    for group in {row["group"] for row in rows}:
        members = [row for row in rows if row["group"] == group]
        expected = {456789} if "policy_only" in group else {456789, 456790}
        if {row["seed"] for row in members} != expected:
            raise ValueError("Missing or duplicated seed in group " + group)
    return rows


def client_for(settings):
    client = Det(settings)
    client.environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return client


def verify_live_groups(selected):
    fields = " ".join('r{}:run(name:{}){{name group}}'.format(index, json.dumps(row["wandb_id"]))
                      for index, row in enumerate(selected))
    query = 'query{project(name:"swarmecho",entityName:"ias_master_thesis_sk"){' + fields + '}}'
    authorization = "Basic " + base64.b64encode(("api:" + local_key()).encode()).decode()
    request = urllib.request.Request("https://api.wandb.ai/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": authorization, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get("errors"):
        raise ValueError("Could not verify live W&B groups.")
    project = result["data"]["project"]
    for index, row in enumerate(selected):
        actual = project["r" + str(index)]
        if actual is None or actual["name"] != row["wandb_id"] or actual["group"] != row["group"]:
            raise ValueError("Live W&B identity/group changed: " + row["name"])
    print("All seventeen live W&B groups match the corrected selection.", flush=True)


def source_records(settings, selected):
    records = {row["experiment"]["id"]: row["experiment"] for row in
               client_for(settings).json("experiment", "describe", ",".join(str(row["source_id"]) for row in selected))}
    for row in selected:
        actual = records[row["source_id"]]
        if (actual["name"] != row["name"] or actual["projectId"] != settings["project_id"]
                or actual["username"] != "ar44ruca" or actual["state"] not in TERMINAL):
            raise ValueError("Source identity/state is not ready for continuation: " + str(row["source_id"]))
    return records


def inventory(settings):
    selected = selection()
    verify_live_groups(selected)
    records = source_records(settings, selected)
    with Transfer(settings, service_file=CLUSTER / "tools/remote_branch_inventory.py") as connection:
        snapshot = connection.request({"persist_root": settings["persist_root"],
            "run_names": [row["name"] for row in selected], "include_prior_sessions": False})
    snapshot["observed_at"] = datetime.now(timezone.utc).isoformat()
    for parent, row in zip(snapshot["parents"], selected):
        if (parent["run_name"] != row["name"] or parent["wandb_id"] != row["wandb_id"]
                or not parent["optimizer_state_present"]):
            raise ValueError("Checkpoint optimizer or W&B identity mismatch: " + row["name"])
        parent["experiment_id"] = row["source_id"]
        parent["experiment_state"] = records[row["source_id"]]["state"]
        print("LATEST: {} {} ; final={}".format(row["source_id"], parent["checkpoint"], parent["final"]), flush=True)
    write_json(INVENTORY, snapshot)
    print("INVENTORY_READY=" + str(INVENTORY), flush=True)


def prepare(settings):
    selected = selection()
    snapshot = read_json(INVENTORY)
    parents = {row["experiment_id"]: row for row in snapshot["parents"]}
    if set(parents) != SOURCE_IDS:
        raise ValueError("Checkpoint inventory differs from the complete seventeen-run selection.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plans = []
    for row in selected:
        parent = parents[row["source_id"]]
        tracked = read_json(ROOT / "outputs" / row["name"] / "cluster_run.json")
        receipt = Path(tracked["submission_dir"]).resolve()
        if (CLUSTER / ".submissions").resolve() not in receipt.parents:
            raise ValueError("Unexpected frozen source location.")
        original = read_json(receipt / "summary.json")
        if (tracked["experiment_id"] != row["source_id"]
                or read_json(receipt / "result.json")["experiment_id"] != row["source_id"]
                or original["run"]["run_name"] != row["name"] or original["settings"] != settings):
            raise ValueError("Frozen source identity/destination mismatch: " + row["name"])
        values = dict(value.split("=", 1) for value in original["run"]["overrides"])
        if int(values["training.seed"]) != row["seed"]:
            raise ValueError("Training seed differs from selected run: " + row["name"])
        spent, clock = checkpoint_clock(parent)
        previous_allowance = float(values["training.max_training_seconds"])
        if not math.isfinite(spent) or spent < 0:
            raise ValueError("Invalid portable checkpoint training clock.")
        unfinished = max(0.0, previous_allowance - spent)
        allowance = unfinished + 43200.0
        own_steps = int(parent["history"]["history"][-1]["steps"])
        remaining_steps = level_steps(original, receipt / "model") - own_steps
        if remaining_steps <= 0:
            raise ValueError("Original lifetime step ceiling exhausted: " + row["name"])
        if row["kind"] == "original":
            name = row["name"][4:].split("_resume36h_", 1)[0]
            postfix = "resume48h_v1"
        else:
            name = row["name"][4:].split("_decom3000_", 1)[0]
            postfix = "decom3000_branch_v3_resume36h_v1"
        run = {key: value for key, value in original["run"].items() if key != "run_name"}
        run.update(run_postfix=postfix, wandb_group=row["live_group"],
            description="Add 12 training hours to experiment {}; preserve any unfinished prior allowance, existing curve and corrected group".format(row["source_id"]),
            overrides=merge_overrides(run["overrides"], [
                "training.checkpoint_path=" + parent["checkpoint"], "training.ckpt_loading_mode=resume",
                "training.max_training_seconds=" + str(allowance), "training.max_wall_seconds=null",
                "training.total_timesteps=" + str(remaining_steps), "training.map_pool_path=" + parent["pool"],
            ]))
        validate(settings, run)
        origin = {"source_experiment_id": row["source_id"], "source_run": row["name"],
            "source_receipt": str(receipt), "checkpoint": parent["checkpoint"],
            "completed_steps": parent["total_steps"], "wandb_id": row["wandb_id"],
            "wandb_group": row["live_group"], "added_training_seconds": 43200,
            "unfinished_training_seconds": unfinished, "remaining_training_seconds": allowance,
            "clock_provenance": clock, "checkpoint_ready_sha256": parent["ready_sha256"]}
        plans.append((name, run, original, receipt, origin))
    for name, run, _, _, _ in plans:
        ensure_unused_run_name(submission_run_name(name, run, stamp))
    entries = []
    for name, run, original, receipt, origin in plans:
        output, config, context = prepare_submission(settings, original["dependency_image"], name, run,
            stamp, False, source_context=(receipt / "model", original["source_files"], original["source_bytes"]))
        install(context[0])
        experiment = read_json(config)
        experiment["environment"]["environment_variables"].extend([
            "WANDB_RUN_ID=" + origin["wandb_id"], "WANDB_RESUME=must"])
        write_json(config, experiment)
        summary = read_json(output / "summary.json")
        summary.update(continuation=origin, lifecycle_patch="preemption-checkpoint-resume-v1")
        write_json(output / "summary.json", summary)
        entries.append({"submission_dir": str(output), "state": "PREPARED", "continuation": origin})
        print("PREPARED {}: {:.3f}h ; group {}".format(origin["source_experiment_id"],
            origin["remaining_training_seconds"] / 3600, origin["wandb_group"]), flush=True)
    path = CLUSTER / ".submissions" / ("batch-g03-promising-plus12h-" + stamp + "-" + uuid.uuid4().hex[:8] + ".json")
    write_json(path, {"sources_observed_at": snapshot["observed_at"], "experiments": entries})
    print("BATCH_READY=" + str(path), flush=True)


def submit_batch(settings, path, allow_queue=False):
    batch = read_json(path)
    selected = selection()
    if (len(batch["experiments"]) != 17 or any(row["state"] != "PREPARED" for row in batch["experiments"])
            or {row["continuation"]["source_experiment_id"] for row in batch["experiments"]} != SOURCE_IDS):
        raise ValueError("Expected exactly seventeen unsubmitted prepared jobs; inspect receipts before retrying.")
    kinds = {row["source_id"]: row["kind"] for row in selected}
    batch["experiments"].sort(key=lambda row:
        kinds[row["continuation"]["source_experiment_id"]] != "original")
    source_records(settings, selected)
    verify_live_groups(selected)
    occupancy = collect(settings, client_for(settings))
    print("LIVE CAPACITY: {} free slots; 17 requested.".format(occupancy["free_slots"]), flush=True)
    if occupancy["free_slots"] < 17 and not allow_queue:
        raise RuntimeError("Fewer than seventeen free slots; use --allow-queue to submit the excess jobs to the scheduler.")
    batch["queue_requested"] = allow_queue
    batch["free_slots_before_submission"] = occupancy["free_slots"]
    write_json(path, batch)
    if allow_queue:
        print("Queue submission authorized: all seventeen jobs will be submitted regardless of immediately free slots.", flush=True)
    print("Submission order: nine main runs first, then eight efficiency branches.", flush=True)
    for entry in batch["experiments"]:
        output = Path(entry["submission_dir"])
        entry["state"] = "SUBMITTING"
        write_json(path, batch)
        try:
            submit(settings, output / "experiment.yaml", output / "model", output)
        except BaseException:
            entry["state"] = "CHECK_RECEIPT"
            write_json(path, batch)
            raise
        entry["state"] = "SUBMITTED"
        entry["experiment_id"] = read_json(output / "result.json")["experiment_id"]
        write_json(path, batch)
    print("All seventeen authorized continuations submitted.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--submit-batch", type=Path)
    parser.add_argument("--allow-queue", action="store_true",
                        help="Submit the entire prepared batch even when fewer than seventeen slots are free.")
    args = parser.parse_args()
    if args.allow_queue and args.submit_batch is None:
        parser.error("--allow-queue requires --submit-batch")
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    settings = read_json(CLUSTER / "settings.json")
    if args.inventory:
        inventory(settings)
    elif args.prepare:
        prepare(settings)
    else:
        submit_batch(settings, args.submit_batch.resolve(), allow_queue=args.allow_queue)


if __name__ == "__main__":
    main()
