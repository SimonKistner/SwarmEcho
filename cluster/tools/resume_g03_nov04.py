"""Prepare or submit the authorized eight +12h continuations and five recoveries."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.request
import uuid

from capacity import Det, collect
from patch_preemption import install
from plan import merge_overrides
from prepare_context import ROOT
from setup_wandb import local_key
from submit import (CLUSTER, ensure_unused_run_name, prepare_submission, read_json,
                    submission_run_name, submit, validate, write_json)


SOURCE_IDS = list(range(55438, 55446)) + list(range(55504, 55509))
INVENTORY = CLUSTER / ".submissions/g03_nov04_resume_sources.json"


def checkpoint_clock(parent):
    metrics = parent.get("ready_metrics", {})
    embedded = metrics.get("cluster_training_seconds", metrics.get("training_seconds"))
    if embedded is not None:
        return float(embedded), {"source": "checkpoint", "seconds": float(embedded)}
    query = ('query($name:String!){project(name:"swarmecho",entityName:"ias_master_thesis_sk")'
             '{run(name:$name){history(samples:100000)}}}')
    authorization = "Basic " + base64.b64encode(("api:" + local_key()).encode()).decode()
    request = urllib.request.Request("https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": {"name": parent["wandb_id"]}}).encode(),
        headers={"Authorization": authorization, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        result = json.load(response)
    if result.get("errors"):
        raise ValueError("W&B clock query failed.")
    rows = [json.loads(row) if isinstance(row, str) else row
            for row in result["data"]["project"]["run"]["history"]]
    # A validation row can merge a late timestamp with an earlier training
    # clock. Use the previous ordinary update to account for that work fully.
    rows = [row for row in rows if row.get("perf/training_seconds") is not None
            and row.get("perf/global_step") is not None
            and row["perf/global_step"] < parent["total_steps"]
            and row.get("validation/success_rate") is None]
    if not rows:
        raise ValueError("No portable clock preceding checkpoint " + parent["checkpoint"])
    row = max(rows, key=lambda item: item["perf/global_step"])
    work = parent["checkpoint_published_at"] - row["_timestamp"]
    if not 0 <= work <= 600:
        raise ValueError("Checkpoint and W&B clock do not align: " + parent["checkpoint"])
    elapsed = float(row["perf/training_seconds"]) + work
    return elapsed, {"source": "preceding W&B update + checkpoint publication", "row": {
        key: row[key] for key in ("perf/global_step", "perf/training_seconds", "_timestamp")},
        "publication_work_seconds": work, "seconds": elapsed}


def level_steps(original, source):
    overrides = dict(item.split("=", 1) for item in original["run"]["overrides"])
    if "training.total_timesteps" in overrides:
        return int(overrides["training.total_timesteps"])
    level = original["run"]["level"]
    level_path = next((source / "src/swarmecho/curriculum_config/levels").rglob(level + ".yaml"))
    match = re.search(r"(?m)^  total_timesteps:\s*([0-9_]+)", level_path.read_text(encoding="utf-8-sig"))
    if not match:
        raise ValueError("Frozen lifetime step allowance is missing.")
    return int(match.group(1).replace("_", ""))


def prepare(settings):
    snapshot = read_json(INVENTORY)
    indexed = {row["experiment_id"]: row for row in snapshot["parents"]}
    if set(indexed) != set(SOURCE_IDS):
        raise ValueError("Inventory differs from the thirteen authorized sources.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plans = []
    for identifier in SOURCE_IDS:
        parent = indexed[identifier]
        tracked = read_json(ROOT / "outputs" / parent["run_name"] / "cluster_run.json")
        receipt = Path(tracked["submission_dir"]).resolve()
        if (CLUSTER / ".submissions").resolve() not in receipt.parents:
            raise ValueError("Unexpected source receipt location.")
        original = read_json(receipt / "summary.json")
        result = read_json(receipt / "result.json")
        if (tracked["experiment_id"] != identifier or result["experiment_id"] != identifier
                or original["run"]["run_name"] != parent["run_name"]
                or original["settings"] != settings or not parent["optimizer_state_present"]):
            raise ValueError("Source identity, optimizer or settings mismatch: " + str(identifier))
        run = {key: value for key, value in original["run"].items() if key != "run_name"}
        values = dict(item.split("=", 1) for item in run["overrides"])
        if identifier < 55500:
            if parent["experiment_state"] != "STATE_COMPLETED" or not parent["final"]:
                raise ValueError("Expected final completed checkpoint: " + str(identifier))
            allowance, consumed, clock = 43200.0, None, None
            name = parent["run_name"][4:-len("_resume12h")]
            postfix = "resume36h_v1"
            purpose = "Another 12h continuation of experiment {} from its final checkpoint".format(identifier)
        else:
            if parent["experiment_state"] != "STATE_ERROR":
                raise ValueError("Expected failed shared-parent source: " + str(identifier))
            consumed, clock = checkpoint_clock(parent)
            allowance = float(values["training.max_training_seconds"]) - consumed
            name = parent["run_name"][4:].split("_decom3000_", 1)[0]
            postfix = "decom3000_branch_v3_shared789_recovery1"
            purpose = "Checkpoint recovery of preempted experiment {}; remaining allowance to 24 cumulative hours".format(identifier)
        if not math.isfinite(allowance) or allowance <= 0:
            raise ValueError("Training allowance exhausted or invalid: " + str(identifier))
        own_steps = int(parent["history"]["history"][-1]["steps"])
        remaining_steps = level_steps(original, receipt / "model") - own_steps
        if remaining_steps <= 0:
            raise ValueError("Lifetime step allowance exhausted: " + str(identifier))
        run.update(run_postfix=postfix, description=purpose,
            overrides=merge_overrides(run["overrides"], [
                "training.checkpoint_path=" + parent["checkpoint"],
                "training.ckpt_loading_mode=resume", "training.max_wall_seconds=null",
                "training.max_training_seconds=" + str(allowance),
                "training.total_timesteps=" + str(remaining_steps),
                "training.map_pool_path=" + parent["pool"],
            ]))
        validate(settings, run)
        origin = {"source_experiment_id": identifier, "source_run": parent["run_name"],
            "source_receipt": str(receipt), "checkpoint": parent["checkpoint"],
            "completed_steps": parent["total_steps"], "wandb_id": parent["wandb_id"],
            "remaining_training_seconds": allowance, "recovered_session_seconds": consumed,
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
        print("PREPARED: source {} -> {}; {:.3f}h remaining; W&B group {}".format(
            origin["source_experiment_id"], run["run_postfix"], origin["remaining_training_seconds"] / 3600,
            run["wandb_group"]), flush=True)
    path = CLUSTER / ".submissions" / ("batch-g03-nov04-" + stamp + "-" + uuid.uuid4().hex[:8] + ".json")
    write_json(path, {"sources_observed_at": snapshot["observed_at"], "experiments": entries})
    print("BATCH_READY=" + str(path), flush=True)


def submit_batch(settings, path):
    batch = read_json(path)
    if len(batch["experiments"]) != 13 or any(row["state"] != "PREPARED" for row in batch["experiments"]):
        raise ValueError("Expected thirteen unsubmitted prepared jobs; inspect receipts before retrying.")
    client = Det(settings)
    client.environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    records = {row["experiment"]["id"]: row["experiment"] for row in
               client.json("experiment", "describe", ",".join(map(str, SOURCE_IDS)))}
    for entry in batch["experiments"]:
        origin = entry["continuation"]
        source = records[origin["source_experiment_id"]]
        if (source["name"] != origin["source_run"] or source["projectId"] != settings["project_id"]
                or source["username"] != "ar44ruca"
                or source["state"] not in {"STATE_COMPLETED", "STATE_ERROR"}):
            raise ValueError("Source experiment identity/state changed.")
    occupancy = collect(settings, client)
    print("LIVE CAPACITY: {} free slots; 13 requested.".format(occupancy["free_slots"]), flush=True)
    if occupancy["free_slots"] < 13:
        raise RuntimeError("Fewer than thirteen free slots; prepared jobs remain unsubmitted.")
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
    print("All thirteen authorized jobs submitted.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--prepare", action="store_true")
    selection.add_argument("--submit-batch", type=Path)
    args = parser.parse_args()
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    settings = read_json(CLUSTER / "settings.json")
    if args.prepare:
        prepare(settings)
    else:
        submit_batch(settings, args.submit_batch.resolve())


if __name__ == "__main__":
    main()
