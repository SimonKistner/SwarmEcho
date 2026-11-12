"""Replace only the twelve identified fresh G03 jobs with matched checkpoint branches.

Read live published PFSS metadata, prepare all replacements, stop experiments
55450..55461, then submit the twelve prepared branches. Parents stay running.
--dry-run prepares without stopping/submitting; --offline additionally uses
the cached parent inventory and performs no network calls. System Python only.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import re
import json
import math
import sys
import time
import uuid
import urllib.request

from capacity import Det
from fetch import Transfer
from plan import check_capacity, expand_config, merge_overrides
from prepare_context import ROOT
from submit import CLUSTER, image_info, prepare_submission, read_json, submit, validate, write_json
from setup_wandb import local_key


PARENTS = (("forward", 456789, 55442), ("forward", 456790, 55443),
           ("reverse", 456789, 55445), ("reverse", 456790, 55446))
CACHE = CLUSTER / ".submissions/g03_efficiency_branch_parents.json"
TERMINAL = {"STATE_COMPLETED", "STATE_CANCELED", "STATE_ERROR", "STATE_DELETED"}


def add_training_budgets(snapshot):
    """Use the inherited final-session clocks and this checkpoint's live clock.

    Legacy checkpoints lack a cumulative clock. W&B records the current session
    clock at every update; readiness mtime adds the checkpoint/validation work
    between that log row and publication. Persist both inputs for review.
    """
    token = base64.b64encode(("api:" + local_key()).encode()).decode()
    query = ('query($name:String!){project(name:"swarmecho",entityName:"ias_master_thesis_sk")'
             '{run(name:$name){name history(samples:100000)}}}')
    for parent in snapshot["parents"]:
        payload = json.dumps({"query": query, "variables": {"name": parent["wandb_id"]}}).encode()
        request = urllib.request.Request("https://api.wandb.ai/graphql", data=payload,
            headers={"Authorization": "Basic " + token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        if data.get("errors"):
            raise ValueError("Could not read the parent W&B clock history.")
        history = data["data"]["project"]["run"]["history"]
        history = [json.loads(row) if isinstance(row, str) else row for row in history]
        rows = [row for row in history if row.get("perf/global_step") == parent["total_steps"]
                and row.get("perf/ppo_updates") == parent["update"] and "perf/training_seconds" in row]
        if not rows:
            raise ValueError("Checkpoint clock not yet available in W&B; retry preparation before stopping jobs.")
        row = rows[-1]
        publication_work = parent["checkpoint_published_at"] - row["_timestamp"]
        if not -5 <= publication_work <= 600:
            raise ValueError("Checkpoint/W&B timestamps do not align; review the clock before replacement.")
        current = float(row["perf/training_seconds"]) + max(0.0, publication_work)
        prior = sum(float(item["training_seconds"]) for item in parent["prior_sessions"])
        cumulative = prior + current
        if not math.isfinite(cumulative) or not 0 < cumulative < 86400:
            raise ValueError("Parent clock is invalid or the 24-hour cumulative budget is exhausted.")
        parent.update(prior_sessions_training_seconds=prior, checkpoint_session_training_seconds=current,
            checkpoint_clock_row={key: row[key] for key in ("perf/training_seconds", "_timestamp", "perf/global_step")},
            cumulative_training_seconds=cumulative, remaining_training_seconds=86400.0-cumulative,
            training_time_limit_seconds=86400)


def describe(client, ids):
    rows = client.json("experiment", "describe", ",".join(str(value) for value in ids))
    return {row["experiment"]["id"]: row["experiment"] for row in rows}


def source_jobs(config, stamp):
    if config.get("run_postfix") != "decom3000_v1":
        raise ValueError("Expected the original decom3000_v1 sweep config; review changed selection.")
    targets = []
    for name, run in expand_config(config, ROOT, stamp):
        remote_name = "CLU_" + name + "_" + run["run_postfix"]
        track = read_json(ROOT / "outputs" / remote_name / "cluster_run.json")
        targets.append({"id": track["experiment_id"], "name": remote_name})
    if len(targets) != 12 or {row["id"] for row in targets} != set(range(55450, 55462)):
        raise ValueError("Replacement targets differ from the twelve authorized experiments 55450..55461.")
    return targets


def inventory(settings, client, targets, offline):
    if offline:
        snapshot = read_json(CACHE)
    else:
        identity = client.text("user", "whoami")
        match = re.search(r"logged in as user ['\"]([^'\"]+)['\"]", identity)
        if not match:
            raise ValueError("Could not identify the authenticated cluster user.")
        parent_names = ["CLU_G03_{}_aux_l0p3_seed{}_resume12h".format(mode, seed)
                        for mode, seed, _ in PARENTS]
        expected = targets + [{"id": row[2], "name": name} for row, name in zip(PARENTS, parent_names)]
        records = describe(client, [row["id"] for row in expected])
        for wanted in expected:
            actual = records[wanted["id"]]
            if (actual["name"] != wanted["name"] or actual["projectId"] != settings["project_id"]
                    or actual["username"] != match.group(1)):
                raise ValueError("Cluster experiment identity/ownership changed: " + str(wanted["id"]))
        with Transfer(settings, service_file=CLUSTER / "tools/remote_branch_inventory.py") as connection:
            snapshot = connection.request({"persist_root": settings["persist_root"], "run_names": parent_names})
        snapshot["observed_at"] = datetime.now(timezone.utc).isoformat()
        for row, (mode, seed, experiment_id) in zip(snapshot["parents"], PARENTS):
            row.update(mode=mode, seed=seed, experiment_id=experiment_id)
        add_training_budgets(snapshot)
        write_json(CACHE, snapshot)
    indexed = {(row["mode"], row["seed"]): row for row in snapshot["parents"]}
    for mode, seed, experiment_id in PARENTS:
        row = indexed[mode, seed]
        expected_name = "CLU_G03_{}_aux_l0p3_seed{}_resume12h".format(mode, seed)
        if (row["experiment_id"] != experiment_id or row["run_name"] != expected_name
                or row["curriculum"]["config"] != {"mode": mode, "success_threshold": 0.75}
                or not row["optimizer_state_present"] or not row["history"]["history"]
                or int(row["total_steps"]) != sum(int(item["steps"]) for item in row["history"]["history"])):
            raise ValueError("Invalid source checkpoint or cumulative history: " + expected_name)
        print("Parent {}: {} ({} total steps), curriculum group {}".format(
            experiment_id, row["checkpoint"], row["total_steps"], row["curriculum"]["group"]), flush=True)
        print("  Cumulative training {:.3f}h; remaining {:.3f}h to 24h.".format(
            row["cumulative_training_seconds"]/3600, row["remaining_training_seconds"]/3600), flush=True)
    return indexed, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--dry-run", action="store_true")
    choice.add_argument("--replace", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline and not args.dry_run:
        parser.error("--offline requires --dry-run")
    settings = read_json(CLUSTER / "settings.json")
    config = read_json(CLUSTER / "config.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets = source_jobs(config, stamp)
    branch_config = dict(config, run_postfix="decom3000_branch_v2",
        description="24h cumulative G03 efficiency branch from the latest matched resumed parent; lambda 0.3; retirement 3000; inherit matching curriculum stage")
    plans = expand_config(branch_config, ROOT, stamp)
    check_capacity(plans, settings["slots_per_trial"])
    for name, run in plans:
        new_name = "CLU_" + name + "_" + run["run_postfix"]
        if (ROOT / "outputs" / new_name / "cluster_run.json").exists():
            raise ValueError("Branch already registered; inspect receipts before retrying: " + new_name)
        for folder in (CLUSTER / ".submissions").glob(new_name + "*"):
            result = folder / "result.json"
            if result.is_file() and read_json(result).get("experiment_id"):
                raise ValueError("Branch already has a submission receipt: " + str(folder))
    client = None if args.offline else Det(settings)
    parents, snapshot = inventory(settings, client, targets, args.offline)
    image = image_info(settings)
    prepared, context = [], None
    for name, run in plans:
        values = dict(value.split("=", 1) for value in run["overrides"])
        parent = parents[values["training.target_curriculum.mode"], int(values["training.seed"])]
        run["overrides"] = merge_overrides(run["overrides"], [
            "training.ckpt_loading_mode=branch", "training.checkpoint_path=" + parent["checkpoint"],
            "training.map_pool_path=" + parent["pool"],
            "training.max_training_seconds=" + str(parent["remaining_training_seconds"]),
        ])
        validate(settings, run)
        output, experiment, context = prepare_submission(
            settings, image, name, run, stamp, args.dry_run, source_context=context)
        summary = read_json(output / "summary.json")
        summary["branch_parent"] = parent
        write_json(output / "summary.json", summary)
        prepared.append((output, experiment, context[0]))
    batch_path = CLUSTER / ".submissions" / ("batch-g03-efficiency-branches-" + stamp + "-" + uuid.uuid4().hex[:8] + ".json")
    batch = {"dry_run": args.dry_run, "parents_observed_at": snapshot["observed_at"],
        "replacement_targets": targets, "parents": snapshot["parents"],
        "experiments": [{"submission_dir": str(folder), "state": "PREPARED"} for folder, _, _ in prepared]}
    write_json(batch_path, batch)
    print("Batch receipt: " + str(batch_path), flush=True)
    if args.dry_run:
        print("DRY RUN: all twelve branches prepared; no training jobs stopped or submitted.", flush=True)
        return
    # Recheck exact identities immediately before any state-changing command.
    records = describe(client, [row["id"] for row in targets])
    for target in targets:
        actual = records[target["id"]]
        if actual["name"] != target["name"] or actual["projectId"] != settings["project_id"]:
            raise ValueError("Replacement identity changed; nothing more will be stopped.")
    for target in targets:
        if records[target["id"]]["state"] not in TERMINAL:
            print(client.text("experiment", "kill", str(target["id"])).strip(), flush=True)
        target["stop_requested"] = True
        write_json(batch_path, batch)
    deadline = time.monotonic() + 90
    while True:
        records = describe(client, [row["id"] for row in targets])
        for target in targets:
            target["state"] = records[target["id"]]["state"]
        write_json(batch_path, batch)
        if all(row["state"] in TERMINAL for row in targets):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Some fresh jobs are still stopping; replacements remain PREPARED. Inspect the batch receipt.")
        time.sleep(3)
    print("All twelve original efficiency jobs stopped; submitting the prepared branches. Parent runs were not stopped.", flush=True)
    for index, (output, experiment, model) in enumerate(prepared):
        record = batch["experiments"][index]
        record["state"] = "SUBMITTING"
        write_json(batch_path, batch)
        try:
            submit(settings, experiment, model, output)
        except Exception:
            record["state"] = "CHECK_RECEIPT"
            write_json(batch_path, batch)
            print("Submission stopped. Inspect receipts/WebUI before retrying; remaining PREPARED jobs were not submitted.", file=sys.stderr)
            raise
        record["state"] = "SUBMITTED"
        write_json(batch_path, batch)


if __name__ == "__main__":
    main()
