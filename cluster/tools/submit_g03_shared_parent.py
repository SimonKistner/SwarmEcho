"""Submit six replacement continuation seeds from the retained branches' parents.

User-executed submission only. Copies each retained job's frozen source/image,
parent checkpoint, training allowance and W&B group into a fresh output name.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

from capacity import Det
from plan import merge_overrides
from prepare_context import ROOT
from submit import (CLUSTER, ensure_unused_run_name, prepare_submission, read_json,
                    submission_run_name, submit, validate, write_json)


# Curriculum, efficiency coefficient, retained experiment, canceled replacement.
PAIRS = (
    ("forward", 25, 55489, 55491),
    ("forward", 50, 55492, 55493),
    ("forward", 100, 55494, 55495),
    ("reverse", 25, 55496, 55497),
    ("reverse", 50, 55498, 55499),
    ("reverse", 100, 55500, 55501),
)
POSTFIX = "decom3000_branch_v3_shared789_v1"
PARENT_CHECKPOINTS = {"forward": "ckpt_002520", "reverse": "ckpt_002880"}
PARENT_ALLOWANCES = {"forward": 34155.4516851604, "reverse": 34102.88276382256}
TERMINAL = {"STATE_CANCELED", "STATE_ERROR", "STATE_COMPLETED", "STATE_DELETED"}


def plans(settings, stamp):
    result = []
    for mode, efficiency, kept_id, stopped_id in PAIRS:
        prefix = "G03_{}_aux_l0p3_eff{}".format(mode, efficiency)
        kept_name = "CLU_" + prefix + "_seed456789_decom3000_branch_v3"
        tracking = read_json(ROOT / "outputs" / kept_name / "cluster_run.json")
        receipt = Path(tracking["submission_dir"]).resolve()
        if (CLUSTER / ".submissions").resolve() not in receipt.parents:
            raise ValueError("Source receipt must be inside cluster/.submissions.")
        original = read_json(receipt / "summary.json")
        source_result = read_json(receipt / "result.json")
        if (tracking["experiment_id"] != kept_id or source_result["experiment_id"] != kept_id
                or original["run"]["run_name"] != kept_name or original["settings"] != settings):
            raise ValueError("Retained submission identity/destination changed: " + str(kept_id))
        run = {key: value for key, value in original["run"].items() if key != "run_name"}
        values = dict(item.split("=", 1) for item in run["overrides"])
        checkpoint = (settings["persist_root"] + "/outputs/CLU_G03_" + mode
                      + "_aux_l0p3_seed456789_resume12h/checkpoints/" + PARENT_CHECKPOINTS[mode])
        group = "CLU_G03_decom3000_branch_v3_{}_aux_l0p3_eff{}".format(mode, efficiency)
        if (values.get("training.seed") != "456789"
                or values.get("training.ckpt_loading_mode") != "branch"
                or values.get("training.checkpoint_path") != checkpoint
                or values.get("training.target_curriculum.mode") != mode
                or float(values["training.max_training_seconds"]) != PARENT_ALLOWANCES[mode]
                or run.get("wandb_group") != group):
            raise ValueError("Retained branch checkpoint, budget, seed or group changed: " + str(kept_id))
        run.update(run_postfix=POSTFIX,
                   description="Continuation seed 456790 from the same frozen parent checkpoint and training allowance as retained experiment {}; replaces {}".format(kept_id, stopped_id),
                   overrides=merge_overrides(run["overrides"], ["training.seed=456790"]))
        validate(settings, run)
        name = prefix + "_seed456790"
        result.append((name, run, original, receipt, {
            "retained_experiment_id": kept_id, "retained_run_name": kept_name,
            "replaced_experiment_id": stopped_id,
            "replaced_run_name": "CLU_" + prefix + "_seed456790_decom3000_branch_v3",
            "parent_checkpoint": checkpoint, "parent_training_seed": 456789,
            "continuation_seed": 456790, "remaining_training_seconds": PARENT_ALLOWANCES[mode],
            "wandb_group": group, "source_receipt": str(receipt),
        }))
    # Confirm all local conflicts before preparing or submitting any member.
    for name, run, _, _, _ in result:
        ensure_unused_run_name(submission_run_name(name, run, stamp))
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    settings = read_json(CLUSTER / "settings.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected = plans(settings, stamp)
    client = Det(settings)
    client.environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    ids = [identifier for row in PAIRS for identifier in row[2:]]
    records = {row["experiment"]["id"]: row["experiment"] for row in
               client.json("experiment", "describe", ",".join(map(str, ids)))}
    for _, _, _, _, origin in selected:
        kept = records[origin["retained_experiment_id"]]
        stopped = records[origin["replaced_experiment_id"]]
        if (kept["name"] != origin["retained_run_name"]
                or stopped["name"] != origin["replaced_run_name"]
                or kept["projectId"] != settings["project_id"]
                or stopped["projectId"] != settings["project_id"]
                or kept["username"] != stopped["username"]
                or stopped["state"] not in TERMINAL):
            raise ValueError("Replacement identity/state differs; no new experiments submitted.")
    prepared = []
    for name, run, original, receipt, origin in selected:
        output, config, context = prepare_submission(
            settings, original["dependency_image"], name, run, stamp, False,
            source_context=(receipt / "model", original["source_files"], original["source_bytes"]))
        summary = read_json(output / "summary.json")
        summary["shared_initialization"] = origin
        write_json(output / "summary.json", summary)
        print("Paired with experiment {}; checkpoint {}; group {}".format(
            origin["retained_experiment_id"], origin["parent_checkpoint"], origin["wandb_group"]), flush=True)
        prepared.append((output, config, context[0], origin))
    batch_path = CLUSTER / ".submissions" / ("batch-g03-shared789-" + stamp + "-" + uuid.uuid4().hex[:8] + ".json")
    batch = {"experiments": [{"submission_dir": str(output), "state": "PREPARED",
                              "shared_initialization": origin} for output, _, _, origin in prepared]}
    write_json(batch_path, batch)
    print("Six fresh continuation runs; retained jobs and parent jobs are unchanged.", flush=True)
    print("Batch receipt: " + str(batch_path), flush=True)
    for index, (output, config, context, _) in enumerate(prepared):
        record = batch["experiments"][index]
        record["state"] = "SUBMITTING"
        write_json(batch_path, batch)
        try:
            submit(settings, config, context, output)
        except BaseException:
            record["state"] = "CHECK_RECEIPT"
            write_json(batch_path, batch)
            print("Batch stopped. Inspect receipts/WebUI before retrying; remaining PREPARED jobs were not submitted.", file=sys.stderr)
            raise
        record["state"] = "SUBMITTED"
        write_json(batch_path, batch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShared-parent submission aborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
