"""Continue canceled experiment 55446 to 24 cumulative training hours.

Run only when ready to submit. Queries the latest complete checkpoint through
a temporary zero-GPU shell, then submits one continuation using frozen source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys

from branch_g03_efficiency import add_training_budgets
from capacity import Det
from fetch import Transfer
from plan import merge_overrides
from prepare_context import ROOT
from submit import (CLUSTER, ensure_unused_run_name, prepare_submission, read_json,
                    submission_run_name, submit, validate, write_json)


SOURCE_ID = 55446
SOURCE_NAME = "CLU_G03_reverse_aux_l0p3_seed456790_resume12h"
NAME = "G03_reverse_aux_l0p3_seed456790"
POSTFIX = "resume24h_recovery1"
LIMIT_SECONDS = 86400.0


def source_state(client, settings):
    records = client.json("experiment", "describe", str(SOURCE_ID))
    if len(records) != 1:
        raise ValueError("Expected exactly one source experiment.")
    source = records[0]["experiment"]
    if (source["name"] != SOURCE_NAME or source["projectId"] != settings["project_id"]
            or source["state"] != "STATE_CANCELED"):
        raise ValueError("Expected canceled experiment 55446 with its original name/project; nothing submitted.")
    return source


def apply_budget(snapshot):
    parent = snapshot["parents"][0]
    # Final checkpoints carry their own precise session clock. Periodic legacy
    # checkpoints need the matching W&B update plus checkpoint publication work.
    current = parent.get("ready_metrics", {}).get("training_seconds")
    if current is None:
        add_training_budgets(snapshot)
        return
    current = float(current)
    prior = sum(float(row["training_seconds"]) for row in parent["prior_sessions"])
    cumulative = prior + current
    if not math.isfinite(cumulative) or current < 0 or not 0 < cumulative < LIMIT_SECONDS:
        raise ValueError("The saved training clock is invalid or 24 cumulative hours are already complete.")
    parent.update(prior_sessions_training_seconds=prior,
                  checkpoint_session_training_seconds=current,
                  cumulative_training_seconds=cumulative,
                  remaining_training_seconds=LIMIT_SECONDS - cumulative,
                  training_time_limit_seconds=LIMIT_SECONDS,
                  checkpoint_clock_source="recovery_ready.json metrics.training_seconds")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    settings = read_json(CLUSTER / "settings.json")
    tracked = read_json(ROOT / "outputs" / SOURCE_NAME / "cluster_run.json")
    if tracked["experiment_id"] != SOURCE_ID:
        raise ValueError("Local source tracking no longer refers to experiment 55446.")
    receipt = Path(tracked["submission_dir"]).resolve()
    if (CLUSTER / ".submissions").resolve() not in receipt.parents:
        raise ValueError("Source receipt must be inside cluster/.submissions.")
    original = read_json(receipt / "summary.json")
    result = read_json(receipt / "result.json")
    if (result["experiment_id"] != SOURCE_ID or original["run"]["run_name"] != SOURCE_NAME
            or original["settings"] != settings):
        raise ValueError("Source submission identity or destination changed.")
    run = {key: value for key, value in original["run"].items() if key != "run_name"}
    run.update(run_postfix=POSTFIX,
               description="Resume canceled experiment 55446 from its latest complete checkpoint to 24 cumulative training hours")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # This confirmation precedes even the temporary metadata shell.
    ensure_unused_run_name(submission_run_name(NAME, run, stamp))
    client = Det(settings)
    client.environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    source = source_state(client, settings)
    with Transfer(settings, service_file=CLUSTER / "tools/remote_branch_inventory.py") as connection:
        snapshot = connection.request({"persist_root": settings["persist_root"],
                                       "run_names": [SOURCE_NAME]})
    snapshot["observed_at"] = datetime.now(timezone.utc).isoformat()
    if len(snapshot["parents"]) != 1:
        raise ValueError("Expected exactly one checkpoint inventory.")
    parent = snapshot["parents"][0]
    expected_wandb = original["continuation"]["wandb_id"]
    if (parent["run_name"] != SOURCE_NAME or not parent["optimizer_state_present"]
            or parent["wandb_id"] != expected_wandb
            or parent["curriculum"]["config"] != {"mode": "reverse", "success_threshold": 0.75}
            or parent["total_steps"] != sum(int(row["steps"]) for row in parent["history"]["history"])):
        raise ValueError("Latest checkpoint optimizer, curriculum, W&B identity or step history is inconsistent.")
    parent["experiment_id"] = SOURCE_ID
    apply_budget(snapshot)
    values = dict(item.split("=", 1) for item in run["overrides"])
    lifetime_step_limit = (int(values["training.total_timesteps"])
                           + int(original["continuation"]["completed_steps"]))
    remaining_steps = lifetime_step_limit - int(parent["total_steps"])
    if remaining_steps <= 0:
        raise ValueError("The original lifetime step allowance is already exhausted.")
    run["overrides"] = merge_overrides(run["overrides"], [
        "training.ckpt_loading_mode=resume",
        "training.checkpoint_path=" + parent["checkpoint"],
        "training.max_training_seconds=" + str(parent["remaining_training_seconds"]),
        "training.max_wall_seconds=null",
        "training.total_timesteps=" + str(remaining_steps),
        "training.map_pool_path=" + parent["pool"],
    ])
    validate(settings, run)
    output, config_path, context = prepare_submission(
        settings, original["dependency_image"], NAME, run, stamp, False,
        source_context=(receipt / "model", original["source_files"], original["source_bytes"]))
    experiment = read_json(config_path)
    experiment["environment"]["environment_variables"].extend([
        "WANDB_RUN_ID=" + expected_wandb, "WANDB_RESUME=must",
    ])
    write_json(config_path, experiment)
    summary = read_json(output / "summary.json")
    summary["continuation"] = {
        "source_experiment_id": SOURCE_ID, "source_run": SOURCE_NAME,
        "source_receipt": str(receipt), "checkpoint": parent["checkpoint"],
        "completed_steps": parent["total_steps"], "wandb_id": expected_wandb,
    }
    summary["recovery_parent"] = parent
    summary["parent_observed_at"] = snapshot["observed_at"]
    write_json(output / "summary.json", summary)
    print("Checkpoint: " + parent["checkpoint"], flush=True)
    print("Training: {:.4f}h inherited + {:.4f}h remaining = 24h total".format(
        parent["cumulative_training_seconds"] / 3600,
        parent["remaining_training_seconds"] / 3600), flush=True)
    print("Continue W&B {} with the original source, image, rewards, optimizer and curriculum.".format(
        expected_wandb), flush=True)
    if source_state(client, settings)["username"] != source["username"]:
        raise ValueError("Source ownership changed; nothing submitted.")
    submit(settings, config_path, context[0], output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRecovery submission aborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
