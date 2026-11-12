"""Prepare/submit the nine G03 continuations using their original source and image.

Standard library only. --dry-run writes inspectable receipts without network,
training imports, the project virtual environment, or cluster submission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import uuid

from plan import check_capacity, merge_overrides
from prepare_context import ROOT
from submit import CLUSTER, prepare_submission, read_json, submit, validate, write_json


# Confirmed final checkpoints from the completed-run inventories fetched 3 Nov.
# Experiment, curriculum, variant, seed, original postfix, final update, W&B ID.
RUNS = (
    (55374, "none", "policy_only", 456789, "curriculum_v2", 2327, "9bnzl3ou"),
    (55377, "none", "aux_l0p3", 456789, "curriculum_v2", 2350, "o5mcki0w"),
    (55378, "none", "aux_l0p3", 456790, "curriculum_v2", 2363, "gtwgu9u2"),
    (55381, "forward", "policy_only", 456789, "curriculum_v2", 2086, "68iyfcr9"),
    (55384, "forward", "aux_l0p3", 456789, "curriculum_v2", 2105, "6789l8na"),
    (55385, "forward", "aux_l0p3", 456790, "curriculum_v2", 2123, "qm968aaz"),
    (55409, "reverse", "policy_only", 456789, "curriculum_v3", 2434, "jpwrj02i"),
    (55418, "reverse", "aux_l0p3", 456789, "curriculum_v3_cachefix", 2400, "ov6vd9q8"),
    (55419, "reverse", "aux_l0p3", 456790, "curriculum_v3_cachefix", 2439, "kchkdgan"),
)


def prepare_plan(settings):
    plans = []
    for experiment_id, curriculum, variant, seed, postfix, update, wandb_id in RUNS:
        old_name = "CLU_G03_auto7_{}_{}_seed{}_{}".format(curriculum, variant, seed, postfix)
        tracked = read_json(ROOT / "outputs" / old_name / "cluster_run.json")
        if tracked["experiment_id"] != experiment_id or tracked["experiment_state"] != "STATE_COMPLETED":
            raise ValueError("Expected the completed source experiment: " + old_name)
        remote = tracked["stages"][old_name]["remote"]
        checkpoint = "ckpt_{:06d}".format(update)
        if not any(row["name"] == checkpoint and row.get("final") and row["update"] == update
                   for row in remote["checkpoints"]):
            raise ValueError("Final checkpoint missing from saved inventory: " + old_name)
        source_receipt = Path(tracked["submission_dir"]).resolve()
        if CLUSTER / ".submissions" not in source_receipt.parents:
            raise ValueError("Source receipt must be inside cluster/.submissions.")
        summary = read_json(source_receipt / "summary.json")
        result = read_json(source_receipt / "result.json")
        if result["experiment_id"] != experiment_id or summary["run"]["run_name"] != old_name:
            raise ValueError("Source receipt identity differs: " + old_name)
        if summary["settings"] != settings:
            raise ValueError("Cluster settings changed; review the source destination before continuing.")
        source = source_receipt / "model"
        level = source / "src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml"
        level_text = level.read_text(encoding="utf-8-sig")
        original = dict(value.split("=", 1) for value in summary["run"]["overrides"])
        def training_int(key):
            value = original.get("training." + key)
            if value is None:
                value = re.search(r"(?m)^  " + key + r":\s*([0-9_]+)", level_text).group(1)
            return int(value.replace("_", ""))
        completed_steps = int(remote["global_step"])
        if completed_steps != update * training_int("num_envs") * training_int("num_steps"):
            raise ValueError("Unexpected step/update geometry: " + old_name)
        # A new output name starts a new history segment. Its remaining allowance
        # keeps the original 5B lifetime ceiling and final update unchanged.
        remaining = training_int("total_timesteps") - completed_steps
        if remaining <= 0 or len(remote["pools"]) != 1:
            raise ValueError("No remaining step allowance or ambiguous dataset: " + old_name)
        name = "G03_{}_{}_seed{}".format(curriculum, variant, seed)
        new_name = "CLU_" + name + "_resume12h"
        if (ROOT / "outputs" / new_name / "cluster_run.json").exists():
            raise ValueError("Continuation already registered; inspect before resubmitting: " + new_name)
        # Also catch a receipt saved before local registration failed.
        for previous in (CLUSTER / ".submissions").glob(new_name + "*"):
            receipt = previous / "result.json"
            if receipt.is_file() and read_json(receipt).get("experiment_id"):
                raise ValueError("Continuation already has a submission receipt: " + str(previous))
        run = {key: value for key, value in summary["run"].items() if key != "run_name"}
        checkpoint_path = settings["persist_root"] + "/outputs/" + old_name + "/checkpoints/" + checkpoint
        run.update(
            description="12h continuation of experiment {}; final update {}; same W&B run {}".format(
                experiment_id, update, wandb_id),
            run_postfix="resume12h",
            wandb_group="CLU_G03_{}_{}".format(curriculum, variant),
            overrides=merge_overrides(run["overrides"], [
                "training.checkpoint_path=" + checkpoint_path,
                "training.ckpt_loading_mode=resume",
                "training.max_training_seconds=43200",
                "training.max_wall_seconds=null",
                "training.total_timesteps=" + str(remaining),
                "training.map_pool_path=" + remote["pools"][0],
            ]),
        )
        validate(settings, run)
        plans.append((name, run, summary, source, {
            "source_experiment_id": experiment_id, "source_run": old_name,
            "source_receipt": str(source_receipt), "checkpoint": checkpoint_path,
            "completed_steps": completed_steps, "wandb_id": wandb_id,
        }))
    check_capacity(plans, settings["slots_per_trial"])
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = read_json(CLUSTER / "settings.json")
    plans = prepare_plan(settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Nine independent one-GPU jobs; account for other jobs within the shared 24-slot limit.")
    prepared = []
    for name, run, original, source, origin in plans:
        output, config_path, context_info = prepare_submission(
            settings, original["dependency_image"], name, run, stamp, args.dry_run,
            source_context=(source, original["source_files"], original["source_bytes"]))
        experiment = read_json(config_path)
        experiment["environment"]["environment_variables"].extend([
            "WANDB_RUN_ID=" + origin["wandb_id"], "WANDB_RESUME=must",
        ])
        write_json(config_path, experiment)
        summary = read_json(output / "summary.json")
        summary["continuation"] = origin
        write_json(output / "summary.json", summary)
        print("Resume:    experiment {}; {} steps; W&B {} (must resume)".format(
            origin["source_experiment_id"], origin["completed_steps"], origin["wandb_id"]))
        prepared.append((output, config_path, context_info[0]))
    batch_path = CLUSTER / ".submissions" / ("batch-g03-resume12h-" + stamp + "-" + uuid.uuid4().hex[:8] + ".json")
    batch = {"dry_run": args.dry_run, "experiments": [
        {"submission_dir": str(output), "state": "PREPARED"} for output, _, _ in prepared]}
    write_json(batch_path, batch)
    print("Batch receipt: " + str(batch_path))
    if args.dry_run:
        print("DRY RUN: nine frozen contexts prepared; no network, det calls or GPU allocation.")
        print("Remote checkpoint availability and runtime restore are not exercised by this local preparation.")
        return
    for index, (output, config_path, context) in enumerate(prepared):
        record = batch["experiments"][index]
        record["state"] = "SUBMITTING"
        write_json(batch_path, batch)
        try:
            submit(settings, config_path, context, output)
        except Exception:
            record["state"] = "CHECK_RECEIPT"
            write_json(batch_path, batch)
            print("Batch stopped. Inspect receipts/WebUI before retrying; remaining PREPARED jobs were not submitted.", file=sys.stderr)
            raise
        record["state"] = "SUBMITTED"
        write_json(batch_path, batch)


if __name__ == "__main__":
    main()
