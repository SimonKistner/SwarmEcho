"""Fetch the four requested G03 best checkpoints; never execute evaluation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath

from fetch import Transfer, run_lock, sync_run, sha256
from tracking import ROOT, now, read_json, save_json


SELECTION = (
    (55559, "CLU_G03_forward_aux_l0p3_eff25_seed456789_decom3000_branch_v3_resume36h_v1",
     "G03 Forward - aux 0.3 - efficiency 25"),
    (55521, "CLU_G03_reverse_aux_l0p3_eff25_seed456790_decom3000_branch_v3_shared789_recovery1",
     "G03 Reverse - aux 0.3 - efficiency 25"),
    (55557, "CLU_G03_reverse_aux_l0p3_seed456789_resume48h_v1", "G03 Reverse - aux 0.3"),
    (55555, "CLU_G03_forward_aux_l0p3_seed456790_resume48h_v1", "G03 Forward - aux 0.3"),
)
MANIFEST = ROOT / "cluster/.fetch/g03_best_replays.json"


def resume_lineage(name):
    """Follow only strict resumes; never cross a reward/seed branch boundary."""
    lineage = []
    while name not in lineage:
        path = ROOT / "outputs" / name / "cluster_run.json"
        state = read_json(path)
        summary = read_json(Path(state["submission_dir"]) / "summary.json")
        if summary["run"]["run_name"] != name:
            raise ValueError("Frozen source mismatch: " + name)
        lineage.append(name)
        values = dict(value.split("=", 1) for value in summary["run"]["overrides"])
        if values.get("training.ckpt_loading_mode") != "resume":
            return lineage
        checkpoint = PurePosixPath(values["training.checkpoint_path"])
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("Unexpected resume checkpoint location")
        name = checkpoint.parent.parent.name
    raise ValueError("Cycle in strict-resume provenance")


def main():
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    settings = read_json(ROOT / "cluster/settings.json")
    chains = {}
    for identifier, name, _ in SELECTION:
        if read_json(ROOT / "outputs" / name / "cluster_run.json")["experiment_id"] != identifier:
            raise ValueError("Requested experiment mismatch: " + name)
        chains[name] = resume_lineage(name)
    names = list(dict.fromkeys(name for chain in chains.values() for name in chain))
    entries = []
    with Transfer(settings) as connection:
        catalog = connection.request({"action": "best_checkpoints", "persist_root": settings["persist_root"],
                                      "run_names": names})
        markers = {row["run_name"]: row["best"] for row in catalog["runs"]}
        plans = []
        for identifier, name, label in SELECTION:
            available = []
            for index, candidate in enumerate(chains[name]):
                marker = markers[candidate]
                if marker is None:
                    continue
                if marker["run_name"] != candidate:
                    raise ValueError("Best-checkpoint run identity mismatch")
                score = float(marker["metrics"]["validation/success_rate"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid best validation score")
                available.append((score, -index, candidate))
            if not available:
                raise ValueError("No published best checkpoint in strict-resume lineage: " + name)
            score, _, source = max(available)
            print("BEST {} -> {} ({:.2%})".format(identifier, source, score), flush=True)
            plans.append((identifier, name, label, source))
        for identifier, requested, label, source in plans:
            run = ROOT / "outputs" / source
            path = run / "cluster_run.json"
            with run_lock(path):
                sync_run(connection, path, False, checkpoint="best_val_checkpoint", eval_maps=1)
            state = read_json(path)
            fetched = state["stages"][state["remote_name"]]["eval_fetches"]["best_val_checkpoint"]
            if not fetched["complete"]:
                raise ValueError("Incomplete evaluation fetch: " + source)
            checkpoint = run / "checkpoints/best_val_checkpoint"
            marker = read_json(checkpoint / "recovery_ready.json")
            if marker["run_name"] != source:
                raise ValueError("Downloaded checkpoint belongs to a different run")
            for alias_run in {source, requested}:
                save_json(ROOT / "outputs" / alias_run / ".cluster/inspector.json", {"display_name": label})
            entries.append({"requested_experiment": identifier, "requested_run": requested,
                "source_experiment": state["experiment_id"], "source_run": source, "display_name": label,
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "update": marker["update"],
                "validation_success": marker["metrics"]["validation/success_rate"],
                "checkpoint_marker_sha256": sha256(checkpoint / "recovery_ready.json"),
                "config_sha256": sha256(run / "config.yaml"),
                "map_sha256": sha256(run / "random_eval_maps/random_eval_0000.yaml")})
    if len({row["map_sha256"] for row in entries}) != 1:
        raise ValueError("Fetched inspection maps differ; choose one shared map before evaluation")
    save_json(MANIFEST, {"observed_at": now(), "runs": entries})
    print("REPLAY_INPUTS_READY=" + str(MANIFEST), flush=True)


if __name__ == "__main__":
    main()
