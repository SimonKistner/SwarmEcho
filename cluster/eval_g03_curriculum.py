"""Evaluate six G03 curriculum checkpoints (seed 456789) on one shared map.

From the repository root in WSL:
    uv run python cluster/eval_g03_curriculum.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    (55374, "none_policy_only_seed456789_curriculum_v2"),
    (55377, "none_aux_l0p3_seed456789_curriculum_v2"),
    (55381, "forward_policy_only_seed456789_curriculum_v2"),
    (55384, "forward_aux_l0p3_seed456789_curriculum_v2"),
    (55409, "reverse_policy_only_seed456789_curriculum_v3"),
    (55418, "reverse_aux_l0p3_seed456789_curriculum_v3_cachefix"),
)
MAP_ID = "random_eval_0000"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepared = []
    map_hashes = set()
    for experiment_id, suffix in RUNS:
        run = ROOT / "outputs" / ("CLU_G03_auto7_" + suffix)
        checkpoint = run / "checkpoints/best_val_checkpoint"
        state = json.loads((run / "cluster_run.json").read_text(encoding="utf-8"))
        if state["experiment_id"] != experiment_id:
            raise ValueError(f"Unexpected experiment in {run}")
        fetched = state["stages"][state["remote_name"]]["eval_fetches"]["best_val_checkpoint"]
        if not fetched["complete"]:
            raise ValueError(f"Incomplete checkpoint fetch: {run}")
        marker = json.loads((checkpoint / "recovery_ready.json").read_text(encoding="utf-8"))
        map_hashes.add(hashlib.sha256((run / f"random_eval_maps/{MAP_ID}.yaml").read_bytes()).hexdigest())
        # Saved configs include descriptive metadata and a resolved map label.
        # Adapt a copy to the level loader; retain the original run config.
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        config.pop("name", None)
        config.pop("building", None)
        config["env"]["map_names"] = []
        level_path = run / ".cluster/replay_eval/G03_CLU_random_buildings.yaml"
        prepared.append((experiment_id, suffix, checkpoint, marker, level_path, config))
    if len(map_hashes) != 1:
        raise ValueError("The frozen maps differ across runs; choose a shared map before evaluation.")
    print(f"Verified {len(prepared)} checkpoints; shared {MAP_ID} SHA256: {next(iter(map_hashes))}", flush=True)
    # Prepare and validate every run before starting the first GPU process.
    commands = []
    for experiment_id, suffix, checkpoint, marker, level_path, config in prepared:
        level_path.parent.mkdir(parents=True, exist_ok=True)
        level_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable, "-m", "swarmecho.training.evaluate",
            f"checkpoint={checkpoint.relative_to(ROOT).as_posix()}",
            f"level={level_path.relative_to(ROOT).as_posix()}",
            "mode=parallel", f"map_suite_map_id={MAP_ID}",
            "replay_after=true", "replay_by_required_agents=true",
            "inspection_evaluation.training_heatmap_creation=true",
            "inspection_evaluation.dynamic_max_geo_distance=true",
            "inspection_evaluation.target_sampling=independent",
            "inspection_evaluation.seed=456789",
            "inspection_evaluation.eval_video=true",
            "inspection_evaluation.map_suite_replay_limit=1",
            "logging.wandb_mode=disabled",
            "eval_name=curriculum_best_one_map",
        ]
        commands.append((experiment_id, suffix, marker, command))
    for experiment_id, suffix, marker, command in commands:
        score = marker["metrics"]["validation/success_rate"]
        print(f"\nex{experiment_id} {suffix}: best update {marker['update']}, validation {score:.2%}", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
