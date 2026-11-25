"""Evaluate the two fetched forward G03 checkpoints with chain-and-cleanup replays.

Run manually in WSL from the repository root:
    uv run python cluster/eval_g03_best_replays.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cluster/.fetch/g03_best_replays.json"
FORWARD_EXPERIMENTS = {55559, 55555}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = [row for row in selection["runs"] if row["requested_experiment"] in FORWARD_EXPERIMENTS]
    if len(selected) != 2 or {row["requested_experiment"] for row in selected} != FORWARD_EXPERIMENTS:
        raise ValueError("Expected both fetched forward replay inputs (55559 and 55555)")
    commands = []
    maps = set()
    for row in selected:
        checkpoint = ROOT / row["checkpoint"]
        run = checkpoint.parents[1]
        for path, expected in ((checkpoint / "recovery_ready.json", row["checkpoint_marker_sha256"]),
                               (run / "config.yaml", row["config_sha256"]),
                               (run / "random_eval_maps/random_eval_0000.yaml", row["map_sha256"])):
            if digest(path) != expected:
                raise ValueError("Fetched replay input changed; fetch again: " + str(path))
        maps.add(row["map_sha256"])
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        config.pop("name", None)
        config.pop("building", None)
        config["env"]["map_names"] = []
        level = run / ".cluster/replay_eval/G03_CLU_random_buildings.yaml"
        level.parent.mkdir(parents=True, exist_ok=True)
        level.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-m", "swarmecho.training.evaluate",
            "checkpoint=" + row["checkpoint"], "level=" + level.relative_to(ROOT).as_posix(),
            "mode=parallel", "map_suite_map_id=random_eval_0000",
            "replay_after=true", "replay_by_required_agents=true",
            "inspection_evaluation.training_heatmap_creation=true",
            "inspection_evaluation.dynamic_max_geo_distance=true",
            "inspection_evaluation.target_sampling=independent",
            "inspection_evaluation.success_condition=chain_held_and_cleand_up",
            "inspection_evaluation.seed=456789", "inspection_evaluation.episodes_per_map=1000",
            "inspection_evaluation.eval_video=true", "inspection_evaluation.map_suite_replay_limit=1",
            "logging.wandb_mode=disabled", "eval_name=chain&cleanUp"]
        commands.append((row, command))
    if len(maps) != 1:
        raise ValueError("The two frozen inspection maps differ; select a shared map before comparing")
    for row, command in commands:
        print("\n{} - chain&cleanUp: saved best {:.2%}, update {}, source experiment {}".format(
            row["display_name"], row["validation_success"], row["update"], row["source_experiment"]), flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nReplays ready. Open with: uv run swarmecho-inspect root=outputs", flush=True)


if __name__ == "__main__":
    main()
