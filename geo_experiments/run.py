"""Sequential frozen spatial probes. CLI accepts training-style key=value overrides."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Permit both `python geo_experiments/run.py` and `python -m geo_experiments.run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from geo_experiments.common import ROOT, experiment_axes, resolve_path, split_episodes, validate_config
from geo_experiments.artifacts import HISTORY_COLUMNS, add_group, save_data, update_run


def main(default_config="config.yaml"):
    arguments = sys.argv[1:]
    config_path = Path(__file__).with_name(default_config)
    overrides = []
    for argument in arguments:
        if "=" not in argument:
            raise ValueError("Use config=<yaml> and key=value overrides, e.g. run.name=pilot dataset.train_samples=10000")
        key, value = argument.split("=", 1)
        if key == "config":
            config_path = resolve_path(value)
        else:
            overrides.append(argument)
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(Path(__file__).with_name("config.yaml")),
                                                   OmegaConf.load(config_path), OmegaConf.from_dotlist(overrides)), resolve=True)
    validate_config(config)
    tasks, sources = experiment_axes(config)
    own = config.get("experiment") == "own_cell"
    output_root = resolve_path(config["run"]["output_root"])
    if not output_root.is_relative_to(ROOT / "geo_experiments"):
        raise ValueError("run.output_root must remain inside geo_experiments/.")
    stamp = datetime.now(ZoneInfo(config["run"]["timezone"])).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{config['run']['name']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    update_run(run_dir, config=config, jobs=[], history_columns=HISTORY_COLUMNS, progress={"stage": "starting"})
    print(f"[experiment] {run_dir}", flush=True)
    try:
        from geo_experiments.collection import prepare_policy, collect_dataset, reuse_dataset
        from geo_experiments.geometry import build_cell_graph, target_labels
        from geo_experiments.probes import train_probe

        model, level, identity = prepare_policy(config, run_dir)
        clearance = config["distance"]["clearance_m"]
        if clearance is None:
            clearance = level.env.drone_radius + level.env.obstacle_planning_clearance_m
        graph = build_cell_graph(level.building, clearance)
        coverage = None
        if own:
            from geo_experiments.own_collection import collect_own_dataset
            dataset, metadata, episodes, audit_result, labels, splits, coverage = collect_own_dataset(
                model, level, config, run_dir, identity, graph)
        elif config["dataset"]["path"]:
            dataset, metadata, episodes, audit_result = reuse_dataset(config["dataset"]["path"], config, run_dir, identity)
        else:
            dataset, metadata, episodes, audit_result = collect_dataset(model, level, config, run_dir, identity)
        if not own:
            labels = target_labels(dataset, graph)
            splits = split_episodes(labels["target_cell"], config["dataset"])
        arrays = {}
        for prefix, values in (("dataset", dataset), ("episodes", episodes), ("geometry", graph),
                               ("labels", labels), ("splits", splits)):
            add_group(arrays, prefix, values)
        if audit_result is not None:
            add_group(arrays, "audit", audit_result[1])
        if coverage is not None:
            add_group(arrays, "coverage", coverage)
        save_data(run_dir, arrays)
        update_run(run_dir, collection=metadata, ambiguity_audit=audit_result[0] if audit_result else None)
        # Frozen features are now on disk. Free rollout compilation/model memory
        # before the independent supervised jobs; no policy optimizer exists.
        del model
        import gc
        import jax
        gc.collect()
        jax.clear_caches()
        total = len(tasks) * len(sources) * len(config["probe"]["seeds"])
        completed = 0
        jobs = []
        for task in tasks:
            for source in sources:
                for seed in config["probe"]["seeds"]:
                    print(f"[experiment] {completed + 1}/{total}: {task}/{source}/seed={seed}", flush=True)
                    result, saved = train_probe(task, source, seed, dataset, labels, splits, graph, config)
                    add_group(arrays, f"probes/{task}/{source}/seed_{seed}", saved)
                    save_data(run_dir, arrays)
                    jobs.append(result)
                    completed += 1
                    update_run(run_dir, jobs=jobs, progress={"stage": "probes", "completed": completed, "total": total})
        from geo_experiments.report import generate_report
        generate_report(run_dir)
        update_run(run_dir, progress={"stage": "complete", "completed": completed, "total": total})
        from geo_experiments.dashboard import dashboard_path
        print(f"[complete] Open {dashboard_path()} (report: {run_dir.name})", flush=True)
    except Exception as exc:
        update_run(run_dir, progress={"stage": "failed", "error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
