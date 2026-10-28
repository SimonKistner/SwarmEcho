"""Refit a saved G01 or G02 dataset without loading or recollecting its policy."""
from __future__ import annotations

import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_experiments.artifacts import add_group, group, load_run, save_data, update_run
from geo_experiments.common import experiment_axes, resolve_path, validate_config


def main():
    from omegaconf import OmegaConf
    options, overrides = {}, []
    for argument in sys.argv[1:]:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("Use run=<saved experiment>, optionally probe.<setting>=<value>.")
        if key == "run":
            options[key] = value
        elif key.startswith("probe.") or key == "report.dpi":
            overrides.append(argument)
        else:
            raise ValueError("Reproduction keeps the original data, labels and splits; only probe settings/report.dpi may change.")
    source = resolve_path(options["run"])
    manifest, arrays, _ = load_run(source)
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(manifest["config"]), OmegaConf.from_dotlist(overrides)), resolve=True)
    validate_config(config)
    stamp = datetime.now(ZoneInfo(config["run"].get("timezone", "Europe/Berlin"))).strftime("%Y%m%d_%H%M%S_%f")
    target = source.parent / f"{config['run']['name']}_refit_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    config["run"]["name"] += "_refit"
    copied = {k: copy.deepcopy(v) for k, v in manifest.items() if k not in ("format", "shared_data", "jobs", "progress", "report")}
    copied["config"] = config
    copied["reproduction"] = {"source": str(source), "protocol": "Original archived features, labels and episode split; no policy rollout"}
    update_run(target, **copied, jobs=[], progress={"stage": "starting"})
    OmegaConf.save(OmegaConf.create(config), target / "config.yaml")
    data = {k: v for k, v in arrays.items() if not k.startswith("probes/")}
    save_data(target, data)
    try:
        if "geometry/map_offsets" in data:
            from geo_experiments.experiments import fit_experiment
            fitted, jobs = fit_experiment(config, data, target, shared_reference=False)
            data.update(fitted)
        else:
            from geo_experiments.probes import train_probe
            dataset, labels, splits, graph = (group(data, k) for k in ("dataset", "labels", "splits", "geometry"))
            tasks, sources = experiment_axes(config)
            jobs = []
            for task in tasks:
                for feature in sources:
                    for seed in config["probe"]["seeds"]:
                        result, fitted = train_probe(task, feature, seed, dataset, labels, splits, graph, config)
                        add_group(data, f"probes/{task}/{feature}/seed_{seed}", fitted)
                        jobs.append(result)
                        save_data(target, data)
                        update_run(target, jobs=jobs, progress={"stage": "probes", "completed": len(jobs)})
        save_data(target, data)
        from geo_experiments.report import generate_report
        generate_report(target)
        update_run(target, jobs=jobs, progress={"stage": "complete", "completed": len(jobs), "total": len(jobs)})
    except Exception as exc:
        update_run(target, progress={"stage": "failed", "error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
