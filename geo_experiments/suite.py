"""Plan and sequentially execute configured readouts, sharing collection by policy."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from geo_experiments.artifacts import update_run
from geo_experiments.common import ROOT, resolve_path, validate_config
from geo_experiments.labels import DECODERS
from geo_experiments.console import terminal_print as print
from geo_experiments.naming import display_name, run_slug


def latest_checkpoint(run):
    candidates = [p for p in (run / "checkpoints").glob("ckpt_*")
                  if p.is_dir() and re.fullmatch(r"ckpt_\d+", p.name)
                  and (p / "_METADATA").is_file() and (p / "training_contract.json").is_file()
                  and (p / "recovery_ready.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint in {run / 'checkpoints'}")
    return max(candidates, key=lambda p: int(p.name.split("_")[-1])).resolve()


def execution_plan(settings):
    if set(settings) != {"run", "defaults", "policies", "experiments", "execute"}:
        raise ValueError("Suite sections must be run, defaults, policies, experiments, execute.")
    selected, resolved = [], {}
    for entry in settings["execute"]:
        if set(entry) != {"policy", "experiment"}:
            raise ValueError("Each execute entry must name policy and experiment.")
        name, experiment = entry["policy"], entry["experiment"]
        if name not in resolved:
            policy = settings["policies"][name]
            run = resolve_path(policy["run"])
            checkpoint = latest_checkpoint(run) if policy["checkpoint"] == "latest" else resolve_path(policy["checkpoint"])
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            resolved[name] = {"run": str(run), "checkpoint": str(checkpoint)}
            if "map_source" in policy:
                resolved[name]["map_source"] = policy["map_source"]
            if policy.get("map_source") == "validation":
                from geo_experiments.validation_maps import frozen_validation_maps
                paths, _ = frozen_validation_maps(checkpoint)
                print(f"[suite] {run.name}: {len(paths)} verified frozen validation maps", flush=True)
        config = OmegaConf.to_container(OmegaConf.merge(settings["defaults"],
                   settings["experiments"][experiment], {"run": {**settings["run"], "name": f"{name}_{experiment}"},
                   "policy": {k: v for k, v in resolved[name].items() if k != "run"}}), resolve=True)
        # Historical saved suite configs omitted this unused field.
        # Modern policies load the checkpoint config rather than a named level.
        config["policy"].setdefault("level", None)
        config["run"]["name"] = run_slug(config)
        validate_config(config)
        if config.get("experiment", "target") != "target" or config["protocol"]["decoder"] not in DECODERS:
            raise ValueError("This suite schedules only the three target-cell alternatives; own-cell is not enabled.")
        if not config["policy"]["from_run_config"] or config["dataset"]["path"] is not None:
            raise ValueError("Suite policies use their saved training config; use resume=<suite folder> to reuse collection.")
        # Decoder/task selection changes fits, never collection. Audit settings
        # are included because conditional history collection is shared too.
        collection_settings = {k: config[k] for k in ("policy", "dataset", "audit", "distance")}
        collection_settings["eligibility"] = config["protocol"]["eligibility"]
        if config["protocol"]["decoder"] == "shared_cell":
            # Small per-map quotas may share coordinate support across maps.
            # Preserve the historical collection key whenever coverage fits.
            if config["policy"].get("map_source") == "validation":
                collection_settings["training_class_coverage"] = "pooled_coordinates"
        key = hashlib.sha256(json.dumps(collection_settings, sort_keys=True).encode()).hexdigest()[:16]
        selected.append({**entry, "collection_key": key, "config": config})
    if not selected:
        raise ValueError("The execute list is empty.")
    names = [(e["policy"], e["experiment"]) for e in selected]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate policy/experiment in execute list.")
    return selected, resolved


def execute_suite(settings, root, plan):
    from geo_experiments.execution import prepare_collection, fit_and_report

    prepared = set()
    for index, job in enumerate(plan):
        config, key = job["config"], job["collection_key"]
        run_dir = root / "runs" / config["run"]["name"]
        if (run_dir / "run.json").is_file():
            previous = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if previous.get("progress", {}).get("stage") == "complete" and (run_dir / "report.html").is_file():
                print(f"[suite] completed, skipping {display_name(previous)}", flush=True)
                continue
        collection_dir = root / "collections" / key
        if key not in prepared:
            decoders = tuple(sorted({j["config"]["protocol"]["decoder"] for j in plan if j["collection_key"] == key}))
            prepare_collection(config, collection_dir, decoders)
            prepared.add(key)
        fit_and_report(config, run_dir, collection_dir)
        update_run(root, progress={"stage": "experiments", "completed": index + 1, "total": len(plan)})
    update_run(root, progress={"stage": "complete", "completed": len(plan), "total": len(plan)})


def main():
    options, overrides = {}, []
    for argument in sys.argv[1:]:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("Use config=<yaml>, plan=true, resume=<suite folder>, or defaults.<key>=<value>.")
        if key in ("config", "plan", "resume"):
            options[key] = value
        else:
            overrides.append(argument)
    if "resume" in options:
        if overrides or "config" in options:
            raise ValueError("A resumed suite uses its frozen config; start a new suite to change settings.")
        root = resolve_path(options["resume"])
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        settings, plan = manifest["suite_config"], manifest["execution_plan"]
    else:
        path = resolve_path(options["config"]) if "config" in options else Path(__file__).with_name("probes.yaml")
        settings = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides)), resolve=True)
        plan, resolved = execution_plan(settings)
        output_root = resolve_path(settings["run"]["output_root"])
        if not output_root.is_relative_to(ROOT / "geo_experiments"):
            raise ValueError("Suite outputs must stay inside geo_experiments/.")
        stamp = datetime.now(ZoneInfo(settings["run"]["timezone"])).strftime("%Y%m%d_%H%M%S_%f")
        root = output_root / f"{settings['run']['name']}_{stamp}"
        # Resolve latest only once; snapshots/resume never chase a newer checkpoint.
        settings["policies"] = resolved
    print(f"[suite] {len(plan)} sequential experiments; {len({j['collection_key'] for j in plan})} shared collections", flush=True)
    for job in plan:
        print(f"  {display_name({'config': job['config']})}: {job['config']['policy']['checkpoint']}", flush=True)
    if options.get("plan", "false").lower() == "true":
        return
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(settings), root / "config.yaml")
    update_run(root, suite_config=settings, execution_plan=plan, progress={"stage": "starting"})
    try:
        execute_suite(settings, root, plan)
    except Exception as exc:
        update_run(root, progress={"stage": "failed", "error": str(exc), "type": type(exc).__name__})
        raise
    print(f"[suite] complete: {root}", flush=True)


if __name__ == "__main__":
    main()
