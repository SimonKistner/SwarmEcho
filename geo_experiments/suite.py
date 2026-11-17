"""Plan and sequentially execute configured readouts, sharing collection by policy."""
from __future__ import annotations

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

from geo_experiments.shared.artifacts import update_run
from geo_experiments.shared.common import ROOT, resolve_path
from geo_experiments.registry import get_probe, job_probe, archived_probe
from geo_experiments.shared.console import terminal_print as print
from geo_experiments.shared.naming import display_name


def latest_checkpoint(run):
    candidates = [p for p in (run / "checkpoints").glob("ckpt_*")
                  if p.is_dir() and re.fullmatch(r"ckpt_\d+", p.name)
                  and (p / "_METADATA").is_file() and (p / "training_contract.json").is_file()
                  and (p / "recovery_ready.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint in {run / 'checkpoints'}")
    return max(candidates, key=lambda p: int(p.name.split("_")[-1])).resolve()


def load_settings(path, overrides):
    settings = OmegaConf.load(path)
    for argument in overrides:
        key, _, raw = argument.partition("=")
        value = OmegaConf.to_container(OmegaConf.from_dotlist(["value=" + raw]), resolve=False)["value"]
        OmegaConf.update(settings, key, value, merge=True)
    return OmegaConf.to_container(settings, resolve=True)


def execution_plan(settings):
    legacy = set(settings) == {"run", "defaults", "policies", "experiments", "execute"}
    if not legacy and set(settings) != {"run", "policies", "execute"}:
        raise ValueError("Suite sections must be run, policies, execute.")
    selected, resolved, verified = [], {}, set()
    for entry in settings["execute"]:
        if legacy:
            if set(entry) != {"policy", "experiment"}:
                raise ValueError("Historical execute entries must name policy and experiment.")
            config = OmegaConf.to_container(OmegaConf.merge(settings["defaults"],
                settings["experiments"][entry["experiment"]]), resolve=True)
            probe_name = {"shared_cell": "target-location/shared-coordinates",
                          "map_specific": "target-location/map-specific",
                          "map_cell": "target-location/map-and-cell"}.get(config["protocol"].get("decoder"))
        else:
            if set(entry) != {"policy", "probe", "repeats"}:
                raise ValueError("Each execute entry must name policy, probe and repeats.")
            if type(entry["repeats"]) is not int or entry["repeats"] < 1:
                raise ValueError("repeats must be a positive integer (decoder seeds 0 through repeats-1).")
            probe_name = entry["probe"]
            config = get_probe(probe_name).settings()
            config["probe"]["seeds"] = list(range(entry["repeats"]))
        probe = archived_probe() if legacy else get_probe(probe_name)
        name = entry["policy"]
        if name not in resolved:
            policy = settings["policies"][name]
            if not legacy and set(policy) != {"run", "checkpoint"}:
                raise ValueError("Suite policies contain only run and checkpoint; sampling belongs to the probe config.")
            run = resolve_path(policy["run"])
            checkpoint = latest_checkpoint(run) if policy["checkpoint"] == "latest" else resolve_path(policy["checkpoint"])
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            resolved[name] = {"run": str(run), "checkpoint": str(checkpoint)}
            if legacy and "map_source" in policy:
                resolved[name]["map_source"] = policy["map_source"]
        config = OmegaConf.to_container(OmegaConf.merge(config,
            {"run": {**settings["run"], "name": name},
             "policy": {k: v for k, v in resolved[name].items() if k != "run"}}), resolve=True)
        adapter = probe.adapter()
        key = adapter.plan_config(config)
        identity = (name, config["policy"].get("map_source", "training"))
        if identity not in verified:
            adapter.verify_policy(config)
            verified.add(identity)
        selected.append({**entry, "probe": probe_name, "collection_key": key, "config": config})
    if not selected:
        raise ValueError("The execute list is empty.")
    names = [(e["policy"], e["probe"]) for e in selected]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate policy/probe in execute list; use repeats for decoder repetitions.")
    return selected, resolved


def execute_suite(settings, root, plan):
    prepared = set()
    for index, job in enumerate(plan):
        config, key = job["config"], job["collection_key"]
        adapter = job_probe(job).adapter()
        run_dir = root / "runs" / config["run"]["name"]
        if (run_dir / "run.json").is_file():
            previous = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if previous.get("progress", {}).get("stage") == "complete" and (run_dir / "report.html").is_file():
                print(f"[suite] completed, skipping {display_name(previous)}", flush=True)
                continue
        collection_dir = root / "collections" / key
        if key not in prepared:
            adapter.prepare(config, collection_dir, [j for j in plan if j["collection_key"] == key])
            prepared.add(key)
        adapter.fit_and_report(config, run_dir, collection_dir)
        update_run(root, progress={"stage": "experiments", "completed": index + 1, "total": len(plan)})
    update_run(root, progress={"stage": "complete", "completed": len(plan), "total": len(plan)})


def main():
    options, overrides = {}, []
    for argument in sys.argv[1:]:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("Use config=<yaml>, resume=<suite folder>, or execute.<index>.repeats=<count>.")
        if key in ("config", "resume"):
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
        settings = load_settings(path, overrides)
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
