"""Shared modern collection, fitting and reporting for suite and compatibility CLIs."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from geo_experiments.artifacts import HISTORY_COLUMNS, archive, load_run, save_data, update_run
from geo_experiments.common import resolve_path
from geo_experiments.console import terminal_print as print


def _reuse_source(path, config, identity):
    from geo_experiments.cache import snapshot_identity_matches

    source = resolve_path(path)
    previous, shared, _ = load_run(source.parent if source.is_file() else source)
    if not snapshot_identity_matches(previous["provenance"]["identity"], identity):
        raise ValueError("Cached collection has a different checkpoint/config/runtime identity.")
    for section in ("dataset", "distance"):
        expected = {k: v for k, v in config[section].items() if k != "path"}
        actual = {k: v for k, v in previous["config"][section].items() if k != "path"}
        if expected != actual:
            raise ValueError(f"Cached collection has different {section} settings.")
    if previous["config"]["protocol"]["eligibility"] != config["protocol"]["eligibility"]:
        raise ValueError("Cached collection has a different episode eligibility rule.")
    return {k: v for k, v in shared.items() if not k.startswith("probes/")}, previous["collection"], previous.get("ambiguity_audit")


def prepare_collection(config, collection_dir, decoders):
    """Collect once, or reuse a completed snapshot with strict identity checks."""
    from omegaconf import OmegaConf
    from geo_experiments.collection import prepare_policy
    from geo_experiments.policy import map_levels
    from geo_experiments.labels import suite_geometry
    from geo_experiments.target_collection import collect_targets
    from geo_experiments.auditing import audit_targets, audit_is_current
    from geo_experiments.cache import snapshot_identity_matches

    collection_dir = Path(collection_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = collection_dir / "run.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    ready = existing.get("progress", {}).get("stage") == "complete"
    collected = existing.get("progress", {}).get("stage") in ("collected", "complete") and (collection_dir / "data.npz").is_file()
    if ready and not collected:
        raise FileNotFoundError(f"Completed collection archive is missing: {collection_dir / 'data.npz'}")
    if not collected:
        OmegaConf.save(OmegaConf.create(config), collection_dir / "config.yaml")
        update_run(collection_dir, config=config, kind="target_collection")
    model, level, identity = prepare_policy(config, None if collected else collection_dir, allow_suite=True)
    try:
        if collected:
            if not snapshot_identity_matches(existing.get("provenance", {}).get("identity"), identity):
                raise ValueError("Shared collection checkpoint/config/runtime changed. Start a new suite.")
            print(f"[collection] reusing {collection_dir}", flush=True)
        if not ready or not audit_is_current(existing.get("ambiguity_audit"), config, decoders):
            levels = map_levels(level)
            if collected:
                shared, collection = archive(collection_dir / "data.npz"), existing["collection"]
                audit = existing.get("ambiguity_audit")
            elif config["dataset"]["path"]:
                shared, collection, audit = _reuse_source(config["dataset"]["path"], config, identity)
            else:
                graph = suite_geometry(levels, config["distance"]["clearance_m"])
                shared, collection = collect_targets(model, levels, config, collection_dir, graph)
                audit = None
            # Retain collection before optional audit replay, including failures.
            save_data(collection_dir, shared)
            update_run(collection_dir, collection=collection, progress={"stage": "collected"})
            if not audit_is_current(audit, config, decoders):
                audit = audit_targets(model, levels, config, shared, decoders=decoders)
            save_data(collection_dir, shared)
            update_run(collection_dir, collection=collection, ambiguity_audit=audit, progress={"stage": "complete"})
    finally:
        del model
        import jax
        gc.collect()
        jax.clear_caches()


def fit_and_report(config, run_dir, collection_dir):
    """Fit one readout against the shared archive, preserving the report contract."""
    from omegaconf import OmegaConf
    from geo_experiments.experiments import fit_experiment
    from geo_experiments.report import generate_report

    run_dir, collection_dir = Path(run_dir), Path(collection_dir)
    manifest = json.loads((collection_dir / "run.json").read_text(encoding="utf-8"))
    shared = archive(collection_dir / "data.npz")
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), run_dir / "config.yaml")
    update_run(run_dir, config=config, kind="target_probe", history_columns=HISTORY_COLUMNS,
               shared_data=os.path.relpath(collection_dir / "data.npz", run_dir).replace("\\", "/"),
               collection=manifest["collection"], ambiguity_audit=manifest.get("ambiguity_audit"),
               provenance=manifest["provenance"], jobs=[], progress={"stage": "starting"})
    try:
        _, jobs = fit_experiment(config, shared, run_dir)
        generate_report(run_dir)
        update_run(run_dir, progress={"stage": "complete", "completed": len(jobs), "total": len(jobs)})
    except Exception as exc:
        update_run(run_dir, progress={"stage": "failed", "error": str(exc), "type": type(exc).__name__})
        raise
    finally:
        del shared
        import jax
        gc.collect()
        jax.clear_caches()


def execute_target(config, run_dir):
    """Compatibility single-readout command using the same suite machinery."""
    collection_dir = Path(run_dir) / "collection"
    prepare_collection(config, collection_dir, (config["protocol"]["decoder"],))
    fit_and_report(config, run_dir, collection_dir)
