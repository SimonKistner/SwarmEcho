"""Suite integration for categorical target-location readouts."""
import hashlib
import json

from geo_experiments.shared.common import uses_bank_maps
from geo_experiments.shared.naming import run_slug
from geo_experiments.archive.target_location.config import validate_config
from geo_experiments.archive.target_location.labels import DECODERS


def verify_policy(config):
    if uses_bank_maps(config):
        from geo_experiments.shared.validation_maps import resolve_probe_maps
        from geo_experiments.shared.common import resolve_path
        from geo_experiments.shared.console import terminal_print
        checkpoint = resolve_path(config["policy"]["checkpoint"])
        paths, metadata = resolve_probe_maps(checkpoint, config["policy"]["map_source"])
        terminal_print(f"[suite] {checkpoint.parent.parent.name}: {len(paths)} verified "
                       f"{metadata.get('source', 'original validation')} maps", flush=True)


def plan_config(config):
    config["policy"].setdefault("level", None)
    config["run"]["name"] = run_slug(config)
    validate_config(config)
    if config.get("experiment", "target") != "target" or config["protocol"]["decoder"] not in DECODERS:
        raise ValueError("Target-location suite probes require a modern target decoder.")
    if not config["policy"]["from_run_config"] or config["dataset"]["path"] is not None:
        raise ValueError("Suite policies use their saved training config; use resume=<suite folder> to reuse collection.")
    collection = {k: config[k] for k in ("policy", "dataset", "audit", "distance")}
    collection["eligibility"] = config["protocol"]["eligibility"]
    if config["protocol"]["decoder"] == "shared_cell" and uses_bank_maps(config):
        collection["training_class_coverage"] = "pooled_coordinates"
    # Preserve existing target collection keys and archive semantics.
    return hashlib.sha256(json.dumps(collection, sort_keys=True).encode()).hexdigest()[:16]


def prepare(config, collection_dir, jobs):
    from geo_experiments.archive.target_location.execution import prepare_collection
    decoders = tuple(sorted({job["config"]["protocol"]["decoder"] for job in jobs}))
    prepare_collection(config, collection_dir, decoders)


def fit_and_report(config, run_dir, collection_dir):
    from geo_experiments.archive.target_location.execution import fit_and_report as execute
    execute(config, run_dir, collection_dir)
