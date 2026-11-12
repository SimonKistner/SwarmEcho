"""One suite adapter for composed input, model, target and report blocks."""
import hashlib
import json

from geo_experiments.shared.common import uses_bank_maps
from geo_experiments.shared.definitions import validate_runtime
from geo_experiments.shared.naming import run_slug


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
    validate_runtime(config)
    definition = config["definition"]
    collection = {key: config[key] for key in ("policy", "dataset", "audit", "distance")}
    collection["eligibility"] = config["protocol"]["eligibility"]
    collection["request"] = {key: definition[key] for key in ("target", "inputs", "output", "sampling")}
    if uses_bank_maps(config) and config.get("experiment") != "path":
        collection["training_class_coverage"] = "pooled_coordinates"
    return hashlib.sha256(json.dumps(collection, sort_keys=True).encode()).hexdigest()[:16]


def prepare(config, collection_dir, jobs):
    validate_runtime(config)
    from geo_experiments.shared.execution import prepare_collection
    prepare_collection(config, collection_dir, ("shared_cell",))


def fit_and_report(config, run_dir, collection_dir):
    validate_runtime(config)
    from geo_experiments.shared.execution import fit_and_report as execute
    execute(config, run_dir, collection_dir)
