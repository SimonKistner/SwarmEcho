"""Target-location axes and validation, including historical own-cell compatibility."""
import re
import numpy as np
from geo_experiments.shared.common import sample_count, uses_bank_maps

SOURCES = ("finder_gru", "finder_message", "base_message")
TASKS = ("target_cell", "target_storey")
SOURCE_LABELS = {
    "finder_gru": "Finder GRU", "finder_message": "Finder message",
    "base_message": "Base stored message",
    "agent_gru": "Agent GRU", "agent_message": "Agent message",
}


def experiment_axes(config):
    if config.get("experiment") == "own_cell":
        return ("own_cell",), ("agent_gru", "agent_message")
    return (TASKS if config.get("protocol", {}).get("target_storey", True) else ("target_cell",)), SOURCES


def validate_config(config):
    # Existing experiment YAML files gain the audit defaults on the next run.
    config.setdefault("audit", {"enabled": True, "samples": 5000, "every": 1, "batch_size": 32})
    config["audit"].setdefault("mode", "full")
    config["audit"].setdefault("duplicate_threshold", 0.01)
    config.setdefault("protocol", {})
    for key, value in {"decoder": "legacy", "target_storey": True, "eligibility": "success"}.items():
        config["protocol"].setdefault(key, value)
    config["policy"].setdefault("from_run_config", False)
    expected = {
        "run": {"name", "output_root", "timezone"},
        "policy": {"checkpoint", "level", "from_run_config"},
        "dataset": {"train_samples", "validation_samples", "test_samples", "eval_batch_size",
                    "max_episodes", "seed", "path", "split_seed"},
        "probe": {"hidden_sizes", "seeds", "epochs", "batch_size", "learning_rate", "weight_decay",
                  "patience", "min_delta"},
        "distance": {"clearance_m"}, "report": {"dpi"},
        "audit": {"enabled", "samples", "every", "batch_size", "mode", "duplicate_threshold"},
        "protocol": {"decoder", "target_storey", "eligibility"},
    }
    own = config.get("experiment") == "own_cell"
    if "experiment" in config:
        if config["experiment"] not in ("own_cell", "target"):
            raise ValueError("experiment must be own_cell or target.")
        expected["experiment"] = None
    if own:
        config["sampling"].setdefault("uniform_cells", True)
        config["sampling"].setdefault("exclude_cells", [])
        if config["audit"]["every"] != 1:
            raise ValueError("Own-cell prefix filtering requires audit.every=1 (full histories).")
        expected["sampling"] = {"buffer_seconds", "uniform_cells", "exclude_cells"}
    if "map_source" in config["policy"]:
        expected["policy"].add("map_source")
        if config["policy"]["map_source"] not in ("training", "validation", "compatible_pool"):
            raise ValueError("policy.map_source must be training, validation or compatible_pool.")
        if uses_bank_maps(config) and (
                not config["policy"]["from_run_config"] or config["protocol"]["decoder"] == "legacy" or own):
            raise ValueError("Validation maps require modern target probes and policy.from_run_config=true.")
    if set(config) != set(expected):
        raise ValueError(f"Expected config sections {sorted(expected)}; received {sorted(config)}")
    for section, keys in expected.items():
        if keys is None:
            continue
        if set(config[section]) != keys:
            raise ValueError(f"Invalid keys in {section}: {sorted(set(config[section]) ^ keys)}")
    name = config["run"]["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
        raise ValueError("run.name must be 1–101 filename-safe letters, numbers, dots, underscores or hyphens.")
    for section, keys in (("dataset", ("train_samples", "validation_samples", "test_samples", "eval_batch_size", "max_episodes")),
                          ("probe", ("epochs", "batch_size", "patience")), ("report", ("dpi",)),
                          ("audit", ("samples", "every", "batch_size"))):
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    d, p = config["dataset"], config["probe"]
    if config["protocol"]["decoder"] not in ("legacy", "map_specific", "map_cell", "shared_cell"):
        raise ValueError("protocol.decoder must be legacy, map_specific, map_cell or shared_cell.")
    if config["protocol"]["eligibility"] not in ("success", "delivered"):
        raise ValueError("protocol.eligibility must be success or delivered.")
    if type(config["protocol"]["target_storey"]) is not bool or type(config["policy"]["from_run_config"]) is not bool:
        raise ValueError("target_storey and from_run_config must be boolean.")
    if config["audit"]["mode"] not in ("full", "conditional"):
        raise ValueError("audit.mode must be full or conditional.")
    if not 0 <= config["audit"]["duplicate_threshold"] <= 1:
        raise ValueError("audit.duplicate_threshold must be a fraction in [0, 1].")
    if type(config["audit"]["enabled"]) is not bool:
        raise ValueError("audit.enabled must be boolean.")
    if not own and d["max_episodes"] < sample_count(d):
        raise ValueError("dataset.max_episodes must cover train + validation + test samples.")
    if (not isinstance(config["policy"]["level"], str) or not config["policy"]["level"].strip()) and not (
            config["policy"]["from_run_config"] and config["policy"]["level"] is None):
        raise ValueError("policy.level must name a normal level definition.")
    if not p["hidden_sizes"] or any(type(x) is not int or x < 1 for x in p["hidden_sizes"]):
        raise ValueError("probe.hidden_sizes must be positive integers.")
    if not p["seeds"] or len(set(p["seeds"])) != len(p["seeds"]) or any(type(x) is not int or x < 0 for x in p["seeds"]):
        raise ValueError("probe.seeds must be distinct nonnegative integers.")
    for key in ("seed", "split_seed"):
        if type(d[key]) is not int or d[key] < 0:
            raise ValueError(f"dataset.{key} must be a nonnegative integer.")
    for key in ("learning_rate", "weight_decay", "min_delta"):
        if not np.isfinite(p[key]) or p[key] < 0 or (key == "learning_rate" and p[key] == 0):
            raise ValueError(f"Invalid probe.{key}.")
    clearance = config["distance"]["clearance_m"]
    if clearance is not None and (not np.isfinite(clearance) or clearance < 0):
        raise ValueError("distance.clearance_m must be null or nonnegative.")
    if own:
        buffer = config["sampling"]["buffer_seconds"]
        excluded = config["sampling"]["exclude_cells"]
        if (not isinstance(excluded, list) or any(not isinstance(cell, list) or len(cell) != 3
                or any(type(x) is not int for x in cell) for cell in excluded)):
            raise ValueError("sampling.exclude_cells must be a list of integer XYZ triples in absolute map-grid coordinates.")
        if len({tuple(cell) for cell in excluded}) != len(excluded):
            raise ValueError("sampling.exclude_cells must not contain duplicates.")
        if type(config["sampling"]["uniform_cells"]) is not bool:
            raise ValueError("sampling.uniform_cells must be boolean.")
        if not np.isfinite(buffer) or buffer < 0:
            raise ValueError("sampling.buffer_seconds must be nonnegative.")
        if d["path"] is not None:
            raise ValueError("Own-cell collection currently requires dataset.path=null; reports can be regenerated from saved data.")


