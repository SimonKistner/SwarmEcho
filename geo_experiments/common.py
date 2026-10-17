"""Configuration, provenance and portable artifacts for the probe suite."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("finder_gru", "finder_message", "base_message")
TASKS = ("target_cell", "target_storey")
SOURCE_LABELS = {
    "finder_gru": "Finder GRU", "finder_message": "Finder message",
    "base_message": "Base stored message",
    "agent_gru": "Agent GRU", "agent_message": "Agent message",
}


def experiment_axes(config):
    return (("own_cell",), ("agent_gru", "agent_message")) if config.get("experiment") == "own_cell" else (TASKS, SOURCES)


def resolve_path(value):
    raw = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", raw):
        raw = f"/mnt/{raw[0].lower()}/{raw[3:]}"
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(config):
    # Existing experiment YAML files gain the audit defaults on the next run.
    config.setdefault("audit", {"enabled": True, "samples": 5000, "every": 1, "batch_size": 32})
    expected = {
        "run": {"name", "output_root", "timezone"},
        "policy": {"checkpoint", "level"},
        "dataset": {"train_samples", "validation_samples", "test_samples", "eval_batch_size",
                    "max_episodes", "seed", "path", "split_seed"},
        "probe": {"hidden_sizes", "seeds", "epochs", "batch_size", "learning_rate", "weight_decay",
                  "patience", "min_delta"},
        "distance": {"clearance_m"}, "report": {"dpi"},
        "audit": {"enabled", "samples", "every", "batch_size"},
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
    if type(config["audit"]["enabled"]) is not bool:
        raise ValueError("audit.enabled must be boolean.")
    if not own and d["max_episodes"] < sample_count(d):
        raise ValueError("dataset.max_episodes must cover train + validation + test samples.")
    if not isinstance(config["policy"]["level"], str) or not config["policy"]["level"].strip():
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


def sample_count(config):
    return sum(config[f"{split}_samples"] for split in ("train", "validation", "test"))


def split_episodes(labels, config):
    """Cell-stratified splits with exact requested sizes, shared by all fits."""
    if len(labels) != sample_count(config):
        raise ValueError("Dataset size must equal train + validation + test samples.")
    rng = np.random.default_rng(config["split_seed"])
    classes, counts = np.unique(labels, return_counts=True)
    if config["train_samples"] < len(classes):
        raise ValueError("Training sample count must cover every observed target cell.")

    def allocate(capacity, total, minimum):
        # Largest-remainder allocation keeps class proportions while meeting
        # exact totals. Training gets at least one episode of each seen cell.
        available = capacity - minimum
        remaining = total - int(minimum.sum())
        ideal = available * (remaining / available.sum()) if available.sum() else available.astype(float)
        allocation = minimum + np.floor(ideal).astype(int)
        remainder = total - int(allocation.sum())
        eligible = np.flatnonzero(allocation < capacity)
        eligible = rng.permutation(eligible)
        order = eligible[np.argsort(-(ideal[eligible] - np.floor(ideal[eligible])), kind="stable")]
        allocation[order[:remainder]] += 1
        return allocation

    train_counts = allocate(counts, config["train_samples"], np.ones(len(classes), dtype=int))
    remaining = counts - train_counts
    validation_counts = allocate(remaining, config["validation_samples"], np.zeros(len(classes), dtype=int))
    splits = {name: [] for name in ("train", "validation", "test")}
    for label, train, validation in zip(classes, train_counts, validation_counts, strict=True):
        indices = rng.permutation(np.flatnonzero(labels == label))
        splits["train"].extend(indices[:train])
        splits["validation"].extend(indices[train:train + validation])
        splits["test"].extend(indices[train + validation:])
    return {name: np.sort(values).astype(np.int32) for name, values in splits.items()}
