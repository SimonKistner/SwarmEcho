"""Configuration, provenance and portable artifacts for the probe suite."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def uses_bank_maps(config):
    return config["policy"].get("map_source", "training") in ("validation", "compatible_pool")


def resolve_path(value):
    raw = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[A-Za-z]/", raw):
        raw = raw[5].upper() + ":/" + raw[7:]
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


def sample_count(config):
    return sum(config[f"{split}_samples"] for split in ("train", "validation", "test"))


def split_episodes(labels, config, *, require_class_coverage=True):
    """Cell-stratified splits with exact requested sizes, shared by all fits."""
    if len(labels) != sample_count(config):
        raise ValueError("Dataset size must equal train + validation + test samples.")
    rng = np.random.default_rng(config["split_seed"])
    classes, counts = np.unique(labels, return_counts=True)
    if require_class_coverage and config["train_samples"] < len(classes):
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

    minimum = np.ones(len(classes), dtype=int) if require_class_coverage else np.zeros(len(classes), dtype=int)
    train_counts = allocate(counts, config["train_samples"], minimum)
    remaining = counts - train_counts
    validation_counts = allocate(remaining, config["validation_samples"], np.zeros(len(classes), dtype=int))
    splits = {name: [] for name in ("train", "validation", "test")}
    for label, train, validation in zip(classes, train_counts, validation_counts, strict=True):
        indices = rng.permutation(np.flatnonzero(labels == label))
        splits["train"].extend(indices[:train])
        splits["validation"].extend(indices[train:train + validation])
        splits["test"].extend(indices[train + validation:])
    return {name: np.sort(values).astype(np.int32) for name, values in splits.items()}
