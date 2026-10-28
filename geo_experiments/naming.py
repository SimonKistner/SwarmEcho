"""Human report labels describe the investigated training run and readout task."""
from __future__ import annotations

from datetime import datetime
import hashlib
import re
from pathlib import Path


def training_run_name(manifest):
    config = manifest["config"]
    checkpoint = manifest.get("provenance", {}).get("checkpoint") or config.get("policy", {}).get("checkpoint", "")
    parts = str(checkpoint).replace("\\", "/").rstrip("/").split("/")
    if "checkpoints" in parts:
        index = len(parts) - 1 - parts[::-1].index("checkpoints")
        if index > 0:
            return parts[index - 1]
    return manifest.get("provenance", {}).get("level") or config["policy"]["level"]


def experiment_description(config, collection=None):
    collection = collection or {}
    if config.get("experiment") == "own_cell":
        balanced = collection.get("uniform_cell_buckets", config.get("sampling", {}).get("uniform_cells", False))
        if balanced:
            count, total = collection.get("eligible_cell_count"), collection.get("total_cell_count")
            subset = f"{count} of {total} cells" if count is not None and total is not None else "cell sampling"
            return "Own cell: balanced " + subset
        return "Own cell: visitation-weighted sampling"
    protocol = config.get("protocol", {})
    task = "Target cell and storey" if protocol.get("target_storey", True) else "Target cell"
    decoder = protocol.get("decoder", "legacy")
    description = {"map_specific": "one decoder per map", "map_cell": "joint map and cell decoder",
                   "shared_cell": "one shared coordinate decoder", "legacy": "single-map decoding"}[decoder]
    return f"{task}: {description}"


def display_name(manifest):
    return training_run_name(manifest) + " — " + experiment_description(manifest["config"], manifest.get("collection"))


def experiment_title(manifest):
    """Short sidebar heading; the investigated run is shown on its own line."""
    config = manifest["config"]
    if config.get("experiment") == "own_cell":
        balanced = manifest.get("collection", {}).get("uniform_cell_buckets",
            config.get("sampling", {}).get("uniform_cells", False))
        return "Balanced Own Cell Decoder" if balanced else "Visitation-Weighted Own Cell Decoder"
    protocol = config.get("protocol", {})
    return {"map_specific": "One Decoder per Map", "map_cell": "Joint Map and Cell Decoder",
            "shared_cell": "One Shared Coordinate Decoder",
            "legacy": "Single-Map Target Cell and Storey Decoder" if protocol.get("target_storey", True)
                      else "Single-Map Target Cell Decoder"}[protocol.get("decoder", "legacy")]


def report_timestamp(run_dir):
    """Use creation time in the run/suite folder, unaffected by report refreshes."""
    if run_dir is not None:
        for path in (Path(run_dir), *Path(run_dir).parents):
            match = re.search(r"_(\d{8})_(\d{6})_(\d{6})$", path.name)
            if match:
                return ''.join(match.groups())
    return ""


def report_subtitle(manifest, run_dir=None):
    checkpoint = manifest.get("provenance", {}).get("checkpoint") or manifest["config"].get("policy", {}).get("checkpoint", "")
    parts = [str(checkpoint).replace("\\", "/").rstrip("/").split("/")[-1]] if checkpoint else []
    maps = manifest.get("provenance", {}).get("identity", {}).get("maps", [])
    if len(maps) > 1:
        parts.append(f"{len(maps)} maps")
    count = manifest.get("collection", {}).get("num_samples")
    if count is not None:
        parts.append(f"{count:,} samples")
    timestamp = report_timestamp(run_dir)
    if timestamp:
        date = datetime.strptime(timestamp, "%Y%m%d%H%M%S%f")
        parts.append(date.strftime("%Y-%m-%d %H:%M"))
    return " · ".join(parts)


def run_slug(config):
    """Readable future folder names; existing frozen plan identifiers stay valid."""
    source = training_run_name({"config": config})
    task = "own_cell" if config.get("experiment") == "own_cell" else "target_cell"
    if task == "target_cell" and config.get("protocol", {}).get("target_storey", True):
        task += "_and_storey"
    decoder = config.get("protocol", {}).get("decoder", "legacy")
    suffix = {"map_specific": "per_map_decoders", "map_cell": "joint_map_and_cell_decoder",
              "shared_cell": "shared_coordinate_decoder", "legacy": "single_map_decoder"}[decoder]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source}_{task}_{suffix}").strip("_.-")
    if len(name) > 101:
        name = name[:92] + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]
    return name
