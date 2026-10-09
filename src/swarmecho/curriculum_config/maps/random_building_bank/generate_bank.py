"""Build and extend a parallel, on-disk pool of random buildings and roadmaps.

Run: uv run python -m swarmecho.curriculum_config.maps.random_building_bank.generate_bank
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import uuid

import numpy as np
import yaml

from swarmecho.core.config import load_level
from swarmecho.core.terminal import terminal_print
from swarmecho.env.buildings import target_spawn_boxes
from swarmecho.env.random_buildings import GENERATOR_VERSION, generate_building, map_seed, save_generated_map
from swarmecho.env.roadmap_cpu import building_roadmap


# Change this constant to change the manual batch size and new-pool minimum.
DEFAULT_POOL_SIZE = 2000
POOL_FORMAT = "swarmecho-random-building-bank/v1"
POOL_ROOT = Path(__file__).resolve().parent / "pools"
_RECORD_KEYS = ("solid_min", "solid_max", "vertices", "distances", "base_position",
                "target_lower", "target_upper", "target_volumes",
                "base_lower", "base_upper", "base_volumes")
_SOURCE_FILES = ("buildings.py", "random_buildings.py", "roadmap_cpu.py")


class CorruptPoolError(RuntimeError):
    """A published pool cannot safely be extended."""


def _source_digest():
    """Invalidate saved roadmaps when their generator or geometry code changes."""
    source_dir = Path(__file__).resolve().parents[3] / "env"
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((source_dir / name).read_bytes())
    return digest.hexdigest()


def required_settings(level):
    generation = asdict(level.random_buildings)
    for runtime_option in ("enabled", "save_training_maps", "load_maps_from_bank"):
        generation.pop(runtime_option)
    cfg = level.env
    plan = (level.reward.chain_reward_system == "obstacle_geodesic"
            or level.reward.enable_chain_efficiency_reward)
    return {
        "format": POOL_FORMAT,
        "generator_version": GENERATOR_VERSION,
        "source_sha256": _source_digest(),
        "random_buildings": generation,
        "roadmap": {
            "plan": plan,
            "approach": cfg.roadmap_approach,
            "node_density": cfg.roadmap_node_density,
            "merge_wall_end_nodes": cfg.roadmap_merge_wall_end_nodes,
            "corner_bonus_m": cfg.roadmap_corner_bonus_m if level.training.minimum_geodesic_separation else 0.,
            "drone_radius": cfg.drone_radius,
            "obstacle_planning_clearance_m": cfg.obstacle_planning_clearance_m,
        },
        "spawn": {"target_wall_buffer_fraction": cfg.target_wall_buffer_fraction},
    }


def _settings_digest(settings):
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _map_digest(data):
    """Hash map content without its display name or generation provenance."""
    geometry = {key: value for key, value in data.items() if key not in {"name", "generation"}}
    payload = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _generation_lock():
    """Serialize pool creation/extension across concurrent training processes."""
    POOL_ROOT.mkdir(parents=True, exist_ok=True)
    with (POOL_ROOT / ".generation.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _differences(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            result.extend(_differences(expected.get(key, "<missing>"), actual.get(key, "<missing>"),
                                       f"{prefix}.{key}" if prefix else key))
        return result
    if expected != actual:
        return [f"{prefix}: bank={actual!r}, requested={expected!r}"]
    return []


def _read_metadata(folder):
    try:
        data = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(data, dict) or data.get("format") != POOL_FORMAT or not data.get("complete")
            or type(data.get("count")) is not int or data["count"] < 1
            or not isinstance(data.get("maps"), list) or len(data["maps"]) != data["count"]):
        return None
    if any(not isinstance(item, dict) or not isinstance(item.get("id"), str)
           or type(item.get("seed")) is not int
           for item in data["maps"]):
        return None
    if any(item["id"] != f"map_{index:05d}" for index, item in enumerate(data["maps"])):
        return None
    if any("hash" in item and not (isinstance(item["hash"], str)
                                   and re.fullmatch(r"[0-9a-f]{64}", item["hash"]))
           for item in data["maps"]):
        return None
    if (not isinstance(data.get("settings"), dict)
            or data.get("settings_sha256") != _settings_digest(data["settings"])):
        return None
    return data


def find_compatible_pool(level, required_count=0, *, exclude=None, log=terminal_print):
    """Find a complete pool and explain every incompatible pool's settings."""
    wanted = required_settings(level)
    if not POOL_ROOT.exists():
        return None
    compatible = []
    for folder in sorted(path for path in POOL_ROOT.iterdir() if path.is_dir()):
        if folder.name.endswith(".incomplete"):
            log(f"Ignoring unfinished map pool: {folder.name}")
            continue
        if exclude is not None and folder.resolve() == Path(exclude).resolve():
            continue
        metadata = _read_metadata(folder)
        if metadata is None:
            message = f"Ignoring incomplete or invalid map pool: {folder.name}"
            terminal_print(f"[BANK WARNING] {message}")
            continue
        disagreements = _differences(wanted, metadata.get("settings", {}))
        if metadata["count"] < required_count:
            disagreements.append(f"count: bank={metadata['count']}, required={required_count}")
        if disagreements:
            message = f"Map pool {folder.name} is incompatible: " + "; ".join(disagreements)
            terminal_print(f"[BANK WARNING] {message}")
        else:
            compatible.append((folder, metadata))
    return max(compatible, key=lambda pair: (pair[1]["count"], pair[0].name)) if compatible else None


def _compile_one(index, child_seed, expected_hash, settings, cfg, plan, corner_bonus_m, folder):
    """Worker: write one YAML map and one unpadded numeric record."""
    name = f"map_{index:05d}"
    data, building = generate_building(settings, child_seed, name=name,
        drone_clearance=cfg.drone_radius + cfg.obstacle_planning_clearance_m)
    if _map_digest(data) != expected_hash:
        raise ValueError(f"Generated map {index} changed between deduplication and roadmap compilation.")
    graph = building_roadmap(building, cfg, plan=plan, corner_bonus_m=corner_bonus_m)
    if plan and (not len(graph.vertices) or np.any(graph.distances[0] >= 1e6)):
        raise ValueError(f"Generated map {index} has a disconnected planning graph (seed {child_seed}).")
    target = target_spawn_boxes(building, cfg)
    base = target_spawn_boxes(building, cfg, include_excluded=True)
    record = {
        "solid_min": graph.solid_min, "solid_max": graph.solid_max,
        "vertices": graph.vertices, "distances": graph.distances,
        "base_position": building.base_position_m,
    }
    for prefix, boxes in (("target", target), ("base", base)):
        for suffix, value in zip(("lower", "upper", "volumes"), boxes):
            record[f"{prefix}_{suffix}"] = value
    if corner_bonus_m:
        record["corner_distances"] = graph.corner_distances
    save_generated_map(data, folder / "maps" / f"{name}.yaml")
    np.savez(folder / "records" / f"{name}.npz", **record)
    return {"id": name, "seed": child_seed, "hash": expected_hash, "nodes": len(graph.vertices)}


def _known_map_hashes(folder, metadata, *, log):
    """Upgrade older manifests while preserving every published map ID."""
    hashes = set()
    existing_duplicates = 0
    for item in metadata["maps"]:
        record = folder / "records" / f"{item['id']}.npz"
        if not record.is_file():
            raise FileNotFoundError(f"Roadmap record is missing: {record}")
        source = folder / "maps" / f"{item['id']}.yaml"
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        digest = _map_digest(data)
        if item.get("hash") not in (None, digest):
            raise ValueError(f"Map definition changed after its roadmap was compiled: {source}")
        item["hash"] = digest
        if digest in hashes:
            existing_duplicates += 1
        hashes.add(digest)
    if existing_duplicates:
        log(f"Pool {folder.name} already contains {existing_duplicates} duplicate maps; new batches will not add more.")
    return hashes


def _write_metadata(folder, metadata):
    temporary = folder / f"metadata.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(folder / "metadata.json")


def _append_batch(level, folder, metadata, count, seed, workers, *, new_pool, fill_unique, log):
    """Append unique maps; publish new IDs only after all records are ready."""
    settings = metadata["settings"]
    try:
        seen_hashes = _known_map_hashes(folder, metadata, log=log) if metadata["maps"] else set()
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
        raise CorruptPoolError(f"Cannot read existing map definitions in {folder}") from exc
    seen_seeds = {item["seed"] for item in metadata["maps"]}
    if seed is None:
        prior = {item.get("seed") for item in metadata.get("generations", [])}
        prior.add(metadata.get("seed"))
        seed = secrets.randbits(32)
        while seed in prior:
            seed = secrets.randbits(32)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    start = metadata["count"]
    planned = []
    skipped = 0
    attempts = 0
    # Explicit seeds cover exactly count candidate positions. Automatically
    # chosen seeds keep sampling until count genuinely new maps are planned.
    limit = max(count * 10, count + 1000) if fill_unique else count
    while attempts < limit and len(planned) < count:
        child_seed = map_seed(seed, attempts)
        attempts += 1
        if child_seed in seen_seeds:
            skipped += 1
            continue
        name = f"map_{start + len(planned):05d}"
        data, _ = generate_building(level.random_buildings, child_seed, name=name,
            drone_clearance=level.env.drone_radius + level.env.obstacle_planning_clearance_m)
        digest = _map_digest(data)
        if digest in seen_hashes:
            skipped += 1
            continue
        seen_seeds.add(child_seed)
        seen_hashes.add(digest)
        planned.append((start + len(planned), child_seed, digest))
        if len(planned) == 1 or len(planned) % 100 == 0 or len(planned) == count:
            log(f"Selected {len(planned)}/{count} unique maps for compilation; skipped {skipped} duplicates")
    if fill_unique and len(planned) < count:
        raise RuntimeError(f"Could find only {len(planned)} unique maps after {attempts} candidates.")
    records = [None] * len(planned)
    if planned:
        log(f"Appending {len(planned)} unique maps to {folder} with {workers} CPU workers")
        plan = settings["roadmap"]["plan"]
        corner_bonus_m = settings["roadmap"]["corner_bonus_m"]
        # Spawn avoids inheriting the training process's JAX runtime in workers.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_compile_one, index, child_seed, digest,
                                       level.random_buildings, level.env, plan, corner_bonus_m, folder): position
                       for position, (index, child_seed, digest) in enumerate(planned)}
            for completed, future in enumerate(as_completed(futures), 1):
                position = futures[future]
                try:
                    records[position] = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"Map pool extension failed; unpublished files remain in {folder}") from exc
                if completed == 1 or completed % 100 == 0 or completed == len(planned):
                    log(f"Compiled pool batch {completed}/{len(planned)}; latest {records[position]['nodes']} roadmap nodes")
    if "generations" not in metadata and metadata["maps"]:
        metadata["generations"] = [{"seed": metadata.get("seed"), "created_utc": metadata.get("created_utc"),
                                    "attempted": metadata["count"], "added": metadata["count"], "legacy": True}]
    metadata.setdefault("generations", []).append({
        "seed": seed, "created_utc": stamp, "attempted": attempts,
        "requested": count, "added": len(records), "duplicates_skipped": skipped,
    })
    metadata["maps"].extend(records)
    metadata["count"] = len(metadata["maps"])
    metadata["complete"] = True
    _write_metadata(folder, metadata)
    if new_pool:
        final = folder.with_name(folder.name.removesuffix(".incomplete"))
        folder.rename(final)
        folder = final
    log(f"Map pool ready: {folder} ({metadata['count']} maps; added {len(records)}, skipped {skipped} duplicates)")
    return folder, metadata


def _new_pool(level, seed):
    settings = required_settings(level)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = _settings_digest(settings)[:12]
    name = f"pool_{stamp}_{fingerprint}_seed{seed}_{uuid.uuid4().hex[:6]}"
    folder = POOL_ROOT / f"{name}.incomplete"
    (folder / "maps").mkdir(parents=True)
    (folder / "records").mkdir()
    metadata = {"format": POOL_FORMAT, "complete": False, "created_utc": stamp,
                "seed": seed, "count": 0, "settings": settings,
                "settings_sha256": _settings_digest(settings), "maps": []}
    return folder, metadata


def build_pool(level, *, seed=None, count=DEFAULT_POOL_SIZE, workers=None, avoid_pool=None, log=terminal_print):
    """Add a manual batch to a compatible pool, or create one if needed."""
    if not level.random_buildings.enabled:
        raise ValueError("A bank requires random_buildings.enabled=true.")
    if type(count) is not int or count < 1:
        raise ValueError("Pool count must be positive.")
    workers = min(4, max(1, (os.cpu_count() or 2) - 1)) if workers is None else workers
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer.")
    with _generation_lock():
        found = find_compatible_pool(level, exclude=avoid_pool, log=log)
        if found is None:
            # The batch seed is picked before naming the new pool.
            batch_seed = secrets.randbits(32) if seed is None else seed
            folder, metadata = _new_pool(level, batch_seed)
            return _append_batch(level, folder, metadata, count, batch_seed, workers,
                                 new_pool=True, fill_unique=seed is None, log=log)
        folder, metadata = found
        try:
            return _append_batch(level, folder, metadata, count, seed, workers,
                                 new_pool=False, fill_unique=seed is None, log=log)
        except CorruptPoolError as exc:
            terminal_print(f"[BANK WARNING] {exc}; creating a new pool.")
            batch_seed = secrets.randbits(32) if seed is None else seed
            folder, metadata = _new_pool(level, batch_seed)
            return _append_batch(level, folder, metadata, count, batch_seed, workers,
                                 new_pool=True, fill_unique=seed is None, log=log)


def ensure_pool(level, required_count, *, log=terminal_print):
    """Reuse a sufficient pool; otherwise extend a matching pool to capacity."""
    if type(required_count) is not int or required_count < 1:
        raise ValueError("required_count must be positive.")
    with _generation_lock():
        found = find_compatible_pool(level, log=log)
        if found is not None and found[1]["count"] >= required_count:
            log(f"Using compatible map pool: {found[0]} ({found[1]['count']} maps)")
            return found
        workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        if found is None:
            message = "No compatible map pool found; generating one before training."
            terminal_print(f"[BANK WARNING] {message}")
            seed = secrets.randbits(32)
            folder, metadata = _new_pool(level, seed)
            return _append_batch(level, folder, metadata, max(DEFAULT_POOL_SIZE, required_count),
                                 seed, workers, new_pool=True, fill_unique=True, log=log)
        folder, metadata = found
        target = max(DEFAULT_POOL_SIZE, required_count)
        additional = target - metadata["count"]
        log(f"Extending compatible map pool {folder.name} from {metadata['count']} to {target} maps")
        try:
            return _append_batch(level, folder, metadata, additional, None, workers,
                                 new_pool=False, fill_unique=True, log=log)
        except CorruptPoolError as exc:
            terminal_print(f"[BANK WARNING] {exc}; generating a new pool.")
            seed = secrets.randbits(32)
            folder, metadata = _new_pool(level, seed)
            return _append_batch(level, folder, metadata, target, seed, workers,
                                 new_pool=True, fill_unique=True, log=log)


def select_maps(folder, metadata, seed, training_count, evaluation_count=0):
    """Select reproducible, distinct training and evaluation maps."""
    needed = training_count + evaluation_count
    if needed > metadata["count"]:
        raise ValueError(f"Pool has {metadata['count']} maps but {needed} distinct maps are required.")
    sequence = np.random.SeedSequence([int(seed), 0xB02B])
    indices = np.random.default_rng(sequence).permutation(metadata["count"])[:needed]
    choices = [metadata["maps"][int(index)] for index in indices]
    for item in choices:
        if not re.fullmatch(r"map_\d{5,}", item["id"]):
            raise ValueError(f"Invalid map ID in {folder / 'metadata.json'}: {item['id']!r}")
    return choices[:training_count], choices[training_count:]


def load_records(folder, choices, *, require_corner=False, include_target=True):
    """Read only selected maps; values retain their natural, unpadded shapes."""
    maps = []
    for item in choices:
        source = folder / "records" / f"{item['id']}.npz"
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) - (set(_RECORD_KEYS) | {"corner_distances"}):
                raise ValueError(f"Unexpected record keys in {source}")
            if not set(_RECORD_KEYS).issubset(archive.files):
                raise ValueError(f"Incomplete roadmap record: {source}")
            if require_corner and "corner_distances" not in archive.files:
                raise ValueError(f"Missing corner-distance matrix in {source}")
            keys = (_RECORD_KEYS if include_target else
                    ("solid_min", "solid_max", "base_position", "base_lower", "base_upper", "base_volumes"))
            record = {key: np.asarray(archive[key], dtype=np.float32) for key in keys}
            if include_target:
                record["corner_distances"] = (np.asarray(archive["corner_distances"], dtype=np.float32)
                                              if "corner_distances" in archive.files else record["distances"])
        graph_valid = True
        if include_target:
            nodes = len(record["vertices"])
            graph_valid = (record["vertices"].shape == (nodes, 3)
                           and record["distances"].shape == (nodes, nodes)
                           and record["corner_distances"].shape == (nodes, nodes))
        prefixes = ("target", "base") if include_target else ("base",)
        boxes_valid = all(
            record[f"{prefix}_lower"].shape == record[f"{prefix}_upper"].shape
            and record[f"{prefix}_lower"].ndim == 2
            and record[f"{prefix}_lower"].shape[1] == 3
            and record[f"{prefix}_volumes"].shape == (len(record[f"{prefix}_lower"]),)
            for prefix in prefixes
        )
        if (not graph_valid or not boxes_valid
                or record["solid_min"].shape != record["solid_max"].shape
                or record["solid_min"].ndim != 2 or record["solid_min"].shape[1] != 3
                or record["base_position"].shape != (3,)):
            raise ValueError(f"Invalid roadmap array shape in {source}")
        maps.append(record)
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="B02a_random_buildings_find_only")
    parser.add_argument("--seed", type=int, help="Override the pool generation seed")
    parser.add_argument("--workers", type=int, help="Parallel CPU workers (default: up to 4)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Level override")
    args = parser.parse_args()
    level = load_level(args.level, args.set)
    build_pool(level, seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
