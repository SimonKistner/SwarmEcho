"""Frozen validation-map selection and the production bank-backed environment."""
from __future__ import annotations

import json
from pathlib import Path
import hashlib
from dataclasses import replace

from geo_experiments.shared.common import fingerprint, resolve_path
from geo_experiments.shared.console import terminal_print as print


def resolve_probe_maps(checkpoint, source="validation", *, level=None):
    if source == "validation":
        return frozen_validation_maps(checkpoint)
    if source != "compatible_pool":
        raise ValueError("Map source must be validation or compatible_pool.")
    return compatible_pool_maps(checkpoint, level=level)


def compatible_pool_maps(checkpoint, *, level=None):
    """Freeze a local matching dataset without changing original run validation."""
    from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
        required_settings, find_compatible_pool, build_pool, select_maps,
    )

    checkpoint = resolve_path(checkpoint)
    run = checkpoint.parent.parent
    if level is None:
        from geo_experiments.shared.policy import load_probe_level
        level, _, _, _ = load_probe_level(
            {"policy": {"from_run_config": True, "map_source": "compatible_pool"}},
            checkpoint, None, resolve_maps=False)
    # A relocated remote pool is often only an inspection subset. Search normal
    # complete local pools rather than treating that path as a pinned full bank.
    level = replace(level, training=replace(level.training, map_pool_path=None))
    wanted = required_settings(level)
    recorded = None
    for path in (checkpoint / "adaptive_map_selection.json", run / "random_buildings.json",
                 run / "adaptive_maps" / "selection.json"):
        if path.is_file():
            item = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(item.get("settings"), dict):
                recorded = item["settings"]
                break
    if not recorded or recorded.get("generator_version") != wanted["generator_version"]:
        raise ValueError("The local generator version must match the checkpoint's recorded generator version.")
    # Source fingerprints also cover geometry/compiler code, so pool reuse below
    # remains stricter than a generator-version-only comparison.
    for key in ("format", "random_buildings", "roadmap", "spawn"):
        if recorded.get(key) != wanted.get(key):
            raise ValueError(f"The checkpoint's {key} settings differ from the requested local pool.")
    count = level.validation.num_maps
    if not level.validation.enabled or type(count) is not int or count < 1:
        raise ValueError("Compatible-pool probes need an enabled positive validation map count in the saved config.")
    seed = level.validation.seed
    stage = level.env.geometric_agent_budget
    contract = {"settings": wanted, "count": count, "seed": seed, "stage": stage}
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    manifest = run / "probe_maps" / f"compatible_pool_{key}.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("contract") != contract:
            raise ValueError("Saved probe-map contract changed.")
    else:
        log = lambda message: print(f"[probe maps] {message}", flush=True)
        found = find_compatible_pool(level, required_count=count, log=log)
        if found is None:
            log(f"No sufficient compatible local pool; preparing {count} maps on CPU.")
            found = build_pool(level, count=count, log=log)
        pool, pool_metadata = found
        _, selected = select_maps(pool, pool_metadata, seed, 0, count)
        paths = [pool / "maps" / f"{item['id']}.yaml" for item in selected]
        if any(not path.is_file() or not (pool / "records" / f"{path.stem}.npz").is_file() for path in paths):
            raise FileNotFoundError(f"Compatible pool is missing selected maps or compiled records: {pool}")
        from geo_experiments.shared.common import ROOT
        pool_path = pool.resolve()
        portable_pool = pool_path.relative_to(ROOT).as_posix() if pool_path.is_relative_to(ROOT) else str(pool_path)
        metadata = {"format": "swarmecho-probe-maps/v1", "source": "compatible_pool",
            "original_validation": False, "pool": portable_pool, "contract": contract,
            "pool_settings_sha256": pool_metadata["settings_sha256"],
            "checkpoint_generator_settings": recorded,
            "config": {"enabled": True, "num_maps": count}, "stage": stage,
            "maps": [{"name": path.stem, "sha256": fingerprint(path)} for path in paths]}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest)
        log(f"Frozen {count} local probe maps in {manifest}")
    paths = []
    for row in metadata["maps"]:
        path = resolve_path(metadata["pool"]) / "maps" / f"{row['name']}.yaml"
        if not path.is_file() or fingerprint(path) != row["sha256"]:
            raise ValueError(f"Frozen local probe map missing or changed: {path}")
        paths.append(path.resolve())
    if len(paths) != count or len(set(paths)) != count:
        raise ValueError("Frozen local probe map count is inconsistent.")
    return paths, metadata


def frozen_validation_maps(checkpoint):
    """Resolve recorded maps only; never generate or select replacements."""
    run = Path(checkpoint).parent.parent
    metadata_path = run / "validation" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identities = metadata["maps"]
    if not metadata["config"]["enabled"] or len(identities) != metadata["config"]["num_maps"]:
        raise ValueError("The saved validation map count is inconsistent.")
    bank_path = run / "random_buildings.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8")) if bank_path.is_file() else {}
    if bank.get("pool"):
        folder = resolve_path(bank["pool"]) / "maps"
        if [row["id"] for row in bank["evaluation_maps"]] != [row["name"] for row in identities]:
            raise ValueError("The bank holdout differs from the frozen validation map order.")
    else:
        folder = run / "validation" / "generated_maps"
    names = [row["name"] for row in identities]
    if not names or len(names) != len(set(names)):
        raise ValueError("Frozen validation map names must be nonempty and unique.")
    paths = []
    for row in identities:
        path = folder / f"{row['name']}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen validation map missing: {path}")
        if fingerprint(path) != row["sha256"]:
            raise ValueError(f"Frozen validation map changed: {path}")
        paths.append(path.resolve())
    return paths, metadata


def make_validation_environment(level, levels):
    """Use the same map-bank preparation and target rules as ValidationRunner.

    Map IDs are dynamic reset inputs, so one collector serves the entire bank.
    Training's adaptive sampling weights and fixed validation cases are unused:
    probes collect fresh episodes with equal retained quotas per map.
    """
    from swarmecho.env.building_bank import prepare_bank, prepare_bank_from_records
    from swarmecho.env.environment import make_env_fns

    paths = tuple(item.map_paths[0] for item in levels)
    buildings, cfg = [item.building for item in levels], level.env
    plan = level.reward.chain_reward_system == "obstacle_geodesic" or level.reward.enable_chain_efficiency_reward
    options = dict(dynamic_max_geo_distance=level.training.dynamic_max_geo_distance.enabled,
                   relay_options=level.training.dynamic_max_geo_distance, randomize_base=level.training.randomize_base)
    log = lambda message: print(f"[collect] {message}", flush=True)
    pool = paths[0].parent.parent
    print(f"[collect] preparing frozen validation bank: {len(paths)} maps", flush=True)
    if all(path.parent == paths[0].parent and (pool / "records" / f"{path.stem}.npz").is_file()
           for path in paths):
        from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import load_records
        records = load_records(pool, [{"id": path.stem} for path in paths], include_target=True)
        bank = prepare_bank_from_records(records, buildings[0], cfg,
            interiors=[b.interior_cells for b in buildings], relay_cache_dir=pool / "relay_cache", log=log, **options)
    else:
        bank = prepare_bank(buildings, cfg, plan=plan, log=log, **options)
    return make_env_fns(buildings[0], cfg, building_bank=bank,
        plan_geodesic=plan, allow_redundancy_reward=level.reward.allow_redundancy_reward,
        target_found_requires_delivery=level.reward.target_found_requires_delivery,
        minimum_geodesic_separation=False, spawn_pair_max_attempts=level.training.spawn_pair_max_attempts,
        memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
        chain_reward_system=level.reward.chain_reward_system,
        progress_based_cluster_tiebreaker_soft=level.reward.progress_based_cluster_tiebreaker_soft,
        delivery_gap_vesting_enabled=level.reward.delivery_gap_vesting_enabled,
        priveleged_vote_reward=level.reward.priveleged_vote_reward, **options)
