"""Frozen validation-map selection and the production bank-backed environment."""
from __future__ import annotations

import json
from pathlib import Path

from geo_experiments.common import fingerprint, resolve_path
from geo_experiments.console import terminal_print as print


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
                   relay_options=level.training.dynamic_max_geo_distance, randomize_base=False)
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
