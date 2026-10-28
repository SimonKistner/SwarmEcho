"""Load frozen run settings without silently substituting today's training level."""
from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path

from geo_experiments.common import resolve_path, fingerprint
from geo_experiments.console import terminal_print as print


def load_probe_level(config, checkpoint, run_dir):
    from omegaconf import OmegaConf
    from swarmecho.core.config import LEVEL_DIR, MAP_DIR, RewardConfig, load_level
    from swarmecho.core.compatibility import resolve_level_file, resolve_map_file

    if config["policy"].get("from_run_config", False):
        source = checkpoint.parent.parent / "config.yaml"
        if not source.is_file():
            raise FileNotFoundError(f"No saved training configuration beside {checkpoint}")
    else:
        source = resolve_level_file(config["policy"]["level"], LEVEL_DIR).resolve()
    original = source.read_text(encoding="utf-8")
    data = OmegaConf.to_container(OmegaConf.create(original), resolve=True)
    # Saved training configs contain descriptive fields outside load_level's schema.
    original_name = data.pop("name", None)
    data.pop("building", None)
    names = data.get("env", {}).get("map_names", [])
    generated = data.get("random_buildings", {}).get("enabled", False)
    validation_maps = config["policy"].get("map_source", "training") == "validation"
    if validation_maps and (not generated or not config["policy"].get("from_run_config")):
        raise ValueError("Validation-map probes require a saved generated-building run config.")
    if generated and not validation_maps:
        raise ValueError("Generated-building probes require policy.map_source=validation.")
    maps = []
    for name in ([] if generated else names):
        explicit = resolve_path(name)
        path = explicit if explicit.is_file() else resolve_map_file(name, MAP_DIR).resolve()
        maps.append(str(path))
    training_maps = checkpoint.parent.parent / "static_maps.json"
    if config["policy"].get("from_run_config") and training_maps.is_file():
        recorded = {row["name"]: row["sha256"] for row in json.loads(training_maps.read_text(encoding="utf-8"))["maps"]}
        for name, path in zip(names, maps, strict=True):
            expected = recorded.get(Path(name).stem)
            if expected is not None and fingerprint(path) != expected:
                raise ValueError(f"Training map changed since this checkpoint was trained: {name}")
    data.setdefault("env", {})["map_names"] = maps
    # Reward amounts are not used by this frozen rollout. Retain every supported
    # behaviour switch while allowing the pre-rename G01 configs to be loaded.
    reward_fields = {f.name for f in fields(RewardConfig)}
    ignored = sorted(set(data.get("reward", {})) - reward_fields)
    historical_payouts = {"rejected_call_penalty", "success_unused_agent_penalty", "success_noncontributor_penalty",
        "chain_efficiency_bonus", "exploration_bonus", "finder_bonus", "target_found_bonus", "success_bonus",
        "success_bonus_as_hold_record", "success_bonus_contributors_only", "peer_informing_bonus",
        "collision_penalty", "time_penalty_per_step", "gap_reduction_meter_bonus", "gap_change_meter_bonus"}
    if set(ignored) - historical_payouts:
        raise ValueError(f"Unknown reward settings in frozen policy config: {sorted(set(ignored) - historical_payouts)}")
    data["reward"] = {k: v for k, v in data.get("reward", {}).items() if k in reward_fields}
    if ignored:
        # These names did not change rollout semantics; they rename unused payouts.
        print(f"[policy] historical reward payout names omitted (no rewards fitted): {', '.join(ignored)}", flush=True)
    if run_dir is None:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="geo-policy-") as temporary:
            snapshot = Path(temporary) / source.name
            OmegaConf.save(OmegaConf.create(data), snapshot)
            level = load_level(snapshot)
    else:
        snapshot = Path(run_dir) / "policy_level.yaml"
        OmegaConf.save(OmegaConf.create(data), snapshot)
        level = load_level(snapshot)
    if original_name:
        level = replace(level, name=original_name)
    if validation_maps:
        from geo_experiments.validation_maps import frozen_validation_maps
        from swarmecho.env.buildings import load_building
        paths, metadata = frozen_validation_maps(checkpoint)
        if not level.validation.enabled or len(paths) != level.validation.num_maps:
            raise ValueError("The run config differs from its frozen validation map count.")
        if metadata["stage"] != level.env.geometric_agent_budget:
            raise ValueError("Frozen validation maps belong to a different agent stage.")
        # Keep trained observation/population settings; inspection overrides must
        # not substitute three inspection maps or disable the stage's target filter.
        level = replace(level, building=load_building(paths[0]), map_paths=tuple(paths),
                        map_names=[path.stem for path in paths])
        print(f"[policy] verified {len(paths)} frozen validation maps", flush=True)
    return level, source, original, ignored


def map_levels(level):
    """One canonical evaluation environment per authored map, with stable suite IDs."""
    from dataclasses import replace
    from swarmecho.env.buildings import load_building

    return [replace(level, building=load_building(path), map_names=[name], map_paths=(path,))
            for name, path in zip(level.map_names, level.map_paths, strict=True)]
