"""Read aliases for names embedded in older runs and artifacts.

Writers use canonical names. These exact aliases keep old files readable;
unrecognized identifiers are left unchanged so normal validation still applies.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_LEVEL_NAME = "G03_CLU_random_buildings"
LEGACY_LOCAL_G03_LEVEL = "G03_legacy_random_buildings"

LEVEL_ALIASES = {
    "G03": DEFAULT_LEVEL_NAME,
    "G03_random_buildings": DEFAULT_LEVEL_NAME,
    "CLU_G01_static": "G01_CLU_static",
}
MAP_ALIASES = {
    "Cellar": "cellar",
    "Custom Building": "custom_building",
    "G01_static_map": "g01_static_map",
    "G02_static_map_01": "g02_static_map_01",
    "G02_static_map_02": "g02_static_map_02",
    "G02_static_map_03": "g02_static_map_03",
    "G02_static_map_04": "g02_static_map_04",
    "G02_static_map_05": "g02_static_map_05",
    "Mirrored Office": "mirrored_office",
    "Open Cuboid": "open_cuboid",
    "Space Station": "space_station",
    "B02c_static_map_noSpawn": "g01_static_map",
    "C03_station": "space_station",
    "C05_cellar": "cellar",
    "B02c_static_map_cellar": "cellar",
    "M00_no_maze_open_cuboid": "open_cuboid",
    "M00_no_maze_open_cuboid_3D": "open_cuboid",
    "office_mirrored": "mirrored_office",
    "custom_3d_building": "custom_building",
}
MARKER_ALIASES = {
    "compact_3d_v1": "compact_v1",
    "swarmecho-3d-obstacle-layout/v1": "swarmecho-obstacle-layout/v1",
    "swarmecho-3d-eval-heatmap/v1": "swarmecho-eval-heatmap/v1",
}


def canonical_level_name(name: str) -> str:
    return LEVEL_ALIASES.get(name, name)


def canonical_map_name(name: str) -> str:
    return MAP_ALIASES.get(name, name)


def canonical_marker(value):
    return MARKER_ALIASES.get(value, value) if isinstance(value, str) else value


def resolve_named_file(name_or_path: str | Path, directory: Path, *, aliases: dict[str, str]) -> Path:
    """Honor existing files first, then try a renamed sibling or bundled file."""
    requested = Path(name_or_path)
    if requested.exists():
        return requested
    canonical = aliases.get(requested.stem, requested.stem)
    sibling = requested.with_name(canonical + (requested.suffix or ".yaml"))
    if sibling.exists():
        return sibling
    return Path(directory) / f"{canonical}.yaml"


def resolve_level_file(name_or_path: str | Path, directory: Path) -> Path:
    """Resolve short names, grouped names, and pre-migration level paths."""
    source = resolve_named_file(name_or_path, directory, aliases=LEVEL_ALIASES)
    if source.is_file():
        return source
    requested = Path(str(name_or_path).replace("\\", "/"))
    canonical = canonical_level_name(requested.stem)
    grouped = Path(directory) / requested.with_name(canonical + ".yaml")
    if grouped.is_file():
        return grouped
    matches = sorted(Path(directory).rglob(f"{canonical}.yaml"))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous level name {canonical!r}; specify its family/path: {matches}")
    return matches[0] if matches else source


def resolve_map_file(name_or_path: str | Path, directory: Path) -> Path:
    return resolve_named_file(name_or_path, directory, aliases=MAP_ALIASES)


REWARD_NAME_ALIASES = {
    "finder_reward": "base_reporter_reward",
    "target_found_reward": "target_delivery_reward",
    "gap_progress_reward_per_meter": "closed_gap_reward_per_meter",
}


def historical_environment_identity(env) -> dict:
    """Keep frozen dataset identities stable across configuration renames."""
    from dataclasses import asdict

    values = asdict(env)
    values["adapt_size"] = values.pop("decommissioning")
    if values.get("radar_mode") == "max_proximity":
        values["radar_mode"] = "legacy"
    return values


def historical_reward_identity(reward) -> dict:
    """Retain the G03 identity format used by frozen validation cases."""
    from dataclasses import asdict

    values = asdict(reward)
    for old, current in REWARD_NAME_ALIASES.items():
        values[old] = values.pop(current)
    values.update(decommission_reward=0, useless_agent_at_success_reward=-3000,
                  useless_agent_at_success_all_agents_reward=-750,
                  useless_agent_at_success_all_agents_reward_cap=-2000)
    return values
