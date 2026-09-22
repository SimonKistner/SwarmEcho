"""Read aliases for names embedded in older runs and artifacts.

Writers use canonical names. These exact aliases keep old files readable;
unrecognized identifiers are left unchanged so normal validation still applies.
"""

from __future__ import annotations

from pathlib import Path


LEVEL_ALIASES = {
    "M00_no_maze_open_cuboid_3D": "M00_no_maze_open_cuboid",
    "M01_no_maze_open_cuboid_tall_3D": "M01_no_maze_open_cuboid_tall",
    "M02_random_cuboid_obstacles_3D": "M02_random_cuboid_obstacles",
}
MAP_ALIASES = {"custom_3d_building": "custom_building"}
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
    return resolve_named_file(name_or_path, directory, aliases=LEVEL_ALIASES)


def resolve_map_file(name_or_path: str | Path, directory: Path) -> Path:
    return resolve_named_file(name_or_path, directory, aliases=MAP_ALIASES)
