"""Path normalization helpers for commands launched from Windows/WSL."""

from __future__ import annotations

import re
from pathlib import Path


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def resolve_project_path(value: str | Path) -> Path:
    """Resolve portable references written by cluster import on Windows or WSL.

    Other saved paths retain their existing interpretation. Imported metadata
    uses repository-relative outputs/pool references instead of PFSS paths.
    """
    raw = str(value).replace("\\", "/")
    if raw == "outputs" or raw.startswith("outputs/") or raw.startswith("src/swarmecho/curriculum_config/maps/random_building_bank/pools/"):
        return Path(__file__).resolve().parents[3] / raw
    return Path(value)


def resolve_saved_paths(value):
    """Resolve portable path strings recursively without changing saved files."""
    if isinstance(value, dict):
        return {key: resolve_saved_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_saved_paths(item) for item in value]
    if isinstance(value, str):
        return str(resolve_project_path(value)) if value.startswith(("outputs/", "src/swarmecho/curriculum_config/maps/random_building_bank/pools/")) else value
    return value


def normalize_wsl_path(value: str | Path) -> Path:
    """Convert a Windows absolute path into its WSL ``/mnt/<drive>`` form.

    Linux/WSL paths are returned unchanged apart from normalizing separators.
    Windows paths should be quoted when passed through a POSIX shell so the
    shell does not consume their backslashes before this function sees them.
    """
    raw = str(value)
    match = _WINDOWS_ABSOLUTE_PATH.match(raw)
    if match:
        rest = match.group("rest").replace("\\", "/")
        return Path(f"/mnt/{match.group('drive').lower()}/{rest}")
    return Path(raw.replace("\\", "/"))
