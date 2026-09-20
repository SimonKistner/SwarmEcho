"""Map filename validation shared by the 3D building API."""

import re

MAP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_map_name(name: str) -> str:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Map name is required.")
    if not MAP_NAME_RE.fullmatch(name):
        raise ValueError("Map name may only contain letters, numbers, underscores, and hyphens.")
    return name
