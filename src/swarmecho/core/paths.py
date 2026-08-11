"""Path normalization helpers for commands launched from Windows/WSL."""

from __future__ import annotations

import re
from pathlib import Path


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


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
