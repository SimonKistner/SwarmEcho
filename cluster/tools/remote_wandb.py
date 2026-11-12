"""Restricted SSH service: status/install for PFSS .wandb_api_key only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import struct
import sys
import uuid


def send(value):
    data = json.dumps(value).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("!Q", len(data)) + data)
    sys.stdout.buffer.flush()


def handle(request):
    root = Path(request["persist_root"])
    if not root.is_absolute() or ".." in root.parts or not root.is_dir():
        raise ValueError("Invalid persistent root")
    target = root / ".wandb_api_key"
    if target.is_symlink():
        raise ValueError("Key path is a symlink")
    exists = target.exists()
    usable = bool(exists and target.is_file() and os.access(str(target), os.R_OK)
                  and stat.S_IMODE(target.stat().st_mode) == 0o600 and target.stat().st_size > 0)
    if request["action"] == "status":
        return {"ok": True, "exists": exists, "usable": usable}
    if request["action"] != "install":
        raise ValueError("Unsupported action")
    if exists and not request.get("replace", False):
        raise ValueError("Existing key is not replaced without explicit --replace")
    key = request["api_key"]
    if (not isinstance(key, str) or len(key) < 8 or any(c.isspace() for c in key)
            or key == "your_wandb_api_key_here"):
        raise ValueError("Invalid key")
    temporary = root / (".wandb_api_key." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(key + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if request.get("replace", False):
            os.replace(str(temporary), str(target))
        else:
            # Atomic, non-overwriting publication if another setup raced this one.
            os.link(str(temporary), str(target))
        return {"ok": True, "usable": True}
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    for line in sys.stdin.buffer:
        try:
            send(handle(json.loads(line)))
        except Exception:
            send({"ok": False, "error": "W&B key operation failed; credentials were not logged."})


if __name__ == "__main__":
    main()
