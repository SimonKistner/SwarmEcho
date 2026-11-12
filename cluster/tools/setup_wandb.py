"""Install an existing local W&B key on PFSS without putting it in job snapshots."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from fetch import Transfer
from tracking import read_json

CLUSTER = Path(__file__).resolve().parents[1]


def local_key():
    value = os.environ.get("WANDB_API_KEY", "").strip()
    env_file = CLUSTER.parent / ".env"
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            key, separator, candidate = line.partition("=")
            if separator and key.strip() == "WANDB_API_KEY":
                value = candidate.strip()
                if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].strip()
    if (not value or value == "your_wandb_api_key_here" or len(value) < 8
            or any(c.isspace() for c in value)):
        raise ValueError("Set WANDB_API_KEY in your local .env or environment first. Never paste the key into chat.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Explicitly replace an existing PFSS key")
    args = parser.parse_args()
    settings = read_json(CLUSTER / "settings.json")
    # Existing retrieval stays read-only. This service can write one named key file only.
    with Transfer(settings, read_only=False, service_file=CLUSTER / "tools/remote_wandb.py") as connection:
        status = connection.request({"action": "status", "persist_root": settings["persist_root"]})
        if status["usable"] and not args.replace:
            print("Cluster W&B key is already configured with owner-only permissions; unchanged.")
            return
        connection.request({"action": "install", "persist_root": settings["persist_root"],
                            "api_key": local_key(), "replace": args.replace})
    print("Cluster W&B key configured with permissions 600. Key was not printed or included in a submission.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError, EOFError):
        # Neither request payloads nor exception details containing credentials reach stdout.
        print("W&B key setup failed. Check cluster connectivity/login, the local key, and PFSS permissions.", file=sys.stderr)
        sys.exit(1)
