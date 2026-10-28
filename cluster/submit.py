"""Standard-library-only preparation and submission; requires Python 3.7+.

No training imports, project environment, Docker daemon or network are used by
--dry-run or --image-info. Paths are resolved relative to this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import uuid

from prepare_context import ROOT, prepare_context


CLUSTER = Path(__file__).resolve().parent
DEPENDENCY_FILES = ("cluster/Dockerfile", "pyproject.toml", "uv.lock")
RESERVED_OVERRIDES = {"level", "logging.run_name", "logging.log_dir", "logging.wandb_mode"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def image_info(settings):
    # Normalizing line endings gives Windows and Linux CI the same tag.
    digest = hashlib.sha256()
    for name in DEPENDENCY_FILES:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    fingerprint = digest.hexdigest()
    repository = settings["image_repository"]
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("settings.image_repository must be a registry repository.")
    return {"repository": repository, "tag": "deps-" + fingerprint[:16],
            "fingerprint": fingerprint, "inputs": list(DEPENDENCY_FILES),
            "image": repository + ":deps-" + fingerprint[:16]}


def validate(settings, run):
    expected_settings = {"master", "determined_version", "workspace", "project", "project_id",
                         "resource_pool", "slots_per_trial", "pfss_mount", "persist_root", "image_repository"}
    if set(settings) != expected_settings:
        raise ValueError("Unexpected/missing settings keys: " + str(set(settings) ^ expected_settings))
    if set(run) != {"description", "level", "wandb_mode", "overrides"}:
        raise ValueError("A run requires exactly description, level, wandb_mode and overrides.")
    for name in ("project_id", "slots_per_trial"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(name + " must be a positive integer.")
    for name in ("master", "determined_version", "workspace", "project", "resource_pool"):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(name + " must be a nonempty string.")
    if not settings["master"].startswith("https://"):
        raise ValueError("Use an HTTPS master URL.")
    mount = PurePosixPath(settings["pfss_mount"])
    persist = PurePosixPath(settings["persist_root"])
    if not mount.is_absolute() or not persist.is_absolute() or ".." in mount.parts + persist.parts:
        raise ValueError("PFSS paths must be absolute Linux paths without '..'.")
    if mount not in persist.parents:
        raise ValueError("persist_root must be a subdirectory of pfss_mount.")
    if run["wandb_mode"] not in {"disabled", "offline", "online"}:
        raise ValueError("wandb_mode must be disabled, offline or online.")
    level = run["level"]
    if not isinstance(level, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", level):
        raise ValueError("level must be a level name, without a path or CLI arguments.")
    levels = list((ROOT / "src/swarmecho/curriculum_config/levels").rglob(level + ".yaml"))
    if len(levels) != 1:
        raise ValueError("Expected one source YAML for level " + level)
    if not isinstance(run["description"], str) or not isinstance(run["overrides"], list):
        raise ValueError("description must be a string and overrides must be a list.")
    seen = set()
    for value in run["overrides"]:
        if not isinstance(value, str) or "=" not in value or "\n" in value or "\r" in value:
            raise ValueError("Overrides must be single-line key=value strings.")
        key = value.split("=", 1)[0]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key) or key in RESERVED_OVERRIDES or key in seen:
            raise ValueError("Invalid, duplicated or reserved override: " + key)
        seen.add(key)


def submit(settings, config_path, context, output):
    det = shutil.which("det")
    if not det:
        raise RuntimeError("det is unavailable. Install determined==" + settings["determined_version"])
    version = subprocess.check_output([det, "--version"], text=True).strip()
    if version != "det " + settings["determined_version"]:
        raise RuntimeError("Expected det " + settings["determined_version"] + "; found " + version)
    command = [det, "experiment", "create", "--project_id", str(settings["project_id"]),
               str(config_path), str(context)]
    env = os.environ.copy()
    env["DET_MASTER"] = settings["master"]
    print("Submitting to " + settings["master"], flush=True)
    # Never retry automatically: a lost response can still mean the job exists.
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    experiment_id = None
    with (output / "submission.log").open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            match = re.search(r"Created experiment (\d+)", line)
            if match:
                experiment_id = int(match.group(1))
        code = process.wait()
    write_json(output / "result.json", {"experiment_id": experiment_id, "exit_code": code})
    if experiment_id is not None:
        print("Experiment: " + settings["master"].rstrip("/") + "/det/experiments/" + str(experiment_id))
    if code:
        raise RuntimeError("det failed; inspect submission.log and the WebUI before submitting again.")
    if experiment_id is None:
        print("Check submission.log/WebUI for the ID; do not resubmit just to recover the ID.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Name of a JSON config under cluster/runs, without .json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--image-info", action="store_true", help="Print image metadata JSON; no writes/network")
    args = parser.parse_args()
    settings = read_json(CLUSTER / "settings.json")
    info = image_info(settings)
    if args.image_info:
        if args.run or args.dry_run:
            parser.error("--image-info cannot be combined with --run/--dry-run")
        print(json.dumps(info, indent=2))
        return
    if not args.run or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run):
        parser.error("--run must name a config such as g03-smoke")
    run = read_json(CLUSTER / "runs" / (args.run + ".json"))
    validate(settings, run)
    unresolved_image = "REPLACE_" in info["repository"]
    if not unresolved_image and (info["repository"] != info["repository"].lower()
                                 or not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+", info["repository"])):
        raise ValueError("image_repository must be lowercase and contain no tag or digest.")
    if unresolved_image and not args.dry_run:
        raise ValueError("Set image_repository in cluster/settings.json and publish the image first.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run + "-" + stamp + "-" + uuid.uuid4().hex[:8]
    output = CLUSTER / ".submissions" / run_name
    context, count, size = prepare_context(output / "model")
    manifest = dict(run, run_name=run_name)
    write_json(context / "cluster/run.json", manifest)
    # JSON is valid YAML, so det can read this without a local YAML package.
    experiment = {
        "name": run_name, "description": run["description"],
        "entrypoint": "python cluster/run_driver.py",
        "environment": {"image": info["image"], "environment_variables": [
            "SWARMECHO_PERSIST_ROOT=" + settings["persist_root"]]},
        "resources": {"resource_pool": settings["resource_pool"], "slots_per_trial": settings["slots_per_trial"]},
        "bind_mounts": [{"host_path": settings["pfss_mount"], "container_path": settings["pfss_mount"], "read_only": False}],
        "searcher": {"name": "single", "metric": "job_completed", "max_length": {"epochs": 1}, "smaller_is_better": False},
        "max_restarts": 0,
    }
    config_path = output / "experiment.yaml"
    write_json(config_path, experiment)
    write_json(output / "summary.json", {"dry_run": args.dry_run, "settings": settings,
               "run": manifest, "dependency_image": info,
               "source_files": count, "source_bytes": size})
    print("Run:       " + run_name)
    print("Target:    {workspace}/{project} (project {project_id}), {resource_pool}, {slots_per_trial} slot(s)".format(**settings))
    print("Level:     " + run["level"] + "; W&B: " + run["wandb_mode"])
    print("Image:     " + info["image"])
    print("PFSS:      " + settings["persist_root"])
    print("Snapshot:  {} ({} source files; {:,} bytes, plus run.json)".format(context, count, size))
    print("Config:    " + str(config_path))
    if args.dry_run:
        print("DRY RUN: local files prepared; no det command, login, network or GPU allocation.")
        if unresolved_image:
            print("IMAGE NOT CONFIGURED: preparation is inspectable; actual submission is blocked until step 2.")
        print("This checks packaging and basic config structure, not training override semantics or cluster compatibility.")
        return
    submit(settings, config_path, context, output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
