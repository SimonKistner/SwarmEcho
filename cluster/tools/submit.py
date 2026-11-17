"""Standard-library-only preparation and submission; requires Python 3.7+.

The --image-info command uses no training imports, project environment, Docker
daemon or network. Paths are resolved relative to this script.
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
import time
import uuid

from prepare_context import ROOT, prepare_context
from tracking import register
from plan import check_capacity, expand_config, level_timestamp_postfix

# The tuning controller's preparation helpers use only the standard library.
sys.path.insert(0, str(ROOT / "src"))
from swarmecho.training.cluster_tuning import validate_tuning
from swarmecho.core.run_names import ppo_name, submission_ppo_parameters


CLUSTER = Path(__file__).resolve().parents[1]
MANIFESTS = CLUSTER / "manifests"
DEFAULT_MANIFEST = MANIFESTS / "g03-default.json"
HISTORICAL_MANIFESTS = MANIFESTS / "historical"
DEPENDENCY_FILES = ("cluster/Dockerfile", "pyproject.toml", "uv.lock")
RESERVED_OVERRIDES = {"level", "levels", "logging.run_name", "logging.log_dir", "logging.wandb_mode",
                      "logging.wandb_group", "logging.use_timestamp_postfix"}
APPROVED_EXISTING_RUNS = {}
PRIORITY = 41


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
    if set(run) - {"tuning", "run_name_template", "wandb_group", "run_postfix"} not in ({"description", "level", "wandb_mode", "overrides"},
                                   {"description", "levels", "wandb_mode", "overrides"}):
        raise ValueError("A run requires description, level OR levels, wandb_mode and overrides.")
    if "wandb_group" in run and (not isinstance(run["wandb_group"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,255}", run["wandb_group"])):
        raise ValueError("wandb_group must contain 1-255 letters, numbers, underscores or hyphens.")
    if "run_name_template" in run:
        template = run["run_name_template"]
        if (not isinstance(template, str) or template.count("{ppo}") != 1
                or not re.fullmatch(r"[A-Za-z0-9_-]+", template.replace("{ppo}", "baseline"))
                or "tuning" in run or "levels" in run):
            raise ValueError("run_name_template requires one {ppo} token and a single non-tuning level; "
                             "otherwise use only letters, numbers, underscores and hyphens.")
    if "tuning" in run:
        tuning = validate_tuning(run["tuning"])
        if run.get("level") != "G01_CLU_static" or "levels" in run or settings["slots_per_trial"] != 1:
            raise ValueError("Tuning requires G01_CLU_static and exactly one slot.")
        if tuning["phase"] == "probe" and run["wandb_mode"] != "disabled":
            raise ValueError("The capacity probe must disable W&B.")
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
    names = run.get("levels", [run.get("level")])
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("levels must be a nonempty list of distinct level names.")
    for level in names:
        if not isinstance(level, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", level):
            raise ValueError("level must be a level name, without a path or CLI arguments.")
        levels = list((ROOT / "src/swarmecho/curriculum_config/levels").rglob(level + ".yaml"))
        if len(levels) != 1:
            raise ValueError("Expected one source YAML for level " + level)
    if "levels" in run and len({name.split("_", 1)[0] for name in names}) != len(names):
        raise ValueError("Curriculum levels must have distinct short names to avoid stage-folder collisions.")
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


def check_det_version(det, expected):
    version = subprocess.check_output([det, "--version"], text=True).strip()
    # argparse includes the launcher name: Windows can print det.EXE.
    match = re.fullmatch(r"det(?:\.exe)?\s+(\S+)", version, flags=re.IGNORECASE)
    if not match or match.group(1) != expected:
        raise RuntimeError("Expected det " + expected + "; found " + version)
    return version


class SubmissionOutput:
    """Collapse det's per-file updates without changing the submission log."""

    def __init__(self, stream):
        self.stream = stream
        self.interactive = stream.isatty()
        self.pending = None
        self.width = 0
        self.updated_at = float("-inf")

    def write(self, line):
        message = line.strip()
        if re.fullmatch(r"Preparing files to send to master\.\.\. .+ and \d+ files", message):
            self.pending = message
            now = time.monotonic()
            if self.interactive and now - self.updated_at >= 0.1:
                self.width = max(self.width, len(message))
                self.stream.write("\r" + message.ljust(self.width))
                self.stream.flush()
                self.updated_at = now
        elif self.pending is not None and not message:
            # det clears its previous progress line with whitespace.
            return
        else:
            self.finish()
            self.stream.write(line)
            self.stream.flush()

    def finish(self):
        if self.pending is None:
            return
        prefix = "\r" if self.interactive else ""
        self.stream.write(prefix + self.pending.ljust(self.width) + "\n")
        self.stream.flush()
        self.pending = None
        self.width = 0
        self.updated_at = float("-inf")


def ensure_unused_run_name(run_name, override_existing=False):
    """Ask before reusing an identity, before preparation or remote creation."""
    conflicts = set()
    tracked = ROOT / "outputs" / run_name / "cluster_run.json"
    if tracked.is_file():
        experiment_id = read_json(tracked).get("experiment_id")
        conflicts.add((str(tracked), experiment_id))
    candidates = [CLUSTER / ".submissions" / run_name]
    candidates.extend((CLUSTER / ".submissions").glob(run_name + "-receipt-*"))
    for directory in candidates:
        result = directory / "result.json"
        if result.is_file():
            experiment_id = read_json(result).get("experiment_id")
            if experiment_id:
                conflicts.add((str(result), experiment_id))
    if not conflicts:
        return False
    if not override_existing and not conflicts.issubset(APPROVED_EXISTING_RUNS.get(run_name, set())):
        print("Local cluster folder belongs to another experiment: " + str(tracked.parent))
        for path, experiment_id in sorted(conflicts):
            print("  Experiment {}: {}".format(experiment_id, path))
        print("Override replaces local tracking after submission and archives the old record.")
        print("PFSS training history is retained; use a new run_postfix for a fresh output directory.")
        while True:
            try:
                response = input("Press Enter to override, or Ctrl+C to abort: ")
            except EOFError:
                raise RuntimeError("Override requires interactive confirmation; submission aborted.")
            if response == "":
                break
    APPROVED_EXISTING_RUNS.setdefault(run_name, set()).update(conflicts)
    return True


def submit(settings, config_path, context, output):
    override_existing = ensure_unused_run_name(read_json(output / "summary.json")["run"]["run_name"])
    det = shutil.which("det")
    if not det:
        raise RuntimeError("det is unavailable. Install determined==" + settings["determined_version"])
    check_det_version(det, settings["determined_version"])
    command = [det, "experiment", "create", "--project_id", str(settings["project_id"]),
               str(config_path), str(context)]
    env = os.environ.copy()
    env["DET_MASTER"] = settings["master"]
    print("Submitting to " + settings["master"], flush=True)
    # Never retry automatically: a lost response can still mean the job exists.
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    experiment_id = None
    console = SubmissionOutput(sys.stdout)
    with (output / "submission.log").open("w", encoding="utf-8") as log:
        try:
            for line in process.stdout:
                console.write(line)
                log.write(line)
                match = re.search(r"Created experiment (\d+)", line)
                if match:
                    experiment_id = int(match.group(1))
                    # Persist the receipt immediately, even if the connection later fails.
                    write_json(output / "result.json", {"experiment_id": experiment_id, "exit_code": None})
                    folder = register(read_json(output / "summary.json"), experiment_id, output,
                                      override_existing=override_existing)
                    print("Local run: " + str(folder.parent), flush=True)
        finally:
            console.finish()
        code = process.wait()
    write_json(output / "result.json", {"experiment_id": experiment_id, "exit_code": code})
    if experiment_id is not None:
        print("Experiment: " + settings["master"].rstrip("/") + "/det/experiments/" + str(experiment_id))
    if code:
        raise RuntimeError("det failed; inspect submission.log and the WebUI before submitting again.")
    if experiment_id is None:
        raise RuntimeError("No experiment ID was returned; inspect submission.log/WebUI before retrying. "
                           "The experiment may already exist.")


def submission_run_name(name, run, stamp):
    if "run_name_template" in run:
        level_path = next((ROOT / "src/swarmecho/curriculum_config/levels").rglob(run["level"] + ".yaml"))
        label = ppo_name(submission_ppo_parameters(level_path, run["overrides"]))
        name = run["run_name_template"].replace("{ppo}", label)
    postfix = run.get("run_postfix", "")
    run_name = "CLU_" + name + ("_" + postfix if postfix else "")
    level = run["level"] if "level" in run else run["levels"][0]
    if level_timestamp_postfix(ROOT, level):
        run_name += "-" + stamp
    return run_name


def prepare_submission(settings, info, name, run, stamp, source_context=None):
    """Freeze each receipt separately, reusing one source snapshot across the batch."""
    run_name = submission_run_name(name, run, stamp)
    ensure_unused_run_name(run_name)
    output = CLUSTER / ".submissions" / run_name
    if output.exists():
        # Keep earlier receipts immutable, including explicitly overridden runs.
        # This local-only suffix cannot isolate a training directory on PFSS.
        output = output.with_name(run_name + "-receipt-" + uuid.uuid4().hex[:8])
    if source_context is None:
        context, count, size = prepare_context(output / "model")
    else:
        previous, count, size = source_context
        context = output / "model"
        shutil.copytree(previous, context)
    manifest = dict(run, run_name=run_name)
    write_json(context / "cluster/run.json", manifest)
    # JSON is valid YAML, so det can read this without a local YAML package.
    experiment = {
        "name": run_name, "description": run["description"],
        "entrypoint": "python cluster/tools/run_driver.py",
        "environment": {"image": info["image"], "environment_variables": [
            "SWARMECHO_PERSIST_ROOT=" + settings["persist_root"]]},
        "resources": {"resource_pool": settings["resource_pool"], "slots_per_trial": settings["slots_per_trial"], "priority": PRIORITY},
        "bind_mounts": [{"host_path": settings["pfss_mount"], "container_path": settings["pfss_mount"], "read_only": False}],
        "searcher": {"name": "single", "metric": "job_completed", "max_length": {"epochs": 1}, "smaller_is_better": False},
        "max_restarts": validate_tuning(run["tuning"])["controller_restarts"] if "tuning" in run else 0,
    }
    config_path = output / "experiment.yaml"
    write_json(config_path, experiment)
    write_json(output / "summary.json", {"settings": settings,
               "run": manifest, "dependency_image": info,
               "source_files": count, "source_bytes": size})
    print("Run:       " + run_name)
    print("Target:    {workspace}/{project} (project {project_id}), {resource_pool}, {slots_per_trial} slot(s)".format(**settings))
    print("Level(s):  " + ", ".join(run.get("levels", [run.get("level")])) + "; W&B: " + run["wandb_mode"])
    if "wandb_group" in run:
        print("W&B group: " + run["wandb_group"])
    if "tuning" in run:
        tuning = validate_tuning(run["tuning"])
        print("Tuning:    " + tuning["phase"] + "; " +
              (str(tuning["probe_seconds"]) + "s controller limit" if tuning["phase"] == "probe" else
               str(tuning["trials"]) + " fresh trials; " + str(tuning["search_seconds"]) + "s overall limit"))
    print("Image:     " + info["image"])
    print("PFSS:      " + settings["persist_root"])
    print("Snapshot:  {} ({} source files; {:,} bytes, plus run.json)".format(context, count, size))
    print("Config:    " + str(config_path))
    return output, config_path, (context, count, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--config", type=Path,
        help=f"Batch config; defaults to {DEFAULT_MANIFEST.relative_to(ROOT).as_posix()}",
    )
    selection.add_argument("--run", help="Historical manifest name without .json, such as g03-smoke")
    parser.add_argument("--override-existing", action="store_true",
                        help="Use existing-name override approval already given to a batch wrapper")
    parser.add_argument("--image-info", action="store_true", help="Print image metadata JSON; no writes/network")
    args = parser.parse_args()
    settings = read_json(CLUSTER / "settings.json")
    if args.image_info:
        if args.run or args.config or args.override_existing:
            parser.error("--image-info cannot be combined with submission options")
        print(json.dumps(image_info(settings), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = stamp + "-" + uuid.uuid4().hex[:8]
    if args.run:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run):
            parser.error("--run must name a historical config such as g03-smoke")
        matches = sorted(
            path for path in HISTORICAL_MANIFESTS.rglob(args.run + ".json") if path.is_file()
        )
        if not matches:
            parser.error(f"No historical manifest named {args.run!r}.")
        if len(matches) > 1:
            parser.error(f"Historical manifest name {args.run!r} is ambiguous.")
        config_path = matches[0]
        plans = [(args.run, read_json(config_path))]
    else:
        config_path = (args.config or DEFAULT_MANIFEST).resolve()
        plans = expand_config(read_json(config_path), ROOT, stamp)
    for _, run in plans:
        validate(settings, run)
    # Resolve and confirm the entire batch before freezing files or submitting
    # its first experiment. A later conflict must not leave a partial batch.
    run_names = [submission_run_name(name, run, stamp) for name, run in plans]
    if len(run_names) != len(set(run_names)):
        raise ValueError("The batch contains duplicate resolved run names; choose distinct names/postfixes.")
    for run_name in run_names:
        ensure_unused_run_name(run_name, override_existing=args.override_existing)
    check_capacity(plans, settings["slots_per_trial"])
    info = image_info(settings)
    repository = info["repository"]
    if "REPLACE_" in repository:
        raise ValueError("Set image_repository in cluster/settings.json and publish the image first.")
    if repository != repository.lower() or not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+", repository):
        raise ValueError("image_repository must be lowercase and contain no tag or digest.")

    print("Config:    " + str(config_path))
    print("Batch:     {} experiment(s), {} GPU slot(s); shared workspace limit 24, including other jobs.".format(
        len(plans), len(plans) * settings["slots_per_trial"]))
    prepared = []
    source_context = None
    for name, run in plans:
        output, experiment_path, source_context = prepare_submission(
            settings, info, name, run, stamp, source_context)
        prepared.append((output, experiment_path, source_context[0]))
    batch_path = CLUSTER / ".submissions" / ("batch-" + batch_id + ".json")
    batch = {"config_path": str(config_path),
             "experiments": [{"submission_dir": str(output), "state": "PREPARED"}
                             for output, _, _ in prepared]}
    write_json(batch_path, batch)
    print("Batch receipt: " + str(batch_path))
    for index, (output, experiment_path, context) in enumerate(prepared):
        record = batch["experiments"][index]
        record["state"] = "SUBMITTING"
        write_json(batch_path, batch)
        try:
            submit(settings, experiment_path, context, output)
        except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError):
            record["state"] = "CHECK_RECEIPT"
            write_json(batch_path, batch)
            print("Batch stopped; inspect receipts before retrying. Remaining PREPARED experiments were not submitted.", file=sys.stderr)
            raise
        record["state"] = "SUBMITTED"
        write_json(batch_path, batch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSubmission aborted.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
