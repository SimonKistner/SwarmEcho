"""Local submission identity and fetch state; standard library only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
TRACK_FILE = "cluster_run.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def register(summary, experiment_id, submission_dir, override_existing=False):
    remote = summary["run"]["run_name"]
    name = remote if remote.startswith("CLU_") else "CLU_" + remote
    path = OUTPUTS / name / TRACK_FILE
    archived = None
    if path.exists():
        previous = read_json(path)
        if previous["experiment_id"] != experiment_id:
            if not override_existing:
                raise ValueError("Local cluster folder belongs to another experiment: " + str(path.parent))
            archived = path.parent / ".cluster/tracking_history" / (
                "experiment-{}-{}.json".format(previous["experiment_id"], uuid.uuid4().hex[:8]))
            save_json(archived, previous)
        else:
            return path
    record = {"format": "swarmecho-cluster-run/v1", "experiment_id": experiment_id,
                    "name": name, "remote_name": remote, "registered_at": now(),
                    "submission_dir": str(submission_dir), "settings": summary["settings"],
                    "run": summary["run"], "dependency_image": summary["dependency_image"],
                    "last_fetch": None, "experiment_state": "SUBMITTED", "stages": {}, "files": {}}
    if archived is not None:
        record["previous_tracking_file"] = str(archived)
    save_json(path, record)
    return path


def discover():
    # Adopt missing tracking from the latest receipt. Existing tracking is
    # authoritative, including when the user explicitly replaced an older ID.
    receipts = []
    for directory in (ROOT / "cluster/.submissions").glob("*"):
        result, summary = directory / "result.json", directory / "summary.json"
        if result.is_file() and summary.is_file():
            receipts.append((result, summary, directory))
    for result, summary_path, directory in sorted(receipts, key=lambda item: item[0].stat().st_mtime,
                                                reverse=True):
        value = read_json(result)
        if value.get("experiment_id"):
            summary = read_json(summary_path)
            remote = summary["run"]["run_name"]
            name = remote if remote.startswith("CLU_") else "CLU_" + remote
            if not (OUTPUTS / name / TRACK_FILE).exists():
                register(summary, value["experiment_id"], directory)
    return sorted((p for p in OUTPUTS.glob("*/" + TRACK_FILE)
                   if read_json(p).get("format") == "swarmecho-cluster-run/v1"),
                  key=lambda p: read_json(p)["registered_at"], reverse=True)
