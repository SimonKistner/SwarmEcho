"""Read-only workspace GPU occupancy, attributed by project and signed-in user.

Standard library only; uses the authenticated det CLI. Generic task responses
can contain connection keys: consume them in memory and retain only the explicit
metadata whitelist below. Never print or persist raw task responses.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from plan import WORKSPACE_SLOT_LIMIT

CLUSTER = Path(__file__).resolve().parents[1]
GENERIC_TASKS = {"TYPE_SHELL": "shell", "TYPE_NOTEBOOK": "notebook",
                 "TYPE_TENSORBOARD": "tensorboard", "TYPE_COMMAND": "command"}


class Det:
    def __init__(self, settings):
        self.executable = shutil.which("det")
        if not self.executable:
            raise RuntimeError("det is unavailable; install the cluster CLI first.")
        self.environment = dict(os.environ, DET_MASTER=settings["master"])

    def text(self, *args, allow_partial=False):
        result = subprocess.run([self.executable, *args], env=self.environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if result.returncode and not allow_partial:
            raise RuntimeError("det " + " ".join(args[:2]) + " failed: " + result.stderr.strip())
        return result.stdout

    def json(self, *args, allow_partial=False):
        try:
            return json.loads(self.text(*args, "--json", allow_partial=allow_partial))
        except json.JSONDecodeError:
            if allow_partial:
                return []
            # Raw stdout can contain credentials, so do not include it in errors.
            raise RuntimeError("Unexpected JSON from det " + " ".join(args[:2])) from None


def task_metadata(records, kind):
    """Whitelist fields; shells/notebooks/commands are workspace-level tasks."""
    return {row["jobId"]: {
        "id": row["id"], "name": row.get("description", ""), "kind": kind,
        "username": row.get("username"), "project_id": row.get("projectId"),
        "project": row.get("projectName"), "state": row.get("state"),
    } for row in records}


def summarize(jobs, metadata, *, workspace_id, project_id, username):
    """Count allocated slots, never requested slots or the number of tasks.

    Outside-project and outside-user are independent, overlapping totals.
    Missing metadata stays unknown instead of silently becoming 'someone else'.
    """
    allocated = outside_project = outside_user = unknown_project = unknown_user = 0
    allocations = []
    shell_count = zero_gpu_shells = 0
    for job in jobs:
        if job.get("workspaceId") != workspace_id:
            continue
        slots = int(job.get("allocatedSlots") or 0)
        info = metadata.get(job["jobId"])
        if job["type"] == "TYPE_SHELL" and info and info.get("state") == "RUNNING":
            shell_count += 1
            zero_gpu_shells += slots == 0
        if slots <= 0:
            continue
        allocated += slots
        if info is None:
            unknown_project += slots
            unknown_user += slots
        else:
            if info["project_id"] != project_id:
                outside_project += slots
            if info["username"] is None:
                unknown_user += slots
            elif info["username"] != username:
                outside_user += slots
        allocations.append(dict(job_id=job["jobId"], slots=slots,
            kind=info["kind"] if info else job["type"],
            id=info["id"] if info else None, name=info["name"] if info else "unknown",
            username=info["username"] if info else None,
            project=info["project"] if info else None,
            project_id=info["project_id"] if info else None,
            metadata_available=info is not None))
    return dict(allocated_slots=allocated, slot_limit=WORKSPACE_SLOT_LIMIT,
        free_slots=max(0, WORKSPACE_SLOT_LIMIT-allocated),
        over_limit_slots=max(0, allocated-WORKSPACE_SLOT_LIMIT),
        outside_project_slots=outside_project, outside_user_slots=outside_user,
        unknown_project_slots=unknown_project, unknown_user_slots=unknown_user,
        running_shells=shell_count, zero_gpu_shells=zero_gpu_shells, allocations=allocations)


def collect(settings, det):
    project = det.json("project", "describe", settings["workspace"], settings["project"])
    if project["id"] != settings["project_id"]:
        raise ValueError("The configured project name and project_id refer to different projects.")
    workspace_id = project["workspaceId"]
    identity = det.text("user", "whoami")
    match = re.search(r"logged in as user ['\"]([^'\"]+)['\"]", identity)
    if not match:
        raise RuntimeError("Cannot determine the authenticated username from det user whoami.")
    username = match.group(1)
    # The CLI's unfiltered job list uses the default pool, not every pool.
    jobs = det.json("job", "list", "--resource-pool", settings["resource_pool"], "--pages", "all")["jobs"]
    checked_at = datetime.now(timezone.utc).isoformat()
    workspace_jobs = [row for row in jobs if row.get("workspaceId") == workspace_id]
    needed = {row["jobId"] for row in workspace_jobs if row.get("allocatedSlots", 0) > 0}
    metadata = {}
    if any(row["type"] == "TYPE_EXPERIMENT" and row["jobId"] in needed for row in workspace_jobs):
        # This cluster's live trial task IDs start with '<experiment ID>.'.
        # Resolve just those IDs, avoiding the very large historical experiment
        # listing. Unknown task formats remain explicitly unclassified below.
        tasks = det.json("task", "list")
        tasks = tasks.values() if isinstance(tasks, dict) else tasks
        candidates = sorted({match.group(1) for row in tasks
            if row.get("resourcePool") == settings["resource_pool"]
            for match in [re.match(r"^(\d+)\.", row.get("taskId", ""))] if match})
        for start in range(0, len(candidates), 40):
            # The CLI emits successful records even if a task disappeared or is
            # inaccessible. Preserve those; missing ownership stays unknown.
            records = det.json("experiment", "describe", ",".join(candidates[start:start+40]), allow_partial=True)
            for record in records:
                row = record["experiment"]
                if row["jobId"] not in needed:
                    continue
                metadata[row["jobId"]] = dict(id=row["id"], name=row["name"], kind="experiment",
                    username=row.get("username"), project_id=row["projectId"], project=row.get("projectName"))
    for job_type, command in GENERIC_TASKS.items():
        if any(row["type"] == job_type for row in workspace_jobs):
            records = det.json(command, "list", "--all", "-w", settings["workspace"])
            metadata.update(task_metadata(records, command))
    result = summarize(workspace_jobs, metadata, workspace_id=workspace_id,
        project_id=project["id"], username=username)
    result.update(workspace=settings["workspace"], workspace_id=workspace_id,
        project=settings["project"], project_id=project["id"], username=username,
        resource_pool=settings["resource_pool"], checked_at=checked_at)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=CLUSTER/"settings.json")
    parser.add_argument("--json", action="store_true", help="Print sanitized machine-readable metadata.")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8-sig"))
    result = collect(settings, Det(settings))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("{workspace} / {resource_pool}: {allocated_slots}/{slot_limit} GPU slots allocated; {free_slots} free".format(**result))
    if result["unknown_project_slots"] or result["unknown_user_slots"]:
        print("Some allocation ownership is unavailable (metadata changed or is inaccessible).")
    print("\nSlots  Type         User          Project             Name")
    for row in sorted(result["allocations"], key=lambda r:(r["kind"], r["name"])):
        print("{:<5}  {:<11}  {:<12}  {:<18}  {} ({})".format(row["slots"],row["kind"],
            row["username"] or "unknown", row["project"] or ("workspace task" if row["metadata_available"] else "unknown"),
            row["name"],row["id"] or row["job_id"]))
    print("\nzero-GPU shells: {zero_gpu_shells}".format(**result))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
