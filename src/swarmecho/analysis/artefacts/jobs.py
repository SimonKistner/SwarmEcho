"""
jobs.py — Job queue read/write/launch for the SwarmEcho Artefacts dashboard.
"""

from __future__ import annotations

import json
import subprocess
import time
import threading
from pathlib import Path


# ---------------------------------------------------------------------------
# PID liveness
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    # Check /proc/<pid>/status on Linux/WSL (avoids keeping zombie processes alive)
    from pathlib import Path
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.exists():
        try:
            content = proc_status.read_text(errors="ignore")
            for line in content.splitlines():
                if line.startswith("State:"):
                    state = line.split()[1]
                    if state.upper() == "Z":  # Zombie state
                        return False
                    return True
        except Exception:
            pass

    try:
        import psutil  # type: ignore[import]
        if not psutil.pid_exists(pid):
            return False
        try:
            status = psutil.Process(pid).status()
            if status == psutil.STATUS_ZOMBIE:
                return False
        except Exception:
            pass
        return True
    except ImportError:
        pass

    import os
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def reap_process(pid: int) -> bool:
    """Check if process is dead, attempting to reap it via waitpid first to clean up zombies on Unix."""
    if pid <= 0:
        return True
    import os
    # Try Unix waitpid to reap zombie child processes
    try:
        res_pid, status = os.waitpid(pid, os.WNOHANG)
        if res_pid == pid:
            return True
        if res_pid == 0:
            return False
    except OSError:
        pass

    # Fallback to general liveness checks
    return not _pid_alive(pid)


def kill_job(pid: int) -> bool:
    """Terminate a running job by PID. Returns True if signal was sent."""
    try:
        import psutil  # type: ignore[import]
        psutil.Process(pid).terminate()
        return True
    except Exception:
        pass
    import os, signal
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Log tail inspection
# ---------------------------------------------------------------------------

def _log_has_error(log_path: str | None) -> bool:
    if not log_path:
        return False
    path = Path(log_path)
    if not path.exists():
        return False
    tail = path.read_text(errors="ignore")[-4_000:].lower()
    return "traceback" in tail or "error:" in tail or "exception" in tail


# ---------------------------------------------------------------------------
# Write / launch
# ---------------------------------------------------------------------------

def start_job(eval_root: Path, name: str, command: list[str], repo_root: Path) -> dict:
    """Queue *command* as a queued job and write a job JSON file."""
    jobs_dir = eval_root / "jobs"
    logs_dir = jobs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"{int(time.time())}_{name}"
    log_path = logs_dir / f"{job_id}.log"

    payload = {
        "id": job_id,
        "name": name,
        "status": "queued",
        "pid": -1,
        "command": command,
        "log_path": str(log_path),
        "started_at": None,
        "finished_at": None,
    }
    _write_job_file(jobs_dir, payload)
    return payload


def _write_job_file(jobs_dir: Path, payload: dict) -> Path:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_path = jobs_dir / f"{payload['id']}.json"
    job_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return job_path


# ---------------------------------------------------------------------------
# Read queue
# ---------------------------------------------------------------------------

def read_queue(eval_root: Path) -> list[dict]:
    """Return all jobs for *eval_root*, updating running→finished/crashed in-place."""
    jobs_dir = eval_root / "jobs"
    if not jobs_dir.exists():
        return []

    items: list[dict] = []
    for job_file in sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(job_file.read_text())
        except Exception:
            continue

        if data.get("status") == "running":
            pid = int(data.get("pid", -1))
            if pid > 0 and reap_process(pid):
                data["status"] = "crashed" if _log_has_error(data.get("log_path")) else "finished"
                data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                job_file.write_text(json.dumps(data, indent=2, sort_keys=True))

        items.append(data)
    return items


def any_running(queue: list[dict]) -> bool:
    return any(j.get("status") in ("running", "queued") for j in queue)


def retry_job(eval_root: Path, job_id: str) -> bool:
    """Read a job by *job_id*, delete its old file, and create a prioritized queued copy."""
    jobs_dir = eval_root / "jobs"
    old_file = jobs_dir / f"{job_id}.json"
    if not old_file.exists():
        return False

    try:
        data = json.loads(old_file.read_text())
        name = data.get("name", "retry")
        command = data.get("command", [])

        # Delete old file
        old_file.unlink()

        # Write new prioritized queued job
        new_job_id = f"0_retry_{int(time.time())}_{name}"
        log_path = jobs_dir / "logs" / f"{new_job_id}.log"

        payload = {
            "id": new_job_id,
            "name": name,
            "status": "queued",
            "pid": -1,
            "command": command,
            "log_path": str(log_path),
            "started_at": None,
            "finished_at": None,
        }
        _write_job_file(jobs_dir, payload)
        return True
    except Exception:
        return False


def clear_queue_history(eval_root: Path):
    """Delete all finished, crashed, or cancelled job files and their logs."""
    jobs_dir = eval_root / "jobs"
    if not jobs_dir.exists():
        return

    for job_file in jobs_dir.glob("*.json"):
        try:
            data = json.loads(job_file.read_text())
            status = data.get("status")
            if status in ("finished", "crashed", "cancelled"):
                # Delete the log file first if it exists
                log_path = data.get("log_path")
                if log_path:
                    p = Path(log_path)
                    if p.exists():
                        p.unlink()
                # Delete the json file
                job_file.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sequential Queue Worker
# ---------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _try_launch_job(job_file: Path, repo_root: Path) -> bool:
    """Attempts to claim and launch a queued job. Returns True if successful."""
    try:
        data = json.loads(job_file.read_text())
        if data.get("status") != "queued":
            return False

        data["status"] = "running"
        data["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job_file.write_text(json.dumps(data, indent=2, sort_keys=True))

        log_path = Path(data["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("w") as log_fh:
            proc = subprocess.Popen(
                data["command"],
                cwd=str(repo_root),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

        data["pid"] = proc.pid
        job_file.write_text(json.dumps(data, indent=2, sort_keys=True))
        return True
    except Exception as e:
        import traceback
        import sys
        print(f"Error launching job {job_file}: {e}", file=sys.stderr)
        traceback.print_exc()
        try:
            data = json.loads(job_file.read_text())
            data["status"] = "crashed"
            data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job_file.write_text(json.dumps(data, indent=2, sort_keys=True))

            log_path = Path(data["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as log_fh:
                log_fh.write(f"\n--- Queue Launcher Exception ---\n")
                traceback.print_exc(file=log_fh)
        except Exception:
            pass
        return False


def _queue_worker_loop():
    repo_root = Path.cwd()
    outputs_dir = repo_root / "outputs"

    while True:
        try:
            time.sleep(1.0)
            if not outputs_dir.exists():
                continue

            # Find all JSON job files
            job_files = list(outputs_dir.glob("**/jobs/*.json"))
            running_jobs = []
            queued_jobs = []

            for jf in job_files:
                try:
                    data = json.loads(jf.read_text())
                except Exception:
                    continue

                status = data.get("status")
                if status == "running":
                    running_jobs.append((jf, data))
                elif status == "queued":
                    queued_jobs.append((jf, data))

            active_running = 0
            for jf, data in running_jobs:
                pid = int(data.get("pid", -1))
                if pid > 0:
                    if not reap_process(pid):
                        active_running += 1
                    else:
                        data["status"] = "crashed" if _log_has_error(data.get("log_path")) else "finished"
                        data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            jf.write_text(json.dumps(data, indent=2, sort_keys=True))
                        except Exception:
                            pass

            if active_running == 0 and queued_jobs:
                # Sort queued jobs by ID (which starts with timestamp)
                queued_jobs.sort(key=lambda item: item[1].get("id", ""))
                next_jf, next_data = queued_jobs[0]
                _try_launch_job(next_jf, repo_root)
        except Exception as e:
            import traceback
            import sys
            print(f"Error in queue worker loop: {e}", file=sys.stderr)
            traceback.print_exc()
            time.sleep(2.0)


def start_queue_worker():
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None:
            return
        _worker_thread = threading.Thread(target=_queue_worker_loop, daemon=True)
        _worker_thread.start()


# Automatically start worker thread on module import
start_queue_worker()
