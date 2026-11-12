"""CPU-only, dependency-free cluster capacity probe and joint KDE-guided search.

Only the disposable worker imports training/JAX. Every trial has an independent
CUDA context, fresh weights, a process-group deadline and persistent receipts.
"""
from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import queue as queue_module
import signal
import statistics
import subprocess
import threading
import time


BASELINE = dict(num_envs=4000, num_steps=100, num_epochs=4, num_minibatches=20)
PARAMETERS = tuple(BASELINE)
DEFAULTS = {
    "probe_seconds": 900, "candidate_seconds": 180, "probe_updates": 3,
    "trials": 16, "startup_trials": 5, "trial_seconds": 3600, "search_seconds": 21600,
    "prune_rungs_seconds": [600, 1200, 2400], "min_updates": 10,
    "min_completed_episodes": 1000, "headroom_fraction": 0.15,
    "envs": [1000, 2000, 4000, 8000, 16000, 32000], "steps": [50, 100, 200, 400],
    "epochs": [2, 4, 6, 8], "minibatches": [5, 10, 20, 40], "seed": 456789,
    "worker_retries": 1, "max_consecutive_errors": 3, "controller_restarts": 2,
    "capacity_seconds": 3600, "capacity_candidate_seconds": 480,
    "startup_trial_seconds": 1800,
}


class StudyFatal(RuntimeError):
    """Shared setup/storage/device failure: keep receipts and stop GPU work."""


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        if default is not None:
            return default
        raise


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def journal(directory, event, **values):
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                 "event": event, **values}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def report_safely(report, metrics):
    try:
        report(metrics)
    except Exception as error:
        # The PFSS receipts are authoritative; dashboard availability is optional.
        print(f"[TUNING REPORT WARNING] {type(error).__name__}: {error}", flush=True)


class ProgressReporter:
    """One bounded daemon; dashboard/network stalls cannot stall GPU supervision."""
    def __init__(self, callback):
        self.callback = callback
        self.pending = queue_module.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def __call__(self, metrics):
        try:
            self.pending.get_nowait()
        except queue_module.Empty:
            pass
        self.pending.put_nowait(dict(metrics))

    def work(self):
        while not self.stopping.is_set():
            try:
                metrics = self.pending.get(timeout=0.5)
            except queue_module.Empty:
                continue
            report_safely(self.callback, metrics)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=1)


@contextmanager
def study_lock(folder):
    import fcntl
    with (folder / "controller.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StudyFatal("Another controller still owns this study; no duplicate GPU workers launched.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_tuning(value):
    if not isinstance(value, dict) or value.get("phase") not in {"probe", "search", "explore"}:
        raise ValueError("tuning.phase must be probe, search or explore.")
    if set(value) - ({"phase"} | set(DEFAULTS)):
        raise ValueError("Unknown tuning fields: " + str(set(value) - ({"phase"} | set(DEFAULTS))))
    config = {**DEFAULTS, **value}
    for key in ("probe_seconds", "candidate_seconds", "probe_updates", "trials", "startup_trials",
                "trial_seconds", "search_seconds", "min_updates", "min_completed_episodes",
                "capacity_seconds", "capacity_candidate_seconds", "startup_trial_seconds"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("tuning." + key + " must be a positive integer.")
    if not 30 <= config["probe_seconds"] <= 900:
        raise ValueError("Probe budget must be 30–900 seconds.")
    if not 0.05 <= config["headroom_fraction"] <= 0.5:
        raise ValueError("headroom_fraction must be between 0.05 and 0.5.")
    for key in ("envs", "steps", "epochs", "minibatches", "prune_rungs_seconds"):
        items = config[key]
        if not isinstance(items, list) or not items or any(type(x) is not int or x < 1 for x in items):
            raise ValueError("tuning." + key + " must be a nonempty positive integer list.")
        if items != sorted(set(items)):
            raise ValueError("tuning." + key + " must be sorted and distinct.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("tuning.seed must be in [0, 2**32).")
    for key, upper in (("worker_retries", 2), ("controller_restarts", 3)):
        if type(config[key]) is not int or not 0 <= config[key] <= upper:
            raise ValueError("tuning." + key + " must be an integer between 0 and " + str(upper))
    if type(config["max_consecutive_errors"]) is not int or config["max_consecutive_errors"] < 1:
        raise ValueError("tuning.max_consecutive_errors must be a positive integer.")
    if config["phase"] == "explore" and config["capacity_seconds"] >= config["search_seconds"] - 60:
        raise ValueError("Exploration must leave at least 60 seconds after its capacity stage.")
    return config


def capture(command):
    try:
        value = subprocess.run(command, capture_output=True, text=True, timeout=2)
        return value.stdout.strip() if value.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_rows():
    output = capture(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                      "--format=csv,noheader,nounits"])
    return [dict(zip(("uuid", "name", "memory_mib", "driver"), (x.strip() for x in row)))
            for row in csv.reader(output.splitlines()) if len(row) == 4]


def hardware():
    gpus = gpu_rows()
    if not gpus:
        raise RuntimeError("nvidia-smi could not identify the allocated GPU. No candidates launched.")
    ram = capture(["cat", "/proc/meminfo"])
    limits = {}
    for name in ("memory.max", "cpu.max", "cpuset.cpus.effective", "memory/memory.limit_in_bytes"):
        try:
            limits[name] = (Path("/sys/fs/cgroup") / name).read_text().strip()
        except OSError:
            pass
    return {"gpus": gpus, "gpu_listing": capture(["nvidia-smi", "-L"]), "cgroup_limits": limits,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "cpu_count": os.cpu_count(), "meminfo": ram,
            "class": sorted({(g["name"], g["memory_mib"], g["driver"]) for g in gpus})}


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.suffix in {".py", ".yaml", ".yml"}
                       and "__pycache__" not in p.parts and "pools" not in p.parts):
        digest.update(file.relative_to(root).as_posix().encode() + b"\0")
        digest.update(file.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def signature(candidate):
    return tuple(candidate[key] for key in PARAMETERS)


def features(candidate):
    return [math.log2(candidate["num_envs"]), math.log2(candidate["num_steps"]),
            math.log2(candidate["num_epochs"]),
            math.log2(candidate["num_envs"] / candidate["num_minibatches"])]


def curve_score(row, until):
    points = sorted((float(p["wall_seconds"]), float(p["eval_success"]))
                    for p in row.get("evaluations", []) if float(p["wall_seconds"]) <= until)
    area, previous_t, previous_y = 0.0, 0.0, 0.0
    for t, y in points:
        area += (t - previous_t) * previous_y
        previous_t, previous_y = t, y
    area += max(0.0, until - previous_t) * previous_y
    return area / max(1.0, until)


def ranking(row, budget):
    if row.get("status") in {"oom", "error", "nonfinite"}:
        return (2, 0)
    if row.get("time_to_95_seconds") is not None:
        return (0, row["time_to_95_seconds"])
    # Missing future progress counts as zero after pruning, not as achieved success.
    measured = min(budget, row.get("wall_seconds", 0))
    return (1, -curve_score(row, measured) * measured / budget)


def suggest(pool, history, config, rng):
    """Small joint Parzen-density search over an explicitly feasible finite pool.

    Fit good/bad mixtures in log-scaled hardware/learning coordinates. Joint
    kernels preserve interactions; uniform exploration avoids local lock-in.
    This is intentionally a compact TPE-style method, not an Optuna dependency.
    """
    if not history:
        point = features(BASELINE)
        chosen = min(pool, key=lambda c: sum((a - b)**2 for a, b in zip(features(c), point)))
        return dict(chosen), "baseline" if chosen == BASELINE else "nearest_safe_baseline"
    if len(history) < config["startup_trials"]:
        # Space-filling warmup; include horizon/epoch/minibatch alternatives.
        previous = [features(row["candidate"]) for row in history]
        samples = rng.sample(pool, min(96, len(pool)))
        chosen = max(samples, key=lambda c: min(sum((a - b)**2 for a, b in zip(features(c), p))
                                               for p in previous))
        return dict(chosen), "space_filling"
    if rng.random() < 0.25:
        return dict(rng.choice(pool)), "exploration"
    ordered = sorted(history, key=lambda r: ranking(r, config["trial_seconds"]))
    cut = max(2, math.ceil(len(ordered) * 0.25))
    good, bad = ordered[:cut], ordered[cut:]
    if not bad:
        return dict(rng.choice(pool)), "exploration"

    def density(point, rows):
        centers = [features(r["candidate"]) for r in rows]
        widths = [max(0.5, statistics.pstdev([p[d] for p in centers]) * len(centers)**(-1 / 8))
                  for d in range(4)]
        return 1e-8 + sum(math.exp(-0.5 * sum(((a - b) / w)**2
                          for a, b, w in zip(point, center, widths)))
                          / math.prod(widths) for center in centers) / len(centers)

    chosen = max(rng.sample(pool, min(128, len(pool))),
                 key=lambda c: density(features(c), good) / density(features(c), bad))
    return dict(chosen), "joint_kde"


def safe_fit(row, config):
    # A timeout after two real optimizer updates is still useful capacity evidence.
    # A timeout before any progress must never be interpreted as an OOM boundary.
    if row.get("status") not in {"probe_ok", "converged", "max_steps", "pruned", "timeout"}:
        return False
    if row.get("status") == "timeout" and row.get("last_train", {}).get("update", 0) < 2:
        return False
    peak, limit = row.get("peak_device_bytes", 0), row.get("allocator_limit_bytes", 0)
    # Without allocator stats the fit is known but its headroom is unverified.
    return bool(peak and limit and peak <= limit * (1 - config["headroom_fraction"]))


def dominated(candidate, measured):
    n, t, _, m = signature(candidate)
    nn, tt, _, mm = signature(measured)
    return n <= nn and t <= tt and n * t <= nn * tt and n * t / m <= nn * tt / mm


def feasible_pool(probe, config):
    safe = [r["candidate"] for r in probe["results"] if safe_fit(r, config)]
    if not safe:
        raise RuntimeError("Probe established no fit with verified memory headroom. Inspect probe results; search was not started.")
    return [dict(zip(PARAMETERS, values)) for values in itertools.product(
        config["envs"], config["steps"], config["epochs"], config["minibatches"])
        if values[0] % values[3] == 0
        and any(dominated(dict(zip(PARAMETERS, values)), fit) for fit in safe)]


def prune_decision(progress, peers, config, checked):
    latest = progress.get("last_train", {})
    evaluations = progress.get("evaluations", [])
    if (int(latest.get("update", 0)) < config["min_updates"]
            or latest.get("completed_episodes", 0) < config["min_completed_episodes"]
            or len(evaluations) < 2 or progress.get("time_to_95_seconds") is not None):
        return False
    elapsed = progress.get("progress", {}).get("wall_seconds", 0)
    for rung in config["prune_rungs_seconds"]:
        if rung in checked or elapsed < rung:
            continue
        checked.add(rung)
        values = [curve_score(p, rung) for p in peers if p.get("wall_seconds", 0) >= rung
                  or p.get("time_to_95_seconds", math.inf) <= rung]
        if len(values) < 3:
            continue
        lower_quartile = sorted(values)[(len(values) - 1) // 4]
        if lower_quartile >= 0.30 and curve_score(progress, rung) < lower_quartile - 0.10:
            return True
    return False


def kill_group(process):
    # All GPU workers use their own POSIX session. Never signal the controller.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        raise StudyFatal("GPU worker could not be reaped after SIGKILL; stopped to avoid overlapping CUDA contexts.") from error


def run_attempt(spec, folder, python, env, seconds, deadline, config, peers, report):
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "trial.json", spec)
    started = time.monotonic()
    child_env = {**env, "PYTHONUNBUFFERED": "1",
                 "SWARMECHO_TRIAL_STARTED_MONOTONIC": str(spec.get("candidate_started_monotonic", started))}
    # Match both phases and the eventual tuning winner; leave physical VRAM free.
    child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    child_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
    child_env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    end = min(deadline, started + seconds)
    checked, forced, request_at, last_report, last_sample = set(), None, None, 0, 0
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(python), "-u", "-m",
            "swarmecho.training.cluster_tuning_worker", str(folder / "trial.json")],
            env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        expired = threading.Event()
        def expire():
            expired.set()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        watchdog = threading.Timer(max(0, end - time.monotonic()), expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= end:
                    forced = forced or "timeout"
                    kill_group(process)
                    break
                progress = read_json(folder / "progress.json", {})
                if now - last_sample >= 5 and end - now > 4:
                    sample = {"wall_seconds": now - started,
                              "gpu": capture(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                              "--format=csv,noheader,nounits"])}
                    try:
                        sample["host_status"] = Path(f"/proc/{process.pid}/status").read_text()
                    except OSError:
                        pass
                    with (folder / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(sample) + "\n")
                    last_sample = now
                if now - last_report >= 30:
                    latest = progress.get("progress", {})
                    print(f"[TUNING {spec['run_name']}] {now-started:.0f}s; "
                          f"update={latest.get('update', 0)}; "
                          f"steps={latest.get('global_step', 0):,.0f}; "
                          f"evals={len(progress.get('evaluations', []))}", flush=True)
                    report_safely(report, {"candidate_elapsed_seconds": now - started,
                            "candidate_update": latest.get("update", 0),
                            "candidate_steps": latest.get("global_step", 0),
                            "candidate_best_success": max((p["eval_success"] for p in
                                                            progress.get("evaluations", [])), default=0)})
                    last_report = now
                if (spec["mode"] == "search" and not spec.get("protected", spec["index"] == 0) and forced is None
                        and prune_decision(progress, peers, config, checked)):
                    forced, request_at = "pruned", now
                    save_json(folder / "stop.json", {"status": forced})
                if request_at is not None and now - request_at >= 30:
                    kill_group(process)
                    break
                time.sleep(min(1, max(0.01, end - time.monotonic())))
        finally:
            watchdog.cancel()
            watchdog.join()
            # Also remove W&B helpers left in the candidate's process group.
            kill_group(process)
    if expired.is_set():
        forced = forced or "timeout"
    result = read_json(folder / "result.json", read_json(folder / "progress.json", {}))
    if result.get("status") == "converged" and forced == "pruned" and not expired.is_set():
        forced = None # Reached the gate while a cooperative prune request was pending.
    if forced is not None:
        result["status"] = forced
    elif result.get("status") not in {"probe_ok", "converged", "max_steps", "pruned", "oom", "error", "nonfinite"}:
        result["status"] = "error"
    if not (folder / "result.json").exists() and forced is None:
        # A progress receipt is never evidence of normal completion.
        result["status"] = "oom" if process.returncode == 86 else "error"
    result.update(candidate=spec["candidate"], run_name=spec["run_name"], index=spec["index"],
                  wall_seconds=time.monotonic() - started, exit_code=process.returncode,
                  trial_dir=str(folder), sampling=spec.get("sampling"))
    # Keep complete tracebacks in worker.log; make failure visible in the summary.
    if result["status"] == "error" and "error" not in result:
        result["error"] = "Worker exited without a result; inspect worker.log."
    save_json(folder / "result.json", result)
    print(f"[TUNING RESULT] {spec['run_name']}: {result['status']}; "
          f"{result['wall_seconds']:.1f}s; {spec['candidate']}", flush=True)
    return result


def run_candidate(spec, folder, python, env, seconds, deadline, config, peers, report):
    """Bounded retries; every attempt owns new output paths and a fresh CUDA process."""
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "trial.json", spec)
    started = time.monotonic()
    end = min(deadline, started + seconds)
    attempts = []
    result = {}
    last_spec = dict(spec, candidate_started_monotonic=started)
    logging_fallback = False
    for attempt in range(config["worker_retries"] + 1):
        if attempt and time.monotonic() + 40 >= end:
            break
        path = folder / f"attempt_{attempt:02d}"
        offset = time.monotonic() - started
        last_spec = dict(last_spec, run_name=spec["run_name"] + f"_a{attempt:02d}")
        journal(folder.parent, "attempt_started", trial=spec["index"], attempt=attempt,
                candidate=spec["candidate"], folder=str(path))
        try:
            result = run_attempt(last_spec, path, python, env, max(0, end - time.monotonic()),
                                 end, config, peers, report)
        except StudyFatal:
            raise
        except OSError as error:
            # Cannot rely on this storage/device environment; don't spend more GPU time.
            raise StudyFatal(f"Worker supervision/storage failed: {error}") from error
        except Exception:
            import traceback
            result = {**read_json(path / "progress.json", {}), "status": "error",
                      "failure_kind": "worker_error", "error": traceback.format_exc(),
                      "trial_dir": str(path), "candidate": spec["candidate"]}
        result["attempt_offset_seconds"] = offset
        attempts.append({"attempt": attempt, "status": result["status"],
                         "failure_kind": result.get("failure_kind"), "folder": str(path),
                         "wall_seconds": time.monotonic() - started - offset,
                         "error": result.get("error")})
        result.update(index=spec["index"], candidate=spec["candidate"], attempts=attempts,
                      run_name=last_spec["run_name"], wall_seconds=time.monotonic() - started)
        save_json(folder / "result.json", result)
        journal(folder.parent, "attempt_finished", trial=spec["index"], attempt=attempt,
                status=result["status"], failure_kind=result.get("failure_kind"))
        if result.get("failure_kind") in {"setup_error", "storage_error"}:
            raise StudyFatal(result.get("error", "Shared study setup/storage failure."))
        if result["status"] != "error":
            break # Never retry the same OOM, NaN, timeout or deliberately pruned trial.
        if (result.get("failure_kind") == "logging_error" and attempt < config["worker_retries"]
                and time.monotonic() + 40 < end):
            last_spec["wandb_mode"] = "disabled"
            logging_fallback = True
            print("[TUNING] W&B failed; retrying once with local logging only.", flush=True)
    result.update(attempts=attempts, wall_seconds=time.monotonic() - started)
    if logging_fallback:
        result["logging_fallback"] = "Local JSON curves; W&B unavailable."
    save_json(folder / "result.json", result)
    return result


def write_results(folder, summary):
    save_json(folder / "summary.json", summary)
    columns = ["index", "run_name", "status", *PARAMETERS, "wall_seconds", "time_to_95_seconds",
               "best_success", "steady_training_sps", "peak_device_gib", "headroom_verified"]
    path = folder / "results.csv"
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summary["results"]:
            training = [p for p in read_progress(row) if p.get("phase") == "training" and p["update"] > 1]
            writer.writerow({"index": row["index"], "run_name": row["run_name"],
                "status": row["status"], **row["candidate"], "wall_seconds": row["wall_seconds"],
                "time_to_95_seconds": row.get("time_to_95_seconds", ""),
                "best_success": max((e["eval_success"] for e in row.get("evaluations", [])), default=0),
                "steady_training_sps": statistics.median(
                    row["candidate"]["num_envs"] * row["candidate"]["num_steps"] / p["timing"]["total_s"]
                    for p in training) if training else "",
                "peak_device_gib": row.get("peak_device_bytes", 0) / 2**30,
                "headroom_verified": safe_fit(row, summary["config"])})
    temp.replace(path)


def read_progress(row):
    try:
        return [json.loads(line) for line in (Path(row["trial_dir"]) / "progress.jsonl").read_text().splitlines()]
    except (OSError, ValueError):
        return []


def recover_pending(summary, details):
    pending = summary.pop("pending_trial", None)
    if pending is None or any(r["index"] == pending["index"] for r in summary["results"]):
        return
    folder = details / f"trial_{pending['index']:03d}"
    result = read_json(folder / "result.json", {})
    attempts = sorted(folder.glob("attempt_*"))
    last = attempts[-1] if attempts else folder
    if not result or Path(result.get("trial_dir", "")).name != last.name:
        prior_attempts = result.get("attempts", [])
        result = read_json(last / "result.json", read_json(last / "progress.json", {}))
        # A worker-only receipt has not been accepted by its CPU supervisor.
        # Keep evidence; do not promote it to a successful/converged trial.
        result.update(status="interrupted", trial_dir=str(last),
                      attempts=prior_attempts,
                      error="Controller/allocation interrupted before accepting this attempt.")
    result.update(index=pending["index"], candidate=pending["candidate"],
                  stage=pending.get("stage", "capacity" if summary["phase"] == "probe" else "learning"),
                  run_name=result.get("run_name", pending["run_name"]),
                  wall_seconds=result.get("wall_seconds", max(0, time.time() - pending["started_at"])))
    if result["status"] == "interrupted":
        result.pop("time_to_95_seconds", None)
    summary["results"].append(result)
    journal(details, "trial_recovered", index=pending["index"], status=result["status"])


def run(manifest, root, python, env, report):
    if os.name != "posix":
        raise RuntimeError("The tuning controller runs in the Linux cluster container only.")
    folder = root / "outputs" / manifest["run_name"]
    folder.mkdir(parents=True, exist_ok=True)
    details = folder / "tuning"
    details.mkdir(exist_ok=True)
    with study_lock(details):
        reporter = ProgressReporter(report)
        try:
            return run_locked(manifest, root, python, env, reporter, folder, details)
        except BaseException as error:
            # Never lose the last trial just because final reporting/cleanup failed.
            summary = read_json(details / "summary.json", {})
            if summary:
                try:
                    recover_pending(summary, details)
                    summary.update(status="failed" if isinstance(error, StudyFatal) else "interrupted",
                                   controller_error=f"{type(error).__name__}: {error}")
                    write_results(details, summary)
                    save_json(folder / "metrics.json", {
                        "tuning_phase": summary["phase"], "tuning_status": summary["status"],
                        "trials": len(summary["results"]), "summary": str(details / "summary.json")})
                except Exception as persistence_error:
                    print(f"[TUNING PERSISTENCE ERROR] {persistence_error}", flush=True)
                    print("[TUNING LAST SUMMARY] " + json.dumps(summary), flush=True)
            raise
        finally:
            reporter.close()


def run_locked(manifest, root, python, env, report, folder, details):
    started = time.monotonic()
    config = validate_tuning(manifest["tuning"])
    phase = config["phase"]
    fingerprint = source_fingerprint()
    previous = read_json(details / "summary.json") if (details / "summary.json").exists() else None
    if previous is not None:
        if previous["config"] != config or previous["source_fingerprint"] != fingerprint:
            raise StudyFatal("Cannot resume this study with different settings/source; use a new submission.")
        if previous["status"] == "completed":
            return previous # Reporting retry: never repeat finished GPU work.
        if previous["status"] == "failed":
            raise StudyFatal(previous.get("controller_error", "This study already stopped on a shared failure."))
    machine = hardware()
    if previous is not None and previous["hardware"]["class"] != json.loads(json.dumps(machine["class"])):
        raise StudyFatal("GPU class changed on allocation restart; retained results, stopped GPU work.")
    exploratory_queue = [dict(zip(PARAMETERS, values)) for values in (
        (4000, 100, 1, 20), (16000, 100, 1, 40), (64000, 25, 1, 160),
        (8000, 400, 1, 40), (2000, 700, 1, 40), (1000, 1400, 1, 40),
        (32000, 50, 1, 80), (12000, 200, 1, 40), (8000, 100, 1, 5), (8000, 100, 1, 80))]
    summary = previous or {"format": "swarmecho-cluster-tuning/v1", "phase": phase, "status": "running",
               "run_name": manifest["run_name"], "created_at": datetime.now(timezone.utc).isoformat(),
               "hardware": machine, "source_fingerprint": fingerprint, "config": config,
               "deadline_epoch": time.time() + (config["probe_seconds"] if phase == "probe" else config["search_seconds"]),
               "capacity_deadline_epoch": time.time() + config["capacity_seconds"],
               "stage": "learning" if phase == "search" else "capacity",
               "wandb_disabled": manifest["wandb_mode"] == "disabled",
               "allocator": {"preallocate": True, "memory_fraction": 0.85}, "results": [],
               "probe_queue": [dict(BASELINE), dict(BASELINE, num_envs=8000),
                               dict(BASELINE, num_steps=200), dict(BASELINE, num_envs=2000, num_steps=400)],
               "restart_count": -1}
    if previous is None and phase == "explore":
        summary["probe_queue"] = exploratory_queue
    summary["restart_count"] = summary.get("restart_count", -1) + 1
    if summary["restart_count"] > config["controller_restarts"]:
        raise StudyFatal("Controller restart limit reached; partial results retained.")
    recover_pending(summary, details)
    summary["status"] = "running"
    if (folder / "metrics.json").exists():
        (folder / "metrics.json").unlink() # Remove the previous interruption's completion marker.
    journal(details, "controller_started", restart=summary["restart_count"], deadline=summary["deadline_epoch"])
    save_json(folder / "tuning_run.json", summary)
    write_results(details, summary)
    save_json(details / "hardware.json", machine)
    latest = root / "tuning" / "G01_CLU_static" / "latest_probe.json"
    rng = random.Random(config["seed"])
    if "rng_state" in summary:
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        rng.setstate(tuples(summary["rng_state"]))
    deadline = time.monotonic() + max(0, summary["deadline_epoch"] - time.time()) - 10
    queue = summary["probe_queue"]
    seen = set()
    expected_runtime = None
    if phase == "search":
        pointer = read_json(latest, {})
        if not pointer:
            raise StudyFatal("No capacity probe found. Run clu-g01-probe first.")
        probe_path = Path(pointer["summary"])
        allowed = root.resolve() / "outputs"
        if allowed not in probe_path.resolve().parents:
            raise ValueError("Probe pointer must refer to this project's outputs.")
        probe = read_json(probe_path)
        if probe["source_fingerprint"] != fingerprint:
            raise StudyFatal("Source changed since capacity probe; rerun clu-g01-probe before searching.")
        if probe["hardware"]["class"] != json.loads(json.dumps(machine["class"])):
            raise StudyFatal("GPU model, VRAM or driver differs from probe; rerun probe on this GPU class.")
        try:
            pool = feasible_pool(probe, config)
        except RuntimeError as error:
            raise StudyFatal(str(error)) from error
        reference = next(row for row in probe["results"] if safe_fit(row, config))
        expected_runtime = {"device": reference["device"], "allocator_limit_bytes": reference["allocator_limit_bytes"]}
        summary["probe_summary"] = str(probe_path)
        summary["feasible_candidates"] = len(pool)
        if not pool:
            raise RuntimeError("No search candidates fit the measured conservative envelope.")
    write_results(details, summary)
    index = max((r["index"] for r in summary["results"]), default=-1) + 1
    errors = summary.get("consecutive_errors", 0)
    while time.monotonic() + 40 < deadline:
        stage = summary["stage"]
        seen = {signature(r["candidate"]) for r in summary["results"]
                if r.get("stage") == stage and r["status"] != "interrupted"}
        if phase == "explore" and stage == "capacity" and (
                time.time() + 40 >= summary["capacity_deadline_epoch"] or not queue):
            summary["stage"] = stage = "learning"
            errors = 0
            journal(details, "stage_changed", stage=stage)
            seen = {signature(r["candidate"]) for r in summary["results"]
                    if r.get("stage") == stage and r["status"] != "interrupted"}
        learning_history = [r for r in summary["results"] if r.get("stage") == "learning"]
        if stage == "learning" and len(learning_history) >= config["trials"]:
            break
        if phase == "explore" and stage == "learning":
            capacity = [r for r in summary["results"] if r.get("stage") == "capacity"]
            try:
                pool = feasible_pool({"results": capacity}, config)
            except RuntimeError as error:
                raise StudyFatal(str(error)) from error
            reference = next(r for r in capacity if safe_fit(r, config))
            expected_runtime = {"device": reference["device"], "allocator_limit_bytes": reference["allocator_limit_bytes"]}
            summary["feasible_candidates"] = len(pool)
        if stage == "capacity":
            queue = [c for c in queue if signature(c) not in seen and not any(
                r["status"] == "oom" and dominated(r["candidate"], c) for r in summary["results"])]
            if not queue:
                if phase == "explore":
                    summary["stage"] = "learning"
                    continue
                break
            candidate, sampling = queue.pop(0), "capacity"
        else:
            remaining = [c for c in pool if signature(c) not in seen and not any(
                r["status"] == "oom" and dominated(r["candidate"], c) for r in summary["results"])]
            if not remaining:
                break
            candidate, sampling = suggest(remaining, learning_history, config, rng)
        seen.add(signature(candidate))
        name = manifest["run_name"] + f"_t{index:03d}_n{candidate['num_envs']}_h{candidate['num_steps']}_e{candidate['num_epochs']}_m{candidate['num_minibatches']}"
        spec = {"mode": "probe" if stage == "capacity" else "search", "index": index,
                "protected": stage == "learning" and not learning_history,
                "sampling": sampling, "candidate": candidate,
                "probe_updates": config["probe_updates"], "outputs": str(root / "outputs"),
                "run_name": name, "group": manifest["run_name"], "wandb_mode": manifest["wandb_mode"],
                "expected_runtime": expected_runtime,
                "overrides": manifest["overrides"] + ["training.seed=" + str(config["seed"])]}
        if stage == "capacity" or summary.get("wandb_disabled"):
            spec["wandb_mode"] = "disabled"
        summary.update(pending_trial={"index": index, "candidate": candidate,
                                      "stage": stage, "run_name": name, "started_at": time.time()},
                       probe_queue=queue, rng_state=rng.getstate())
        write_results(details, summary)
        journal(details, "trial_started", index=index, candidate=candidate)
        seconds = config["candidate_seconds"] if stage == "capacity" else config["trial_seconds"]
        candidate_deadline = deadline
        if phase == "explore":
            if stage == "capacity":
                seconds = config["capacity_candidate_seconds"]
                candidate_deadline = min(deadline, time.monotonic() + max(0, summary["capacity_deadline_epoch"] - time.time()))
            elif len(learning_history) < config["startup_trials"]:
                seconds = min(config["trial_seconds"], config["startup_trial_seconds"])
        result = run_candidate(spec, details / f"trial_{index:03d}", python, env,
            seconds, candidate_deadline, config, learning_history, report)
        result["stage"] = stage
        if result.get("logging_fallback"):
            summary["wandb_disabled"] = True
        summary["results"].append(result)
        summary.pop("pending_trial", None)
        write_results(details, summary)
        report_safely(report, {"candidates_completed": index + 1,
                "candidates_converged": sum(r["status"] == "converged" for r in summary["results"]),
                "candidate_failed": int(result["status"] in {"oom", "error", "nonfinite"})})
        if stage == "capacity":
            n, t, _, m = signature(candidate)
            fit = result["status"] == "probe_ok" or safe_fit(result, config)
            if phase == "explore":
                same_shape_oom = [r["candidate"]["num_envs"] for r in summary["results"]
                                  if r["status"] == "oom" and r["candidate"]["num_steps"] == t
                                  and r["candidate"]["num_minibatches"] == m and r["candidate"]["num_envs"] > n]
                if fit and n < max(config["envs"]):
                    if not safe_fit(result, config):
                        queue.insert(0, dict(candidate, num_envs=max(m, int(n * 0.8) // m * m)))
                    next_n = (n + min(same_shape_oom)) // 2 if same_shape_oom else min(max(config["envs"]), n * 2)
                    next_n = max(m, next_n // m * m)
                    if next_n > n:
                        queue.append(dict(candidate, num_envs=next_n))
                elif result["status"] == "oom" and n > min(config["envs"]):
                    lower = [r["candidate"]["num_envs"] for r in summary["results"]
                             if r.get("stage") == "capacity" and r["status"] == "probe_ok"
                             and r["candidate"]["num_steps"] == t and r["candidate"]["num_minibatches"] == m
                             and r["candidate"]["num_envs"] < n]
                    next_n = (max(lower) + n) // 2 if lower else max(min(config["envs"]), n // 2)
                    next_n = max(m, next_n // m * m)
                    if next_n < n:
                        queue.insert(0, dict(candidate, num_envs=next_n))
            elif index == 0 and fit:
                queue = [dict(candidate, num_envs=n * 2),
                         dict(candidate, num_envs=n, num_steps=200),
                         dict(candidate, num_envs=max(1000, n // 2), num_steps=400),
                         dict(candidate, num_minibatches=40), dict(candidate, num_minibatches=10)]
            elif fit and t == 100 and m == 20 and n < max(config["envs"]):
                queue.append(dict(candidate, num_envs=min(max(config["envs"]), n * 2)))
            elif result["status"] == "oom" and n > min(config["envs"]):
                smaller = dict(candidate, num_envs=max(min(config["envs"]), n // 2))
                if signature(smaller) not in seen:
                    queue.insert(0, smaller)
            if phase != "explore" and index == 0 and result["status"] != "probe_ok":
                queue += [dict(BASELINE, num_envs=1000, num_steps=200),
                          dict(BASELINE, num_envs=1000, num_steps=400)]
        errors = errors + 1 if result["status"] == "error" else 0
        summary.update(probe_queue=queue, consecutive_errors=errors)
        write_results(details, summary)
        if errors >= config["max_consecutive_errors"]:
            raise StudyFatal("Repeated worker errors triggered the circuit breaker; partial results retained.")
        index += 1
    if summary["status"] == "running":
        summary["status"] = "completed"
    summary["wall_seconds"] = (config["probe_seconds"] if phase == "probe" else config["search_seconds"]) - max(0, summary["deadline_epoch"] - time.time())
    summary["ranking"] = [r["run_name"] for r in sorted(
        (r for r in summary["results"] if r.get("stage") == "learning"),
        key=lambda r: ranking(r, config["trial_seconds"]))] if phase != "probe" else []
    summary["safe_probe_candidates"] = sum(safe_fit(r, config) for r in summary["results"])
    if phase == "probe" and not summary["safe_probe_candidates"]:
        summary["status"] = "failed"
    write_results(details, summary)
    if phase != "probe":
        winners = [r for r in summary["results"] if r["status"] == "converged"
                   and r.get("time_to_95_seconds") is not None]
        if winners:
            best = min(winners, key=lambda r: r["time_to_95_seconds"])
            save_json(details / "best_run.json", {
                "description": "Best observed G01 configuration; confirm across seeds before production",
                "level": "G01_CLU_static", "wandb_mode": "online",
                "overrides": [f"training.{key}={value}" for key, value in best["candidate"].items()]})
    save_json(folder / "metrics.json", {"tuning_phase": phase, "tuning_status": summary["status"],
                                      "trials": len(summary["results"]), "summary": str(details / "summary.json")})
    if phase == "probe" and summary["status"] == "completed" and summary["safe_probe_candidates"]:
        save_json(latest, {"summary": str(details / "summary.json"), "run_name": manifest["run_name"]})
    print(f"[TUNING] {phase} {summary['status']}; results: {details / 'results.csv'}", flush=True)
    if summary["status"] != "completed":
        raise StudyFatal("Study did not establish a verified safe capacity; partial results retained. Inspect summary.json and worker logs.")
    if phase == "probe" and not summary["safe_probe_candidates"]:
        raise StudyFatal("Probe finished without a verified safe fit; latest successful probe was not replaced.")
    return summary
