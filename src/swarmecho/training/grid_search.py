"""
General sequential grid search for SwarmEcho training configurations.

Examples
--------
Sweep two training parameters:

    uv run swarmecho-grid-search M01_small_maze \
        --grid training.lr=0.0001,0.0003 \
        --grid training.ent_coef=0.0,0.01

Add fixed overrides, randomise run order, and cap each run at 20 minutes:

    uv run swarmecho-grid-search M01_small_maze \
        --name m04_optimizer \
        --grid training.lr=0.0001,0.0003 \
        --grid training.num_epochs=1,5 \
        --set evaluation.eval_video=false \
        --shuffle \
        --timeout-seconds 1200

Each ``--grid`` value is passed directly as an OmegaConf/Hydra override.
Results are resumable and written to ``outputs/grid_search_<name>.json`` plus
a human-readable Markdown summary.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return cleaned or "search"


def _parse_assignment(text: str, *, multiple_values: bool) -> tuple[str, list[str]]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, received: {text!r}")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key or "." not in key:
        raise argparse.ArgumentTypeError(
            f"Configuration key must include its section, for example training.lr: {key!r}"
        )
    values = [value.strip() for value in raw_value.split(",")] if multiple_values else [raw_value.strip()]
    if any(value == "" for value in values):
        raise argparse.ArgumentTypeError(f"Empty value in assignment: {text!r}")
    return key, values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a general SwarmEcho configuration grid search.")
    parser.add_argument("level", help="Level name passed as level=<name> to train.py.")
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="KEY=V1,V2",
        help="Repeatable parameter axis. Values are comma-separated.",
    )
    parser.add_argument(
        "--set",
        dest="fixed_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable fixed override applied to every run.",
    )
    parser.add_argument("--name", help="Search/output name; defaults to the level name.")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Per-run wall-clock limit.")
    parser.add_argument("--total-timesteps", type=int, help="Override training.total_timesteps for every run.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle pending combinations.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for --shuffle.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    args = parser.parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.total_timesteps is not None and args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    return args


def _build_spec(args: argparse.Namespace) -> tuple[dict[str, list[str]], list[str]]:
    axes: dict[str, list[str]] = {}
    for text in args.grid:
        key, values = _parse_assignment(text, multiple_values=True)
        if key in axes:
            raise ValueError(f"Duplicate --grid key: {key}")
        axes[key] = values

    fixed: list[str] = []
    fixed_keys: set[str] = set()
    for text in args.fixed_overrides:
        key, values = _parse_assignment(text, multiple_values=False)
        if key in axes:
            raise ValueError(f"{key} cannot be both --grid and --set")
        if key in fixed_keys:
            raise ValueError(f"Duplicate --set key: {key}")
        fixed_keys.add(key)
        fixed.append(f"{key}={values[0]}")
    return axes, fixed


def _combo_id(combo: dict[str, str]) -> str:
    payload = json.dumps(combo, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _enqueue_output(stream, output_queue: queue.Queue[str]) -> None:
    for line in iter(stream.readline, ""):
        output_queue.put(line)
    stream.close()


def _run_command(cmd: list[str], timeout_seconds: float | None) -> tuple[str, bool, int | None]:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_enqueue_output, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    output: list[str] = []
    started = time.perf_counter()
    timed_out = False
    try:
        while True:
            try:
                line = output_queue.get(timeout=1.0)
                sys.stdout.write(line)
                sys.stdout.flush()
                output.append(line)
            except queue.Empty:
                if process.poll() is not None:
                    break
            if timeout_seconds is not None and time.perf_counter() - started > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise

    while not output_queue.empty():
        line = output_queue.get_nowait()
        sys.stdout.write(line)
        output.append(line)
    return "".join(output), timed_out, process.poll()


def _parse_metrics(output: str) -> dict:
    metrics = {
        "steps": 0,
        "sps": 0.0,
        "ep_return": "----",
        "coverage": "----",
        "found": "----",
        "success": "----",
    }
    lines = output.splitlines()
    for line in reversed(lines):
        if "steps=" in line and "sps=" in line:
            patterns = {
                "steps": r"steps=\s*([\d,]+)",
                "sps": r"sps=\s*([\d,]+)",
                "coverage": r"cov=\s*([^\s]+)",
                "found": r"found=\s*([^\s]+)",
                "success": r"succ=\s*([^\s]+)",
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = match.group(1)
                    if key == "steps":
                        metrics[key] = int(value.replace(",", ""))
                    elif key == "sps":
                        metrics[key] = float(value.replace(",", ""))
                    else:
                        metrics[key] = value
            break
    for line in reversed(lines):
        if "[eval]" in line and "ep_return=" in line:
            match = re.search(r"ep_return=\s*([-\d.]+)", line)
            if match:
                metrics["ep_return"] = match.group(1)
            break
    return metrics


def _failure_status(output: str, return_code: int | None) -> tuple[str, str]:
    allocation = re.search(
        r"(?:trying to allocate|allocate)\s+([\d.]+\s*(?:GiB|MiB|KiB|B|GB|MB|KB))",
        output,
        re.IGNORECASE,
    )
    if allocation or "RESOURCE_EXHAUSTED" in output or "Out of memory" in output:
        return "OOM", f"Allocation failed: {allocation.group(1) if allocation else 'unknown size'}"
    if "ValueError" in output:
        return "INVALID_CONFIG", "Validation ValueError"
    return "FAILED", f"Exit code {return_code}"


def _write_outputs(path_base: Path, payload: dict) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = path_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    axes = list(payload["axes"])
    results = payload["results"]

    def coverage_key(result: dict) -> float:
        raw = str(result["metrics"]["coverage"]).replace("%", "")
        try:
            return float(raw)
        except ValueError:
            return -1.0

    lines = [
        f"# Grid Search: {payload['name']}",
        "",
        f"Level: `{payload['level']}`",
        "",
        "| Rank | Run | " + " | ".join(axes) + " | Status | Steps | SPS | Coverage | Found | Success | Eval Return | Details |",
        "|---|---|" + "---|" * len(axes) + "---|---|---|---|---|---|---|---|",
    ]
    for rank, result in enumerate(sorted(results, key=coverage_key, reverse=True), 1):
        values = " | ".join(str(result["combo"][key]).replace("|", "\\|") for key in axes)
        metrics = result["metrics"]
        lines.append(
            f"| {rank} | `{result['run_name']}` | {values} | **{result['status']}** | "
            f"{metrics['steps']:,} | {metrics['sps']:,.0f} | {metrics['coverage']} | "
            f"{metrics['found']} | {metrics['success']} | {metrics['ep_return']} | {result['details']} |"
        )
    path_base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    axes, fixed = _build_spec(args)
    search_name = _safe_name(args.name or args.level)
    output_base = Path("outputs") / f"grid_search_{search_name}"
    spec = {
        "name": search_name,
        "level": args.level,
        "axes": axes,
        "fixed_overrides": fixed,
        "total_timesteps": args.total_timesteps,
    }

    payload = {**spec, "results": []}
    json_path = output_base.with_suffix(".json")
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        existing_spec = {key: payload.get(key) for key in spec}
        if existing_spec != spec:
            raise ValueError(
                f"Existing search {json_path} has a different specification. "
                "Choose another --name or restore the original grid."
            )

    axis_names = list(axes)
    combinations = [
        dict(zip(axis_names, values))
        for values in itertools.product(*(axes[key] for key in axis_names))
    ]
    completed_ids = {result["combo_id"] for result in payload["results"]}
    pending = [combo for combo in combinations if _combo_id(combo) not in completed_ids]
    if args.shuffle:
        random.Random(args.seed).shuffle(pending)

    print(f"Grid search: {search_name}")
    print(f"Level: {args.level}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Pending: {len(pending)}")

    for index, combo in enumerate(pending, 1):
        combo_id = _combo_id(combo)
        run_name = f"grid_{search_name}_{combo_id}"
        overrides = [f"{key}={value}" for key, value in combo.items()]
        cmd = [
            "uv",
            "run",
            "swarmecho-train",
            f"level={args.level}",
            f"logging.run_name={run_name}",
            f"logging.wandb_group=grid_search_{search_name}",
            "logging.use_timestamp_postfix=false",
            *fixed,
            *overrides,
        ]
        if args.total_timesteps is not None:
            cmd.append(f"training.total_timesteps={args.total_timesteps}")

        print(f"\n[{index}/{len(pending)}] {run_name}")
        print("  " + " ".join(overrides))
        if args.dry_run:
            print("  " + subprocess.list2cmdline(cmd))
            continue

        started = time.perf_counter()
        try:
            output, timed_out, return_code = _run_command(cmd, args.timeout_seconds)
        except KeyboardInterrupt:
            print("\nGrid search interrupted; completed results were already saved.")
            return

        metrics = _parse_metrics(output)
        if timed_out:
            status = "TIMEOUT"
            details = f"Exceeded {_format_duration(args.timeout_seconds)}"
        elif return_code == 0:
            status = "SUCCESS"
            details = "Completed budget or early exit"
        else:
            status, details = _failure_status(output, return_code)

        payload["results"].append(
            {
                "combo_id": combo_id,
                "combo": combo,
                "run_name": run_name,
                "status": status,
                "details": details,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "metrics": metrics,
            }
        )
        _write_outputs(output_base, payload)
        print(f"{status}: {details}")

    if args.dry_run:
        print("\nDry run only; no result files were written.")
    else:
        _write_outputs(output_base, payload)
        print(f"\nResults: {output_base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
