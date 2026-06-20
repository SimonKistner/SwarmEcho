"""
training/grid_search_followup.py
================================
Sequentially launches SwarmEcho training runs for 6 specifically chosen 
followup combinations of num_envs and num_steps.
Terminates each run after a 30-minute wallclock timeout, logging and parsing 
the results to compare exploration performance (map coverage, target found).

Minibatch size is calculated dynamically at runtime for each run to keep 
the transitions per minibatch strictly under the VRAM safety limit (30,000).

MEM_SHARE_COMM_MERGES = ["residual", "concat"]
MEM_SHARE_COMM_ATTENTION_MODES = [
    "attend_global_learned_query",
    "attend_cur_obs_query",
    "attend_mem_query",
    "attend_cur_obs_and_mem_query",
]
MEM_SHARE_COMM_GRADIENT_MODES = ["rial", "dial"]

expected
expected 40M -> concat global rial
expected 20M -> concat obs+mem dial

"""

import sys
from pathlib import Path
import itertools
import subprocess
import time
import re
import random
import queue
import threading
import os
import math

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config

# --- Grid Parameters (Constants) ---
COMBINATIONS = [
    (2000, 50),
    (4000, 50),
    (2000, 100),
    (4000, 100),
    (2000, 200),
    (4000, 200),
]

TIMEOUT_SECONDS = None  # Disabled (no wallclock limit)

def calculate_optimal_minibatches(num_envs, num_steps, recurrent, max_transitions_per_mb=30000):
    total_batch_size = num_envs * num_steps
    if recurrent:
        min_mb = math.ceil(total_batch_size / max_transitions_per_mb)
        mb = max(1, min_mb)
        if mb > num_envs:
            mb = num_envs
        while num_envs % mb != 0:
            mb += 1
    else:
        min_mb = math.ceil(total_batch_size / max_transitions_per_mb)
        mb = max(1, min_mb)
        while total_batch_size % mb != 0:
            mb += 1
    return mb

def shorten_num(val):
    if val >= 1000:
        return f"{val//1000}k" if val % 1000 == 0 else f"{val/1000:.1f}k"
    return str(val)

def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def enqueue_output(out, q):
    for line in iter(out.readline, ''):
        q.put(line)
    out.close()

def run_command_realtime_logging(cmd, timeout_seconds):
    # Set PYTHONUNBUFFERED=1 environment variable to force real-time flushing
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    t.daemon = True
    t.start()
    
    output_lines = []
    start_time = time.perf_counter()
    timed_out = False
    
    try:
        while True:
            try:
                line = q.get(timeout=1.0)
                sys.stdout.write(line)
                sys.stdout.flush()
                output_lines.append(line)
            except queue.Empty:
                # Check if the process has terminated naturally
                if process.poll() is not None:
                    break
                    
            elapsed = time.perf_counter() - start_time
            if timeout_seconds is not None and elapsed > timeout_seconds:
                timed_out = True
                print(f"\n⚠️ Run exceeded timeout limit of {format_duration(timeout_seconds)}. Terminating...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("⚠️ Process did not exit cleanly. Force killing...")
                    process.kill()
                break
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt caught. Terminating child process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️ Process did not exit cleanly on interrupt. Force killing...")
            process.kill()
        raise
            
    # Drain any remaining output from the queue
    while not q.empty():
        try:
            line = q.get_nowait()
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
        except queue.Empty:
            break
            
    return_code = process.poll()
    full_output = "".join(output_lines)
    
    return full_output, timed_out, return_code

def extract_oom_size(output_text: str) -> str | None:
    pattern = r"(?:trying to allocate|allocate)\s+([\d\.]+\s*(?:GiB|MiB|KiB|B|GB|MB|KB))"
    match = re.search(pattern, output_text, re.IGNORECASE)
    if match:
        return match.group(1)
    if "RESOURCE_EXHAUSTED" in output_text or "Out of memory" in output_text:
        return "Unknown Size"
    return None

def parse_metrics(output_text: str):
    stats = {
        "steps": 0,
        "sps": 0.0,
        "ep_return": "----",
        "cov": "----",
        "found": "----",
        "chain": "---%",
        "succ": "----"
    }
    
    lines = output_text.splitlines()
    for line in reversed(lines):
        if "steps=" in line and "sps=" in line:
            m_steps = re.search(r"steps=\s*([\d,]+)", line)
            if m_steps:
                stats["steps"] = int(m_steps.group(1).replace(",", ""))
            m_sps = re.search(r"sps=\s*([\d,]+)", line)
            if m_sps:
                stats["sps"] = float(m_sps.group(1).replace(",", ""))
            m_cov = re.search(r"cov=\s*([^\s]+)", line)
            if m_cov:
                stats["cov"] = m_cov.group(1)
            m_found = re.search(r"found=\s*([^\s]+)", line)
            if m_found:
                stats["found"] = m_found.group(1)
            m_chain = re.search(r"chain=\s*([^\s]+)", line)
            if m_chain:
                stats["chain"] = m_chain.group(1)
            m_succ = re.search(r"succ=\s*([^\s]+)", line)
            if m_succ:
                stats["succ"] = m_succ.group(1)
            break
            
    for line in reversed(lines):
        if "[eval]" in line and "ep_return=" in line:
            m_ret = re.search(r"ep_return=\s*([-\d\.]+)", line)
            if m_ret:
                stats["ep_return"] = m_ret.group(1)
            break
            
    return stats

def write_summary_markdown(level, results):
    def get_sort_key(res):
        try:
            cov_str = res["metrics"]["cov"]
            cov_val = float(cov_str.replace("%", "")) if cov_str != "----" else -1.0
        except ValueError:
            cov_val = -1.0
            
        is_oom = (res["status"] == "OOM")
        oom_bytes = 0.0
        if is_oom:
            details = res["details"]
            match = re.search(r"Allocation failed:\s*([\d\.]+)\s*(GiB|MiB|KiB|B|GB|MB|KB)", details, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                unit = match.group(2).lower()
                if "g" in unit:
                    oom_bytes = val * 1024 * 1024 * 1024
                elif "m" in unit:
                    oom_bytes = val * 1024 * 1024
                elif "k" in unit:
                    oom_bytes = val * 1024
                else:
                    oom_bytes = val
            else:
                oom_bytes = 999999999999.0
                
        return (cov_val, -1.0 if is_oom else 0.0, -oom_bytes)
        
    sorted_results = sorted(results, key=get_sort_key, reverse=True)
    summary_path = Path("outputs") / f"grid_search_followup_{level}_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Followup Grid Search Results for Level: {level}\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Runs are sorted by **Map Coverage** (descending). OOMs at the bottom are sorted by allocation size (ascending).\n\n")
        f.write("| Rank | Run Name | num_envs | num_steps | num_epochs | num_minibatches | Status | Steps | SPS | Map Coverage | Target Found | Success Rate | Eval Return | Details |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for rank, res in enumerate(sorted_results, 1):
            run_name = res["run_name"]
            envs = res["envs"]
            steps = res["steps"]
            epochs = res["epochs"]
            minibatches = res["minibatches"]
            status = res["status"]
            steps_done = f"{res['metrics']['steps']:,}"
            sps = f"{res['metrics']['sps']:,.0f}" if res['metrics']['sps'] > 0 else "----"
            cov = res["metrics"]["cov"]
            found = res["metrics"]["found"]
            succ = res["metrics"]["succ"]
            ret = res["metrics"]["ep_return"]
            details = res["details"]
            
            f.write(f"| {rank} | `{run_name}` | {envs} | {steps} | {epochs} | {minibatches} | **{status}** | {steps_done} | {sps} | {cov} | {found} | {succ} | {ret} | {details} |\n")

def load_existing_results(level):
    results = []
    summary_path = Path("outputs") / f"grid_search_followup_{level}_summary.md"
    if not summary_path.exists():
        return results
        
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line.startswith("|") or "Rank" in line or set(line.replace(" ", "")).issubset({"|", "-", ":"}):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 15:
                continue
            
            run_name = parts[2].strip("`")
            if not run_name.startswith(f"grid_{level}_followup_"):
                continue
            
            try:
                envs = int(parts[3])
                steps = int(parts[4])
                epochs = int(parts[5])
                minibatches = int(parts[6])
            except ValueError:
                continue
                
            status = parts[7].strip("*")
            
            raw_steps = parts[8].replace(",", "")
            steps_done = int(raw_steps) if raw_steps.isdigit() else 0
            
            raw_sps = parts[9].replace(",", "")
            sps = float(raw_sps) if raw_sps.replace(".", "", 1).isdigit() else 0.0
            
            cov = parts[10]
            found = parts[11]
            succ = parts[12]
            ret = parts[13]
            details = parts[14]
            
            results.append({
                "run_name": run_name,
                "envs": envs,
                "steps": steps,
                "epochs": epochs,
                "minibatches": minibatches,
                "status": status,
                "details": details,
                "metrics": {
                    "steps": steps_done,
                    "sps": sps,
                    "ep_return": ret,
                    "cov": cov,
                    "found": found,
                    "chain": "---%",
                    "succ": succ
                }
            })
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse existing summary file: {e}. Starting fresh.")
        results = []
        
    return results

def print_progress_bar(completed, total, elapsed_time, session_completed):
    percent = (completed / total) * 100
    bar_length = 20
    filled_length = int(bar_length * completed // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    
    if session_completed > 0:
        avg_time = elapsed_time / session_completed
        remaining = total - completed
        eta_seconds = avg_time * remaining
        eta_str = format_duration(eta_seconds)
    else:
        eta_str = "----"
    
    print("\n" + "=" * 80)
    print(f"FOLLOWUP PROGRESS: [{bar}] {percent:.1f}% ({completed}/{total} Runs Completed)")
    print(f"Session Elapsed Time: {format_duration(elapsed_time)} | ETA: {eta_str}")
    print("=" * 80 + "\n")

def run_benchmarks():
    level = "M02"
    extra_args = []
    if len(sys.argv) > 1:
        if "=" not in sys.argv[1]:
            level = sys.argv[1]
            extra_args = sys.argv[2:]
        else:
            extra_args = sys.argv[1:]

    # Load level configuration once to check for recurrent settings and get epochs
    try:
        level_cfg = load_config(overrides=[f"level={level}"], cli_overrides=True)
        recurrent = bool(level_cfg.network.actor_memory) or bool(level_cfg.network.critic_memory)
        epochs = level_cfg.training.num_epochs
    except Exception as e:
        print(f"⚠️ Warning: Could not load config for level {level}: {e}. Defaulting to non-recurrent and epochs=2.")
        recurrent = False
        epochs = 2

    # Calculate optimal minibatches for our combinations
    combinations = []
    for envs, steps in COMBINATIONS:
        mb = calculate_optimal_minibatches(envs, steps, recurrent)
        combinations.append((envs, steps, mb))

    total_runs = len(combinations)
    
    # Load already-completed results to support resume functionality
    results = load_existing_results(level)
    completed_combos = {(r["envs"], r["steps"], r["minibatches"]) for r in results}
    
    # Filter remaining combinations
    remaining_combinations = [c for c in combinations if (c[0], c[1], c[2]) not in completed_combos]
    
    # We do NOT shuffle these to keep the order clean and predictable
    
    print(f"🚀 Starting/Resuming followup grid search for Level: {level}")
    print(f"   Level Epochs (from config)       : {epochs}")
    print(f"   Recurrent Memory mode            : {recurrent}")
    print(f"   Total combinations to test       : {total_runs}")
    print(f"   Completed in previous runs       : {len(completed_combos)}")
    print(f"   Remaining runs to evaluate       : {len(remaining_combinations)}")
    print("=" * 80)
    
    if len(remaining_combinations) == 0:
        print("✅ All combinations have already been completed!")
        return
        
    session_completed = 0
    start_session_time = time.perf_counter()
    
    for envs, steps, minibatches in remaining_combinations:
        envs_str = shorten_num(envs)
        steps_str = shorten_num(steps)
        mb_str = shorten_num(minibatches)
        
        run_name = f"grid_{level}_followup_env{envs_str}_step{steps_str}_mb{mb_str}"
        
        current_idx = len(results) + 1
        print(f"\n[{current_idx}/{total_runs}] Launching run: {run_name}")
        print(f"          Environments : {envs:<5} | Steps : {steps:<4} | Minibatches : {minibatches}")
        print("-" * 80)
        
        # Build training CLI command
        # Epochs is not passed so it falls back to the level's yaml value.
        cmd = [
            "uv", "run", "python", "-u", "src/training/train.py",
            f"level={level}",
            f"training.num_envs={envs}",
            f"training.num_steps={steps}",
            f"training.num_minibatches={minibatches}",
            "training.total_timesteps=50000000",  # prevent natural exit
            f"logging.run_name={run_name}",
            f"logging.wandb_group=grid_search_{level}_followup",
            "logging.use_timestamp_postfix=False"
        ]
        
        cmd.extend(extra_args)
        
        status = "UNKNOWN"
        details = "-"
        metrics = {
            "steps": 0,
            "sps": 0.0,
            "ep_return": "----",
            "cov": "----",
            "found": "----",
            "chain": "---%",
            "succ": "----"
        }
        
        try:
            full_output, timed_out, return_code = run_command_realtime_logging(cmd, TIMEOUT_SECONDS)
            metrics = parse_metrics(full_output)
            
            if timed_out:
                status = "TIMEOUT"
                details = f"Reached {format_duration(TIMEOUT_SECONDS)} wallclock limit"
                print(f"✅ {run_name} timed out after the limit.")
            elif return_code == 0:
                status = "SUCCESS"
                details = "Completed budget or early exit"
                print(f"✅ {run_name} completed successfully.")
            else:
                oom_size = extract_oom_size(full_output)
                if oom_size:
                    status = "OOM"
                    details = f"Allocation failed: {oom_size}"
                    print(f"❌ {run_name} failed with Out Of Memory. Size: {oom_size}")
                elif "ValueError" in full_output:
                    status = "INVALID_CONFIG"
                    details = "Validation ValueError"
                    print(f"❌ {run_name} failed with config validation error.")
                else:
                    status = "FAILED"
                    details = f"Exit code {return_code}"
                    print(f"❌ {run_name} crashed (exit code {return_code}).")
                    
        except KeyboardInterrupt:
            print("\n🛑 Followup search manually interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            status = "ERROR"
            details = str(e)
            print(f"❌ {run_name} encountered execution error: {e}")
            
        results.append({
            "run_name": run_name,
            "envs": envs,
            "steps": steps,
            "epochs": epochs,
            "minibatches": minibatches,
            "status": status,
            "details": details,
            "metrics": metrics
        })
        
        session_completed += 1
        
        # Update Markdown table in real-time
        write_summary_markdown(level, results)
        
        # Print progress bar and ETA
        elapsed_session = time.perf_counter() - start_session_time
        print_progress_bar(len(results), total_runs, elapsed_session, session_completed)

if __name__ == "__main__":
    run_benchmarks()
