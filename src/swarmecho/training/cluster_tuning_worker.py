"""Disposable single-GPU tuning worker. The controller never imports JAX."""
from __future__ import annotations

import json
import errno
import math
import os
from pathlib import Path
import sys
import time
import traceback

from swarmecho.training.cluster_tuning import read_json, save_json


class TrialStop(Exception):
    def __init__(self, status):
        self.status = status


class SetupError(RuntimeError):
    pass


def main():
    config_path = Path(sys.argv[1])
    spec = read_json(config_path)
    folder = config_path.parent
    started = float(os.environ["SWARMECHO_TRIAL_STARTED_MONOTONIC"])
    result = {"status": "error", "candidate": spec["candidate"], "evaluations": []}
    last_train = {}
    device = None
    peak_bytes = 0

    def observe(row):
        nonlocal last_train, peak_bytes
        import jax
        elapsed = time.monotonic() - started
        memory = device.memory_stats() or {}
        peak_bytes = max(peak_bytes, int(memory.get("peak_bytes_in_use", memory.get("bytes_in_use", 0))))
        row = {**row, "wall_seconds": elapsed, "device_memory": memory}
        if row["phase"] == "training":
            last_train = row
            for key in ("policy_loss", "value_loss", "approx_kl"):
                if not math.isfinite(float(row[key])):
                    raise TrialStop("nonfinite")
        elif row["phase"] == "evaluation":
            result["evaluations"].append(row)
            if row.get("converged"):
                result["time_to_95_seconds"] = elapsed
        result.update(progress=row, last_train=last_train, peak_device_bytes=peak_bytes,
                      allocator_limit_bytes=int(memory.get("bytes_limit", 0)))
        save_json(folder / "progress.json", result)
        with (folder / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if spec["wandb_mode"] != "disabled":
            import wandb
            if wandb.run is not None:
                # Match ordinary training: global environment steps remain the
                # default x-axis. Wall time is available as an optional metric.
                wandb.log({"perf/wall_seconds": elapsed,
                           "tuning/device_peak_gib": peak_bytes / 2**30},
                          step=int(row["global_step"]), commit=False)
                if result.get("time_to_95_seconds") is not None or row["phase"] == "complete":
                    wandb.run.summary.update({
                        "tuning/status": "converged" if "time_to_95_seconds" in result else "max_steps",
                        "tuning/time_to_95_seconds": result.get("time_to_95_seconds"),
                        "tuning/wall_seconds": elapsed})
        if spec["mode"] == "probe" and row["phase"] == "training" and row["update"] >= spec["probe_updates"]:
            raise TrialStop("probe_ok")
        request = read_json(folder / "stop.json", {})
        if request and "time_to_95_seconds" not in result and row["phase"] != "complete":
            raise TrialStop(request["status"])

    try:
        import jax
        from swarmecho.core.config import load_level
        try:
            devices = jax.devices("gpu")
        except RuntimeError as error:
            raise SetupError("JAX could not initialize the GPU backend: " + str(error)) from error
        if len(devices) != 1:
            raise SetupError(f"Tuning requires exactly one visible GPU; found {devices}.")
        device = devices[0]
        result["device"] = {"kind": device.device_kind, "jax": jax.__version__,
                            "platform": device.platform, "python": sys.version.split()[0]}
        jax.device_put(0, device).block_until_ready()
        limit = int((device.memory_stats() or {}).get("bytes_limit", 0))
        expected = spec.get("expected_runtime")
        if expected and (expected["device"] != result["device"] or expected["allocator_limit_bytes"] != limit):
            raise SetupError("GPU runtime or allocator limit differs from the capacity probe (including MIG slice size). Rerun the probe.")
        save_json(folder / "progress.json", result)
        overrides = [*spec["overrides"],
                     *[f"training.{key}={value}" for key, value in spec["candidate"].items()],
                     "training.checkpoint_path=null", "training.oom_recovery.enabled=false",
                     "training.agent_count_curriculum.enabled=false",
                     "training.max_wall_seconds=null",
                     "training.adaptive_target_spawn.curriculum_start_agents=null",
                     "inspection_evaluation.target_sampling=independent",
                     "validation.enabled=false",
                     "logging.run_name=" + spec["run_name"],
                     "logging.log_dir=" + spec["outputs"],
                     "logging.use_timestamp_postfix=false",
                     "logging.wandb_group=" + spec["group"],
                     "logging.wandb_mode=" + spec["wandb_mode"]]
        if spec["mode"] == "probe":
            overrides += ["adapt_size.initial_agents=7", "env.spawn_delay=0",
                          "adapt_size.decommission_hold_steps=701",
                          "inspection_evaluation.save_model=false",
                          "inspection_evaluation.eval_video=false",
                          "inspection_evaluation.eval_every_env_steps=2147483647",
                          "inspection_evaluation.early_exit=false",
                          "training.total_timesteps=" + str(
                              spec["candidate"]["num_envs"] * spec["candidate"]["num_steps"]
                              * (spec["probe_updates"] + 1))]
        level = load_level("G01_CLU_static", overrides=overrides)
        if (level.training.gamma, level.training.gae_lambda) != (0.99, 0.95):
            raise SetupError("Cluster tuning fixes gamma=0.99 and gae_lambda=0.95.")
        if level.random_buildings.enabled or level.validation.enabled or len(level.map_names) != 1:
            raise SetupError("Cluster tuning requires one authored map and no held-out validation.")
        from swarmecho.training.train import train
        checkpoint, metrics = train(level, on_progress=observe)
        result.update(status="converged" if "time_to_95_seconds" in result else "max_steps",
                      checkpoint=str(checkpoint) if checkpoint else None, metrics=metrics)
    except TrialStop as stop:
        result["status"] = stop.status
    except Exception as error:
        message = traceback.format_exc()
        result["failure_kind"] = (
            "setup_error" if isinstance(error, SetupError) else
            "storage_error" if isinstance(error, OSError) and error.errno in
                {errno.ENOSPC, errno.EIO, errno.EROFS, errno.EACCES, errno.EPERM} else
            "logging_error" if type(error).__module__.startswith("wandb") or
                any(token in message.lower() for token in ("wandb.errors", "wandb.sdk")) else
            "worker_error")
        oom = isinstance(error, MemoryError) or any(
            word in message.lower() for word in ("out of memory", "cuda_error_out_of_memory",
                                                "cudaerrormemoryallocation", "cublas_status_alloc_failed",
                                                "cudnn_status_alloc_failed"))
        result.update(status="oom" if oom and result["failure_kind"] == "worker_error" else "error", error=message)
        print(message, flush=True)
    finally:
        if device is not None:
            try:
                memory = device.memory_stats() or {}
                result.update(peak_device_bytes=max(peak_bytes, int(memory.get("peak_bytes_in_use", 0))),
                              allocator_limit_bytes=int(memory.get("bytes_limit", 0)))
            except Exception:
                pass # Device may already be lost; retain the last successful measurement.
        result["wall_seconds"] = time.monotonic() - started
        save_json(folder / "result.json", result)
        # Explicitly close pruned runs too; hard deadlines remain the parent's job.
        if spec["wandb_mode"] != "disabled":
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary.update({"tuning/status": result["status"],
                                             "tuning/wall_seconds": result["wall_seconds"],
                                             "tuning/time_to_95_seconds": result.get("time_to_95_seconds")})
                    wandb.log({}, commit=True)
                    wandb.finish(exit_code=0 if result["status"] in
                                 {"converged", "max_steps", "pruned"} else 1)
            except Exception as error:
                # A failed dashboard shutdown must not overwrite a real training result.
                result["logging_warning"] = f"{type(error).__name__}: {error}"
                save_json(folder / "result.json", result)
        if result["status"] == "oom":
            # Failed CUDA allocator state must never be reused or retained by hooks.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(86)


if __name__ == "__main__":
    main()
