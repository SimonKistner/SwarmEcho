"""CPU-only training supervisor; a failed CUDA process is never reused."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

from swarmecho.core.config import load_level_cli, validate_oom_recovery
from swarmecho.core.terminal import terminal_print
from swarmecho.training.run_lifecycle import resolve_run_layout


def is_out_of_memory(error):
    """Recognize allocation failures, including asynchronous JAX device errors."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error).lower()
        if isinstance(error, MemoryError) or any(text in message for text in (
            "out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation",
            "cudnn_status_alloc_failed", "cublas_status_alloc_failed",
        )):
            return True
        error = error.__cause__ or error.__context__
    return False


PPO_PARAMETERS = ("num_envs", "num_steps", "num_epochs", "num_minibatches")


def ppo_settings(training):
    return {name: getattr(training, name) for name in PPO_PARAMETERS}


def next_oom_training(training):
    """Apply the selected memory adjustment without breaking recurrent minibatches."""
    policy = training.oom_recovery
    validate_oom_recovery(policy)
    current = getattr(training, policy.parameter)
    if policy.parameter == "optimizer_microbatches":
        batch = training.num_envs // training.num_minibatches
        target = max(current+1,math.ceil(current*policy.factor))
        value = next((n for n in range(target,batch+1) if batch%n==0),None)
        if value is None:
            raise RuntimeError("OOM recovery exhausted: one environment sequence per optimizer microbatch still does not fit.")
    elif policy.parameter == "num_minibatches":
        target = math.ceil(current * policy.factor)
        value = next((count for count in range(target, training.num_envs + 1)
                      if training.num_envs % count == 0), None)
        if value is None:
            raise RuntimeError("OOM recovery exhausted: no larger valid minibatch count remains.")
    else:
        value = (current - policy.decrement if policy.factor is None
                 else math.floor(current * policy.factor))
        if policy.parameter == "num_envs":
            if policy.factor is None and value % training.num_minibatches:
                raise ValueError("OOM environment decrement would produce invalid recurrent minibatches.")
            value -= value % training.num_minibatches
        if value < 1 or value >= current:
            raise RuntimeError(f"OOM recovery exhausted: cannot reduce {policy.parameter} from {current} further.")
    return replace(training, **{policy.parameter: value})


def next_memory_level(level, phase):
    """Keep scientific PPO settings fixed under the execution-only policy."""
    if level.training.oom_recovery.parameter != "optimizer_microbatches":
        return replace(level,training=next_oom_training(level.training))
    factor=level.training.oom_recovery.factor
    if phase == "validation":
        current=min(level.validation.parallel_envs,level.validation.num_episodes)
        if current<=1:
            raise RuntimeError("OOM recovery exhausted at one validation lane.")
        return replace(level,validation=replace(level.validation,parallel_envs=max(1,int(current/factor))))
    if phase == "rollout":
        current=level.training.rollout_batch_size or level.training.num_envs
        if current<=1:
            raise RuntimeError("OOM recovery exhausted at one rollout lane.")
        return replace(level,training=replace(level.training,rollout_batch_size=max(1,int(current/factor))))
    return replace(level,training=next_oom_training(level.training))


def execution_settings(level):
    return {"optimizer_microbatches":level.training.optimizer_microbatches,
        "rollout_batch_size":level.training.rollout_batch_size,
        "validation_parallel_envs":level.validation.parallel_envs}


def _worker(connection, level, output_dir, recovery_id, retry, recovery_started, requested_ppo):
    os.environ["SWARMECHO_RECOVERY_ID"] = recovery_id
    os.environ["SWARMECHO_RECOVERY_RETRY"] = "1" if retry else "0"
    try:
        # Importing the environment can initialize JAX, so even imports belong
        # in the disposable worker rather than in this supervisor.
        from swarmecho.training.train import train as train_once
        checkpoint, metrics = train_once(level, output_dir=output_dir,
                                        wall_clock_started=recovery_started,
                                        requested_ppo=requested_ppo)
        connection.send(("ok", checkpoint, metrics))
    except Exception as error:
        if is_out_of_memory(error):
            connection.send(("oom", f"{type(error).__name__}: {error}",getattr(error,"swarmecho_oom_phase","optimizer")))
            connection.close()
            sys.stdout.flush()
            sys.stderr.flush()
            # Bypass JAX/W&B shutdown hooks holding the failed allocator. The
            # OS reclaims the process's complete CUDA context before retry.
            os._exit(86)
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _attempt(level, output_dir, recovery_id, retry, recovery_started, requested_ppo):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(
        sender, level, output_dir, recovery_id, retry, recovery_started, requested_ppo))
    process.start()
    sender.close()
    try:
        while not receiver.poll(0.5):
            if not process.is_alive():
                break
        try:
            result = receiver.recv()
        except EOFError:
            result = ("error", f"Training worker exited with code {process.exitcode} without reporting an exception. No OOM retry was assumed.")
        process.join()
        return result
    except BaseException:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    finally:
        receiver.close()
        process.close()


def _latest_checkpoint(directory, recovery_id):
    candidates = []
    for path in directory.glob("ckpt_*"):
        marker = path / "recovery_ready.json"
        if not marker.is_file() or not (path / "step_history.json").is_file():
            continue
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data["recovery_id"] == recovery_id:
                candidates.append((int(data["update"]), path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def train(level, *, output_dir=None, checkpoint_path=None):
    """Route curricula or run one stage, retrying only recognized OOM errors."""
    if checkpoint_path is not None:
        level = replace(level, training=replace(level.training, checkpoint_path=str(checkpoint_path)))
    if level.training.agent_count_curriculum.enabled:
        from swarmecho.training.curriculum import train_agent_counts
        return train_agent_counts(level, output_dir=output_dir)
    if not level.training.oom_recovery.enabled:
        from swarmecho.training.train import train as train_once
        return train_once(level, output_dir=output_dir)

    recovery_started = time.perf_counter()
    requested_ppo = ppo_settings(level.training)
    validate_oom_recovery(level.training.oom_recovery)
    layout = resolve_run_layout(level.logging, level.evaluation, media_dir="replays", training=level.training)
    level = replace(level, logging=replace(level.logging, run_name=layout.run_name, use_timestamp_postfix=False))
    destination = Path(output_dir).absolute() if output_dir is not None else layout.run_dir
    checkpoints = layout.checkpoint_dir if output_dir is None else destination / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    journal = destination / "oom_recovery.jsonl"
    recovery_id = uuid.uuid4().hex
    # Retain a previous reduction when explicitly resuming this run manually.
    if level.training.ckpt_loading_mode == "resume" and journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                settings = row.get("next_ppo")
                if settings is None:
                    settings = {"num_envs": row["next_num_envs"]}
                if (not isinstance(settings, dict) or any(
                        name not in PPO_PARAMETERS or type(value) is not int or value < 1
                        for name, value in settings.items())):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            retained = {name: (max(getattr(level.training, name), value)
                               if name == "num_minibatches" else min(getattr(level.training, name), value))
                        for name, value in settings.items() if name != "num_epochs"}
            level = replace(level, training=replace(level.training, **retained))
            execution = row.get("next_execution",{})
            if execution:
                level=replace(level,training=replace(level.training,
                    optimizer_microbatches=max(level.training.optimizer_microbatches,execution["optimizer_microbatches"]),
                    rollout_batch_size=execution["rollout_batch_size"]),
                    validation=replace(level.validation,parallel_envs=execution["validation_parallel_envs"]))
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("Recovered environment count must divide into recurrent minibatches.")
    retry = False
    retries = 0
    while True:
        status, *payload = _attempt(level, output_dir, recovery_id, retry,
                                   recovery_started, requested_ppo)
        if status == "ok":
            checkpoint, metrics = payload
            return checkpoint, metrics
        if status != "oom":
            raise RuntimeError(payload[0])
        policy = level.training.oom_recovery
        elapsed = time.perf_counter() - recovery_started
        if level.training.max_wall_seconds is not None and elapsed >= level.training.max_wall_seconds:
            raise RuntimeError(f"OOM recovery exhausted the shared wall-time budget. Last error: {payload[0]}")
        clock = destination / "oom_recovery_clock.json"
        if level.training.max_training_seconds is not None and clock.is_file():
            timing = json.loads(clock.read_text(encoding="utf-8"))
            if (timing["recovery_id"] == recovery_id and
                    time.perf_counter() - timing["training_started"] >= level.training.max_training_seconds):
                raise RuntimeError(f"OOM recovery exhausted the shared training-time budget. Last error: {payload[0]}")
        if policy.max_retries is not None and retries >= policy.max_retries:
            raise RuntimeError(f"OOM recovery exhausted its {policy.max_retries} retries. Last error: {payload[0]}")
        phase = payload[1] if len(payload)>1 else "optimizer"
        next_level = next_memory_level(level,phase)
        training = next_level.training
        checkpoint = _latest_checkpoint(checkpoints, recovery_id)
        if checkpoint is not None:
            training = replace(training, checkpoint_path=str(checkpoint), ckpt_loading_mode="resume", checkpoint_step_offset=None)
        row = {"utc": datetime.now(timezone.utc).isoformat(), "recovery_id": recovery_id,
               "run_name": layout.run_name, "agents": level.env.num_agents,
               "previous_num_envs": level.training.num_envs, "next_num_envs": training.num_envs,
               "parameter": policy.parameter, "decrement": policy.decrement,
               "factor": policy.factor, "retry": retries + 1,
               "requested_ppo": requested_ppo,
               "previous_ppo": ppo_settings(level.training), "next_ppo": ppo_settings(training),
               "elapsed_wall_seconds": elapsed,
               "checkpoint": training.checkpoint_path, "mode": training.ckpt_loading_mode,
               "error": payload[0]}
        row.update(phase=phase,previous_execution=execution_settings(level),next_execution=execution_settings(next_level))
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        terminal_print(f"[OOM] {payload[0]}")
        terminal_print(f"[OOM RETRY] {layout.run_name}: {policy.parameter} "
                       f"{getattr(level.training, policy.parameter):,} -> {getattr(training, policy.parameter):,}; "
                       f"{training.ckpt_loading_mode}; checkpoint: {training.checkpoint_path or 'fresh initialization'}")
        terminal_print(f"[OOM EXECUTION] phase={phase}; {execution_settings(next_level)}; scientific PPO batch unchanged="
                       f"{ppo_settings(training)==requested_ppo}")
        level = replace(next_level, training=training)
        retry = True
        retries += 1


def main():
    level = load_level_cli(sys.argv[1:])
    if level.logging.terminal_log_init:
        terminal_print("[INIT] Level and building loaded.")
    train(level)


if __name__ == "__main__":
    main()
