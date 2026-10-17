"""CPU-only training supervisor; a failed CUDA process is never reused."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
import uuid

from swarmecho.core.config import load_level_cli
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


def _worker(connection, level, output_dir, recovery_id, retry):
    os.environ["SWARMECHO_RECOVERY_ID"] = recovery_id
    os.environ["SWARMECHO_RECOVERY_RETRY"] = "1" if retry else "0"
    try:
        # Importing the environment can initialize JAX, so even imports belong
        # in the disposable worker rather than in this supervisor.
        from swarmecho.training.train import train as train_once
        checkpoint, metrics = train_once(level, output_dir=output_dir)
        connection.send(("ok", checkpoint, metrics))
    except Exception as error:
        if is_out_of_memory(error):
            connection.send(("oom", f"{type(error).__name__}: {error}"))
            connection.close()
            sys.stdout.flush()
            sys.stderr.flush()
            # Bypass JAX/W&B shutdown hooks holding the failed allocator. The
            # OS reclaims the process's complete CUDA context before retry.
            os._exit(86)
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _attempt(level, output_dir, recovery_id, retry):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, level, output_dir, recovery_id, retry))
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

    layout = resolve_run_layout(level.logging, level.evaluation, media_dir="replays")
    level = replace(level, logging=replace(level.logging, run_name=layout.run_name, use_timestamp_postfix=False))
    destination = Path(output_dir).absolute() if output_dir is not None else layout.run_dir
    checkpoints = layout.checkpoint_dir if output_dir is None else destination / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    journal = destination / "oom_recovery.jsonl"
    recovery_id = uuid.uuid4().hex
    incoming = level.training
    # Retain a previous reduction when explicitly resuming this run manually.
    if incoming.ckpt_loading_mode == "resume" and journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                count = int(json.loads(line)["next_num_envs"])
            except (ValueError, KeyError, TypeError):
                continue
            level = replace(level, training=replace(level.training, num_envs=min(level.training.num_envs, count)))
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("Recovered environment count must divide into recurrent minibatches.")
    retry = False
    while True:
        status, *payload = _attempt(level, output_dir, recovery_id, retry)
        if status == "ok":
            checkpoint, metrics = payload
            metrics["effective_num_envs"] = level.training.num_envs
            return checkpoint, metrics
        if status != "oom":
            raise RuntimeError(payload[0])
        count = level.training.num_envs - 500
        if count < 1:
            raise RuntimeError(f"OOM recovery exhausted: cannot reduce {level.training.num_envs} environments by another 500. Last error: {payload[0]}")
        if count % level.training.num_minibatches:
            raise ValueError("Reducing environments by 500 would produce invalid recurrent minibatches.")
        checkpoint = _latest_checkpoint(checkpoints, recovery_id)
        training = replace(incoming, num_envs=count)
        if checkpoint is not None:
            training = replace(training, checkpoint_path=str(checkpoint), ckpt_loading_mode="resume", checkpoint_step_offset=None)
        row = {"utc": datetime.now(timezone.utc).isoformat(), "recovery_id": recovery_id,
               "run_name": layout.run_name, "agents": level.env.num_agents,
               "previous_num_envs": level.training.num_envs, "next_num_envs": count,
               "checkpoint": training.checkpoint_path, "mode": training.ckpt_loading_mode,
               "error": payload[0]}
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        terminal_print(f"[OOM] {payload[0]}")
        terminal_print(f"[OOM RETRY] {layout.run_name}: {level.training.num_envs:,} -> {count:,} environments; {training.ckpt_loading_mode}; checkpoint: {training.checkpoint_path or 'fresh initialization'}")
        level = replace(level, training=training)
        retry = True


def main():
    level = load_level_cli(sys.argv[1:])
    if level.logging.terminal_log_init:
        terminal_print("[INIT] Level and building loaded.")
    train(level)


if __name__ == "__main__":
    main()
