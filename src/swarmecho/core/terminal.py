"""Shared timestamp and column formatting for training console output."""

import builtins
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
import re

_INIT_ENABLED = ContextVar("terminal_init_enabled", default=False)
_TIMESTAMP = re.compile(r"^\[(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}:\d{2}\]\s*")


def terminal_print(*values, sep=" ", end="\n", flush=True):
    message = sep.join(str(value) for value in values)
    for line in message.splitlines():
        builtins.print(f"[{datetime.now():%H:%M:%S}] " + _TIMESTAMP.sub("", line), end=end, flush=flush)


@contextmanager
def init_logging(enabled):
    token = _INIT_ENABLED.set(enabled)
    try:
        yield
    finally:
        _INIT_ENABLED.reset(token)


def init_message(message):
    if _INIT_ENABLED.get():
        terminal_print(f"[INIT] {message}")


def display_path(path):
    """Display project-relative paths, retaining absolute external paths."""
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)


def evaluation_row(metrics, *, label, update, total, episodes, tail=""):
    # Replace the training update counter with the evaluation label. The longer
    # par_envs label borrows five spaces from the otherwise blank steps column;
    # its numeric field and every metric then align with the unchanged train row.
    counter_width = len(f"[{update:>4}/{total}]") if total else 11
    prefix_width = counter_width + 2 + 18 + 2 + 10
    throughput = f"envs={episodes:>6,}"
    prefix = f"{label:<{prefix_width - len(throughput)}}{throughput}"
    terminal_print(
        f" {prefix}  ep_len={metrics['eval_episode_length']:>6.0f}  "
        f"cov={metrics['eval_coverage']:>5.1%}  "
        f"found={metrics['eval_target_found_rate']:>5.1%}  "
        f"chain={metrics['eval_chain_progress_pct']:>5.1f}%  "
        f"succ={metrics['eval_success']:>5.1%}  {tail}"
    )
