"""Keep terminal cancellation responsive while the evaluator runs native code."""

import os
import signal
import subprocess
import sys


def main() -> int:
    interrupted = False

    def cancel(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, cancel)
    try:
        posix = os.name == "posix"
        options = ({"start_new_session": True} if posix else
                   {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP})
        worker = subprocess.Popen([
            sys.executable, "-c",
            "from swarmecho.training.evaluate import _run_evaluation; _run_evaluation()",
            *sys.argv[1:],
        ], **options)
        while not interrupted:
            try:
                status = worker.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue
            if not interrupted:
                return status if status >= 0 else 128 - status

        print("\n[EVAL] Cancelling evaluation...", file=sys.stderr, flush=True)
        if posix:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        else:
            worker.terminate()
        worker.wait()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
