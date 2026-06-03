"""
training/multi_train.py
=======================
Run the same training config sequentially across N different random seeds.

Each run gets a unique, non-overlapping seed and a run-name postfix of
``_seed_1``, ``_seed_2``, ... ``_seed_N`` (where the number is the run
index, NOT the actual seed value).

Usage
-----
    uv run python training/multi_train.py level=M01 logging.run_name=maze01_v2 seeds=5

    # Specify the base seed used to generate the per-run seeds:
    uv run python training/multi_train.py level=M01 logging.run_name=maze01_v2 seeds=5 base_seed=42

    # Any other training/env overrides work as usual:
    uv run python training/multi_train.py level=A01 logging.run_name=warehouse seeds=3 \\
        training.total_timesteps=30000000 logging.wandb_mode=offline

Notes
-----
- ``seeds=N``      — how many sequential runs to launch (default: 3).
- ``base_seed=N``  — seed for the seed-generator itself (default: 42).
  Controls which batch of per-run seeds is produced.  Changing this gives
  you a completely different, reproducible set of seeds.
- Seeds are drawn from a shared random pool, guaranteed unique within the
  batch.  They are printed at startup so results are always reproducible.
- ``logging.run_name`` is required.  If not supplied, a default of
  ``multi_run`` is used.
- ``base_seed`` is a multi_train-only argument and does NOT appear in any
  individual run's config or checkpoint.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from core.config import load_config, validate_config
from training.runner import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pop_arg(args: list[str], key: str, default):
    """Remove ``key=value`` from args list and return (value, remaining_args)."""
    value = default
    remaining = []
    for arg in args:
        if arg.startswith(f"{key}="):
            try:
                value = type(default)(arg.split("=", 1)[1])
            except (ValueError, TypeError):
                value = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
    return value, remaining


def _generate_unique_seeds(n: int, base_seed: int) -> list[int]:
    """
    Draw N unique integers in [1, 2**31 - 1] using Python's random module
    seeded by *base_seed*, so the sequence is itself reproducible.
    """
    rng = random.Random(base_seed)
    seeds = set()
    while len(seeds) < n:
        seeds.add(rng.randint(1, 2**31 - 1))
    # Deterministic ordering: sort so the sequence is stable
    return sorted(seeds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_args = sys.argv[1:]

    # --- Pull out multi_train-specific args before passing to load_config ---
    num_seeds, raw_args = _pop_arg(raw_args, "seeds", 3)
    base_seed, raw_args = _pop_arg(raw_args, "base_seed", 42)

    # Load config once (without per-run overrides) to read the base run_name
    # and validate level= etc.  base_seed is already resolved above.
    print("\n══════════════════════════════════════════════════════")
    print("  SwarmEcho — multi_train  (loading base config)")
    print("══════════════════════════════════════════════════════")
    base_cfg = load_config(cli_overrides=True, overrides=raw_args)
    validate_config(base_cfg)

    base_run_name = str(base_cfg.logging.run_name) if base_cfg.logging.run_name else "multi_run"

    seeds = _generate_unique_seeds(num_seeds, base_seed)

    print(f"  Multi-run config:")
    print(f"    num_runs   : {num_seeds}")
    print(f"    base_seed  : {base_seed}  (seed-generator seed — set via base_seed=N)")
    print(f"    run seeds  : {seeds}")
    print(f"    base name  : {base_run_name}")
    print()

    # --- Sequential runs ---------------------------------------------------
    for run_idx, seed in enumerate(seeds, start=1):
        run_name = f"{base_run_name}_seed_{run_idx}"

        print(f"\n{'═'*54}")
        print(f"  Run {run_idx}/{num_seeds}  |  seed={seed}  |  name={run_name}")
        print(f"{'═'*54}\n")

        # Build a fresh config for this run by re-loading with the seed and
        # run_name injected as extra overrides.  This guarantees each call to
        # train() receives a completely independent, readonly config object.
        run_overrides = raw_args + [
            f"training.seed={seed}",
            f"logging.run_name={run_name}",
        ]
        cfg = load_config(cli_overrides=True, overrides=run_overrides)
        validate_config(cfg)

        train(cfg)

    print(f"\n{'═'*54}")
    print(f"  multi_train complete — {num_seeds} run(s) finished.")
    print(f"{'═'*54}\n")
