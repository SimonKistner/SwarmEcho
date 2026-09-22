"""Sequential multi-seed training entry point for the trainer."""

from __future__ import annotations

import sys

from swarmecho.core.config import load_level_cli
from swarmecho.training.orchestration import (
    explicit_override,
    generate_unique_seeds,
    pop_arg,
)
from swarmecho.training.train import train


def main() -> None:
    """Train one level sequentially across reproducible random seeds."""
    raw_args = sys.argv[1:]
    count, raw_args = pop_arg(raw_args, "seeds", 3)
    base_seed, raw_args = pop_arg(raw_args, "base_seed", 42)

    base_level = load_level_cli(raw_args)
    base_run_name = base_level.logging.run_name or "multi_run"
    seeds = generate_unique_seeds(count, base_seed)
    explicit_group = explicit_override(raw_args, "logging.wandb_group")

    print("\n" + "=" * 60)
    print("  SwarmEcho multi-train")
    print("=" * 60)
    print(f"  level      : {base_level.name}")
    print(f"  num_runs   : {count}")
    print(f"  base_seed  : {base_seed}")
    print(f"  run seeds  : {seeds}")
    print(f"  base name  : {base_run_name}")
    print("=" * 60)

    for run_index, seed in enumerate(seeds, start=1):
        run_name = f"{base_run_name}_seed_{run_index}"
        run_overrides = raw_args + [
            f"training.seed={seed}",
            f"logging.run_name={run_name}",
        ]
        if base_level.logging.wandb_mode != "disabled" and explicit_group is None:
            run_overrides.append(f"logging.wandb_group={base_run_name}")
        level = load_level_cli(run_overrides)

        print(f"\n  Run {run_index}/{count} | seed={seed} | name={run_name}")
        train(level)

    print(f"\nmulti-train complete: {count} run(s) finished.")


if __name__ == "__main__":
    main()
